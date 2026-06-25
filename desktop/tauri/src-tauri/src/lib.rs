use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Mutex, OnceLock},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

const FLASK_URL: &str = "http://127.0.0.1:5000";
const FLASK_HOST_PORT: &str = "127.0.0.1:5000";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(90);

static FLASK_PROCESS: OnceLock<Mutex<Option<Child>>> = OnceLock::new();
static STARTUP_LOGS: OnceLock<StartupLogs> = OnceLock::new();
static BACKEND_ROOT: OnceLock<PathBuf> = OnceLock::new();

#[derive(Clone)]
struct StartupLogs {
    stdout: PathBuf,
    stderr: PathBuf,
}

struct PythonCommand {
    program: String,
    args: Vec<String>,
}

impl PythonCommand {
    fn display(&self) -> String {
        std::iter::once(self.program.as_str())
            .chain(self.args.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            match ensure_flask_ready(app) {
                Ok(()) => create_main_window(app)?,
                Err(error) => create_error_window(app, &error.to_string())?,
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                if window.app_handle().webview_windows().len() <= 1 {
                    stop_managed_flask();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running QuantAgent desktop app");
}

fn ensure_flask_ready(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let resource_dir = app.path().resource_dir()?;
    let bundled_backend = bundled_backend_root(&resource_dir)?;
    let expected_backend_root = bundled_backend.join("backend");
    let _ = BACKEND_ROOT.set(expected_backend_root.clone());

    if is_expected_flask_ready(&expected_backend_root) {
        return Ok(());
    }

    start_flask(app)?;
    wait_for_flask(&expected_backend_root)
}

fn start_flask(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let resource_dir = app.path().resource_dir()?;
    let bundled_backend = bundled_backend_root(&resource_dir)?;
    let backend_root = bundled_backend.join("backend");
    let site_packages = bundled_backend.join("python/site-packages");
    let python = python_command(&bundled_backend)?;
    let script = backend_root.join("web_interface.py");

    if !backend_root.exists() {
        return Err(format!("Bundled backend not found: {}", backend_root.display()).into());
    }

    if !script.exists() {
        return Err(format!("Flask entry file not found: {}", script.display()).into());
    }

    if !site_packages.exists() {
        return Err(format!(
            "Bundled Python dependencies not found: {}",
            site_packages.display()
        )
        .into());
    }

    let _ = BACKEND_ROOT.set(backend_root.clone());
    let logs = startup_logs()?;
    let stdout = File::create(&logs.stdout)?;
    let stderr = File::create(&logs.stderr)?;
    let python_path = std::env::join_paths([&site_packages, &backend_root])?;
    let mut command = Command::new(&python.program);
    command.args(&python.args);
    let child = command
        .arg(&script)
        .current_dir(&backend_root)
        .env("PYTHONPATH", python_path)
        .env("MPLCONFIGDIR", log_dir()?.join("matplotlib"))
        .env("QUANTAGENT_DESKTOP", "1")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|error| {
            format!(
                "Failed to start Flask with {} {}: {error}",
                python.display(),
                script.display()
            )
        })?;

    let _ = STARTUP_LOGS.set(logs);
    let process = FLASK_PROCESS.get_or_init(|| Mutex::new(None));
    *process.lock().expect("Flask process lock poisoned") = Some(child);

    Ok(())
}

fn wait_for_flask(expected_backend_root: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let started_at = Instant::now();

    while started_at.elapsed() < STARTUP_TIMEOUT {
        if is_expected_flask_ready(expected_backend_root) {
            return Ok(());
        }

        if let Some(process) = FLASK_PROCESS.get() {
            let mut guard = process.lock().expect("Flask process lock poisoned");
            if let Some(child) = guard.as_mut() {
                if let Some(status) = child.try_wait()? {
                    return Err(format!(
                        "Flask exited before {FLASK_URL} became ready.\nStatus: {status}\n{}",
                        log_summary()
                    )
                    .into());
                }
            }
        }

        thread::sleep(Duration::from_millis(400));
    }

    Err(format!("Timed out waiting for {FLASK_URL}.\n{}", log_summary()).into())
}

fn is_expected_flask_ready(expected_backend_root: &PathBuf) -> bool {
    let Some(body) = request_flask_health() else {
        return false;
    };
    let expected = expected_backend_root.display().to_string();
    let escaped_expected = expected.replace('\\', "\\\\");
    body.contains("\"status\":\"ok\"")
        && body.contains("\"desktop\":true")
        && (body.contains(&expected) || body.contains(&escaped_expected))
}

fn request_flask_health() -> Option<String> {
    FLASK_HOST_PORT
        .to_socket_addrs()
        .ok()
        .into_iter()
        .flatten()
        .find_map(|addr| {
            let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(300)) else {
                return None;
            };

            let _ = stream.set_read_timeout(Some(Duration::from_millis(1200)));
            let _ = stream.set_write_timeout(Some(Duration::from_millis(1200)));

            if stream
                .write_all(b"GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:5000\r\nConnection: close\r\n\r\n")
                .is_err()
            {
                return None;
            }

            let mut response = String::new();
            stream.read_to_string(&mut response).ok()?;

            if response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200") {
                Some(response)
            } else {
                None
            }
        })
}

fn create_main_window(app: &mut tauri::App) -> tauri::Result<()> {
    WebviewWindowBuilder::new(
        app,
        "main",
        WebviewUrl::External(FLASK_URL.parse().expect("valid Flask URL")),
    )
    .title("QuantAgent")
    .inner_size(1280.0, 860.0)
    .min_inner_size(1024.0, 700.0)
    .resizable(true)
    .build()?;

    Ok(())
}

fn create_error_window(app: &mut tauri::App, message: &str) -> tauri::Result<()> {
    let error_file =
        write_error_page(message).map_err(|error| tauri::Error::Anyhow(error.into()))?;
    let error_url = format!("file://{}", error_file.display())
        .parse()
        .expect("valid error file URL");

    WebviewWindowBuilder::new(app, "startup-error", WebviewUrl::External(error_url))
        .title("QuantAgent startup error")
        .inner_size(920.0, 620.0)
        .min_inner_size(720.0, 480.0)
        .resizable(true)
        .build()?;

    Ok(())
}

fn write_error_page(message: &str) -> std::io::Result<PathBuf> {
    let path = std::env::temp_dir().join("quantagent-tauri-startup-error.html");
    let logs = STARTUP_LOGS.get();
    let stdout = logs
        .map(|logs| logs.stdout.display().to_string())
        .unwrap_or_default();
    let stderr = logs
        .map(|logs| logs.stderr.display().to_string())
        .unwrap_or_default();
    let html = format!(
        r#"<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>QuantAgent startup error</title>
    <style>
      body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #15171a; }}
      main {{ max-width: 820px; margin: 56px auto; padding: 0 28px; }}
      h1 {{ font-size: 28px; margin: 0 0 16px; }}
      p {{ line-height: 1.6; }}
      pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #fff; border: 1px solid #d8dde5; border-radius: 8px; padding: 16px; }}
      .meta {{ color: #4d5662; }}
    </style>
  </head>
  <body>
    <main>
      <h1>QuantAgent 启动失败</h1>
      <p>桌面端没有等到 Flask 在 <strong>{url}</strong> 就绪。</p>
      <pre>{message}</pre>
      <p class="meta">项目路径：{project_root}</p>
      <p class="meta">stdout 日志：{stdout}</p>
      <p class="meta">stderr 日志：{stderr}</p>
    </main>
  </body>
</html>"#,
        url = FLASK_URL,
        message = escape_html(message),
        project_root = escape_html(&backend_root_display()),
        stdout = escape_html(&stdout),
        stderr = escape_html(&stderr),
    );

    fs::write(&path, html)?;
    Ok(path)
}

fn startup_logs() -> std::io::Result<StartupLogs> {
    let dir = log_dir()?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(StartupLogs {
        stdout: dir.join(format!("flask-{stamp}.stdout.log")),
        stderr: dir.join(format!("flask-{stamp}.stderr.log")),
    })
}

fn log_dir() -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join("quantagent-tauri-logs");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn log_summary() -> String {
    STARTUP_LOGS
        .get()
        .map(|logs| {
            format!(
                "stdout log: {}\nstderr log: {}",
                logs.stdout.display(),
                logs.stderr.display()
            )
        })
        .unwrap_or_else(|| "No Flask logs were created.".to_string())
}

fn backend_root_display() -> String {
    BACKEND_ROOT
        .get()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "Bundled backend was not resolved.".to_string())
}

fn python_command(bundled_backend: &PathBuf) -> Result<PythonCommand, Box<dyn std::error::Error>> {
    let version_file = bundled_backend.join("python/version.txt");
    let required_version = fs::read_to_string(&version_file)
        .map_err(|error| format!("Could not read {}: {error}", version_file.display()))?
        .trim()
        .to_string();

    let mut candidates = Vec::new();

    #[cfg(target_os = "windows")]
    {
        candidates.push(PythonCommand {
            program: bundled_backend
                .join("python-runtime/python.exe")
                .display()
                .to_string(),
            args: vec![],
        });
        candidates.push(PythonCommand {
            program: "py".to_string(),
            args: vec![format!("-{required_version}")],
        });
        candidates.push(PythonCommand {
            program: "python".to_string(),
            args: vec![],
        });
        candidates.push(PythonCommand {
            program: "python3".to_string(),
            args: vec![],
        });
    }

    #[cfg(target_os = "macos")]
    {
        for program in [
            format!("/Library/Frameworks/Python.framework/Versions/{required_version}/bin/python{required_version}"),
            format!("/opt/homebrew/bin/python{required_version}"),
            format!("/usr/local/bin/python{required_version}"),
            format!("python{required_version}"),
            "python3".to_string(),
        ] {
            candidates.push(PythonCommand {
                program,
                args: vec![],
            });
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        for program in [format!("python{required_version}"), "python3".to_string()] {
            candidates.push(PythonCommand {
                program,
                args: vec![],
            });
        }
    }

    for candidate in candidates {
        if python_matches_version(&candidate, &required_version) {
            return Ok(candidate);
        }
    }

    Err(format!(
        "Python {required_version} is required by the bundled dependencies but was not found."
    )
    .into())
}

fn python_matches_version(command: &PythonCommand, required_version: &str) -> bool {
    let output = Command::new(&command.program)
        .args(&command.args)
        .arg("--version")
        .output();

    let Ok(output) = output else {
        return false;
    };
    let version_output = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    output.status.success() && version_output.starts_with(&format!("Python {required_version}."))
}

fn bundled_backend_root(resource_dir: &PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut candidates = vec![
        resource_dir.join("bundled-backend"),
        resource_dir.join("_up_/bundled-backend"),
        resource_dir.join("resources/bundled-backend"),
        resource_dir.join("resources/_up_/bundled-backend"),
    ];

    if let Some(parent) = resource_dir.parent() {
        candidates.push(parent.join("bundled-backend"));
        candidates.push(parent.join("_up_/bundled-backend"));
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join("bundled-backend"));
            candidates.push(exe_dir.join("_up_/bundled-backend"));
            candidates.push(exe_dir.join("resources/bundled-backend"));
            candidates.push(exe_dir.join("resources/_up_/bundled-backend"));
        }
    }

    candidates
        .into_iter()
        .find(|candidate| candidate.join("backend/web_interface.py").exists())
        .ok_or_else(|| {
            format!(
                "Bundled backend not found under resources directory: {}",
                resource_dir.display()
            )
            .into()
        })
}

fn stop_managed_flask() {
    if let Some(process) = FLASK_PROCESS.get() {
        if let Some(mut child) = process.lock().expect("Flask process lock poisoned").take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
