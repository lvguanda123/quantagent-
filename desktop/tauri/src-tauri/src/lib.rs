use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
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

#[derive(Clone)]
struct StartupLogs {
    stdout: PathBuf,
    stderr: PathBuf,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            match ensure_flask_ready() {
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

fn ensure_flask_ready() -> Result<(), Box<dyn std::error::Error>> {
    if is_flask_ready() {
        return Ok(());
    }

    start_flask()?;
    wait_for_flask()
}

fn start_flask() -> Result<(), Box<dyn std::error::Error>> {
    let project_root = project_root();
    let python = python_path(&project_root);
    let script = project_root.join("web_interface.py");

    if !python.exists() {
        return Err(format!("Python not found: {}", python.display()).into());
    }

    if !script.exists() {
        return Err(format!("Flask entry file not found: {}", script.display()).into());
    }

    let logs = startup_logs()?;
    let stdout = File::create(&logs.stdout)?;
    let stderr = File::create(&logs.stderr)?;
    let child = Command::new(&python)
        .arg(&script)
        .current_dir(&project_root)
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

fn wait_for_flask() -> Result<(), Box<dyn std::error::Error>> {
    let started_at = Instant::now();

    while started_at.elapsed() < STARTUP_TIMEOUT {
        if is_flask_ready() {
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

    Err(format!(
        "Timed out waiting for {FLASK_URL}.\n{}",
        log_summary()
    )
    .into())
}

fn is_flask_ready() -> bool {
    FLASK_HOST_PORT
        .to_socket_addrs()
        .ok()
        .into_iter()
        .flatten()
        .any(|addr| {
            let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(300)) else {
                return false;
            };

            let _ = stream.set_read_timeout(Some(Duration::from_millis(1200)));
            let _ = stream.set_write_timeout(Some(Duration::from_millis(1200)));

            if stream
                .write_all(b"GET / HTTP/1.1\r\nHost: 127.0.0.1:5000\r\nConnection: close\r\n\r\n")
                .is_err()
            {
                return false;
            }

            let mut response = [0; 128];
            let Ok(count) = stream.read(&mut response) else {
                return false;
            };

            response[..count].starts_with(b"HTTP/1.1 200")
                || response[..count].starts_with(b"HTTP/1.0 200")
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
    let error_file = write_error_page(message).map_err(|error| tauri::Error::Anyhow(error.into()))?;
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
    let stdout = logs.map(|logs| logs.stdout.display().to_string()).unwrap_or_default();
    let stderr = logs.map(|logs| logs.stderr.display().to_string()).unwrap_or_default();
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
        project_root = escape_html(&project_root().display().to_string()),
        stdout = escape_html(&stdout),
        stderr = escape_html(&stderr),
    );

    fs::write(&path, html)?;
    Ok(path)
}

fn startup_logs() -> std::io::Result<StartupLogs> {
    let dir = project_root().join("desktop/tauri/logs");
    fs::create_dir_all(&dir)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(StartupLogs {
        stdout: dir.join(format!("flask-{stamp}.stdout.log")),
        stderr: dir.join(format!("flask-{stamp}.stderr.log")),
    })
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

fn project_root() -> PathBuf {
    PathBuf::from(env!("QUANTAGENT_PROJECT_ROOT"))
}

fn python_path(project_root: &Path) -> PathBuf {
    project_root.join(".venv/bin/python")
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
