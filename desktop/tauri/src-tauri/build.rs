fn main() {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir
        .parent()
        .and_then(std::path::Path::parent)
        .and_then(std::path::Path::parent)
        .expect("src-tauri should live under desktop/tauri")
        .canonicalize()
        .expect("failed to resolve QuantAgent project root");

    assert!(
        project_root.join(".venv/bin/python").exists(),
        "failed to find .venv/bin/python under {}",
        project_root.display()
    );
    assert!(
        project_root.join("web_interface.py").exists(),
        "failed to find web_interface.py under {}",
        project_root.display()
    );

    println!(
        "cargo:rustc-env=QUANTAGENT_PROJECT_ROOT={}",
        project_root.display()
    );
    tauri_build::build()
}
