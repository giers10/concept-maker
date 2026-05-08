use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use tauri::{
    menu::{Menu, MenuItem, Submenu},
    AppHandle, Emitter, Manager,
};

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap_or(&manifest_dir).to_path_buf()
}

enum Backend {
    Sidecar(PathBuf),
    Source { python: String, script: PathBuf },
}

fn sidecar_names() -> Vec<&'static str> {
    if cfg!(target_os = "windows") {
        vec![
            "concept-api.exe",
            "concept-api-x86_64-pc-windows-msvc.exe",
            "concept-api-aarch64-pc-windows-msvc.exe",
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            "concept-api",
            "concept-api-aarch64-apple-darwin",
            "concept-api-x86_64-apple-darwin",
        ]
    } else {
        vec![
            "concept-api",
            "concept-api-x86_64-unknown-linux-gnu",
            "concept-api-aarch64-unknown-linux-gnu",
        ]
    }
}

fn push_sidecar_candidates(candidates: &mut Vec<PathBuf>, dir: &Path) {
    for name in sidecar_names() {
        candidates.push(dir.join(name));
    }
}

fn sidecar_candidates(app: &AppHandle) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let root = repo_root();
    push_sidecar_candidates(&mut candidates, &root.join("src-tauri").join("binaries"));

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            push_sidecar_candidates(&mut candidates, exe_dir);
            if let Some(contents_dir) = exe_dir.parent() {
                push_sidecar_candidates(&mut candidates, &contents_dir.join("Resources"));
            }
        }
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        push_sidecar_candidates(&mut candidates, &resource_dir);
        push_sidecar_candidates(&mut candidates, &resource_dir.join("binaries"));
    }

    candidates
}

fn bundled_sidecar(app: &AppHandle) -> Option<PathBuf> {
    sidecar_candidates(app)
        .into_iter()
        .find(|candidate| candidate.is_file())
}

fn source_backend_script() -> PathBuf {
    repo_root().join("concept_api.py")
}

fn backend(app: &AppHandle) -> Result<Backend, String> {
    let script = source_backend_script();
    if cfg!(debug_assertions) && script.exists() {
        return Ok(Backend::Source {
            python: "python3".to_string(),
            script,
        });
    }

    if let Some(sidecar) = bundled_sidecar(app) {
        return Ok(Backend::Sidecar(sidecar));
    }

    if script.exists() {
        return Ok(Backend::Source {
            python: "python3".to_string(),
            script,
        });
    }

    Err("Python backend not found. Rebuild the app to package the backend sidecar.".to_string())
}

fn backend_data_dir(app: &AppHandle, backend: &Backend) -> PathBuf {
    match backend {
        Backend::Source { .. } => repo_root().join(".idea-hole"),
        Backend::Sidecar(_) => app
            .path()
            .app_data_dir()
            .unwrap_or_else(|_| repo_root().join(".idea-hole")),
    }
}

fn command_for_backend(backend: &Backend) -> ProcessCommand {
    match backend {
        Backend::Sidecar(path) => ProcessCommand::new(path),
        Backend::Source { python, script } => {
            let mut command = ProcessCommand::new(python);
            command.arg(script);
            command
        }
    }
}

fn run_backend(app: &AppHandle, backend: &Backend, input: &str) -> Result<Vec<u8>, String> {
    let data_dir = backend_data_dir(app, backend);
    if let Err(err) = std::fs::create_dir_all(&data_dir) {
        return Err(format!("Failed to create backend data directory: {err}"));
    }
    let cwd = match backend {
        Backend::Source { .. } => repo_root(),
        Backend::Sidecar(_) => data_dir.clone(),
    };

    let mut child = match command_for_backend(backend)
        .current_dir(cwd)
        .env("CONCEPT_MAKER_DATA_DIR", &data_dir)
        .env(
            "PATH",
            "/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return Err(format!("Failed to start Python backend: {err}"));
        }
    };

    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(input.as_bytes()) {
            return Err(format!("Failed to write to Python stdin: {err}"));
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|err| format!("Python backend failed to finish: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python backend failed: {stderr}"));
    }

    Ok(output.stdout)
}

#[tauri::command]
fn run_python_action(app: AppHandle, action: String, payload: Value) -> Result<Value, String> {
    let req = serde_json::json!({
      "action": action,
      "payload": payload,
    });
    let input = serde_json::to_string(&req).map_err(|e| e.to_string())?;
    let backend = backend(&app)?;
    let stdout = run_backend(&app, &backend, &input)?;
    let resp: Value = serde_json::from_slice(&stdout).map_err(|e| e.to_string())?;
    if resp.get("ok") == Some(&Value::Bool(true)) {
        Ok(resp.get("data").cloned().unwrap_or(Value::Null))
    } else {
        let error = resp
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown backend error");
        Err(error.to_string())
    }
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let path = PathBuf::from(path);
    if !path.exists() {
        return Err(format!("Path does not exist: {}", path.display()));
    }

    let mut command = if cfg!(target_os = "macos") {
        let mut command = ProcessCommand::new("open");
        command.arg(&path);
        command
    } else if cfg!(target_os = "windows") {
        let mut command = ProcessCommand::new("cmd");
        command.arg("/C").arg("start").arg("").arg(&path);
        command
    } else {
        let mut command = ProcessCommand::new("xdg-open");
        command.arg(&path);
        command
    };

    command
        .spawn()
        .map(|_| ())
        .map_err(|err| format!("Failed to open {}: {err}", path.display()))
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .menu(|handle| {
            let quit_item =
                MenuItem::with_id(handle, "app-quit", "Quit", true, Some("CmdOrCtrl+Q"))?;
            let app_menu = Submenu::with_items(handle, "Concept Maker", true, &[&quit_item])?;

            let new_item = MenuItem::with_id(handle, "file-new", "New", true, None::<&str>)?;
            let open_item = MenuItem::with_id(handle, "file-open", "Open", true, None::<&str>)?;
            let save_item = MenuItem::with_id(handle, "file-save", "Save", true, None::<&str>)?;
            let file_menu =
                Submenu::with_items(handle, "File", true, &[&new_item, &open_item, &save_item])?;

            let settings_item =
                MenuItem::with_id(handle, "settings-open", "Settings", true, None::<&str>)?;
            let settings_menu = Submenu::with_items(handle, "Settings", true, &[&settings_item])?;

            Menu::with_items(handle, &[&app_menu, &file_menu, &settings_menu])
        })
        .on_menu_event(|app, event| match event.id().as_ref() {
            "app-quit" => {
                app.exit(0);
            }
            "file-new" | "file-open" | "file-save" | "settings-open" => {
                let _ = app.emit("app-menu-action", event.id().as_ref());
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![run_python_action, open_path])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
