use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::{
    menu::{Menu, MenuItem, Submenu},
    Emitter,
};

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap_or(&manifest_dir).to_path_buf()
}

fn sessions_file() -> PathBuf {
    repo_root().join(".idea-hole").join("sessions.jsonl")
}

#[derive(Deserialize)]
struct SessionFields {
    title: Option<String>,
    description: Option<String>,
    saved_at: Option<i64>,
}

#[derive(Serialize)]
struct SessionSummary {
    title: String,
    description: String,
    saved_at: i64,
}

fn open_sessions_file() -> Result<Option<File>, String> {
    match File::open(sessions_file()) {
        Ok(file) => Ok(Some(file)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(format!("Failed to open sessions file: {err}")),
    }
}

fn session_summary_from_line(line: &str) -> Option<SessionSummary> {
    let fields: SessionFields = serde_json::from_str(line).ok()?;
    let title = fields.title.unwrap_or_default().trim().to_string();
    if title.is_empty() {
        return None;
    }

    Some(SessionSummary {
        title,
        description: fields.description.unwrap_or_default(),
        saved_at: fields.saved_at.unwrap_or_default(),
    })
}

fn list_sessions_fast() -> Result<Vec<SessionSummary>, String> {
    let Some(file) = open_sessions_file()? else {
        return Ok(Vec::new());
    };

    let mut sessions = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|err| format!("Failed to read sessions file: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(summary) = session_summary_from_line(&line) {
            sessions.push(summary);
        }
    }

    Ok(sessions)
}

fn load_session_fast(title: &str) -> Result<Value, String> {
    let target = title.trim();
    if target.is_empty() {
        return Ok(Value::Null);
    }

    let Some(file) = open_sessions_file()? else {
        return Ok(Value::Null);
    };

    for line in BufReader::new(file).lines() {
        let line = line.map_err(|err| format!("Failed to read sessions file: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }

        let Some(summary) = session_summary_from_line(&line) else {
            continue;
        };
        if summary.title == target {
            return serde_json::from_str(&line)
                .map_err(|err| format!("Failed to parse saved session: {err}"));
        }
    }

    Ok(Value::Null)
}

fn run_python(script: &str, input: &str) -> Result<Vec<u8>, String> {
    let cwd = repo_root();
    let mut last_err: Option<String> = None;
    for candidate in ["python3", "python"] {
        let mut child = match Command::new(candidate)
            .arg(script)
            .current_dir(&cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) => {
                last_err = Some(format!("{candidate}: {err}"));
                continue;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            if let Err(err) = stdin.write_all(input.as_bytes()) {
                return Err(format!("Failed to write to Python stdin: {err}"));
            }
        }

        let output = match child.wait_with_output() {
            Ok(output) => output,
            Err(err) => {
                last_err = Some(format!("{candidate}: {err}"));
                continue;
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Python backend failed: {stderr}"));
        }

        return Ok(output.stdout);
    }

    Err(last_err.unwrap_or_else(|| "Python not found".to_string()))
}

#[tauri::command]
fn run_python_action(action: String, payload: Value) -> Result<Value, String> {
    match action.as_str() {
        "list_sessions" => {
            return serde_json::to_value(list_sessions_fast()?).map_err(|err| err.to_string());
        }
        "load_session" => {
            let title = payload.get("title").and_then(Value::as_str).unwrap_or("");
            return load_session_fast(title);
        }
        _ => {}
    }

    let script = repo_root().join("concept_api.py");
    if !script.exists() {
        return Err("concept_api.py not found".to_string());
    }

    let req = serde_json::json!({
      "action": action,
      "payload": payload,
    });
    let input = serde_json::to_string(&req).map_err(|e| e.to_string())?;
    let stdout = run_python(script.to_string_lossy().as_ref(), &input)?;
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
        let mut command = Command::new("open");
        command.arg(&path);
        command
    } else if cfg!(target_os = "windows") {
        let mut command = Command::new("cmd");
        command.arg("/C").arg("start").arg("").arg(&path);
        command
    } else {
        let mut command = Command::new("xdg-open");
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
            let new_item = MenuItem::with_id(handle, "file-new", "New", true, None::<&str>)?;
            let open_item = MenuItem::with_id(handle, "file-open", "Open", true, None::<&str>)?;
            let save_item = MenuItem::with_id(handle, "file-save", "Save", true, None::<&str>)?;
            let file_menu =
                Submenu::with_items(handle, "File", true, &[&new_item, &open_item, &save_item])?;

            let open_settings =
                MenuItem::with_id(handle, "settings-open", "Open Settings", true, None::<&str>)?;
            let settings_menu = Submenu::with_items(handle, "Settings", true, &[&open_settings])?;

            Menu::with_items(handle, &[&file_menu, &settings_menu])
        })
        .on_menu_event(|app, event| match event.id().as_ref() {
            "file-new" | "file-open" | "file-save" | "settings-open" => {
                let _ = app.emit("app-menu-action", event.id().as_ref());
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![run_python_action, open_path])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
