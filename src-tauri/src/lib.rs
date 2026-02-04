use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
  let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  manifest_dir.parent().unwrap_or(&manifest_dir).to_path_buf()
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

pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .invoke_handler(tauri::generate_handler![run_python_action])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
