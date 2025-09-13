# Concept Maker GUI

A desktop tool to turn raw ideas, notes, and files into a clear, actionable project concept. Drag in files, jot notes, build a compact knowledge base, and generate an editable Markdown concept using a local Ollama model.

## Features
- Drag and drop files/folders into a table (falls back to add buttons).
- Freeform notes area for thoughts and fragments.
- Builds a JSONL knowledge base from your files (uses `corpus_builder.py` when present; includes a simple fallback).
- Generates a polished Concept (Markdown) via a local Ollama model.
- Prior-art search via SearXNG: generates smart queries, fetches pages, embeds + reranks, and shows a results window.
- Edit the result in‑app, then save it into a local Git repo and optionally push to a remote.
- Remembers sessions; re‑open and continue later.
- Uses `icon.png` for the window/taskbar/dock icon.

## Requirements
- Python 3.9+
- Ollama running locally (default `http://localhost:11434`)
  - Suggested default model: `mistral3.2-small:24b` (configurable in the UI)
- OS packages as needed for Tk:
  - macOS: included with the official Python.org installer
  - Ubuntu/Debian: `sudo apt-get install python3-tk`
  - Windows: included with the official Python.org installer

Optional Python packages improve file ingestion:
- `tkinterdnd2` (native drag & drop)
- `pymupdf` (PDF text extraction)
- `beautifulsoup4` (better HTML text extraction)

Prior‑art search requires API access to a SearXNG instance:
- Recommended: run SearXNG locally via Docker (installation/compose details are out of scope for this repo).
- Project site: https://searxng.org

Note: `requirements.txt` also lists some optional extras and system tools. If installation fails on entries that are not Python packages (e.g., system binaries), install them via your OS package manager or comment those lines out.

## Quick Start
- With the convenience script:
  - `bash run.sh`
- Manual steps:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate` (Windows: `.\\.venv\\Scripts\\activate`)
  - `python -m pip install --upgrade pip`
  - `pip install -r requirements.txt`
  - `python concept-maker_gui.py`

Ensure Ollama is running before generating concepts:
- Install: https://ollama.com
- Pull a model: `ollama pull mistral3.2-small:24b`
- Start service (if not auto-started): `ollama serve`

## Usage
- Add files/folders via drag & drop or the Add buttons.
- Write or paste notes in the Notes panel.
- Choose the Ollama model (or keep your default).
- Click to generate the Concept; edit as needed.
- Save: the app can save to a local repo and push to a remote.

### Prior‑Art Search (SearXNG)
- In the Notes section, click "Find Prior Art".
- Configure the SearXNG endpoint in the bottom bar (default `http://localhost:8888`).
- The app will:
  - Use the selected Ollama model to generate exactly three search queries (based on your Notes, Knowledge Base, and asset filenames).
  - Query SearXNG, fetch promising pages, embed them with Ollama embeddings (`bge-m3:latest` by default), and rerank by relevance.
  - Open a results window showing URLs, scores, and content snippets; double‑click a row to open in your browser.

## Configuration
Environment variables (optional):
- `OLLAMA_HOST`: override Ollama URL (default `http://localhost:11434`).
- `IDEA_HOLE_MODEL`: default model name shown in the UI.
- `IDEA_HOLE_REMOTE`: default Git remote URL.
- `SEARX_URL`: default SearXNG base URL for prior‑art search (default `http://localhost:8888`).

## Data & Storage
- Working data lives under `./.idea-hole/`:
  - `files/`: symlinks/copies of added files
  - `corpus.jsonl`: unified knowledge base
  - `sessions.jsonl`: saved sessions metadata

## Platform Notes
- macOS dock icon: if you install PyObjC (`pip install pyobjc`), the app also sets the dock icon from `icon.png`.
- Drag & drop requires `tkinterdnd2`; otherwise, the app uses the add buttons.
- PDF extraction works best with `pymupdf` installed.

## Troubleshooting
- Tk import errors on Linux: install `python3-tk` via your package manager.
- Ollama connection errors: ensure `ollama serve` is running and the model is pulled.
- Prior‑art search returns nothing: ensure your SearXNG instance is reachable at the configured URL and supports `/search?format=json`.
- Pip fails on non-Python entries in `requirements.txt`: install those tools via your OS, or comment out the offending lines and rerun the install.

## Development
- Code entry point: `concept-maker_gui.py`
- Launcher script: `run.sh`
- Optional corpus builder: `corpus_builder.py` (if present) is invoked for richer ingestion; otherwise, a simple built‑in fallback is used.

Prior‑art module: `websearch.py`
