# Concept Maker

A Tauri desktop app that turns raw ideas, notes, URLs, and files into a clear project concept. The UI is React/Vite, the desktop shell is Tauri, and the Python backend is exposed through `concept_api.py`.

## Features

- Add files, folders, and URLs as source material.
- Write freeform notes and ask a local Ollama model to rephrase, extend, or generate a concept.
- Build a JSONL knowledge base from selected files using `corpus_builder.py` when available, with a lightweight fallback in the backend.
- Search for prior art through SearXNG and Ollama embeddings.
- Preview concepts as Markdown/PDF and save them into a local concepts repo.
- Generate an image prompt and optional image asset for a concept.
- Save and reopen local sessions.

## Requirements

- Node.js and npm
- Rust toolchain
- Python 3.9+
- Ollama running locally, defaulting to `http://localhost:11434`
- Optional local SearXNG instance for prior-art search, defaulting to `http://localhost:8888`

Python packages are listed in `requirements.txt`. Some features also depend on external command-line tools:

- `pandoc` or `wkhtmltopdf` for higher-quality Markdown to PDF export
- `tesseract` and `ocrmypdf` for OCR-heavy ingestion
- `ffmpeg`/`ffprobe` for media handling

## Setup

Install JavaScript dependencies:

```bash
npm install
```

Install Python backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure Ollama is running before using model-backed actions:

```bash
ollama serve
```

## Development

Run the web UI only:

```bash
npm run dev
```

Run the desktop app:

```bash
npm run tauri dev
```

Build the frontend:

```bash
npm run build
```

Build the desktop bundle:

```bash
npm run tauri build
```

## Project Layout

- `src/`: React frontend
- `src-tauri/`: Tauri desktop shell and Rust commands
- `concept_api.py`: JSON action backend invoked by Tauri
- `corpus_builder.py`: optional richer file ingestion pipeline
- `websearch.py`: prior-art search helpers
- `requirements.txt`: Python backend dependencies

## Configuration

Environment variables:

- `OLLAMA_HOST`: override the Ollama URL
- `IDEA_HOLE_MODEL`: default model name
- `IDEA_HOLE_REMOTE`: default Git remote URL
- `SEARX_URL`: default SearXNG base URL

Runtime data is stored in `.idea-hole/`, and generated concepts are stored under `concepts/` unless another repo folder is selected in the app.
