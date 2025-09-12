#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a JSONL corpus from a folder (recurses subdirectories).

What it does (type-specific):
  • PDF: PyMuPDF extraction (multi-column); OCR scanned PDFs via ocrmypdf.
  • HTML: strip chrome; split into H1/H2 sections.
  • Text: encoding-sniffed read.
  • EPUB: extract spine sections (BS4) + OCR embedded images; optional EPUB→PDF fallback.
  • Audio/Video: ffmpeg → mono 16k WAV → slice into N overlapping parts → multi-process Whisper (base) → merge.
  • Images: detect text-like → Tesseract OCR; otherwise VLM description via Ollama (qwen2.5vl); OCR→VLM fallback if empty.
  • Code: summarize with Ollama (qwen3:4b), no code copied into text (only description).

RAG-friendly emission:
  • --emit {per-file, per-page, per-section, auto}
      - PDF per-page (auto, with optional per-PDF page threads)
      - EPUB/HTML per-section (auto)
      - everything else per-file
  • A/V can emit per-slice and/or joined via --emit-av {joined, slices, both}

LLM hygiene:
  • Strips <think>…</think>, code fences, normalizes whitespace before writing JSONL.

Language detection:
  • Uses langid or langdetect (if installed). Store `lang` per record.

Concurrency:
  • ThreadPoolExecutor for files and per-PDF page extraction (safe variant).
  • Multiprocessing for Whisper slices.
  • Bounded semaphore for Ollama calls.

External tools:
  • ocrmypdf, tesseract, ffmpeg, ffprobe
  • (optional) Calibre `ebook-convert` or `pandoc` for EPUB→PDF fallback
  • Ollama running qwen2.5vl:7b and qwen3:4b models

Python deps (install as needed):
  pymupdf beautifulsoup4 ebooklib chardet pillow numpy requests tqdm
  openai-whisper
  langid (or langdetect)
  opencv-python-headless (optional, improves image text-detect)
"""

from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import base64
import csv
import mimetypes
import threading
import queue
import multiprocessing as mp
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Any
import faulthandler, signal

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# -------------------------
# Async writer (chunked + optional rotation)
# -------------------------

_writer_q: Optional[queue.Queue] = None
_writer_thread: Optional[threading.Thread] = None

def start_writer(out_path: Path, rotate_mb: int, queue_max: int):
    """Background writer with bounded queue and optional file rotation."""
    global _writer_q, _writer_thread
    _writer_q = queue.Queue(maxsize=max(1, queue_max))

    def _run():
        bytes_since_rotate = 0
        fh = open(out_path, "a", encoding="utf-8", buffering=1<<20)  # 1 MiB buffer
        try:
            while True:
                chunk = _writer_q.get()
                if chunk is None:
                    break
                fh.write(chunk)
                bytes_since_rotate += len(chunk.encode("utf-8", "ignore"))
                if rotate_mb and bytes_since_rotate >= rotate_mb * 1024 * 1024:
                    fh.flush()
                    fh.close()
                    fh = open(out_path, "a", encoding="utf-8", buffering=1<<20)
                    bytes_since_rotate = 0
        finally:
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    _writer_thread = threading.Thread(target=_run, daemon=True)
    _writer_thread.start()

def enqueue_records_chunked(records: List["Record"], chunk_size: int):
    """Serialize records in small batches to keep latency/GC sane."""
    if not records:
        return
    step = max(1, int(chunk_size))
    for i in range(0, len(records), step):
        batch = records[i:i+step]
        chunk = "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in batch)
        _writer_q.put(chunk)

def stop_writer():
    if _writer_q is not None:
        _writer_q.put(None)
    if _writer_thread is not None:
        _writer_thread.join()

# -------------------------
# Crash diagnostics
# -------------------------

try:
    faulthandler.enable()
    for _sig in (signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT):
        try:
            faulthandler.register(_sig, chain=True)
        except Exception:
            pass
except Exception:
    pass

# -------------------------
# Subprocess isolation helper (for crashy libs)
# -------------------------

def _subproc_entry(conn, func, path, args):
    """Run `func(path, args)` in a clean process and send back (status, payload)."""
    try:
        recs = func(path, args)
        conn.send(("ok", recs))
    except Exception as e:
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass

def run_isolated(func, path, args, *, timeout=900):
    """
    Run a CPU/IO-heavy function in a child process.
    If the child segfaults, times out, or crashes, we return a synthetic error.
    """
    ctx = mp.get_context("fork" if sys.platform == "darwin" else "spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_subproc_entry, args=(child_conn, func, path, args), daemon=True)
    p.start()
    try:
        child_conn.close()
        status, payload = ("err", "crash")
        if parent_conn.poll(timeout):
            status, payload = parent_conn.recv()
        else:
            status, payload = ("err", f"timeout after {timeout}s")
    except EOFError:
        status, payload = ("err", "eof")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        if p.is_alive():
            p.terminate()
        p.join()

    if status == "ok":
        return payload, None
    else:
        return [], f"isolated-{status}: {payload}"

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

# ---- Required core deps
try:
    import fitz  # PyMuPDF
except ImportError:
    print("[ERROR] PyMuPDF (fitz) is required. Install with: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("[ERROR] BeautifulSoup is required. Install with: pip install beautifulsoup4", file=sys.stderr)
    sys.exit(1)

# ---- Optional but recommended
try:
    from ebooklib import epub
except ImportError:
    epub = None

try:
    import chardet
except ImportError:
    chardet = None

try:
    from PIL import Image, ImageOps, ImageChops
except ImportError:
    Image = None
    ImageOps = None
    ImageChops = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2  # optional
except ImportError:
    cv2 = None

# Whisper (OpenAI)
try:
    import whisper
except ImportError:
    whisper = None

# Optional: device hinting for Whisper
try:
    import torch
except Exception:
    torch = None

# Optional language detection (either works)
try:
    import langid
except ImportError:
    langid = None
try:
    from langdetect import detect as _ld_detect, DetectorFactory as _ld_factory
    _ld_factory.seed = 42
except Exception:
    _ld_detect = None

# Progress
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # fallback to simple prints

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# -------------------------
# CLI args
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build a JSONL corpus from a folder")

    # Root input (recurses)
    p.add_argument("--root", help="Path to input root directory")
    p.add_argument("--mirror", help="(Deprecated) Path to website mirror root (alias of --root)")
    p.add_argument("--out", required=True, help="Output JSONL file path")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Concurrent per-file workers")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Emission granularity
    p.add_argument("--emit", choices=["per-file", "per-page", "per-section", "auto"], default="auto",
                   help="Granularity: per-file, per-page (PDF), per-section (EPUB/HTML), or auto")
    p.add_argument("--emit-av", choices=["joined", "slices", "both"], default="joined",
                   help="For audio/video: emit one joined record, per-slice records, or both")

    # PDF/EPUB/HTML specifics
    p.add_argument("--ocr-page-jobs", type=int, default=1, help="Per-PDF page concurrency for ocrmypdf --jobs")
    p.add_argument("--ocr-lang", default="eng", help="Tesseract language(s), e.g. 'eng+deu'")
    p.add_argument("--max-cols", type=int, default=4, help="Maximum columns to consider per PDF page")
    p.add_argument("--epub-strategy", choices=["direct", "pdf-fallback", "force-pdf"], default="pdf-fallback",
                   help="EPUB handling: try direct, fallback to PDF; or always convert to PDF")
    p.add_argument("--pdf-page-workers", type=int, default=0,
                   help="Threads per PDF for page extraction (0=auto: min(4, cpu)). Only used when emitting per-page/auto.")
    p.add_argument("--html-section-workers", type=int, default=0,
                   help="Threads per HTML for per-section record building (0=auto: min(4, cpu)).")

    # Include/Exclude
    p.add_argument(
        "--include",
        default=(
            r".*\.(?:pdf|html?|txt|md|rst|epub|"
            r"png|jpe?g|gif|bmp|tiff?|webp|heic|"
            r"mp3|wav|m4a|flac|ogg|opus|aac|"
            r"mp4|mkv|mov|webm|avi|ts|"
            r"py|ipynb|js|ts|tsx|jsx|java|c|cpp|rs|go|rb|php|cs|swift|kt|m|sh|bat|ps1|sql)$"
        ),
        help="Regex for files to include"
    )
    p.add_argument(
        "--exclude",
        default=r"(^|[\\/])\.|__MACOSX([\\/]|$)|\.DS_Store$|\.ocr\.txt$",
        help="Regex for files/paths to exclude"
    )

    # ASR (Whisper-base, multi-process slices)
    p.add_argument("--whisper-model", default="base", help="OpenAI Whisper model size (tiny, base, small, …)")
    p.add_argument("--num-slices", type=int, default=8, help="Number of equal slices per media file")
    p.add_argument("--overlap-sec", type=float, default=1.0, help="Overlap seconds between slices")
    p.add_argument("--max-overlap-words", type=int, default=7, help="Max words to align/dedup across slice boundaries")
    p.add_argument("--mp-workers", type=int, default=0, help="Multiprocessing workers (0 -> use num-slices)")
    p.add_argument("--asr-task", choices=["transcribe", "translate"], default="transcribe",
                   help="Whisper task: transcribe (original language) or translate (to English)")
    p.add_argument("--max-av-duration", type=float, default=5*3600, help="Hard cap (seconds) for audio/video")

    # NEW: device control (avoid MPS crash by default)
    p.add_argument("--whisper-device", choices=["auto","cpu","cuda","mps"], default="auto",
                   help="Device for Whisper slices. Default 'auto' prefers CUDA, otherwise CPU (not MPS).")

    # Ollama (images, code)
    p.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL")
    p.add_argument("--vlm-model", default="qwen2.5vl:7b", help="Vision LLM model for image description")
    p.add_argument("--code-llm", default="qwen3:4b", help="Code summarizer model")
    p.add_argument("--llm-parallel", type=int, default=1, help="Parallel LLM calls (Ollama)")

    # Images
    p.add_argument("--image-max-edge", type=int, default=1600, help="Resize longest edge before VLM to save VRAM")

    # Image OCR gate + thresholds
    p.add_argument("--image-text-gate",
                   choices=["tesseract-conf", "vlm-gate", "always-ocr", "always-vlm"],
                   default="tesseract-conf",
                   help="How to decide OCR vs VLM for images.")
    p.add_argument("--ocr-psms", default="6,11",
                   help="Comma-separated PSMs to probe for OCR gating (e.g. '6,11').")
    p.add_argument("--ocr-min-conf", type=int, default=55,
                   help="Minimum median word confidence to accept OCR.")
    p.add_argument("--ocr-min-words", type=int, default=10,
                   help="Minimum word count to accept OCR.")
    p.add_argument("--ocr-min-alnum", type=float, default=0.55,
                   help="Minimum alnum ratio over non-space printable chars to accept OCR.")

    # Code
    p.add_argument("--code-max-bytes", type=int, default=200_000, help="Read at most N bytes from code files")

    # Language hints/detection
    p.add_argument("--lang-hint", default=None, help="Optional language hint for OCR")
    p.add_argument("--lang-detect", action="store_true", default=True, help="Detect language of each record")
    p.add_argument("--no-lang-detect", dest="lang_detect", action="store_false")

    # Writer tuning
    p.add_argument("--writer-queue", type=int, default=64, help="Max queued chunks to the writer thread")
    p.add_argument("--writer-chunk", type=int, default=256, help="Records per JSONL chunk enqueued to writer")
    p.add_argument("--writer-rotate-mb", type=int, default=0, help="Rotate (close/reopen) writer every N MB; 0=off")

    # External tools
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "/usr/bin/ffmpeg", help="Path to ffmpeg")
    p.add_argument("--ffprobe", default=shutil.which("ffprobe") or "/usr/bin/ffprobe", help="Path to ffprobe")
    p.add_argument("--tesseract", default=shutil.which("tesseract") or "/usr/bin/tesseract", help="Path to tesseract")
    p.add_argument("--ebook-convert", dest="ebook_convert", default=shutil.which("ebook-convert"), help="Path to Calibre's ebook-convert (optional)")
    p.add_argument("--pandoc", default=shutil.which("pandoc"), help="Path to pandoc (optional)")

    return p.parse_args()

# -------------------------
# Utilities
# -------------------------

def log(msg: str, *, verbose: bool = True):
    if verbose:
        print(msg, flush=True)

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def detect_encoding(b: bytes) -> str:
    if chardet is None:
        return "utf-8"
    guess = chardet.detect(b) or {}
    enc = guess.get("encoding") or "utf-8"
    return enc

def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    enc = detect_encoding(data)
    try:
        return data.decode(enc, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")

def run_cmd(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def ffprobe_json(ffprobe_bin: str, media_path: Path) -> Optional[Dict]:
    cmd = [ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(media_path)]
    res = run_cmd(cmd)
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout)
    except Exception:
        return None

def extract_audio_wav(ffmpeg_bin: str, input_path: Path, out_wav: Path, *, samplerate=16000) -> bool:
    cmd = [ffmpeg_bin, "-y", "-i", str(input_path), "-ac", "1", "-ar", str(samplerate), "-f", "wav", str(out_wav)]
    res = run_cmd(cmd)
    return res.returncode == 0

def try_mutool_clean(in_pdf: Path) -> Optional[Path]:
    if not shutil.which("mutool"): return None
    tmp = Path(tempfile.mkstemp(suffix=".clean.pdf")[1])
    res = run_cmd(["mutool", "clean", "-gg", str(in_pdf), str(tmp)])
    return tmp if res.returncode == 0 and tmp.exists() else None

def pdftotext_fallback(in_pdf: Path) -> str:
    if not shutil.which("pdftotext"): return ""
    tmp = Path(tempfile.mkstemp(suffix=".txt")[1])
    try:
        run_cmd(["pdftotext", "-layout", "-enc", "UTF-8", str(in_pdf), str(tmp)])
        return tmp.read_text("utf-8", errors="ignore")
    finally:
        try: tmp.unlink()
        except Exception: pass

# ---- Ollama HTTP helpers
def ollama_generate(host: str, model: str, prompt: str, images_b64: Optional[List[str]] = None, options: Optional[Dict]=None, stream: bool=False) -> str:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("The 'requests' package is required for Ollama calls. Install with: pip install requests") from e
    payload = {"model": model, "prompt": prompt, "stream": stream}
    if images_b64:
        payload["images"] = images_b64
    if options:
        payload["options"] = options
    resp = requests.post(f"{host.rstrip('/')}/api/generate", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")

def encode_image_b64(path: Path, max_edge: int = 1600) -> str:
    if Image is None:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    w, h = img.size
    scale = max(w, h)
    if scale > max_edge:
        ratio = max_edge / float(scale)
        img = img.resize((int(w*ratio), int(h*ratio)))
    buf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        img.save(buf.name, format="JPEG", quality=90)
        b = Path(buf.name).read_bytes()
        return base64.b64encode(b).decode("ascii")
    finally:
        try:
            os.unlink(buf.name)
        except Exception:
            pass

# ---- LLM hygiene / language detection
def sanitize_llm_text(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S|re.I)
    s = re.sub(r"^\s*```(?:\w+)?\s*|\s*```\s*$", "", s, flags=re.M)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def detect_language(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    n = len(text)
    if n > 3000:
        head = text[:1000]; mid = text[n//2:n//2+1000]; tail = text[-1000:]
        sample = head + "\n" + mid + "\n" + tail
    else:
        sample = text
    try:
        if langid is not None:
            lang, _ = langid.classify(sample)
            return lang
        if _ld_detect is not None:
            return _ld_detect(sample)
    except Exception:
        pass
    return None

# -------------------------
# Image text-likeness detection (optional)
# -------------------------

def image_is_textlike(path: Path) -> bool:
    try:
        if cv2 is not None and np is not None:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            h, w = img.shape[:2]
            scale = max(h, w)
            if scale > 1800:
                r = 1800.0 / scale
                img = cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)
            thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 11)
            contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False
            areas = [cv2.contourArea(c) for c in contours]
            small = [a for a in areas if 10 < a < 5000]
            density = len(small) / (img.shape[0]*img.shape[1] / 1e5)
            return density > 8
        else:
            if Image is None or np is None:
                return False
            img = Image.open(path).convert("L")
            w, h = img.size
            if max(w, h) > 1800:
                r = 1800.0 / max(w, h)
                img = img.resize((int(w*r), int(h*r)))
            arr = np.array(img, dtype=np.float32)
            dx = np.abs(np.diff(arr, axis=1))
            dy = np.abs(np.diff(arr, axis=0))
            edge_ratio = (np.pad((dx[:, :-1]**2 + dy[:-1, :]**2)**0.5, ((0,1),(0,1))) > 25).mean()
            thresh = (arr > 200).mean() + (arr < 55).mean()
            return (edge_ratio > 0.15) and (thresh > 0.25)
    except Exception:
        return False

# -------------------------
# PDF helpers
# -------------------------

def is_probably_scanned(pdf_path: Path, sample_pages: int = 3) -> bool:
    try:
        with fitz.open(pdf_path) as doc:
            n = min(len(doc), max(1, sample_pages))
            text_len = 0
            for i in range(n):
                page = doc.load_page(i)
                txt = page.get_text("text")
                text_len += len(txt.strip())
            return text_len < 50 * n
    except Exception:
        return True

def ocrmypdf_searchable(in_pdf: Path, out_pdf: Path, lang: str, page_jobs: int, verbose: bool) -> Tuple[bool, str]:
    base_cmd = [
        "ocrmypdf",
        "--skip-text",
        "--optimize", "0",
        "--rotate-pages",
        "--deskew",
        "--jobs", str(max(1, page_jobs)),
        "--tesseract-timeout", "120",
        "--output-type", "pdf",
        "--language", lang,
    ]
    base_cmd.append("--verbose" if verbose else "-q")
    cmd = base_cmd + [str(in_pdf), str(out_pdf)]
    res = run_cmd(cmd)
    out = res.stdout or ""
    if "NotImplementedError: --remove-background" in out or "--remove-background is temporarily not implemented" in out:
        log(f"[INFO] {in_pdf.name}: retrying without --remove-background", verbose=verbose)
        res = run_cmd(cmd)
        out = res.stdout or ""
    ok = res.returncode == 0
    if not ok and "NotImplementedError" in out:
        log(f"[INFO] {in_pdf.name}: quality retry (psm=3, cleanup=on)", verbose=verbose)
        cmd_retry = base_cmd + ["--tesseract-pagesegmode", "3", "--clean-final"] + [str(in_pdf), str(out_pdf)]
        res = run_cmd(cmd_retry)
        out = res.stdout or ""
        ok = res.returncode == 0
    return ok, out

def segment_columns(blocks: List[Tuple], max_cols: int) -> List[List[Tuple]]:
    if not blocks:
        return []
    tblocks = [b for b in blocks if isinstance(b[4], str) and b[4].strip()]
    if not tblocks:
        return []
    xs = []
    for b in tblocks:
        x0, y0, x1, y1, txt, *_ = b
        xs.append(((x0 + x1) / 2.0, b))
    xs.sort(key=lambda t: t[0])
    centers = [v for v,_ in xs]
    gaps = []
    for i in range(1, len(centers)):
        gaps.append((centers[i] - centers[i-1], i))
    gaps.sort(reverse=True, key=lambda t: t[0])
    splits = sorted(idx for _, idx in gaps[:max(0, max_cols-1)])
    columns: List[List[Tuple]] = []
    last = 0
    for s in splits:
        col = [b for _, b in xs[last:s]]
        if col:
            columns.append(col)
        last = s
    col = [b for _, b in xs[last:]]
    if col:
        columns.append(col)
    if len(columns) <= 1:
        columns = [[b for _, b in xs]]
    for col in columns:
        col.sort(key=lambda b: (b[1], b[0]))
    return columns

def extract_pdf_text(pdf_path: Path, max_cols: int, verbose: bool) -> str:
    texts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            blocks = page.get_text("blocks")
            if not blocks:
                continue
            blocks = [b for b in blocks if isinstance(b[4], str) and b[4].strip()]
            if not blocks:
                continue
            cols = segment_columns(blocks, max_cols=max_cols)
            page_lines: List[str] = []
            for col in cols:
                for x0,y0,x1,y1,txt,*_ in col:
                    t = re.sub(r"\s+", " ", txt.strip())
                    if t:
                        page_lines.append(t)
            if page_lines:
                texts.append("\n".join(page_lines))
    return "\n\n".join(texts).strip()

# -------------------------
# HTML helpers
# -------------------------

def split_html_sections(html_text: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()

    sections: List[Dict[str, Any]] = []
    current = {"title": None, "parts": []}

    def flush():
        if current["parts"] or current["title"]:
            txt = "\n".join(current["parts"]).strip()
            sections.append({"title": current["title"] or None, "text": txt})
            current["title"], current["parts"] = None, []

    for el in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","blockquote","pre","code"]):
        if el.name in {"h1","h2"}:
            flush()
            t = el.get_text(separator=" ", strip=True)
            current["title"] = t or None
        else:
            t = el.get_text(separator=" ", strip=True)
            if t:
                current["parts"].append(t)
    flush()
    return sections

# -------------------------
# Records
# -------------------------

@dataclass
class Record:
    id: str
    parent_id: Optional[str]
    source_path: str
    url: Optional[str]
    mime: str
    record_type: str       # "file" | "page" | "section" | "av" | "image" | "code-summary" | "html-section"
    title: Optional[str]
    text: str
    span: Optional[Dict[str, Any]] = None
    lang: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

# -------------------------
# Processors
# -------------------------

def _extract_single_pdf_page(pdf_path: Path, pno: int, max_cols: int) -> Tuple[int, str, str]:
    """Open the PDF in THIS thread, extract one page. Returns (page_index, title_guess, text)."""
    title = None
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            if pno < 0 or pno >= len(doc):
                return (pno, "", "")
            page = doc.load_page(pno)
            blocks = page.get_text("blocks") or []
            blocks = [b for b in blocks if isinstance(b[4], str) and b[4].strip()]
            if not blocks:
                return (pno, "", "")
            cols = segment_columns(blocks, max_cols=max_cols)
            lines: List[str] = []
            for col in cols:
                for x0, y0, x1, y1, txt, *_ in col:
                    t = re.sub(r"\s+", " ", txt.strip())
                    if t:
                        lines.append(t)
            text = "\n".join(lines).strip()
            for line in text.splitlines():
                if line.strip():
                    title = line.strip()
                    break
        return (pno, title or "", text)
    except Exception:
        return (pno, "", "")

def process_pdf(path: Path, args) -> List[Record]:
    """
    PDF: if emit=per-page/auto → one record per page (with optional page threads);
         else single record.
    Also uses ocrmypdf --jobs for scanned PDFs (already parallel).
    """
    verbose = args.verbose
    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = Path(tmpdir_obj.name)
    records: List[Record] = []
    try:
        src = path
        work_pdf = src
        # (1) Make searchable if scanned
        if is_probably_scanned(src):
            out_pdf = tmpdir / f"{src.stem}.ocr.pdf"
            ok, _ocr_log = ocrmypdf_searchable(src, out_pdf, args.lang_hint or args.ocr_lang, args.ocr_page_jobs, verbose)
            if ok:
                work_pdf = out_pdf

        per_page = (args.emit in ("per-page", "auto"))
        if per_page:
            # Determine page worker count
            page_workers = args.pdf_page_workers or min(4, (os.cpu_count() or 4))
            try:
                # First open once to count pages
                with fitz.open(work_pdf) as d:
                    n_pages = len(d)
                if page_workers > 1 and n_pages > 1:
                    # Threaded per-page extraction (safe: each worker opens the doc)
                    results: List[Tuple[int, str, str]] = []
                    with cf.ThreadPoolExecutor(max_workers=max(1, page_workers)) as ex:
                        futs = {ex.submit(_extract_single_pdf_page, work_pdf, pno, args.max_cols): pno for pno in range(n_pages)}
                        for fut in cf.as_completed(futs):
                            results.append(fut.result())
                    results.sort(key=lambda t: t[0])
                else:
                    # Single-threaded per-page
                    results = []
                    with fitz.open(work_pdf) as d:
                        for pno in range(len(d)):
                            page = d.load_page(pno)
                            blocks = page.get_text("blocks") or []
                            blocks = [b for b in blocks if isinstance(b[4], str) and b[4].strip()]
                            if not blocks:
                                text = ""
                            else:
                                cols = segment_columns(blocks, max_cols=args.max_cols)
                                lines = []
                                for col in cols:
                                    for x0,y0,x1,y1,txt,*_ in col:
                                        t = re.sub(r"\s+", " ", txt.strip())
                                        if t: lines.append(t)
                                text = "\n".join(lines).strip()
                            title = None
                            for line in text.splitlines():
                                if line.strip():
                                    title = line.strip(); break
                            results.append((pno, title or "", text))
                for (pno, title, text) in results:
                    lang = detect_language(text) if args.lang_detect else None
                    records.append(Record(
                        id=f"{path.as_posix()}#page={pno+1}",
                        parent_id=str(path.as_posix()),
                        source_path=str(path.resolve()),
                        url=None,
                        mime="application/pdf",
                        record_type="page",
                        title=title or f"{path.stem} — p.{pno+1}",
                        text=text,
                        span={"page_start": pno+1, "page_end": pno+1},
                        lang=lang,
                        meta=None
                    ))
                return records
            except Exception:
                pass  # fallthrough to file-level

        # (2) File-level extraction
        text = extract_pdf_text(work_pdf, max_cols=args.max_cols, verbose=verbose)
        title = None
        for line in text.splitlines():
            if line.strip():
                title = line.strip()
                break
        lang = detect_language(text) if args.lang_detect else None
        records.append(Record(
            id=str(path.as_posix()),
            parent_id=None,
            source_path=str(path.resolve()),
            url=None,
            mime="application/pdf",
            record_type="file",
            title=title,
            text=text,
            span=None,
            lang=lang,
            meta=None
        ))
        return records
    finally:
        tmpdir_obj.cleanup()

def process_html(path: Path, args) -> List[Record]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    per_section = (args.emit in ("per-section", "auto"))
    if per_section:
        secs = split_html_sections(html)
        secs = [s for s in secs if (s.get("text") or "").strip()]
        if secs:
            sec_workers = args.html_section_workers or min(4, (os.cpu_count() or 4))

            def _build(idx: int, s: Dict[str, Any]) -> Record:
                text = s["text"]
                title = s["title"] or f"{path.stem} — section {idx+1}"
                lang = detect_language(text) if args.lang_detect else None
                return Record(
                    id=f"{path.as_posix()}#section={idx+1}",
                    parent_id=str(path.as_posix()),
                    source_path=str(path.resolve()),
                    url=None,
                    mime="text/html",
                    record_type="html-section",
                    title=title,
                    text=text,
                    span={"section_idx": idx+1, "section_title": s["title"]},
                    lang=lang,
                    meta=None
                )

            records: List[Tuple[int, Record]] = []
            with cf.ThreadPoolExecutor(max_workers=max(1, sec_workers)) as ex:
                futs = {ex.submit(_build, i, s): i for i, s in enumerate(secs)}
                for fut in cf.as_completed(futs):
                    i = futs[fut]
                    records.append((i, fut.result()))
            records.sort(key=lambda t: t[0])
            return [r for _, r in records]

    # file-level fallback
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    texts: List[str] = []
    for el in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","blockquote","pre","code"]):
        t = el.get_text(separator=" ", strip=True)
        if t:
            texts.append(t)
    text = "\n".join(texts).strip()
    title = None
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        for line in text.splitlines():
            if line.strip():
                title = line.strip()
                break
    lang = detect_language(text) if args.lang_detect else None
    return [Record(
        id=str(path.as_posix()),
        parent_id=None,
        source_path=str(path.resolve()),
        url=None,
        mime="text/html",
        record_type="file",
        title=title or path.stem,
        text=text,
        span=None,
        lang=lang,
        meta=None
    )]

def preprocess_image_for_ocr(img_path: Path, upsample_min_edge: int = 900) -> Path:
    if Image is None:
        return img_path
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if ImageChops is not None:
        corners = [(0,0), (w-1,0), (0,h-1), (w-1,h-1)]
        bboxes = []
        for cx, cy in corners:
            try:
                bg = Image.new(img.mode, img.size, img.getpixel((cx, cy)))
                diff = ImageChops.difference(img, bg)
                bbox = diff.getbbox()
                if bbox: bboxes.append(bbox)
            except Exception:
                pass
        if bboxes:
            left  = max(b[0] for b in bboxes)
            top   = max(b[1] for b in bboxes)
            right = min(b[2] for b in bboxes)
            bottom= min(b[3] for b in bboxes)
            if 0 <= left < right <= w and 0 <= top < bottom <= h:
                if (right-left) >= 0.7*w and (bottom-top) >= 0.7*h:
                    img = img.crop((left, top, right, bottom))
    img = ImageOps.grayscale(img)
    try:
        img = ImageOps.autocontrast(img, cutoff=1)
    except Exception:
        pass
    W, H = img.size
    if max(W, H) < upsample_min_edge:
        scale = float(upsample_min_edge) / float(max(W, H))
        img = img.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
    img.save(tmp)
    return Path(tmp)

def tesseract_ocr_image(tesseract_bin: str, img_path: Path, lang: str, psm: Optional[int] = None) -> str:
    pre = preprocess_image_for_ocr(img_path)
    try:
        cmd = [tesseract_bin, str(pre), "stdout", "-l", lang]
        if psm is not None:
            cmd += ["--psm", str(psm)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        if res.returncode != 0:
            return ""
        return res.stdout.strip()
    finally:
        if pre != img_path:
            try: pre.unlink()
            except Exception: pass

def _alnum_ratio(s: str) -> float:
    chars = [c for c in s if c.isprintable() and not c.isspace()]
    if not chars:
        return 0.0
    alnum = sum(1 for c in chars if c.isalnum())
    return float(alnum) / float(len(chars))

def _looks_like_garbage(text: str, *, require_lang: bool, args) -> bool:
    t = (text or "").strip()
    if len(t) < 20:
        return True
    toks = re.findall(r"\w+|\S", t)
    avg_tok = sum(len(x) for x in toks) / max(1, len(toks))
    uniq_ratio = len(set(t)) / max(1, len(t))
    if uniq_ratio > 0.6 and avg_tok < 2.2:
        return True
    if re.search(r"[|—\-]{5,}", t):
        return True
    if require_lang and args.lang_detect and (detect_language(t) is None):
        return True
    return False

def _tesseract_probe_tsv(tesseract_bin: str, img_path: Path, lang: str, psm: Optional[int] = None) -> Dict[str, Any]:
    pre = preprocess_image_for_ocr(img_path)
    tmpdir = Path(tempfile.mkdtemp(prefix="tsv_"))
    try:
        base = tmpdir / "probe"
        cmd = [tesseract_bin, str(pre), str(base), "-l", lang]
        if psm is not None:
            cmd += ["--psm", str(psm)]
        cmd += ["tsv"]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            return {"psm": psm, "words": 0, "conf_median": 0.0, "conf_mean": 0.0, "text": "", "alnum_ratio": 0.0}
        tsv_path = base.with_suffix(".tsv")
        if not tsv_path.exists():
            return {"psm": psm, "words": 0, "conf_median": 0.0, "conf_mean": 0.0, "text": "", "alnum_ratio": 0.0}
        words, confs, tokens = 0, [], []
        with open(tsv_path, "r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                txt = (row.get("text") or "").strip()
                try:
                    conf = float(row.get("conf") or -1)
                except Exception:
                    conf = -1.0
                if txt and conf >= 0:
                    words += 1
                    confs.append(conf)
                    tokens.append(txt)
        text = " ".join(tokens).strip()
        conf_median = float(np.median(confs)) if confs else 0.0
        conf_mean = float(np.mean(confs)) if confs else 0.0
        return {
            "psm": psm,
            "words": words,
            "conf_median": conf_median,
            "conf_mean": conf_mean,
            "text": text,
            "alnum_ratio": _alnum_ratio(text),
        }
    finally:
        try:
            if pre != img_path:
                pre.unlink()
        except Exception:
            pass
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

def process_image(path: Path, args) -> List[Record]:
    def vlm_describe() -> Tuple[str, str, Dict[str, Any]]:
        img_b64 = encode_image_b64(path, args.image_max_edge)
        prompt = (
            "Decide first if the image is primarily TEXT or not.\n"
            "- If TEXT: output exactly:\n"
            "TYPE: TEXT\nCONTENT:\n<verbatim transcription with line breaks preserved>\n"
            "- If not: output exactly:\n"
            "TYPE: DESCRIPTION\nCONTENT:\n<concise description of the scene, objects, layout; include short visible text>\n"
            "Do not add extra headers, markdown, or commentary."
        )
        if LLM_SEM is not None:
            with LLM_SEM:
                resp = ollama_generate(args.ollama_host, args.vlm_model, prompt, images_b64=[img_b64], options={"temperature": 0.2})
        else:
            resp = ollama_generate(args.ollama_host, args.vlm_model, prompt, images_b64=[img_b64], options={"temperature": 0.2})
        resp = sanitize_llm_text(resp)
        kind = "DESCRIPTION"
        content = resp.strip()
        m = re.search(r"TYPE:\s*(TEXT|DESCRIPTION)", resp, re.I)
        if m:
            kind = m.group(1).upper()
        m2 = re.search(r"CONTENT:\s*(.*)", resp, re.S)
        if m2:
            content = m2.group(1).strip()
        meta = {"vlm_kind": kind}
        return sanitize_llm_text(content), f"vlm:{kind}", meta

    if args.image_text_gate == "always-vlm":
        text, mode, meta_extra = vlm_describe()
    else:
        if args.image_text_gate == "always-ocr":
            psms = [int(x) for x in str(args.ocr_psms).split(",") if str(x).strip().isdigit()]
            best_txt, best_psm = "", None
            for psm in psms or [6]:
                txt = tesseract_ocr_image(args.tesseract, path, args.lang_hint or args.ocr_lang, psm=psm).strip()
                if len(txt) > len(best_txt):
                    best_txt, best_psm = txt, psm
            text = sanitize_llm_text(best_txt)
            if _looks_like_garbage(text, require_lang=True, args=args):
                vlm_text, vlm_mode, meta_extra = vlm_describe()
                text, mode = vlm_text, vlm_mode
                meta_extra = {"fallback": "vlm_garbage_filter"}
            else:
                mode, meta_extra = "tesseract", {"ocr_psm": best_psm}
        elif args.image_text_gate in ("tesseract-conf", "vlm-gate"):
            gate_decision = None
            gate_meta: Dict[str, Any] = {}
            if args.image_text_gate == "vlm-gate":
                img_b64 = encode_image_b64(path, args.image_max_edge)
                gate_prompt = (
                    "Is this image primarily text (documents, slides, screenshots) or not?\n"
                    "Answer with EXACTLY one word: TEXT or DESCRIPTION."
                )
                if LLM_SEM is not None:
                    with LLM_SEM:
                        g = ollama_generate(args.ollama_host, args.vlm_model, gate_prompt, images_b64=[img_b64], options={"temperature": 0.0})
                else:
                    g = ollama_generate(args.ollama_host, args.vlm_model, gate_prompt, images_b64=[img_b64], options={"temperature": 0.0})
                g = sanitize_llm_text(g).split()[0].upper() if g.strip() else "DESCRIPTION"
                if g not in {"TEXT", "DESCRIPTION"}:
                    g = "DESCRIPTION"
                gate_decision = g
                gate_meta["vlm_gate"] = g

            if gate_decision == "DESCRIPTION":
                text, mode, meta_extra = vlm_describe()
                meta_extra.update({"image_gate": "vlm-gate"})
            else:
                psms = [int(x) for x in str(args.ocr_psms).split(",") if str(x).strip().isdigit()] or [6, 11]
                probes = [_tesseract_probe_tsv(args.tesseract, path, args.lang_hint or args.ocr_lang, psm=psm) for psm in psms]
                best = max(probes, key=lambda d: (d.get("conf_median", 0.0), d.get("words", 0)))
                accept = (
                    best.get("conf_median", 0.0) >= float(args.ocr_min_conf) and
                    best.get("words", 0) >= int(args.ocr_min_words) and
                    best.get("alnum_ratio", 0.0) >= float(args.ocr_min_alnum)
                )
                if accept:
                    best_psm = best.get("psm") or 6
                    text = tesseract_ocr_image(args.tesseract, path, args.lang_hint or args.ocr_lang, psm=best_psm).strip()
                    text = sanitize_llm_text(text)
                    if _looks_like_garbage(text, require_lang=True, args=args):
                        vlm_text, vlm_mode, meta_extra = vlm_describe()
                        text, mode = vlm_text, vlm_mode
                        meta_extra = {"fallback": "vlm_garbage_filter", "image_gate": "tesseract-conf"}
                        meta_extra.update(gate_meta)
                    else:
                        mode, meta_extra = "tesseract", {"image_gate": "tesseract-conf", "ocr_psm": best_psm}
                        meta_extra.update({
                            "ocr_words": best.get("words", 0),
                            "ocr_conf_median": round(best.get("conf_median", 0.0), 2),
                            "ocr_conf_mean": round(best.get("conf_mean", 0.0), 2),
                            "alnum_ratio": round(best.get("alnum_ratio", 0.0), 3),
                        })
                        meta_extra.update(gate_meta)
                else:
                    vlm_text, vlm_mode, meta_extra = vlm_describe()
                    text, mode = vlm_text, vlm_mode
                    meta_extra.update({
                        "image_gate": "tesseract-conf",
                        "fallback": "vlm_conf_too_low",
                        "ocr_words": best.get("words", 0),
                        "ocr_conf_median": round(best.get("conf_median", 0.0), 2),
                        "ocr_conf_mean": round(best.get("conf_mean", 0.0), 2),
                        "alnum_ratio": round(best.get("alnum_ratio", 0.0), 3),
                    })
        else:
            text, mode, meta_extra = vlm_describe()

    text = sanitize_llm_text(text)
    mime = mimetypes.guess_type(str(path))[0] or "image/*"
    title = (text.splitlines()[0].strip() if text else path.stem)[:200]
    lang = detect_language(text) if args.lang_detect else None

    meta = {"image_mode": mode}
    if "meta_extra" in locals() and isinstance(meta_extra, dict):
        meta.update(meta_extra)

    return [Record(
        id=f"{path.as_posix()}",
        parent_id=None,
        source_path=str(path.resolve()),
        url=None,
        mime=mime,
        record_type="image",
        title=title or path.stem,
        text=text,
        span=None,
        lang=lang,
        meta=meta
    )]

def extract_epub_sections(path: Path, args) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    if epub is None:
        return sections
    book = epub.read_epub(str(path))
    tmpdir = Path(tempfile.mkdtemp(prefix="epub_"))
    try:
        order = []
        for itemref in book.spine or []:
            idref = itemref[0] if isinstance(itemref, (list, tuple)) else itemref
            it = book.get_item_with_id(idref)
            if it: order.append(it)
        if not order:
            order = [it for it in book.get_items() if it.get_type() == 9]
        for idx, it in enumerate(order):
            html = it.get_content().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
                tag.decompose()
            texts: List[str] = []
            for el in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","blockquote","pre","code"]):
                t = el.get_text(separator=" ", strip=True)
                if t:
                    texts.append(t)
            title = None
            for el in soup.find_all(["h1","h2"]):
                t = el.get_text(separator=" ", strip=True)
                if t:
                    title = t
                    break
            if not title:
                title = it.get_id() or f"Section {idx+1}"
            sections.append({"idx": idx, "title": title, "text": "\n".join(texts).strip(), "images": []})
        images = []
        for item in book.get_items():
            if item.get_type() == 3:
                fp = tmpdir / f"{item.get_id()}"
                with open(fp, "wb") as fh:
                    fh.write(item.get_content())
                images.append(fp)
        if sections and images:
            sections[0]["images"] = images
        return sections
    except Exception:
        return sections
    finally:
        pass

def process_epub(path: Path, args) -> List[Record]:
    per_section = (args.emit in ("per-section", "auto"))
    if per_section:
        secs = extract_epub_sections(path, args)
        records: List[Record] = []
        if not secs:
            per_section = False
        else:
            for sec in secs:
                texts = sec["text"]
                img_texts: List[str] = []
                for img in sec.get("images") or []:
                    ocr_txt = tesseract_ocr_image(args.tesseract, img, args.lang_hint or args.ocr_lang)
                    if ocr_txt:
                        img_texts.append(ocr_txt)
                final_text = (texts + ("\n\n" + "\n\n".join(img_texts) if img_texts else "")).strip()
                rid = f"{path.as_posix()}#section={sec['idx']+1}"
                lang = detect_language(final_text) if args.lang_detect else None
                records.append(Record(
                    id=rid,
                    parent_id=str(path.as_posix()),
                    source_path=str(path.resolve()),
                    url=None,
                    mime="application/epub+zip",
                    record_type="section",
                    title=sec["title"] or f"{path.stem} — section {sec['idx']+1}",
                    text=final_text,
                    span={"section_idx": sec['idx']+1, "section_title": sec["title"]},
                    lang=lang,
                    meta={"epub_strategy": "direct"}
                ))
        if records:
            return records

    texts = ""
    img_texts: List[str] = []
    tmp_pdf = None
    if args.epub_strategy in ("direct", "pdf-fallback"):
        secs = extract_epub_sections(path, args)
        texts = "\n\n".join([s["text"] for s in secs]) if secs else ""
        for s in secs:
            for img in s.get("images") or []:
                ocr_txt = tesseract_ocr_image(args.tesseract, img, args.lang_hint or args.ocr_lang)
                if ocr_txt:
                    img_texts.append(ocr_txt)
        combined = (texts + ("\n\n" + "\n\n".join(img_texts) if img_texts else "")).strip()
        if len(combined) < 500 and args.epub_strategy == "pdf-fallback":
            tmp_pdf = path.with_suffix(".epub.tmp.pdf")
    else:
        tmp_pdf = path.with_suffix(".epub.tmp.pdf")

    if tmp_pdf:
        converted = False
        if args.ebook_convert:
            res = run_cmd([args.ebook_convert, str(path), str(tmp_pdf)])
            converted = (res.returncode == 0 and tmp_pdf.exists())
        elif args.pandoc:
            res = run_cmd([args.pandoc, str(path), "-o", str(tmp_pdf)])
            converted = (res.returncode == 0 and tmp_pdf.exists())
        if converted:
            try:
                recs = process_pdf(tmp_pdf, args)
                try: tmp_pdf.unlink(missing_ok=True)
                except Exception: pass
                return recs
            except Exception:
                try: tmp_pdf.unlink(missing_ok=True)
                except Exception: pass

    final_text = (texts + ("\n\n" + "\n\n".join(img_texts) if img_texts else "")).strip()
    title = None
    for line in final_text.splitlines():
        if line.strip():
            title = line.strip()
            break
    lang = detect_language(final_text) if args.lang_detect else None
    return [Record(
        id=str(path.as_posix()),
        parent_id=None,
        source_path=str(path.resolve()),
        url=None,
        mime="application/epub+zip",
        record_type="file",
        title=title or path.stem,
        text=final_text,
        span=None,
        lang=lang,
        meta={"epub_strategy": args.epub_strategy}
    )]

def process_text(path: Path, args) -> List[Record]:
    txt = read_text_file(path)
    title = None
    for line in txt.splitlines():
        if line.strip():
            title = line.strip()
            break
    mime = mimetypes.guess_type(str(path))[0] or "text/plain"
    lang = detect_language(txt) if args.lang_detect else None
    return [Record(
        id=str(path.as_posix()),
        parent_id=None,
        source_path=str(path.resolve()),
        url=None,
        mime=mime,
        record_type="file",
        title=title or path.stem,
        text=txt,
        span=None,
        lang=lang,
        meta=None
    )]

# Global semaphore for LLM calls (set in main)
LLM_SEM: Optional[threading.BoundedSemaphore] = None

CODE_SUFFIX_LANG = {
    ".py":"Python",".ipynb":"Jupyter",".js":"JavaScript",".ts":"TypeScript",".tsx":"TSX",".jsx":"JSX",
    ".java":"Java",".c":"C",".cpp":"C++",".cc":"C++",".h":"C/C++ header",".hpp":"C++ header",
    ".rs":"Rust",".go":"Go",".rb":"Ruby",".php":"PHP",".cs":"C#",".swift":"Swift",".kt":"Kotlin",".m":"Objective-C",
    ".sh":"Shell",".bat":"Batch",".ps1":"PowerShell",".sql":"SQL"
}

def process_code_llm(path: Path, args) -> List[Record]:
    maxb = max(1, args.code_max_bytes)
    b = path.read_bytes()
    trunc = False
    if len(b) > maxb:
        b = b[:maxb]; trunc = True
    try:
        content = b.decode("utf-8")
    except Exception:
        content = b.decode("latin-1", errors="replace")
    suffix = path.suffix.lower()
    lang_hint = CODE_SUFFIX_LANG.get(suffix, "Code")
    prompt = (
        f"File: {path.name} (language: {lang_hint})\n"
        "Task: Explain what this file does in 5–10 tight bullet points.\n"
        "Include: purpose, key functions/classes, inputs/outputs, side effects (I/O, network, env), external deps.\n"
        "Avoid: stylistic critique and rewrites. Be precise.\n\n"
        "Code:\n" + content + ("\n\n[TRUNCATED]" if trunc else "")
    )
    if LLM_SEM is not None:
        with LLM_SEM:
            resp = ollama_generate(args.ollama_host, args.code_llm, prompt, options={"temperature": 0.2})
    else:
        resp = ollama_generate(args.ollama_host, args.code_llm, prompt, options={"temperature": 0.2})
    text = sanitize_llm_text(resp.strip())
    title = f"{path.name} — summary"
    lang = detect_language(text) if args.lang_detect else None
    return [Record(
        id=str(path.as_posix()),
        parent_id=None,
        source_path=str(path.resolve()),
        url=None,
        mime="text/x-code-summary",
        record_type="code-summary",
        title=title,
        text=text,
        span=None,
        lang=lang,
        meta={"model": args.code_llm, "truncated": "yes" if trunc else "no", "lang_hint": lang_hint}
    )]

# -------------------------
# Whisper-base ASR
# -------------------------

def get_audio_duration(audio_path: Path, ffprobe_bin: str) -> float:
    info = ffprobe_json(ffprobe_bin, audio_path)
    if not info:
        return 0.0
    try:
        return float(info.get("format", {}).get("duration") or 0.0)
    except Exception:
        return 0.0

def slice_audio(audio_path: Path, out_dir: Path, num_slices: int, overlap_sec: float, ffprobe_bin: str, ffmpeg_bin: str) -> List[Tuple[Path, float, float]]:
    duration = get_audio_duration(audio_path, ffprobe_bin)
    if duration <= 0:
        return [(audio_path, 0.0, 0.0)]
    length = duration / max(1, num_slices)
    slices: List[Tuple[Path, float, float]] = []
    for i in range(num_slices):
        start = max(0.0, i * length - (overlap_sec if i > 0 else 0.0))
        end = min(duration, (i + 1) * length + (overlap_sec if i < num_slices - 1 else 0.0))
        fn = out_dir / f"slice_{i:02d}.wav"
        cmd = [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start}", "-to", f"{end}",
            "-i", str(audio_path), "-acodec", "copy", str(fn)
        ]
        res = run_cmd(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg slice failed for {audio_path.name} [{i}]")
        slices.append((fn, start, end))
    return slices

_WHISPER_MODEL = None

def _resolve_whisper_device(flag: str) -> Optional[str]:
    if flag and flag != "auto":
        return flag
    try:
        if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def _whisper_pool_init(model_name: str, device: Optional[str] = None):
    global _WHISPER_MODEL
    if whisper is None:
        raise RuntimeError("Whisper package is required (pip install -U openai-whisper)")
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    if device in (None, "auto"):
        device = _resolve_whisper_device("auto")
    try:
        _WHISPER_MODEL = whisper.load_model(model_name, device=device)
    except TypeError:
        _WHISPER_MODEL = whisper.load_model(model_name)

def _transcribe_slice(task: str, tup: Tuple[Path, int, str]) -> Tuple[int, str]:
    global _WHISPER_MODEL
    slice_path, idx, _vid = tup
    res = _WHISPER_MODEL.transcribe(str(slice_path), task=task)
    text = (res.get("text") or "").strip()
    return idx, text

def merge_transcripts(files_idx_text: List[Tuple[int, str]], max_overlap_words: int) -> str:
    files_idx_text.sort(key=lambda x: x[0])
    merged_words: List[str] = []
    prev_words: List[str] = []
    for i, txt in files_idx_text:
        words = (txt or "").split()
        if merged_words and prev_words:
            p_tail = prev_words[-max_overlap_words:]
            c_head = words[:max_overlap_words]
            L = min(len(p_tail), len(c_head))
            best = 0
            for n in range(L, 4, -1):
                if p_tail[-n:] == c_head[:n]:
                    best = n
                    break
            if best:
                words = words[best:]
        merged_words += words
        prev_words = words
    return " ".join(merged_words).strip()

def process_media(path: Path, args) -> List[Record]:
    probe = ffprobe_json(args.ffprobe, path)
    duration_s = None
    if probe:
        try:
            duration_s = float(probe.get("format", {}).get("duration") or 0.0)
        except Exception:
            duration_s = None
    if duration_s and duration_s > args.max_av_duration:
        raise RuntimeError(f"Media too long ({duration_s:.1f}s > cap {args.max_av_duration}s)")

    tmpdir = Path(tempfile.mkdtemp(prefix="av_"))
    wav_path = tmpdir / "audio.wav"
    ok = extract_audio_wav(args.ffmpeg, path, wav_path)
    if not ok or not wav_path.exists():
        try: shutil.rmtree(tmpdir)
        except Exception: pass
        raise RuntimeError("ffmpeg audio extraction failed")

    slice_dir = tmpdir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    nslices = max(1, args.num_slices)
    slices = slice_audio(wav_path, slice_dir, nslices, args.overlap_sec, args.ffprobe, args.ffmpeg)

    mpw = args.mp_workers or len(slices)
    device = _resolve_whisper_device(args.whisper_device)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=mpw, initializer=_whisper_pool_init, initargs=(args.whisper_model, device))
    try:
        jobs = [(fp, i, path.stem) for i, (fp, _s, _e) in enumerate(slices)]
        results = pool.starmap(_transcribe_slice, [(args.asr_task, j) for j in jobs])
    except BaseException:
        try:
            pool.terminate()
        finally:
            pool.join()
        raise
    else:
        pool.close()
        pool.join()

    joined_text = merge_transcripts(results, args.max_overlap_words)
    joined_text = sanitize_llm_text(joined_text)
    lang = "en" if args.asr_task == "translate" else (detect_language(joined_text) if args.lang_detect else None)
    mime = mimetypes.guess_type(str(path))[0] or "audio/wav"

    records: List[Record] = []
    if args.emit_av in ("slices", "both"):
        for i, (fp, s, e) in enumerate(slices):
            seg_txt = next((t for idx, t in results if idx == i), "")
            seg_txt = sanitize_llm_text(seg_txt)
            seg_lang = "en" if args.asr_task == "translate" else (detect_language(seg_txt) if args.lang_detect else None)
            records.append(Record(
                id=f"{path.as_posix()}#slice={i+1}",
                parent_id=str(path.as_posix()),
                source_path=str(path.resolve()),
                url=None,
                mime=mime,
                record_type="av",
                title=f"{path.stem} — slice {i+1}",
                text=seg_txt,
                span={"time_start": s, "time_end": e},
                lang=seg_lang,
                meta={"duration_s": f"{duration_s:.1f}" if duration_s else "", "asr_model": f"whisper-{args.whisper_model}", "asr_task": args.asr_task}
            ))
    if args.emit_av in ("joined", "both"):
        records.append(Record(
            id=str(path.as_posix()),
            parent_id=None,
            source_path=str(path.resolve()),
            url=None,
            mime=mime,
            record_type="av",
            title=path.stem,
            text=joined_text,
            span={"duration_s": duration_s},
            lang=lang,
            meta={"duration_s": f"{duration_s:.1f}" if duration_s else "", "asr_model": f"whisper-{args.whisper_model}", "asr_task": args.asr_task}
        ))

    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    return records

# -------------------------
# IO
# -------------------------

def iter_files(root: Path, include_rgx: re.Pattern, exclude_rgx: re.Pattern) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if exclude_rgx.search(rel):
            continue
        if include_rgx.search(rel):
            yield p

# -------------------------
# Main
# -------------------------

def main():
    global LLM_SEM
    args = parse_args()

    root_arg = args.root or args.mirror
    if not root_arg:
        print("[ERROR] Please provide --root <dir> (or legacy --mirror).", file=sys.stderr)
        sys.exit(2)

    root = Path(root_arg).expanduser().resolve()
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (Path(__file__).parent / out_path).resolve()

    ensure_parent(out_path)
    open(out_path, "w", encoding="utf-8").close()
    start_writer(out_path, rotate_mb=args.writer_rotate_mb, queue_max=args.writer_queue)
    print(f"[INFO] Writing JSONL to: {out_path}", flush=True)

    include_rgx = re.compile(args.include, flags=re.I)
    exclude_rgx = re.compile(args.exclude, flags=re.I)

    files = list(iter_files(root, include_rgx, exclude_rgx))
    if not files:
        print("[WARN] No matching files found.", file=sys.stderr)
        stop_writer()
        return

    # Sort for deterministic order with size tiebreaker (small-first inside type)
    priority = {
        ".pdf": 0, ".html": 1, ".htm": 1, ".txt": 2, ".md": 2, ".rst": 2, ".epub": 3,
        ".png": 4, ".jpg": 4, ".jpeg": 4, ".gif": 4, ".bmp": 4, ".tif": 4, ".tiff": 4, ".webp": 4, ".heic": 4,
        ".mp3": 5, ".wav": 5, ".m4a": 5, ".flac": 5, ".ogg": 5, ".opus": 5, ".aac": 5,
        ".mp4": 6, ".mkv": 6, ".mov": 6, ".webm": 6, ".avi": 6, ".ts": 6
    }
    priority.update({k: 7 for k in CODE_SUFFIX_LANG.keys()})
    files.sort(key=lambda p: (priority.get(p.suffix.lower(), 9),
                              (p.stat().st_size if p.exists() else 0),
                              str(p).lower()))

    # Limit parallel LLM calls
    LLM_SEM = threading.BoundedSemaphore(max(1, args.llm_parallel))

    def worker(path: Path) -> Tuple[Path, List[Record], Optional[str]]:
        try:
            suf = path.suffix.lower()
            if suf == ".pdf":
                recs, perr = run_isolated(process_pdf, path, args, timeout=1200)
                if perr:
                    cleaned = try_mutool_clean(path)
                    if cleaned:
                        recs2, perr2 = run_isolated(process_pdf, cleaned, args, timeout=1200)
                        try: cleaned.unlink(missing_ok=True)
                        except Exception: pass
                        if not perr2:
                            return (path, recs2, None)
                    txt = pdftotext_fallback(path)
                    if txt.strip():
                        lang = detect_language(txt) if args.lang_detect else None
                        return (path, [Record(
                            id=str(path.as_posix()),
                            parent_id=None,
                            source_path=str(path.resolve()),
                            url=None,
                            mime="application/pdf",
                            record_type="file",
                            title=(txt.splitlines()[0].strip() if txt else path.stem)[:200],
                            text=txt,
                            span=None,
                            lang=lang,
                            meta={"fallback":"pdftotext"}
                        )], None)
                    return (path, [], perr)
            elif suf in {".html", ".htm"}:
                recs = process_html(path, args)
            elif suf in {".txt", ".md", ".rst"}:
                recs = process_text(path, args)
            elif suf == ".epub":
                recs = process_epub(path, args)
            elif suf in {".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".webp",".heic"}:
                recs = process_image(path, args)
            elif suf in {".mp3",".wav",".m4a",".flac",".ogg",".opus",".aac",".mp4",".mkv",".mov",".webm",".avi",".ts"}:
                recs = process_media(path, args)
            elif suf in set(CODE_SUFFIX_LANG.keys()):
                recs = process_code_llm(path, args)
            else:
                recs = process_text(path, args)
            return (path, recs, None)
        except Exception as e:
            return (path, [], f"{type(e).__name__}: {e}")

    total = len(files)
    iterator = files
    progress = None
    if tqdm is not None:
        progress = tqdm(total=total, desc="Building corpus (per-file)", unit="file")

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(worker, p): p for p in iterator}
        for fut in cf.as_completed(futures):
            path, recs, err = fut.result()
            if err:
                print(f"[ERROR] {path.name}: {err}", file=sys.stderr)
            else:
                enqueue_records_chunked(recs, args.writer_chunk)
            if progress:
                progress.update(1)

    stop_writer()
    if progress:
        progress.close()
    print("[DONE] Corpus build complete.", flush=True)

if __name__ == "__main__":
    main()