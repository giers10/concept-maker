#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Idea → Concept GUI

What this provides
- Drag/drop or add files and folders into a table
- Freeform notes box for thoughts and fragments
- Builds a JSONL knowledge base using corpus_builder.py if present (fallback included)
- Generates a polished "Concept" using a local Ollama model
- Lets you edit, then save into a local Git repo and push to a remote

Notes
- Requires a local Ollama daemon (default http://localhost:11434)
- Default model is the one you requested: "mistral3.2-small:24b" (editable in UI)

Dependencies
- Standard library only for HTTP (urllib); requests is NOT required.
- Optional: tkinterdnd2 for native drag+drop (falls back to buttons if missing)

Run
  python3 concept_gui.py
"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import shutil
import threading
import traceback
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Set

# --- GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Optional native drag/drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    _TKDND_AVAILABLE = True
except Exception:
    TkinterDnD = tk  # type: ignore
    DND_FILES = None  # type: ignore
    _TKDND_AVAILABLE = False

import subprocess
import tempfile
import re
import html

# --- HTTP (stdlib)
import urllib.request
import urllib.error
import webbrowser
import websearch


# -----------------------------
# Simple utilities
# -----------------------------

def human_size(n: int) -> str:
    if n <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    k = 1024.0
    i = int(math.floor(math.log(n, k)))
    i = max(0, min(i, len(units) - 1))
    return f"{n / (k**i):.1f} {units[i]}"


def safe_symlink(src: Path, dst: Path) -> bool:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        return True
    except Exception:
        return False


def copy_or_link(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if safe_symlink(src, dst):
        return dst
    # fallback to copy
    shutil.copy2(src, dst)
    return dst


def read_text_guess(path: Path) -> str:
    try:
        b = path.read_bytes()
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return b.decode(enc)
            except Exception:
                pass
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# -----------------------------
# Corpus building
# -----------------------------

@dataclass
class Record:
    id: str
    title: str
    text: str
    source_path: Optional[str] = None
    mime: Optional[str] = None


class SimpleCorpusBuilder:
    """Very lightweight fallback if corpus_builder.py or deps are unavailable.

    Supports: txt, md, rst, html (strip tags naively), pdf (if PyMuPDF installed).
    """

    def __init__(self) -> None:
        self._fitz = None
        try:
            import fitz  # PyMuPDF
            self._fitz = fitz
        except Exception:
            self._fitz = None

    def build(self, root: Path, out_jsonl: Path) -> List[Record]:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        records: List[Record] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            try:
                if suf in {".txt", ".md", ".rst"}:
                    text = read_text_guess(p)
                    if text.strip():
                        records.append(Record(id=str(p), title=p.stem, text=text, source_path=str(p)))
                elif suf in {".html", ".htm"}:
                    raw = read_text_guess(p)
                    text = self._strip_html(raw)
                    if text.strip():
                        records.append(Record(id=str(p), title=p.stem, text=text, source_path=str(p)))
                elif suf == ".pdf" and self._fitz is not None:
                    text = self._pdf_text(p)
                    if text.strip():
                        records.append(Record(id=str(p), title=p.stem, text=text, source_path=str(p)))
                else:
                    # just record stub entry for unsupported types
                    records.append(Record(id=str(p), title=p.stem, text=f"[Unsupported file type: {suf}]", source_path=str(p)))
            except Exception:
                records.append(Record(id=str(p), title=p.stem, text=f"[Error reading file: {p.name}]", source_path=str(p)))

        # write JSONL
        with out_jsonl.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
        return records

    def _strip_html(self, html: str) -> str:
        # very naive fallback without bs4
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
            return text
        except Exception:
            import re
            txt = re.sub(r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", " ", html, flags=re.S|re.I)
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = re.sub(r"\s+", " ", txt)
            return txt.strip()

    def _pdf_text(self, path: Path) -> str:
        try:
            doc = self._fitz.open(str(path))
            out = []
            for i in range(len(doc)):
                page = doc.load_page(i)
                out.append(page.get_text("text"))
            return "\n\n".join(out)
        except Exception:
            return ""


class ExternalCorpusBuilder:
    """Invokes corpus_builder.py as a subprocess to build a JSONL corpus."""

    def __init__(self, script_path: Path) -> None:
        self.script = script_path

    def build(self, root: Path, out_jsonl: Path, *, workers: int = 4, verbose: bool = False) -> bool:
        cmd = [
            sys.executable,
            str(self.script),
            "--root", str(root),
            "--out", str(out_jsonl),
            "--emit", "auto",
            "--workers", str(max(1, workers)),
            "--llm-parallel", "1",
        ]
        if verbose:
            cmd.append("--verbose")
        try:
            import subprocess
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            ok = proc.returncode == 0 and out_jsonl.exists() and out_jsonl.stat().st_size > 0
            return ok
        except Exception:
            return False


# -----------------------------
# Ollama client (stdlib HTTP)
# -----------------------------

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 600):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, model: str, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error {e.code}: {e.read().decode('utf-8', 'ignore')}")
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")
        try:
            obj = json.loads(body.decode("utf-8", "ignore"))
        except Exception:
            raise RuntimeError("Invalid JSON from Ollama")
        return (obj.get("response") or "").strip()


def _parse_json_strict(s: str) -> Optional[Dict[str, str]]:
    try:
        s = sanitize_llm_text_simple(s)
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


# (Discord webhook removed; now using Git push flow)


# -----------------------------
# Prompting
# -----------------------------
'''
PROMPT_TEMPLATE = """
You are an experienced product strategist and technical writer.
Write a clear, compelling PROJECT CONCEPT based only on the information in Notes and Knowledge Base below.

Output in Markdown with these sections:
- Overview & Problem
- Vision & Goals
- Target Users & Use Cases
- Key Features
- Non-Goals / Out of Scope
- Constraints & Risks
- Rough Architecture / Approach
- Milestones & Next Steps
- Success Criteria & Metrics

Guidelines:
- Be pragmatic, specific, and actionable. Do not fabricate details not present in the sources.
- Pull concrete facts, quotes, and constraints from the knowledge base.
- If something is unknown or ambiguous, add a TODO with a short question.
- Use short paragraphs and bullets; avoid marketing fluff.

Assets to include:
- The following files will be committed alongside README.md in the same folder.
- When relevant, embed images using Markdown (e.g., `![Label](./FILENAME)`), and link documents/other files using `[Label](./FILENAME)`.
- Place links or images in the appropriate sections where they add clarity.

Assets Provided:
{ASSETS}

Notes (from user):
{NOTES}

Knowledge Base (source excerpts):
{KB}
""".strip()
'''

PROMPT_TEMPLATE = """
You are a cross-domain concept developer (product strategist, creative producer, research lead, grant writer).
Turn the sources into a concise, presentable CONCEPT document. Adapt to the domain.

INSTRUCTIONS
1) Detect IDEA TYPE (pick one primary; if unclear, choose closest and add a TODO):
   {Product/Software, Service, Research/Study, Policy/Proposal, Art/Exhibition/Performance, Event/Program,
    Education/Curriculum, Media/Film/Publication, Campaign/Nonprofit, Data/ML/Infrastructure, Game/Interactive,
    Writing/Book/Article, Other}

2) Tone & register:
   - Product/Software → pragmatic PM/tech brief
   - Research → neutral academic project brief
   - Policy → policy memo
   - Art/Exhibition/Performance → curator/producer note (clear, not flowery)
   - Event → producer’s run-of-show style
   - Education → syllabus brief
   - Media/Publication → one-sheet
   - Campaign/Nonprofit → strategy brief
   - Data/ML/Infrastructure → engineering design note
   - Game/Interactive → design doc overview
   - Writing/Book/Article → proposal overview

3) Output Markdown using these core sections (use these exact headings; include only relevant ones):
- Overview & Intent
- Context / Problem (or Opportunity)
- Audience / Stakeholders
- Deliverables / Outputs & Scope
- Approach / Method  (rename to “Methodology”, “Implementation Plan”, “Format & Installation Plan”, etc., to fit the idea type)
- Resources / Budget / Tools  (only if present; else add a short TODO)
- Timeline & Milestones
- Risks, Ethics & Constraints
- Success Criteria / Evaluation
- Open Questions (TODOs)

Add one domain-specific block (only if relevant and supported by sources):
- Product/Software: Key Features; Non-Goals; Rough Architecture; Dependencies & Integration; License.
- Research/Study: Research Questions; Methodology & Data; Expected Contributions; References/Citations.
- Policy/Proposal: Policy Mechanism; Legal/Standards; Impact Assessment; Implementation Steps.
- Art/Exhibition/Performance: Conceptual Frame & References; Medium/Materials; Venue/Spatial Requirements; Tech/AV; Rights/Permissions.
- Event/Program: Programme Outline / Run-of-Show; Roles & Staffing; Logistics & Venue.
- Education/Curriculum: Learning Objectives; Syllabus Outline; Assessment & Materials.
- Media/Film/Publication: Logline & Synopsis; Format; Production Plan; Distribution.
- Campaign/Nonprofit: Theory of Change; Channels & Tactics; KPIs; Partnerships.
- Data/ML/Infrastructure: Data Sources; Models; Architecture Diagram (describe); Privacy & Compliance; Ops/Monitoring.
- Game/Interactive: Core Loop; Mechanics; Narrative; Tech; Monetization (if relevant).
- Writing/Book/Article: Thesis; Outline/Chapters; Sources; Target Readers.

4) Evidence use:
- Use only facts in Notes/KB. If missing, add short TODOs instead of inventing.
- Where a claim relies on a specific source, include a short inline blockquote with “Source: <Path or Title>”.

5) Assets:
- These files are committed alongside README.md. Embed images with Markdown and link documents where they help clarity.

STYLE
- Short paragraphs and bullets; concrete, specific, and actionable. Avoid marketing fluff.
- If dates/budget/ownership are uncertain, show ranges or TODOs.
- Keep a neutral, professional tone adapted to the idea type.

TITLE
- Generate a neutral 2-4 words working title.
- Begin the document with “# {Title}”.

Assets Provided:
{ASSETS}

Notes (from user):
{NOTES}

Knowledge Base (source excerpts):
{KB}
""".strip()


def build_kb_string(records: List[Record], *, max_chars: int = 80000, per_record_cap: int = 4000) -> str:
    """Format records into a compact KB string with caps to avoid overlong prompts."""
    parts: List[str] = []
    budget = max_chars
    for r in records:
        if budget <= 0:
            break
        text = (r.text or "").strip()
        if not text:
            continue
        if len(text) > per_record_cap:
            text = text[:per_record_cap] + "\n...[truncated]"
        title = r.title or (Path(r.source_path).name if r.source_path else r.id)
        header = f"\n---\nSource: {title}\nPath: {r.source_path or ''}\n\n"
        chunk = header + text.strip() + "\n"
        if len(chunk) > budget:
            chunk = chunk[:budget]
        parts.append(chunk)
        budget -= len(chunk)
    return ("\n".join(parts)).strip()


def sanitize_llm_text_simple(s: str) -> str:
    """Remove <think> blocks and surrounding code fences from LLM responses."""
    try:
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S|re.I)
        s = re.sub(r"^\s*```(?:\w+)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        return s.strip()
    except Exception:
        return (s or "").strip()


def md_heading_replace_or_insert(md: str, title: str) -> str:
    """Ensure the first-level heading is the provided title, replacing '# PROJECT CONCEPT' if present."""
    if not md:
        return f"# {title}\n\n"
    lines = md.splitlines()
    # Replace '# PROJECT CONCEPT' (case-insensitive)
    if lines and re.match(r"^\s*#\s+project\s+concept\s*$", lines[0], flags=re.I):
        lines[0] = f"# {title}"
        return "\n".join(lines)
    # If first line is already a H1, leave it as-is; else insert
    if lines and re.match(r"^\s*#\s+", lines[0]):
        return md
    return f"# {title}\n\n" + md


def strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\"'“”‘’]+", "", s)
    s = re.sub(r"[\"'“”‘’]+$", "", s)
    return s


# -----------------------------
# GUI Application
# -----------------------------

class App(TkinterDnD.Tk):  # type: ignore
    def __init__(self):
        super().__init__()
        self.title("Idea → Concept")
        self.geometry("1080x720")
        # Enforce a sensible minimum window size (as before)
        try:
            self.minsize(900, 600)
        except Exception:
            pass

        # Try to set application/window icon from icon.png
        try:
            self._set_app_icon()
        except Exception:
            # Non-fatal: keep running with default icon
            pass

        # State
        self.files: List[Path] = []
        self.records: List[Record] = []
        # Legacy fields kept but no longer used for per-session staging
        self.staging_dir: Optional[Path] = None
        self.corpus_path: Optional[Path] = None
        self.include_map: Dict[str, bool] = {}
        self.file_hashes: Dict[str, str] = {}  # path -> sha256
        self._seen_hashes: Set[str] = set()    # hashes present in corpus.jsonl
        self._ingesting: Set[str] = set()      # hashes currently being ingested
        self._base_dir: Path = Path.cwd() / ".idea-hole"
        self._files_dir: Path = self._base_dir / "files"
        self._corpus_file: Path = self._base_dir / "corpus.jsonl"
        self._sessions_file: Path = self._base_dir / "sessions.jsonl"

        # Defaults
        self.ollama_host = tk.StringVar(value=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        # Default model prompt suggests explicit selection
        self.ollama_model = tk.StringVar(value=os.environ.get("IDEA_HOLE_MODEL", "Select model..."))
        self.git_remote_url = tk.StringVar(value=os.environ.get("IDEA_HOLE_REMOTE", ""))
        # SearXNG URL (for prior-art search)
        try:
            _searx_default = websearch.SEARX_DEFAULT_URL
        except Exception:
            _searx_default = "http://localhost:8888"
        self.searx_url = tk.StringVar(value=os.environ.get("SEARX_URL", _searx_default))

        # Concept metadata
        self.title_var = tk.StringVar(value="")
        self.desc_var = tk.StringVar(value="")

        # Load persisted settings (if any)
        self._load_settings()
        # Prepare unified storage and indexes
        self._init_storage()
        self._build_ui()
        self._maybe_enable_dnd()
        # Dirty tracking and close handler
        self._dirty = False
        try:
            self.protocol('WM_DELETE_WINDOW', self.on_close)
        except Exception:
            pass
        # Initialize last-saved snapshot to current clean state
        try:
            self._last_saved = self._snapshot_session_state()
        except Exception:
            self._last_saved = {
                "title": "",
                "description": "",
                "notes": "",
                "concept": "",
                "files": [],
            }

    def _set_app_icon(self) -> None:
        """Set the window/taskbar/dock icon from local icon.png when possible.

        - Uses Tk's iconphoto on all platforms (PNG supported on Tk 8.6+).
        - On macOS, also tries to set the dock icon via AppKit if available.
        """
        icon_path = Path(__file__).parent / "icon.png"
        if not icon_path.exists():
            return

        # Keep a reference to avoid GC
        try:
            self._icon_photoimage = tk.PhotoImage(file=str(icon_path))
            # Set for this window and as the default for future toplevels
            try:
                self.iconphoto(True, self._icon_photoimage)
            except Exception:
                # Older Tk variants
                self.tk.call('wm', 'iconphoto', self._w, self._icon_photoimage)
        except Exception:
            # PhotoImage may fail if Tk lacks PNG support
            self._icon_photoimage = None

        # On macOS, set the dock icon using AppKit if available (optional)
        if sys.platform == 'darwin':
            try:
                from AppKit import NSImage, NSApplication
                app = NSApplication.sharedApplication()
                nsimg = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
                if nsimg is not None:
                    app.setApplicationIconImage_(nsimg)
            except Exception:
                # AppKit (pyobjc) not available; ignore
                pass

    # --- UI construction
    def _build_ui(self):
        root = self
        # Top controls
        top = ttk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        # Session file actions at top-left
        ttk.Button(top, text="New", command=self.on_new_session).pack(side=tk.LEFT)
        ttk.Button(top, text="Open", command=self.on_open_session).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(top, text="Save", command=self.on_save_session).pack(side=tk.LEFT, padx=(6,0))
        # Status moved to top-right
        self.status = ttk.Label(top, text="Ready", anchor=tk.E)
        self.status.pack(side=tk.RIGHT)

        # Paned layout: left (files + notes), right (concept)
        paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=1)

        # Files table (with controls inside the same frame)
        files_frame = ttk.LabelFrame(left, text="Files (drag & drop or use buttons)")
        files_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=(0,8))

        # Inner container for tree + scrollbar to allow controls below
        files_inner = ttk.Frame(files_frame)
        files_inner.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Prevent scrollbar squashing by using grid with a fixed minsize column
        try:
            files_inner.rowconfigure(0, weight=1)
            files_inner.columnconfigure(0, weight=1)
            files_inner.columnconfigure(1, minsize=14)
        except Exception:
            pass

        cols = ("name", "path", "type", "size", "include")
        self.tree = ttk.Treeview(files_inner, columns=cols, show="headings", height=8)
        for c, w in ("name", 80), ("path", 100), ("type", 20), ("size", 30), ("include", 30):
            heading = "Add to Repo" if c == "include" else c.capitalize()
            self.tree.heading(c, text=heading)
            self.tree.column(c, width=w, anchor=tk.W)
        try:
            self.tree.grid(row=0, column=0, sticky='nsew')
        except Exception:
            self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree_vsb = ttk.Scrollbar(files_inner, orient=tk.VERTICAL, command=self.tree.yview)
        try:
            self.tree_vsb.grid(row=0, column=1, sticky='ns')
        except Exception:
            # Fallback if grid not available
            self.tree_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        # Connect scrollbar normally (always visible; resilient to squashing)
        self.tree.configure(yscrollcommand=self.tree_vsb.set)
        # Click to toggle include checkbox-like column
        self.tree.bind('<Button-1>', self.on_tree_click)
        # No auto-hide for reliability

        # File table controls inside the files frame border
        file_controls = ttk.Frame(files_frame)
        # Slightly more padding at bottom than top
        file_controls.pack(side=tk.TOP, fill=tk.X, padx=(8,0), pady=(4,8))
        ttk.Button(file_controls, text="Add Files", command=self.on_add_files).pack(side=tk.LEFT)
        ttk.Button(file_controls, text="Add Folder", command=self.on_add_folder).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(file_controls, text="Remove Selected", command=self.on_remove_selected).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(file_controls, text="Clear All", command=self.on_clear_all).pack(side=tk.LEFT, padx=(6,0))

        # Notes (with Generate Concept button inside the same frame)
        notes_frame = ttk.LabelFrame(left, text="Notes / Thoughts")
        notes_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8,0))
        # Notes text area with ttk scrollbar that auto-hides
        notes_container = ttk.Frame(notes_frame)
        notes_container.pack(fill=tk.BOTH, expand=True)
        try:
            notes_container.rowconfigure(0, weight=1)
            notes_container.columnconfigure(0, weight=1)
            notes_container.columnconfigure(1, minsize=14)
        except Exception:
            pass
        self.notes = tk.Text(notes_container, height=12, wrap=tk.WORD, undo=True)
        try:
            self.notes.grid(row=0, column=0, sticky='nsew')
        except Exception:
            self.notes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.notes_vsb = ttk.Scrollbar(notes_container, orient=tk.VERTICAL, command=self.notes.yview)
        try:
            self.notes_vsb.grid(row=0, column=1, sticky='ns')
        except Exception:
            # Fallback
            self.notes_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        # Connect scrollbar normally (always visible; resilient to squashing)
        self.notes.configure(yscrollcommand=self.notes_vsb.set)
        # No auto-hide for reliability
        #
        notes_actions = ttk.Frame(notes_frame)
        notes_actions.pack(side=tk.TOP, fill=tk.X, padx=(8,0), pady=(6,6))
        ttk.Button(notes_actions, text="Generate Concept", command=self.on_generate).pack(side=tk.LEFT)
        ttk.Button(notes_actions, text="Find Prior Art", command=self.on_prior_art).pack(side=tk.LEFT, padx=(6,0))

        # Concept editor + metadata
        concept_frame = ttk.LabelFrame(right, text="Concept (editable)")
        concept_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        meta = ttk.Frame(concept_frame)
        # Stack Title and Description vertically; align with section inset
        meta.pack(side=tk.TOP, fill=tk.X, padx=(8,8), pady=(6,6))
        ttk.Label(meta, text="Title:").pack(anchor=tk.W)
        self.title_entry = tk.Entry(meta, textvariable=self.title_var)
        self.title_entry.pack(fill=tk.X, expand=True)
        ttk.Label(meta, text="Description:").pack(anchor=tk.W, pady=(6,0))
        self.desc_entry = tk.Entry(meta, textvariable=self.desc_var)
        self.desc_entry.pack(fill=tk.X, expand=True)

        # Concept text area with ttk scrollbar that auto-hides
        concept_container = ttk.Frame(concept_frame)
        concept_container.pack(fill=tk.BOTH, expand=True)
        try:
            concept_container.rowconfigure(0, weight=1)
            concept_container.columnconfigure(0, weight=1)
            concept_container.columnconfigure(1, minsize=14)
        except Exception:
            pass
        self.concept = tk.Text(concept_container, wrap=tk.WORD, undo=True)
        try:
            self.concept.grid(row=0, column=0, sticky='nsew')
        except Exception:
            self.concept.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.concept_vsb = ttk.Scrollbar(concept_container, orient=tk.VERTICAL, command=self.concept.yview)
        try:
            self.concept_vsb.grid(row=0, column=1, sticky='ns')
        except Exception:
            # Fallback
            self.concept_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        # Connect scrollbar normally (always visible; resilient to squashing)
        self.concept.configure(yscrollcommand=self.concept_vsb.set)
        # No auto-hide for reliability
        #
        # Match Title/Description entry backgrounds to textarea background
        try:
            _bg = self.concept.cget('background')
            self.title_entry.configure(bg=_bg)
            self.desc_entry.configure(bg=_bg)
        except Exception:
            pass

        # Push to Repo under the concept editor (aligned horizontally with Generate Concept)
        concept_actions = ttk.Frame(concept_frame)
        concept_actions.pack(side=tk.TOP, fill=tk.X, padx=(8,0), pady=(6,6))
        # PDF preview button
        self.preview_btn = ttk.Button(concept_actions, text="Preview PDF", command=self.on_preview)
        self.preview_btn.pack(side=tk.LEFT)
        self.push_btn = ttk.Button(concept_actions, text="Push to Repo", command=self.on_push)
        self.push_btn.pack(side=tk.LEFT, padx=(6,0))

        # Bottom bar
        bottom = ttk.Frame(root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0,8))
        # Moved Ollama/Model/Remote inputs to bottom bar
        ttk.Label(bottom, text="Ollama host:").pack(side=tk.LEFT)
        ttk.Entry(bottom, textvariable=self.ollama_host, width=22).pack(side=tk.LEFT, padx=(4,10))
        ttk.Label(bottom, text="Model:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(bottom, textvariable=self.ollama_model, state="readonly", width=20, values=self._get_model_values())
        self.model_combo.pack(side=tk.LEFT, padx=(4,10))
        ttk.Label(bottom, text="Remote git repository:").pack(side=tk.LEFT)
        ttk.Entry(bottom, textvariable=self.git_remote_url, width=40).pack(side=tk.LEFT, padx=(4,10))
        ttk.Label(bottom, text="SearXNG URL:").pack(side=tk.LEFT)
        ttk.Entry(bottom, textvariable=self.searx_url, width=26).pack(side=tk.LEFT, padx=(4,0))

        # Enable/disable push button based on fields
        self.title_var.trace_add('write', lambda *a: (self.update_push_state(), self._set_dirty(True)))
        self.desc_var.trace_add('write', lambda *a: (self.update_push_state(), self._set_dirty(True)))
        self.concept.bind('<<Modified>>', self._on_concept_modified)
        try:
            self.notes.bind('<<Modified>>', self._on_notes_modified)
        except Exception:
            pass
        # Persist settings
        self.ollama_host.trace_add('write', lambda *a: self._save_settings())
        self.ollama_model.trace_add('write', lambda *a: self._save_settings())
        self.git_remote_url.trace_add('write', lambda *a: self._save_settings())
        self.searx_url.trace_add('write', lambda *a: self._save_settings())
        # Optional tool overrides would be persisted if we expose them later
        self.update_push_state()
        # Ensure placeholder if saved model not present
        try:
            values = list(self.model_combo["values"]) if hasattr(self, 'model_combo') else []
            if self.ollama_model.get() not in values:
                self.ollama_model.set("Select model...")
        except Exception:
            pass

    def _extract_title_desc(self, concept_md: str, client: Optional[OllamaClient] = None) -> (Optional[str], Optional[str]):
        """Second-pass LLM call to extract title and description as JSON."""
        try:
            client = client or OllamaClient(host=self.ollama_host.get())
            prompt = (
                "Extract a concise title and a one-sentence description from the following concept.\n"
                "- Title: <= 50 chars (3-5 words).\n- Description: <= 120 chars, (one sentence) no trailing period.\n"
                "Return ONLY strict JSON with keys 'title' and 'description'.\n\nCONCEPT:\n" + concept_md
            )
            raw = client.generate(self.ollama_model.get(), prompt)
            obj = _parse_json_strict(raw) or {}
            title = strip_wrapping_quotes(str(obj.get('title') or '').strip()) or None
            desc = strip_wrapping_quotes(str(obj.get('description') or '').strip()) or None
            return title, desc
        except Exception:
            return None, None

    def _maybe_enable_dnd(self):
        if _TKDND_AVAILABLE and DND_FILES is not None:
            try:
                self.tree.drop_target_register(DND_FILES)
                self.tree.dnd_bind('<<Drop>>', self._on_drop)
                self._set_status("Drag & drop enabled")
            except Exception:
                self._set_status("Drag & drop not available (fallback to buttons)")
        else:
            self._set_status("Drag & drop not available (fallback to buttons)")

    # --- File ops
    def _on_drop(self, event):  # type: ignore
        raw = event.data
        paths = self._parse_dnd_paths(raw)
        self._add_paths(paths)

    @staticmethod
    def _parse_dnd_paths(s: str) -> List[Path]:
        # Handles Tcl-style space-separated paths with braces
        out: List[Path] = []
        buf = ''
        in_brace = False
        for ch in s:
            if ch == '{':
                in_brace = True
                if buf:
                    buf += ch
                continue
            if ch == '}':
                in_brace = False
                if buf:
                    buf += ch
                continue
            if ch == ' ' and not in_brace:
                p = buf.strip().strip('{}')
                if p:
                    out.append(Path(p))
                buf = ''
            else:
                buf += ch
        if buf.strip():
            out.append(Path(buf.strip().strip('{}')))
        # Expand directories
        final: List[Path] = []
        for p in out:
            if p.is_dir():
                for q in p.rglob('*'):
                    if q.is_file():
                        final.append(q)
            elif p.exists():
                final.append(p)
        return final

    def _add_paths(self, paths: List[Path]):
        # Expand directories into files
        expanded: List[Path] = []
        for p in paths:
            if p.is_dir():
                for q in p.rglob('*'):
                    if q.is_file():
                        expanded.append(q)
            elif p.is_file():
                expanded.append(p)

        added = 0
        new_files: List[Path] = []
        for p in expanded:
            if p.is_file() and p not in self.files:
                self.files.append(p)
                self.include_map[str(p)] = True
                try:
                    size = human_size(p.stat().st_size)
                except Exception:
                    size = "?"
                self.tree.insert('', tk.END, values=(p.name, str(p), p.suffix.lower(), size, '✓'))
                added += 1
                new_files.append(p)
        if added:
            self._set_status(f"Added {added} item(s)")
            self._set_dirty(True)
            # Kick off background ingest for any new files not in corpus
            threading.Thread(target=self._ensure_corpus_for_files, args=(new_files,), kwargs={"blocking": True}, daemon=True).start()
        #

    def on_add_files(self):
        paths = filedialog.askopenfilenames(title="Select files")
        if not paths:
            return
        self._add_paths([Path(p) for p in paths])

    def on_add_folder(self):
        path = filedialog.askdirectory(title="Select folder")
        if not path:
            return
        self._add_paths([Path(path)])

    def on_remove_selected(self):
        sels = self.tree.selection()
        if not sels:
            return
        removed = 0
        for item in sels:
            vals = self.tree.item(item, 'values')
            if vals:
                p = Path(vals[1])
                if p in self.files:
                    self.files.remove(p)
                try:
                    self.include_map.pop(str(p), None)
                except Exception:
                    pass
            self.tree.delete(item)
            removed += 1
        if removed:
            self._set_status(f"Removed {removed} item(s)")
            self._set_dirty(True)
        #

    def on_clear_all(self):
        self.tree.delete(*self.tree.get_children())
        self.files.clear()
        self.include_map.clear()
        self._set_status("Cleared")
        self._set_dirty(True)
        #

    def on_tree_click(self, event):
        # Toggle include column when clicked
        region = self.tree.identify_region(event.x, event.y)
        if region != 'cell':
            return
        column = self.tree.identify_column(event.x)  # e.g., '#1' .. '#5'
        if column != '#5':  # include column index
            return
        item = self.tree.identify_row(event.y)
        if not item:
            return
        vals = list(self.tree.item(item, 'values'))
        if not vals or len(vals) < 5:
            return
        path = str(vals[1])
        current = self.include_map.get(path, True)
        new = not current
        self.include_map[path] = new
        vals[4] = '✓' if new else ''
        self.tree.item(item, values=tuple(vals))
        self._set_dirty(True)
        return 'break'

    # --- Unified storage helpers
    def _init_storage(self):
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._files_dir.mkdir(parents=True, exist_ok=True)
            if not self._corpus_file.exists():
                self._corpus_file.write_text("", encoding="utf-8")
            if not self._sessions_file.exists():
                self._sessions_file.write_text("", encoding="utf-8")
            # Build in-memory index of seen file hashes
            self._seen_hashes = set()
            with self._corpus_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line or not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    h = obj.get("file_hash")
                    if h:
                        self._seen_hashes.add(str(h))
        except Exception:
            # best effort; index stays possibly empty
            self._seen_hashes = set()

    def _compute_file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        try:
            with path.open("rb") as fh:
                while True:
                    b = fh.read(1024 * 1024)
                    if not b:
                        break
                    h.update(b)
        except Exception:
            # Fall back to name+mtime if unreadable
            st = None
            try:
                st = path.stat()
            except Exception:
                pass
            h.update((str(path) + "|" + str(getattr(st, 'st_mtime', 0.0))).encode("utf-8", "ignore"))
        return h.hexdigest()

    def _ensure_file_symlink(self, src: Path, file_hash: str) -> Path:
        # name pattern: {hash}__basename
        dst = self._files_dir / f"{file_hash}__{src.name}"
        try:
            if not dst.exists():
                copy_or_link(src, self._files_dir)
        except Exception:
            pass
        return dst

    def _ingest_single_file(self, src: Path, file_hash: str, *, verbose: bool = False) -> bool:
        # Build corpus entries for a single file into the unified corpus
        try:
            tmp_dir = Path.cwd() / ".idea-hole" / "ingest_tmp" / file_hash
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass
            tmp_dir.mkdir(parents=True, exist_ok=True)
            # Put a copy or symlink inside tmp_dir
            local_file = copy_or_link(src, tmp_dir)

            # Try external builder first
            external = None
            script = Path(__file__).parent / "corpus_builder.py"
            if script.exists():
                external = ExternalCorpusBuilder(script)

            tmp_out = tmp_dir / "out.jsonl"
            ok = False
            if external is not None:
                self._set_status(f"Indexing {src.name} (external)…")
                ok = external.build(tmp_dir, tmp_out, workers=2, verbose=verbose)
            if not ok:
                self._set_status(f"Indexing {src.name} (simple)…")
                try:
                    simple = SimpleCorpusBuilder()
                    recs = simple.build(tmp_dir, tmp_out)
                    ok = bool(recs)
                except Exception:
                    ok = False

            # Append to unified corpus with added metadata
            if ok and tmp_out.exists():
                ts = int(time.time())
                with tmp_out.open("r", encoding="utf-8") as fh_in, self._corpus_file.open("a", encoding="utf-8") as fh_out:
                    for line in fh_in:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        obj["file_hash"] = file_hash
                        # Ensure source_path is the real source
                        obj["source_path"] = str(src.resolve())
                        obj.setdefault("mime", obj.get("mime") or None)
                        obj["added_at"] = ts
                        try:
                            fh_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        except Exception:
                            # As last resort, write ascii-only
                            fh_out.write(json.dumps(obj) + "\n")
                # Update in-memory index
                self._seen_hashes.add(file_hash)
                return True
            return False
        finally:
            # Cleanup tmp_dir
            try:
                shutil.rmtree(Path.cwd() / ".idea-hole" / "ingest_tmp" / file_hash)
            except Exception:
                pass

    def _ensure_corpus_for_files(self, paths: List[Path], *, blocking: bool = True):
        if not paths:
            return
        # Compute hashes and create symlinks; ingest missing
        to_ingest: List[tuple[Path, str]] = []
        for p in paths:
            try:
                h = self._compute_file_hash(p)
            except Exception:
                continue
            self.file_hashes[str(p)] = h
            self._ensure_file_symlink(p, h)
            if h not in self._seen_hashes and h not in self._ingesting:
                to_ingest.append((p, h))

        if not to_ingest:
            return

        def _run():
            try:
                for src, h in to_ingest:
                    self._ingesting.add(h)
                    try:
                        self._ingest_single_file(src, h, verbose=False)
                    finally:
                        try:
                            self._ingesting.remove(h)
                        except Exception:
                            pass
                self._set_status("Corpus up to date")
            except Exception:
                self._set_status("Corpus ingest failed (see logs)")

        if blocking:
            _run()
        else:
            threading.Thread(target=_run, daemon=True).start()

    def _load_records_for_hashes(self, hashes: Set[str]) -> List[Record]:
        out: List[Record] = []
        if not hashes:
            return out
        try:
            with self._corpus_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if str(obj.get("file_hash") or "") not in hashes:
                        continue
                    out.append(Record(
                        id=str(obj.get("id", "")),
                        title=str(obj.get("title", "")),
                        text=str(obj.get("text", "")),
                        source_path=str(obj.get("source_path", "")) if obj.get("source_path") else None,
                        mime=str(obj.get("mime", "")) if obj.get("mime") else None,
                    ))
        except Exception:
            pass
        return out

    def ensure_and_load_kb_for_current(self) -> List[Record]:
        # Ensure corpus contains all current files, then load those records
        self._set_status("Checking corpus…")
        self._ensure_corpus_for_files(self.files, blocking=True)
        hashes = {self.file_hashes.get(str(p)) for p in self.files}
        hashes = {h for h in hashes if h}
        recs = self._load_records_for_hashes(hashes)
        self.records = recs
        self.corpus_path = self._corpus_file
        self._set_status(f"KB ready with {len(recs)} records")
        return recs

    # --- Concept generation
    def on_generate(self):
        # Ensure a model is selected
        model = (self.ollama_model.get() or "").strip()
        if not model or model == "Select model...":
            self._ui(lambda: messagebox.showinfo("Select model", "Please select a model first."))
            return
        if not self.files and not self.notes.get("1.0", tk.END).strip():
            self._ui(lambda: messagebox.showinfo("Nothing to do", "Add files or write some notes first."))
            return
        threading.Thread(target=self._generate_concept_thread, daemon=True).start()

    def _generate_concept_thread(self):
        try:
            self._set_status("Preparing knowledge base…")
            records = self.ensure_and_load_kb_for_current()
            notes = self.notes.get("1.0", tk.END).strip()
            kb = build_kb_string(records)
            assets = [p for p in self.files if self.include_map.get(str(p), True)]
            assets_str = "\n".join(f"- {Path(p).name}" for p in assets) or "(none)"
            prompt = (
                PROMPT_TEMPLATE
                .replace("{NOTES}", notes or "(none)")
                .replace("{KB}", kb or "(empty)")
                .replace("{ASSETS}", assets_str)
            )
            self._set_status("Querying Ollama…")
            client = OllamaClient(host=self.ollama_host.get())
            concept_md = client.generate(model=self.ollama_model.get(), prompt=prompt)
            concept_md = sanitize_llm_text_simple(concept_md)
            # Extract title/description via second structured call
            title, desc = self._extract_title_desc(concept_md, client)
            if not desc:
                desc = ""
            # Ensure top heading matches title
            concept_md = md_heading_replace_or_insert(concept_md, title)
            if not concept_md.strip():
                raise RuntimeError("Empty response from model")
            self._set_status("Concept generated")
            # schedule UI update (concept, title, desc)
            def _apply():
                self.concept.delete("1.0", tk.END)
                self.concept.insert("1.0", concept_md)
                self.title_var.set(title)
                self.desc_var.set(strip_wrapping_quotes(desc)[:120])
                self.update_push_state()
                self._set_dirty(True)
            self._ui(_apply)
        except Exception as e:
            self._set_status("Generation failed")
            msg = f"Failed to generate concept:\n{e}"
            self._ui(lambda m=msg: messagebox.showerror("Error", m))
            return
        # After a successful apply, attempt autosave shortly after
        try:
            self.after(150, self._autosave_after_generation)
        except Exception:
            pass

    # --- Rebuild KB only
    def on_rebuild_kb(self):
        threading.Thread(target=self._rebuild_kb_thread, daemon=True).start()

    def _rebuild_kb_thread(self):
        try:
            self.ensure_and_load_kb_for_current()
        except Exception as e:
            self._set_status("KB refresh failed")
            msg = f"KB refresh failed:\n{e}"
            self._ui(lambda m=msg: messagebox.showerror("Error", m))

    # --- Sessions: New / Open / Save
    def on_new_session(self):
        # Prompt to save if there are unsaved changes
        if not self._maybe_save_if_dirty():
            return
        try:
            self.on_clear_all()
            self.title_var.set("")
            self.desc_var.set("")
            self.notes.delete("1.0", tk.END)
            self.concept.delete("1.0", tk.END)
            self._set_status("New session")
            self._set_dirty(False)
            # Reset last-saved snapshot to clean baseline
            self._last_saved = self._snapshot_session_state()
            self.update_push_state()
        except Exception:
            pass

    def on_save_session(self):
        ok = self._save_session(confirm_overwrite=True, autosave=False)
        if ok:
            self._set_dirty(False)

    def on_open_session(self):
        # Guard unsaved changes
        if not self._maybe_save_if_dirty():
            return
        sessions = self._load_all_sessions()
        if not sessions:
            self._ui(lambda: messagebox.showinfo("No sessions", "No sessions found. Save one first."))
            return

        # Simple chooser dialog
        dlg = tk.Toplevel(self)
        dlg.title("Open Session")
        dlg.geometry("600x360")
        frm = ttk.Frame(dlg)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        cols = ("title", "saved")
        tv = ttk.Treeview(frm, columns=cols, show="headings")
        tv.heading("title", text="Title")
        tv.heading("saved", text="Saved")
        tv.column("title", width=360)
        tv.column("saved", width=180)
        vs = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tv.yview)
        tv.configure(yscrollcommand=vs.set)
        tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vs.pack(side=tk.RIGHT, fill=tk.Y)

        def fmt_ts(ts: int) -> str:
            try:
                return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            except Exception:
                return str(ts)

        def refresh(selected_saved_at: Optional[int] = None):
            nonlocal sessions
            tv.delete(*tv.get_children())
            sessions = self._load_all_sessions()
            for idx, s in enumerate(sessions):
                tv.insert('', tk.END, iid=str(idx), values=(s.get("title",""), fmt_ts(int(s.get("saved_at", 0)))))
            # Try to reselect
            if selected_saved_at is not None:
                for iid in tv.get_children(''):
                    i = int(iid)
                    try:
                        if int(sessions[i].get("saved_at", -1)) == int(selected_saved_at):
                            tv.selection_set(iid)
                            tv.see(iid)
                            break
                    except Exception:
                        pass

        refresh()

        sel_idx = {"i": None}

        def _choose_and_close():
            sel = tv.selection()
            if not sel:
                return
            try:
                sel_idx["i"] = int(sel[0])
            except Exception:
                sel_idx["i"] = None
            dlg.destroy()

        tv.bind('<Double-1>', lambda _e: _choose_and_close())
        btns = ttk.Frame(dlg)
        btns.pack(fill=tk.X, padx=8, pady=8)
        def _rename():
            sel = tv.selection()
            if not sel:
                return
            i = int(sel[0]); s = sessions[i]
            # Prompt for new title
            ttl = tk.Toplevel(dlg); ttl.title("Rename Session"); ttl.geometry("420x120")
            f = ttk.Frame(ttl); f.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            ttk.Label(f, text="New title:").pack(anchor=tk.W)
            var = tk.StringVar(value=s.get("title",""))
            ent = ttk.Entry(f, textvariable=var)
            ent.pack(fill=tk.X)
            ent.focus_set()
            btnf = ttk.Frame(f); btnf.pack(fill=tk.X, pady=(8,0))
            def _ok():
                new_title = (var.get() or "").strip()
                if not new_title:
                    ttl.destroy(); return
                # If another session with new_title exists (not this one), confirm overwrite
                exists = any((x.get("title") == new_title and int(x.get("saved_at",-1)) != int(s.get("saved_at",-1))) for x in sessions)
                if exists:
                    if not messagebox.askyesno("Overwrite", f"A session titled '{new_title}' exists. Overwrite it?"):
                        return
                # Rewrite sessions: remove all with new_title (except current); then update current title
                all_s = self._load_all_sessions()
                cur_sa = int(s.get("saved_at", -1))
                filtered = [x for x in all_s if x.get("title") != new_title and int(x.get("saved_at",-1)) != cur_sa]
                s2 = dict(s); s2["title"] = new_title
                filtered.append(s2)
                self._write_all_sessions(filtered)
                ttl.destroy()
                refresh(selected_saved_at=cur_sa)
            ttk.Button(btnf, text="OK", command=_ok).pack(side=tk.RIGHT)
            ttk.Button(btnf, text="Cancel", command=ttl.destroy).pack(side=tk.RIGHT, padx=(0,8))
            ttl.transient(dlg); ttl.grab_set(); self.wait_window(ttl)

        def _delete():
            sel = tv.selection()
            if not sel:
                return
            i = int(sel[0]); s = sessions[i]
            if not messagebox.askyesno("Delete Session", f"Delete '{s.get('title','')}'?"):
                return
            all_s = self._load_all_sessions()
            sa = int(s.get("saved_at", -1))
            kept = [x for x in all_s if int(x.get("saved_at", -1)) != sa]
            self._write_all_sessions(kept)
            refresh()

        ttk.Button(btns, text="Delete", command=_delete).pack(side=tk.LEFT)
        ttk.Button(btns, text="Rename", command=_rename).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(btns, text="Open", command=_choose_and_close).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=(0,8))

        dlg.transient(self)
        dlg.grab_set()
        self.wait_window(dlg)

        i = sel_idx.get("i")
        if i is None:
            return
        s = sessions[i]
        # Apply session
        try:
            self.on_clear_all()
            self.title_var.set(s.get("title", ""))
            self.desc_var.set(s.get("description", ""))
            self.notes.delete("1.0", tk.END)
            self.notes.insert("1.0", s.get("notes", ""))
            self.concept.delete("1.0", tk.END)
            self.concept.insert("1.0", s.get("concept", ""))

            files = s.get("files") or []
            # Re-add files; prefer original path, fallback to symlink by hash
            resolved: List[Path] = []
            for f in files:
                p = Path(f.get("path") or "")
                h = str(f.get("file_hash") or "")
                if not p.exists() and h:
                    # try to locate symlink with hash prefix
                    try:
                        for q in self._files_dir.glob(f"{h}__*"):
                            p = q
                            break
                    except Exception:
                        pass
                if p.exists():
                    resolved.append(p)
            self._add_paths(resolved)
            # Restore include flags
            for f in files:
                p = str(f.get("path") or "")
                inc = bool(f.get("include", True))
                if p:
                    self.include_map[p] = inc
            # Reflect include flags in UI rows
            for item in self.tree.get_children(''):
                vals = list(self.tree.item(item, 'values'))
                if vals and len(vals) >= 5:
                    path = str(vals[1])
                    vals[4] = '✓' if self.include_map.get(path, True) else ''
                    self.tree.item(item, values=tuple(vals))

            self.update_push_state()
            self._set_status("Session loaded")
            self._set_dirty(False)
            # Save snapshot of the loaded state for accurate dirty detection
            self._last_saved = self._snapshot_session_state()
        except Exception as e:
            self._ui(lambda m=f"Failed to open session:\n{e}": messagebox.showerror("Error", m))

    # (Description generation now runs automatically after concept generation.)

    # --- Push to Repo
    def update_push_state(self):
        title_ok = bool(self.title_var.get().strip())
        desc_ok = bool(self.desc_var.get().strip())
        concept_ok = bool(self.concept.get("1.0", tk.END).strip())
        state = tk.NORMAL if (title_ok and desc_ok and concept_ok) else tk.DISABLED
        try:
            self.push_btn.configure(state=state)
        except Exception:
            pass

    def _on_concept_modified(self, _evt=None):
        try:
            if self.concept.edit_modified():
                self.update_push_state()
                self._set_dirty(True)
                self.concept.edit_modified(False)
        except Exception:
            pass

    def _on_notes_modified(self, _evt=None):
        try:
            if self.notes.edit_modified():
                self._set_dirty(True)
                self.notes.edit_modified(False)
        except Exception:
            pass

    def _open_path_default(self, path: Path) -> bool:
        try:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', str(path)])
                return True
            elif os.name == 'nt':
                os.startfile(str(path))  # type: ignore[attr-defined]
                return True
            else:
                subprocess.Popen(['xdg-open', str(path)])
                return True
        except Exception:
            return False

    def on_preview(self):
        """Generate a PDF preview using the same conversion flow as Push, but in a temp folder.
        - Writes a temporary README.md and copies selected assets into .idea-hole/preview/<slug>-preview
        - Exports PDF into that same folder and opens it with the OS default viewer
        """
        concept_text = self.concept.get("1.0", tk.END).strip()
        if not concept_text:
            self._ui(lambda: messagebox.showinfo("No concept", "Please generate or paste the concept text to preview."))
            return
        threading.Thread(target=self._preview_thread, daemon=True).start()

    def _preview_thread(self):
        try:
            title = (self.title_var.get() or "").strip()
            slug = self._slug(title or "preview")

            # Prepare preview workspace
            base = Path.cwd() / ".idea-hole" / "preview" / f"{slug}-preview"
            try:
                if base.exists():
                    shutil.rmtree(base)
            except Exception:
                pass
            base.mkdir(parents=True, exist_ok=True)

            # Write Markdown
            md_path = base / "README.md"
            md_text = self.concept.get("1.0", tk.END).strip()
            md_path.write_text(md_text, encoding="utf-8")

            # Copy selected assets into preview workspace
            assets = [p for p in self.files if self.include_map.get(str(p), True)]
            for src in assets:
                try:
                    dst = base / src.name
                    if dst.name.lower() in {"readme.md", f"{slug}-concept.pdf".lower(), f"{slug}-preview.pdf".lower()}:
                        dst = base / f"asset-{src.name}"
                    shutil.copy2(src, dst)
                except Exception:
                    pass

            # Export PDF in the same workspace folder
            pdf_path = base / f"{slug}-preview.pdf"
            self._set_status("Exporting PDF preview…")
            ok_pdf = self._convert_markdown_to_pdf(md_path, pdf_path)
            if not ok_pdf:
                logs_dir = Path.cwd() / ".idea-hole" / "logs"
                log_path = logs_dir / f"pdf_export_{base.name}.log"
                def _show_pdf_error():
                    msg = "PDF preview failed."
                    if log_path.exists():
                        try:
                            data = log_path.read_text(encoding='utf-8')
                            snippet = data[-2000:] if len(data) > 2000 else data
                            rel = Path('.idea-hole') / 'logs' / log_path.name
                            msg = f"PDF preview failed. See {rel} for details.\n\nLast output:\n{snippet}"
                        except Exception:
                            pass
                    messagebox.showerror("PDF Preview", msg)
                self._ui(_show_pdf_error)
                self._set_status("Preview failed")
                return

            # Try to open the generated PDF
            opened = self._open_path_default(pdf_path)
            if not opened:
                self._ui(lambda: messagebox.showinfo("Preview ready", f"Preview PDF saved to:\n{pdf_path}"))
            self._set_status("Preview ready")
        except Exception as e:
            self._set_status("Preview failed")
            self._ui(lambda m=f"Failed to preview PDF:\n{e}": messagebox.showerror("Error", m))

    def on_push(self):
        title = self.title_var.get().strip()
        desc = self.desc_var.get().strip()
        if not title or not desc:
            self._ui(lambda: messagebox.showinfo("Missing fields", "Please fill Title and Description."))
            return
        if not self.concept.get("1.0", tk.END).strip():
            self._ui(lambda: messagebox.showinfo("No concept", "Please generate or paste the concept text."))
            return
        threading.Thread(target=self._push_thread, daemon=True).start()

    # --- Git helpers
    def _run_git(self, repo_dir: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=str(repo_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _ensure_repo_initialized(self, repo_dir: Path):
        repo_dir.mkdir(parents=True, exist_ok=True)
        if not (repo_dir / ".git").exists():
            _ = subprocess.run(["git", "init"], cwd=str(repo_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _ensure_remote_origin(self, repo_dir: Path, remote_url: str):
        if not remote_url:
            return
        res = self._run_git(repo_dir, "remote", "get-url", "origin")
        if res.returncode == 0:
            current = (res.stdout or "").strip()
            if current != remote_url:
                _ = self._run_git(repo_dir, "remote", "set-url", "origin", remote_url)
        else:
            _ = self._run_git(repo_dir, "remote", "add", "origin", remote_url)

    def _ensure_branch_master(self, repo_dir: Path):
        """Ensure the default branch is master even on unborn HEAD."""
        # Try to read symbolic HEAD (works even if unborn)
        res = self._run_git(repo_dir, "symbolic-ref", "-q", "HEAD")
        headref = (res.stdout or "").strip() if res.returncode == 0 else ""
        if not headref:
            # Unborn or detached; set HEAD to refs/heads/master
            self._run_git(repo_dir, "symbolic-ref", "HEAD", "refs/heads/master")
            return
        if headref.endswith("/master"):
            return
        # Otherwise rename current branch to master
        self._run_git(repo_dir, "branch", "-M", "master")

    def _slug(self, s: str) -> str:
        s = re.sub(r"[\s]+", "-", s.strip())
        s = re.sub(r"[^a-zA-Z0-9._-]", "-", s)
        return re.sub(r"-+", "-", s).strip("-_")

    # --- Concepts index helpers
    def _build_slug_map_from_sessions(self) -> Dict[str, Dict[str, str]]:
        """Return mapping: slug -> {title, description}; prefer most recent saved_at per slug."""
        entries = self._load_all_sessions()
        best: Dict[str, Dict[str, str]] = {}
        best_ts: Dict[str, int] = {}
        for e in entries:
            title = (e.get("title") or "").strip()
            if not title:
                continue
            slug = self._slug(title)
            ts = 0
            try:
                ts = int(e.get("saved_at") or 0)
            except Exception:
                ts = 0
            if slug not in best or ts >= best_ts.get(slug, 0):
                best[slug] = {"title": title, "description": (e.get("description") or "").strip()}
                best_ts[slug] = ts
        return best

    def _write_concepts_index(self, repo_dir: Path):
        try:
            # Map available descriptions from sessions.jsonl by slug
            slug_map = self._build_slug_map_from_sessions()
            items = []
            seen: Set[str] = set()
            for child in sorted(repo_dir.iterdir(), key=lambda p: p.name.lower()):
                if not child.is_dir():
                    continue
                name = child.name
                if name.startswith('.') or name in {".git", "node_modules"}:
                    continue
                slug = name
                if slug in seen:
                    continue
                seen.add(slug)
                title = slug_map.get(slug, {}).get("title") or re.sub(r"[-_]+", " ", slug).strip().title()
                desc = slug_map.get(slug, {}).get("description") or ""
                items.append((slug, title, desc))

            intro = (
                "This folder contains a library of project concepts created with the Idea → Concept tool. "
                "Each entry links to its folder with the original concept README and related assets."
            )
            lines: List[str] = []
            lines.append("# Concepts Index")
            lines.append("")
            lines.append(intro)
            lines.append("")
            for slug, title, desc in items:
                if desc:
                    lines.append(f"- [{title}](./{slug}/) — {desc}")
                else:
                    lines.append(f"- [{title}](./{slug}/)")
            (repo_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        except Exception:
            # Non-fatal; skip index update on error
            pass

    def _convert_markdown_to_pdf(self, md_file: Path, out_pdf: Path) -> bool:
        """Single-path conversion: pandoc + tectonic with image compatibility.
        - Converts unsupported image formats to PNG in a temp folder
        - Uses a temporary Markdown copy for PDF only (original README.md untouched)
        - Sans-serif font, 20mm margins
        - Logs to .idea-hole/logs (never the concepts repo)
        """
        concept_dir = out_pdf.parent
        concept_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = Path.cwd() / ".idea-hole" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"pdf_export_{concept_dir.name}.log"

        def _resolve(name: str) -> Optional[str]:
            for base in [None, "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]:
                p = shutil.which(name) if base is None else os.path.join(base, name)
                if p and os.path.exists(p):
                    return p
            return None

        pandoc = _resolve("pandoc")
        tectonic = _resolve("tectonic")

        lines: List[str] = []
        lines.append(f"PATH={os.environ.get('PATH','')}")
        lines.append(f"md_file={md_file}")
        lines.append(f"resolved pandoc={pandoc}")
        lines.append(f"resolved tectonic={tectonic}")

        if not pandoc or not tectonic:
            lines.append("Missing required tools: pandoc and/or tectonic.")
            try:
                log_path.write_text("\n".join(lines), encoding="utf-8")
            except Exception:
                pass
            return False

        # Build temp workspace for PDF (images + markdown copy)
        tmp_base = Path.cwd() / ".idea-hole" / "tmp_pdf" / concept_dir.name
        try:
            if tmp_base.exists():
                shutil.rmtree(tmp_base)
        except Exception:
            pass
        tmp_base.mkdir(parents=True, exist_ok=True)

        # Rewrite Markdown image refs to PDF-compatible formats
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            lines.append(f"read error: {e}")
            try: log_path.write_text("\n".join(lines), encoding="utf-8")
            except Exception: pass
            return False

        # Regex for Markdown images ![alt](path "title")
        img_rgx = re.compile(r"!\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")
        allowed_ext = {".png", ".jpg", ".jpeg", ".pdf", ".eps"}

        def ensure_image_available(src: str) -> str:
            """Copy or convert image into tmp_base and return the tmp filename to use in Markdown."""
            p = Path(src)
            if not p.is_absolute():
                p = (concept_dir / p).resolve()
            if not p.exists():
                # Fallback: if src had subdirs, try by basename in concept_dir (assets were copied flat)
                alt = (concept_dir / Path(src).name).resolve()
                if alt.exists():
                    p = alt
                else:
                    lines.append(f"missing image: {src}")
                    return src  # leave as-is; LaTeX will likely show caption only
            ext = p.suffix.lower()
            # Destination filename in tmp
            if ext in allowed_ext:
                out_name = p.name
                out_path = tmp_base / out_name
                try:
                    if not out_path.exists():
                        shutil.copy2(str(p), str(out_path))
                    return out_name
                except Exception as e:
                    lines.append(f"copy fail: {src} -> {out_name} ({e})")
                    return src
            # Special-case SVG: try CairoSVG or rsvg-convert/ImageMagick
            if ext == ".svg":
                out_name = p.stem + ".png"
                out_path = tmp_base / out_name
                # Try CairoSVG (python lib)
                try:
                    from cairosvg import svg2png  # type: ignore
                    svg2png(url=str(p), write_to=str(out_path))
                    return out_name
                except Exception as e_svg_py:
                    lines.append(f"cairosvg unavailable or failed: {e_svg_py}")
                # Try rsvg-convert CLI
                try:
                    tool = shutil.which("rsvg-convert")
                    if tool:
                        res = subprocess.run([tool, "-f", "png", "-o", str(out_path), str(p)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        if res.returncode == 0 and out_path.exists():
                            return out_name
                        lines.append(f"rsvg-convert failed: exit {res.returncode}, {res.stdout}")
                except Exception as e_svg_cli:
                    lines.append(f"rsvg-convert error: {e_svg_cli}")
                # Try ImageMagick convert
                try:
                    tool = shutil.which("magick") or shutil.which("convert")
                    if tool:
                        res = subprocess.run([tool, str(p), str(out_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        if res.returncode == 0 and out_path.exists():
                            return out_name
                        lines.append(f"imagemagick failed: exit {res.returncode}, {res.stdout}")
                except Exception as e_im:
                    lines.append(f"imagemagick error: {e_im}")
                # fallthrough to Pillow/general path

            # Try to convert unsupported formats to PNG via Pillow
            try:
                from PIL import Image
                img = Image.open(str(p))
                try:
                    img.seek(0)
                except Exception:
                    pass
                out_name = p.stem + ".png"
                out_path = tmp_base / out_name
                img.convert("RGBA" if img.mode in ("P", "LA") else "RGB").save(str(out_path), format="PNG")
                return out_name
            except Exception as e:
                lines.append(f"convert fail: {src} -> png ({e})")
                # Last resort: copy original and hope engine supports it
                out_name = p.name
                out_path = tmp_base / out_name
                try:
                    shutil.copy2(str(p), str(out_path))
                    return out_name
                except Exception as e2:
                    lines.append(f"final copy fail: {src} ({e2})")
                    return src

        # Build a modified markdown string with tmp-local image paths
        def _repl(m: re.Match) -> str:
            orig = m.group(0)
            path = m.group(1)
            rep = ensure_image_available(path)
            # Replace the path with the tmp filename (no directories)
            return orig.replace(path, rep)

        mod_text = img_rgx.sub(_repl, text)

        # Preserve multiple consecutive blank lines by inserting raw TeX vspace.
        # Pandoc collapses multiple blank lines into a single paragraph break in LaTeX.
        # We translate extra blank lines into additional vertical space so visual spacing matches editing.
        def _preserve_extra_blank_lines(s: str) -> str:
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            lines_in = s.split("\n")
            out_lines: List[str] = []
            in_fence = False
            blank_run = 0
            for ln in lines_in:
                stripped = ln.lstrip()
                # Toggle code fence state on ``` or ~~~ lines
                if stripped.startswith("```") or stripped.startswith("~~~"):
                    # flush any pending blanks before fence delimiter
                    if blank_run > 0:
                        out_lines.append("")
                        for _ in range(blank_run - 1):
                            out_lines.append("\\vspace{1em}")
                        blank_run = 0
                    out_lines.append(ln)
                    in_fence = not in_fence
                    continue
                if in_fence:
                    # pass through exactly
                    if blank_run > 0:
                        out_lines.append("")
                        for _ in range(blank_run - 1):
                            out_lines.append("\\vspace{1em}")
                        blank_run = 0
                    out_lines.append(ln)
                    continue
                # Outside fences: track blank runs
                if stripped == "":
                    blank_run += 1
                    continue
                # Non-blank line: flush blanks first
                if blank_run > 0:
                    out_lines.append("")
                    for _ in range(blank_run - 1):
                        out_lines.append("\\vspace{1em}")
                    blank_run = 0
                out_lines.append(ln)
            # flush at EOF
            if blank_run > 0:
                out_lines.append("")
                for _ in range(blank_run - 1):
                    out_lines.append("\\vspace{1em}")
            return "\n".join(out_lines)

        mod_text = _preserve_extra_blank_lines(mod_text)
        tmp_md = tmp_base / "README_pdf.md"
        tmp_md.write_text(mod_text, encoding="utf-8")

        # Run pandoc (tectonic engine) in tmp_base with resource-path pointing to concept_dir and tmp_base
        cmd = [
            pandoc,
            str(tmp_md),
            "-f", "markdown+hard_line_breaks+raw_tex",
            "-s",
            "--pdf-engine=tectonic",
            "-V", "mainfont=Helvetica",
            "-V", "monofont=Menlo",
            "-V", "geometry:margin=20mm",
            "-V", "fontsize=11pt",
            "--resource-path", f"{str(tmp_base)}:{str(concept_dir)}",
            "-o", str(out_pdf),
        ]
        res = subprocess.run(cmd, cwd=str(tmp_base), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines.append("$ " + " ".join(cmd))
        lines.append(f"(exit {res.returncode})")
        lines.append(res.stdout or "")
        ok = (res.returncode == 0 and out_pdf.exists())
        # Fallback: retry without hard_line_breaks if first attempt failed (older pandoc versions)
        if not ok:
            try:
                cmd_fallback = [
                    pandoc,
                    str(tmp_md),
                    "-f", "markdown+hard_line_breaks",
                    "-s",
                    "--pdf-engine=tectonic",
                    "-V", "mainfont=Helvetica",
                    "-V", "monofont=Menlo",
                    "-V", "geometry:margin=20mm",
                    "-V", "fontsize=11pt",
                    "--resource-path", f"{str(tmp_base)}:{str(concept_dir)}",
                    "-o", str(out_pdf),
                ]
                res2 = subprocess.run(cmd_fallback, cwd=str(tmp_base), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                lines.append("$ " + " ".join(cmd_fallback))
                lines.append(f"(exit {res2.returncode})")
                lines.append(res2.stdout or "")
                ok = (res2.returncode == 0 and out_pdf.exists())
            except Exception as _e_fallback:
                lines.append(f"fallback error: {_e_fallback}")

        if not ok:
            try:
                log_path.write_text("\n".join(lines), encoding="utf-8")
            except Exception:
                pass

        # Cleanup temp workspace
        try:
            shutil.rmtree(tmp_base)
        except Exception:
            pass

        return ok

    def _push_thread(self):
        try:
            self._set_status("Preparing repo…")
            repo_dir = Path.cwd() / "concepts"
            self._ensure_repo_initialized(repo_dir)

            # Create concept folder and files
            title = self.title_var.get().strip()
            slug = self._slug(title)
            concept_dir = repo_dir / slug
            concept_dir.mkdir(parents=True, exist_ok=True)
            md_path = concept_dir / "README.md"
            md_text = self.concept.get("1.0", tk.END).strip()
            md_path.write_text(md_text, encoding="utf-8")

            # (No meta.json written; index derives data from sessions.jsonl)

            # Copy selected assets into concept folder BEFORE PDF generation
            assets = [p for p in self.files if self.include_map.get(str(p), True)]
            for src in assets:
                try:
                    dst = concept_dir / src.name
                    if dst.name.lower() in {"readme.md", f"{slug}-concept.pdf".lower()}:
                        # avoid clobbering the generated files
                        dst = concept_dir / f"asset-{src.name}"
                    shutil.copy2(src, dst)
                except Exception:
                    # ignore individual copy failures but continue
                    pass

            # Now export PDF (images are present in concept_dir)
            pdf_path = concept_dir / f"{slug}-concept.pdf"
            self._set_status("Exporting PDF…")
            ok_pdf = self._convert_markdown_to_pdf(md_path, pdf_path)
            if not ok_pdf:
                logs_dir = Path.cwd() / ".idea-hole" / "logs"
                log_path = logs_dir / f"pdf_export_{concept_dir.name}.log"
                def _show_pdf_error():
                    msg = "PDF export failed."
                    if log_path.exists():
                        try:
                            data = log_path.read_text(encoding='utf-8')
                            snippet = data[-2000:] if len(data) > 2000 else data
                            rel = Path('.idea-hole') / 'logs' / log_path.name
                            msg = f"PDF export failed. See {rel} for details.\n\nLast output:\n{snippet}"
                        except Exception:
                            pass
                    messagebox.showerror("PDF Export", msg)
                self._ui(_show_pdf_error)

            # Update concepts root README index before committing
            try:
                self._write_concepts_index(repo_dir)
            except Exception:
                pass

            # Git add/commit
            self._set_status("Committing…")
            add_res = self._run_git(repo_dir, "add", ".")
            if add_res.returncode != 0:
                raise RuntimeError(add_res.stdout)
            commit_msg = f"{title} - {self.desc_var.get().strip()}"
            commit_res = self._run_git(repo_dir, "commit", "-m", commit_msg)
            if commit_res.returncode != 0:
                # If nothing to commit
                if "nothing to commit" not in (commit_res.stdout or "").lower():
                    raise RuntimeError(commit_res.stdout)

            # Ensure master branch and remote
            remote = (self.git_remote_url.get() or "").strip()
            self._ensure_branch_master(repo_dir)
            if remote:
                self._set_status("Pushing…")
                self._ensure_remote_origin(repo_dir, remote)
                push_res = self._run_git(repo_dir, "push", "-u", "origin", "master")
                if push_res.returncode != 0:
                    raise RuntimeError(push_res.stdout)
                self._set_status("Pushed to remote")
            else:
                self._set_status("Committed locally (no remote set)")

            self._ui(lambda: messagebox.showinfo("Success", "Concept saved and repository updated."))
        except Exception as e:
            self._set_status("Push failed")
            msg = f"Failed to push to repo:\n{e}"
            self._ui(lambda m=msg: messagebox.showerror("Error", m))

    # --- Status helper
    def _set_status(self, s: str):
        self._ui(lambda: (self.status.configure(text=s), self.status.update_idletasks()))

    def _ui(self, fn):
        try:
            self.after(0, fn)
        except Exception:
            # best-effort fallback
            try:
                fn()
            except Exception:
                pass

    # --- Tk helpers
    def _auto_hide_scrollbar(self, sb: ttk.Scrollbar, lo, hi):
        """Auto-hide a ttk.Scrollbar when content fits; show when it overflows.
        Works with both pack and grid-managed scrollbars.
        """
        try:
            flo, fhi = float(lo), float(hi)
        except Exception:
            try:
                flo, fhi = float(str(lo)), float(str(hi))
            except Exception:
                flo, fhi = 0.0, 1.0
        # Update the scrollbar range
        try:
            sb.set(flo, fhi)
        except Exception:
            pass
        needs = not (flo <= 0.0 and fhi >= 1.0)
        try:
            mgr = sb.winfo_manager()
        except Exception:
            mgr = ''
        try:
            if needs:
                # show
                if mgr == 'grid':
                    if not sb.winfo_ismapped():
                        sb.grid()
                elif mgr == 'pack':
                    try:
                        sb.pack_info()
                    except Exception:
                        sb.pack(side=tk.RIGHT, fill=tk.Y)
                else:
                    # default to pack
                    sb.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                # hide
                if mgr == 'grid':
                    sb.grid_remove()
                elif mgr == 'pack':
                    sb.pack_forget()
        except Exception:
            pass

    # --- Ollama model listing
    def _list_models(self) -> List[str]:
        try:
            res = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=8)
            if res.returncode != 0:
                return []
            lines = [ln.strip() for ln in (res.stdout or "").splitlines()]
            out: List[str] = []
            for ln in lines:
                if not ln or ln.lower().startswith("name"):
                    continue
                # first column is model name
                name = ln.split()[0]
                if name and name not in out:
                    out.append(name)
            return out
        except Exception:
            return []

    # --- Dirty snapshot helpers
    def _snapshot_session_state(self) -> Dict:
        try:
            files_list = [{
                "path": str(p),
                "include": bool(self.include_map.get(str(p), True))
            } for p in self.files]
            # stable order
            files_list.sort(key=lambda x: (x["path"], not x["include"]))
            return {
                "title": (self.title_var.get() or "").strip(),
                "description": (self.desc_var.get() or "").strip(),
                "notes": self.notes.get("1.0", tk.END).strip(),
                "concept": self.concept.get("1.0", tk.END).strip(),
                "files": files_list,
            }
        except Exception:
            return {"title":"","description":"","notes":"","concept":"","files":[]}

    def _is_effectively_dirty(self) -> bool:
        try:
            if not getattr(self, "_dirty", False):
                return False
            now = self._snapshot_session_state()
            last = getattr(self, "_last_saved", None)
            if last is None:
                return any([now.get("title"), now.get("description"), now.get("notes"), now.get("concept"), now.get("files")])
            return now != last
        except Exception:
            # On any error, be conservative and assume not dirty to avoid noisy prompts
            return False

    def _get_model_values(self) -> List[str]:
        models = self._list_models()
        return ["Select model..."] + models

    # --- Settings persistence
    def _config_path(self) -> Path:
        base = Path.cwd() / ".idea-hole"
        base.mkdir(parents=True, exist_ok=True)
        return base / "settings.json"

    def _load_settings(self):
        try:
            p = self._config_path()
            if p.exists():
                obj = json.loads(p.read_text(encoding='utf-8'))
                if isinstance(obj, dict):
                    if obj.get('ollama_host'):
                        self.ollama_host.set(obj['ollama_host'])
                    if obj.get('ollama_model'):
                        self.ollama_model.set(obj['ollama_model'])
                    if obj.get('git_remote_url'):
                        self.git_remote_url.set(obj['git_remote_url'])
                    if obj.get('searx_url'):
                        self.searx_url.set(obj['searx_url'])
                    if obj.get('pandoc_path'):
                        # Only set if attribute exists
                        try: self.pandoc_path.set(obj['pandoc_path'])
                        except Exception: pass
                    if obj.get('wkhtmltopdf_path'):
                        try: self.wkhtmltopdf_path.set(obj['wkhtmltopdf_path'])
                        except Exception: pass
        except Exception:
            pass

    # --- Dirty/session helpers
    def _set_dirty(self, flag: bool = True):
        try:
            self._dirty = bool(flag)
        except Exception:
            self._dirty = True

    def _load_all_sessions(self) -> List[Dict]:
        entries: List[Dict] = []
        try:
            with self._sessions_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and obj.get("title"):
                            entries.append(obj)
                    except Exception:
                        continue
        except Exception:
            pass
        return entries

    def _write_all_sessions(self, entries: List[Dict]):
        tmp = self._sessions_file.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for obj in entries:
                    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            tmp.replace(self._sessions_file)
        except Exception:
            # fallback non-atomic
            with self._sessions_file.open("w", encoding="utf-8") as fh:
                for obj in entries:
                    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _session_title_exists(self, title: str) -> bool:
        t = (title or "").strip()
        if not t:
            return False
        try:
            with self._sessions_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and (obj.get("title") or "").strip() == t:
                            return True
                    except Exception:
                        continue
        except Exception:
            pass
        return False

    def _build_session_payload(self) -> Dict:
        # Ensure corpus coverage before saving
        self._ensure_corpus_for_files(self.files, blocking=True)
        files_meta = []
        for p in self.files:
            h = self.file_hashes.get(str(p)) or self._compute_file_hash(p)
            self.file_hashes[str(p)] = h
            files_meta.append({
                "path": str(p),
                "file_hash": h,
                "include": bool(self.include_map.get(str(p), True)),
            })
        return {
            "title": (self.title_var.get() or "").strip(),
            "description": (self.desc_var.get() or "").strip(),
            "notes": self.notes.get("1.0", tk.END).strip(),
            "concept": self.concept.get("1.0", tk.END).strip(),
            "files": files_meta,
            "saved_at": int(time.time()),
        }

    def _save_session(self, *, confirm_overwrite: bool, autosave: bool) -> bool:
        title = (self.title_var.get() or "").strip()
        if not title:
            self._ui(lambda: messagebox.showinfo("Title required", "Please enter a Title before saving the session."))
            return False
        exists = self._session_title_exists(title)
        if exists and not confirm_overwrite and autosave:
            # On autosave, do not overwrite; inform user once
            self._ui(lambda: messagebox.showinfo("Autosave skipped", f"Autosave not performed: a session titled '{title}' already exists."))
            return False
        if exists and confirm_overwrite:
            ok = messagebox.askyesno("Overwrite", f"A session titled '{title}' exists. Overwrite it?")
            if not ok:
                return False
        # Prepare payload and rewrite file
        payload = self._build_session_payload()
        try:
            all_entries = self._load_all_sessions()
            if exists:
                all_entries = [e for e in all_entries if (e.get("title") or "") != title]
            all_entries.append(payload)
            self._write_all_sessions(all_entries)
            self._set_status("Session saved")
            # Update last-saved snapshot for clean state detection
            try:
                self._last_saved = self._snapshot_session_state()
            except Exception:
                self._last_saved = {
                    "title": (self.title_var.get() or "").strip(),
                    "description": (self.desc_var.get() or "").strip(),
                    "notes": self.notes.get("1.0", tk.END).strip(),
                    "concept": self.concept.get("1.0", tk.END).strip(),
                    "files": [{"path": str(p), "include": bool(self.include_map.get(str(p), True))} for p in self.files],
                }
            return True
        except Exception as e:
            self._ui(lambda m=f"Failed to save session:\n{e}": messagebox.showerror("Error", m))
            return False

    def _maybe_save_if_dirty(self) -> bool:
        try:
            # Only prompt if content truly differs from last saved snapshot
            if not self._is_effectively_dirty():
                return True
            ans = messagebox.askyesnocancel("Unsaved changes", "Save changes to the current session?")
            if ans is None:
                return False
            if ans is False:
                return True
            # ans is True -> attempt to save; confirm overwrite if needed
            saved = self._save_session(confirm_overwrite=True, autosave=False)
            if saved:
                self._set_dirty(False)
                return True
            # Save not performed (likely declined overwrite) -> block
            return False
        except Exception:
            return True

    def _autosave_after_generation(self):
        try:
            if not (self.title_var.get() or "").strip():
                return
            saved = self._save_session(confirm_overwrite=False, autosave=True)
            if saved:
                self._set_dirty(False)
        except Exception:
            pass

    def on_close(self):
        if self._maybe_save_if_dirty():
            try:
                self.destroy()
            except Exception:
                os._exit(0)

    def _save_settings(self):
        try:
            obj = {
                'ollama_host': self.ollama_host.get(),
                'ollama_model': self.ollama_model.get(),
                'git_remote_url': self.git_remote_url.get(),
                'searx_url': self.searx_url.get(),
                'pandoc_path': getattr(self, 'pandoc_path', tk.StringVar(value='')).get() if hasattr(self, 'pandoc_path') else '',
                'wkhtmltopdf_path': getattr(self, 'wkhtmltopdf_path', tk.StringVar(value='')).get() if hasattr(self, 'wkhtmltopdf_path') else '',
            }
            self._config_path().write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass


def main():
    try:
        app = App()
        app.mainloop()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
