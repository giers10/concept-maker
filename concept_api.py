#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Headless backend actions for the Concept Maker app.

This module exposes JSON actions for the Tauri UI without desktop toolkit imports.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import websearch

# -----------------------------
# Paths
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parent
IDEA_HOLE_DIR = REPO_ROOT / ".idea-hole"


# -----------------------------
# Utilities
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
    """Very lightweight fallback if corpus_builder.py or deps are unavailable."""

    def __init__(self) -> None:
        self._fitz = None
        try:
            import fitz  # type: ignore
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
                    records.append(Record(id=str(p), title=p.stem, text=f"[Unsupported file type: {suf}]", source_path=str(p)))
            except Exception:
                records.append(Record(id=str(p), title=p.stem, text=f"[Error reading file: {p.name}]", source_path=str(p)))

        with out_jsonl.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
        return records

    def _strip_html(self, html_text: str) -> str:
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(html_text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
            return text
        except Exception:
            txt = re.sub(r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", " ", html_text, flags=re.S | re.I)
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
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            ok = proc.returncode == 0 and out_jsonl.exists() and out_jsonl.stat().st_size > 0
            return ok
        except Exception:
            return False


# -----------------------------
# Ollama client
# -----------------------------

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 600):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, model: str, prompt: str) -> str:
        import urllib.request
        import urllib.error

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


# -----------------------------
# Prompting
# -----------------------------

PROMPT_TEMPLATE = """
You are a cross-domain concept developer (product strategist, creative producer, research lead, grant writer).
Turn the sources into a concise, presentable CONCEPT document. Adapt to the domain.

INSTRUCTIONS
1) Detect IDEA TYPE (pick one primary; if unclear, choose closest and add a TODO):
   {Product/Software, Service, Research/Study, Policy/Proposal, Art/Exhibition/Performance, Event/Program,
    Education/Curriculum, Media/Film/Publication, Campaign/Nonprofit, Data/ML/Infrastructure, Game/Interactive,
    Writing/Book/Article, Other}

2) Tone & register:
   - Product/Software -> pragmatic PM/tech brief
   - Research -> neutral academic project brief
   - Policy -> policy memo
   - Art/Exhibition/Performance -> curator/producer note (clear, not flowery)
   - Event -> producer's run-of-show style
   - Education -> syllabus brief
   - Media/Publication -> one-sheet
   - Campaign/Nonprofit -> strategy brief
   - Data/ML/Infrastructure -> engineering design note
   - Game/Interactive -> design doc overview
   - Writing/Book/Article -> proposal overview

3) Output Markdown using these core sections (use these exact headings; include only relevant ones):
- Overview & Intent
- Context / Problem (or Opportunity)
- Audience / Stakeholders
- Deliverables / Outputs & Scope
- Approach / Method  (rename to "Methodology", "Implementation Plan", "Format & Installation Plan", etc., to fit the idea type)
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
- Where a claim relies on a specific source, include a short inline blockquote with "Source: <Path or Title>".

5) Assets:
- These files are committed alongside README.md. Embed images with Markdown and link documents where they help clarity.

STYLE
- Short paragraphs and bullets; concrete, specific, and actionable. Avoid marketing fluff.
- If dates/budget/ownership are uncertain, show ranges or TODOs.
- Keep a neutral, professional tone adapted to the idea type.

TITLE
- Generate a neutral 2-4 words working title.
- Begin the document with "# {Title}".

Assets Provided:
{ASSETS}

Notes (from user):
{NOTES}

Knowledge Base (source excerpts):
{KB}
""".strip()

REPHRASE_LENSES = [
    {
        "key": "neutral",
        "label": "Neutral Clarification / Expansion",
        "prompt": """Take the following rough note and turn it into a single clear, concise paragraph that captures the main idea.
- Keep a neutral, explanatory tone.
- Don't add new features or speculation, only clarify and connect what is already there.
- Output exactly one paragraph.

Note:
{USER_NOTE}
""",
    },
    {
        "key": "problem_solution",
        "label": "Problem-Solution Framing",
        "prompt": """Rewrite the following note as a single paragraph that clearly describes:
1. What problem or frustration exists,
2. For whom,
3. How the idea could solve it in principle.
Keep it concrete but high-level, no implementation details.
Output exactly one paragraph.

Note:
{USER_NOTE}
""",
    },
    {
        "key": "user_story",
        "label": "User Story / Scenario",
        "prompt": """Rewrite the following note as a single paragraph that describes a short scenario from a user's point of view.
Show how a specific person encounters the situation and how this idea helps them.
Keep it realistic and simple, not hype-y.
Output exactly one paragraph.

Note:
{USER_NOTE}
""",
    },
    {
        "key": "value_prop",
        "label": "Value Proposition / Pitch",
        "prompt": """Rewrite the following note as a single paragraph that sounds like a clear, simple pitch of the idea.
Explain what it is, who it's for, and why it's valuable or interesting.
Avoid buzzwords; keep it grounded and concrete.
Output exactly one paragraph.

Note:
{USER_NOTE}
""",
    },
    {
        "key": "implementation",
        "label": "Implementation / Next Steps",
        "prompt": """Rewrite the following note as a single paragraph that keeps the original idea but focuses on how one might start implementing or exploring it.
Mention 2-3 plausible first steps or components without going into deep technical detail.
Output exactly one paragraph.

Note:
{USER_NOTE}
""",
    },
]

EXTEND_PROMPT = """
You are continuing the user's own note. Keep writing in the same language, tone, and formatting style they used.

Instructions:
- Extend the idea with additional possibilities, use cases, angles, or problems to consider.
- Preserve the author's voice: match their formality, punctuation habits, and quirks (e.g., all lowercase, terse bullets, or formal sentences).
- Do not summarize or rewrite the original; add new material that flows naturally after it.
- Keep it concise (2-5 sentences or a few short bullet points).
- If the input is in bullet form, continue the bullets; otherwise, continue the paragraph.

Original note:
{USER_NOTE}
""".strip()

IMAGE_PROMPT_PLACEHOLDER = "Generated image prompt will appear here."


class IdeaCategory(str, Enum):
    APP_OR_TOOL = "APP_OR_TOOL"
    DASHBOARD_OR_ANALYTICS = "DASHBOARD_OR_ANALYTICS"
    DEV_TOOL_OR_API = "DEV_TOOL_OR_API"
    PHYSICAL_PRODUCT = "PHYSICAL_PRODUCT"
    SYSTEM_OR_WORKFLOW = "SYSTEM_OR_WORKFLOW"
    ABSTRACT_FRAMEWORK = "ABSTRACT_FRAMEWORK"
    SERVICE_OR_EVENT = "SERVICE_OR_EVENT"
    SPATIAL_DESIGN_OR_INSTALLATION = "SPATIAL_DESIGN_OR_INSTALLATION"
    GAME_OR_WORLD = "GAME_OR_WORLD"
    BRAND_OR_CAMPAIGN = "BRAND_OR_CAMPAIGN"
    EDUCATIONAL_TOOL = "EDUCATIONAL_TOOL"
    DATA_INFRASTRUCTURE = "DATA_INFRASTRUCTURE"


VISUALIZATION_HINTS: Dict[IdeaCategory, str] = {
    IdeaCategory.APP_OR_TOOL: "Hero UI screen on a device mockup, showing the main interface and color palette.",
    IdeaCategory.DASHBOARD_OR_ANALYTICS: "Full-screen dashboard view with charts, cards, widgets and clear information hierarchy.",
    IdeaCategory.DEV_TOOL_OR_API: "Stylized developer scene with screens and terminal, or a clean system architecture diagram.",
    IdeaCategory.PHYSICAL_PRODUCT: "Hero product shot of the object, centered, photorealistic, materials and key features clearly visible.",
    IdeaCategory.SYSTEM_OR_WORKFLOW: "Isometric system diagram showing entities and arrows, clean infographic look.",
    IdeaCategory.ABSTRACT_FRAMEWORK: "Metaphorical, atmospheric scene representing the idea using one strong visual metaphor.",
    IdeaCategory.SERVICE_OR_EVENT: "Lifestyle scene with people interacting in an environment, representing the experience.",
    IdeaCategory.SPATIAL_DESIGN_OR_INSTALLATION: "Hero shot of the space or installation, wide view, with lighting and geometry clearly visible.",
    IdeaCategory.GAME_OR_WORLD: "In-game style scene showing a player's point of view or isometric world with the core mechanic visible.",
    IdeaCategory.BRAND_OR_CAMPAIGN: "Bold key visual / poster with strong graphic composition and a central symbol or logo-like element.",
    IdeaCategory.EDUCATIONAL_TOOL: "Scene with a learner interacting with an interface, or a clear diagram of the method.",
    IdeaCategory.DATA_INFRASTRUCTURE: "Network-like visualization with nodes and connections, or a dense monitoring dashboard.",
}


def classify_idea(idea_text: str) -> Dict[str, Any]:
    text = (idea_text or "").lower()

    def has(*phrases: str) -> bool:
        return any(p in text for p in phrases)

    if has("dashboard", "analytics", "kpi", "monitoring panel", "business intelligence"):
        category = IdeaCategory.DASHBOARD_OR_ANALYTICS
    elif has("observability", "traces", "logs", "infrastructure", "nodes", "monitoring", "telemetry"):
        category = IdeaCategory.DATA_INFRASTRUCTURE
    elif has("api", "sdk", "cli", "developer", "framework", "backend"):
        category = IdeaCategory.DEV_TOOL_OR_API
    elif has("device", "hardware", "physical", "furniture", "wearable", "sensor"):
        category = IdeaCategory.PHYSICAL_PRODUCT
    elif has("workflow", "pipeline", "automation", "process", "system architecture", "orchestration"):
        category = IdeaCategory.SYSTEM_OR_WORKFLOW
    elif has("mobile app", "ios", "android", "web app", "saas", "desktop app", "tool", "platform", "software"):
        category = IdeaCategory.APP_OR_TOOL
    elif has("framework", "philosophy", "mindset", "mental model", "metaphor"):
        category = IdeaCategory.ABSTRACT_FRAMEWORK
    elif has("event", "workshop", "conference", "service", "consulting", "community"):
        category = IdeaCategory.SERVICE_OR_EVENT
    elif has("space", "room", "gallery", "installation", "architecture", "store layout"):
        category = IdeaCategory.SPATIAL_DESIGN_OR_INSTALLATION
    elif has("game", "player", "level", "world", "gamified"):
        category = IdeaCategory.GAME_OR_WORLD
    elif has("brand", "campaign", "logo", "poster", "identity"):
        category = IdeaCategory.BRAND_OR_CAMPAIGN
    elif has("learning", "course", "students", "tutorial", "study", "education", "teaching"):
        category = IdeaCategory.EDUCATIONAL_TOOL
    else:
        product_like = has("app", "tool", "product", "platform", "saas", "software")
        category = IdeaCategory.APP_OR_TOOL if product_like else IdeaCategory.ABSTRACT_FRAMEWORK

    return {
        "category": category,
        "visualization_hint": VISUALIZATION_HINTS.get(category, ""),
    }


def build_image_prompt_system_message(category: IdeaCategory, visualization_hint: str) -> str:
    return f"""
You are an expert concept artist and image prompt writer for text-to-image models.

The user will give you a description of an idea: a product, project, app, physical object, system, or abstract concept.

Your job:
- Understand the idea and decide the single most effective visualization for it.
- Then write ONE powerful, detailed image prompt.

The idea has been classified as:
- Category: {category.value}
- Recommended visualization style: {visualization_hint}

You must:
1. Internally figure out:
   - What matters most to show for this category (function, form, context of use, mood, or metaphor).
   - How to best apply the recommended visualization style: {visualization_hint}.

2. Choose ONE visualization approach that feels most natural and expressive for this idea.
   You can pick any camera angle, composition, style, and mood you like, as long as it serves the idea and stays consistent with the recommended visualization style.

3. In the final answer, output ONLY a single image description, as one paragraph, around 40-80 words, ready to send to an image generation model.

4. In that paragraph, clearly specify:
   - Main subject and what is happening
   - Environment / background context
   - Camera angle and shot type (e.g. "isometric view", "over-the-shoulder", "close-up")
   - Art style / medium (e.g. "clean flat vector illustration", "photorealistic 3D render", "anime style", "technical blueprint")
   - Lighting and color mood
   - Level of detail (e.g. "highly detailed", "minimalist")
   - Optional negative constraints if useful (e.g. "no text, no logos")

5. Do NOT mention the words "user", "idea", "prompt", "concept art", or "text-to-image model".
   Just describe the image directly.

ASSISTANT:
(One single paragraph image description)
""".strip()


def generate_image_prompt_for_idea(idea_text: str, *, client: OllamaClient, model: str) -> str:
    cleaned = (idea_text or "").strip()
    if not cleaned:
        raise ValueError("Idea text is empty")
    details = classify_idea(cleaned)
    category: IdeaCategory = details.get("category", IdeaCategory.APP_OR_TOOL)
    visualization_hint = details.get("visualization_hint", VISUALIZATION_HINTS.get(category, ""))
    system_message = build_image_prompt_system_message(category, visualization_hint)
    prompt = f"{system_message}\n\nUSER IDEA:\n{cleaned}\n\nASSISTANT:"
    raw = client.generate(model=model, prompt=prompt)
    return sanitize_llm_text_simple(raw)


def build_kb_string(records: List[Record], *, max_chars: int = 80000, per_record_cap: int = 4000) -> str:
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
    try:
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
        s = re.sub(r"^\s*```(?:\w+)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        return s.strip()
    except Exception:
        return (s or "").strip()


def md_heading_replace_or_insert(md: str, title: str) -> str:
    if not md:
        return f"# {title}\n\n"
    lines = md.splitlines()
    if lines and re.match(r"^\s*#\s+project\s+concept\s*$", lines[0], flags=re.I):
        lines[0] = f"# {title}"
        return "\n".join(lines)
    if lines and re.match(r"^\s*#\s+", lines[0]):
        return md
    return f"# {title}\n\n" + md


def strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\"'""'']+", "", s)
    s = re.sub(r"[\"'""'']+$", "", s)
    return s


# -----------------------------
# Core engine
# -----------------------------

class ConceptEngine:
    def __init__(self, *, status_cb: Optional[Any] = None) -> None:
        self.status_cb = status_cb
        self.files: List[Path] = []
        self.websites: List[str] = []
        self.records: List[Record] = []
        self.include_map: Dict[str, bool] = {}
        self.file_hashes: Dict[str, str] = {}
        self._seen_hashes: Set[str] = set()
        self._ingesting: Set[str] = set()
        self._base_dir: Path = IDEA_HOLE_DIR
        self._files_dir: Path = self._base_dir / "files"
        self._corpus_file: Path = self._base_dir / "corpus.jsonl"
        self._sessions_file: Path = self._base_dir / "sessions.jsonl"
        self._init_storage()

    def _status(self, msg: str) -> None:
        if self.status_cb:
            try:
                self.status_cb(msg)
            except Exception:
                pass

    def _init_storage(self) -> None:
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._files_dir.mkdir(parents=True, exist_ok=True)
            if not self._corpus_file.exists():
                self._corpus_file.write_text("", encoding="utf-8")
            if not self._sessions_file.exists():
                self._sessions_file.write_text("", encoding="utf-8")
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
            st = None
            try:
                st = path.stat()
            except Exception:
                pass
            h.update((str(path) + "|" + str(getattr(st, "st_mtime", 0.0))).encode("utf-8", "ignore"))
        return h.hexdigest()

    def _compute_url_hash(self, url: str) -> str:
        try:
            return hashlib.sha256(url.strip().encode("utf-8", "ignore")).hexdigest()
        except Exception:
            return hashlib.sha256(url.encode("utf-8", "ignore")).hexdigest()

    def _ensure_file_symlink(self, src: Path, file_hash: str) -> Path:
        dst = self._files_dir / f"{file_hash}__{src.name}"
        try:
            if not dst.exists():
                copy_or_link(src, self._files_dir)
        except Exception:
            pass
        return dst

    def _ingest_single_file(self, src: Path, file_hash: str, *, verbose: bool = False) -> bool:
        try:
            tmp_dir = self._base_dir / "ingest_tmp" / file_hash
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass
            tmp_dir.mkdir(parents=True, exist_ok=True)
            copy_or_link(src, tmp_dir)

            external = None
            script = REPO_ROOT / "corpus_builder.py"
            if script.exists():
                external = ExternalCorpusBuilder(script)

            tmp_out = tmp_dir / "out.jsonl"
            ok = False
            if external is not None:
                self._status(f"Indexing {src.name} (external)...")
                ok = external.build(tmp_dir, tmp_out, workers=2, verbose=verbose)
            if not ok:
                self._status(f"Indexing {src.name} (simple)...")
                try:
                    simple = SimpleCorpusBuilder()
                    recs = simple.build(tmp_dir, tmp_out)
                    ok = bool(recs)
                except Exception:
                    ok = False

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
                        obj["source_path"] = str(src.resolve())
                        obj.setdefault("mime", obj.get("mime") or None)
                        obj["added_at"] = ts
                        try:
                            fh_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        except Exception:
                            fh_out.write(json.dumps(obj) + "\n")
                self._seen_hashes.add(file_hash)
                return True
            return False
        finally:
            try:
                shutil.rmtree(self._base_dir / "ingest_tmp" / file_hash)
            except Exception:
                pass

    def _ingest_single_url(self, url: str, url_hash: str) -> bool:
        try:
            self._status(f"Fetching {url}...")
            try:
                html_text, _hdrs = websearch._http_get(url, timeout=25)
            except Exception:
                return False
            text = websearch._extract_text(html_text)
            if not text.strip():
                return False
            title = self._friendly_url_name(url)
            try:
                m = re.search(r"<title>(.*?)</title>", html_text, flags=re.I | re.S)
                if m:
                    raw_title = m.group(1)
                    cleaned = re.sub(r"\s+", " ", raw_title)
                    try:
                        cleaned = html.unescape(cleaned)
                    except Exception:
                        pass
                    cleaned = cleaned.strip()
                    if cleaned:
                        title = cleaned
            except Exception:
                pass

            ts = int(time.time())
            obj = {
                "id": url,
                "title": title,
                "text": text,
                "source_path": url,
                "mime": "text/html",
                "file_hash": url_hash,
                "added_at": ts,
            }
            with self._corpus_file.open("a", encoding="utf-8") as fh_out:
                fh_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._seen_hashes.add(url_hash)
            return True
        except Exception:
            return False

    @staticmethod
    def _friendly_url_name(url: str) -> str:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.netloc or url
            path = (parsed.path or "").strip("/").split("/")
            if path and path[0]:
                first = path[0][:40]
                return f"{host}/{first}"
            return host
        except Exception:
            return url

    def _ensure_corpus_for_files(self, paths: List[Path]) -> None:
        if not paths:
            return
        to_ingest: List[Tuple[Path, str]] = []
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

        for src, h in to_ingest:
            self._ingesting.add(h)
            try:
                self._ingest_single_file(src, h, verbose=False)
            finally:
                try:
                    self._ingesting.remove(h)
                except Exception:
                    pass

    def _ensure_corpus_for_urls(self, urls: List[str]) -> None:
        if not urls:
            return
        to_ingest: List[Tuple[str, str]] = []
        for u in urls:
            if not u:
                continue
            h = self._compute_url_hash(u)
            self.file_hashes[u] = h
            if h not in self._seen_hashes and h not in self._ingesting:
                to_ingest.append((u, h))

        if not to_ingest:
            return

        for url, h in to_ingest:
            self._ingesting.add(h)
            try:
                self._ingest_single_url(url, h)
            finally:
                try:
                    self._ingesting.remove(h)
                except Exception:
                    pass

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

    def build_kb_records(self, files: List[str], websites: List[str]) -> List[Record]:
        paths = [Path(p) for p in files]
        self._ensure_corpus_for_files(paths)
        self._ensure_corpus_for_urls(websites)
        hashes = {self.file_hashes.get(str(p)) for p in paths}
        hashes.update({self.file_hashes.get(u) for u in websites})
        hashes = {h for h in hashes if h}
        self.records = self._load_records_for_hashes(hashes)
        return self.records

    # --- Sessions
    def _load_all_sessions(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
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

    def _write_all_sessions(self, entries: List[Dict[str, Any]]) -> None:
        tmp = self._sessions_file.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for obj in entries:
                    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            tmp.replace(self._sessions_file)
        except Exception:
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

    def list_sessions(self) -> List[Dict[str, Any]]:
        out = []
        for e in self._load_all_sessions():
            out.append({
                "title": e.get("title") or "",
                "description": e.get("description") or "",
                "saved_at": e.get("saved_at") or 0,
            })
        return out

    def load_session(self, title: str) -> Optional[Dict[str, Any]]:
        t = (title or "").strip()
        if not t:
            return None
        for e in self._load_all_sessions():
            if (e.get("title") or "").strip() == t:
                return e
        return None

    def save_session(self, payload: Dict[str, Any], *, allow_overwrite: bool) -> Dict[str, Any]:
        title = (payload.get("title") or "").strip()
        if not title:
            raise RuntimeError("Title is required to save a session.")
        exists = self._session_title_exists(title)
        if exists and not allow_overwrite:
            raise RuntimeError("Session already exists")

        files_list = payload.get("files") or []
        websites_list = payload.get("websites") or []

        self._ensure_corpus_for_files([Path(f["path"]) for f in files_list if f.get("path")])
        self._ensure_corpus_for_urls([w.get("url") for w in websites_list if w.get("url")])

        files_meta = []
        for f in files_list:
            path = f.get("path")
            if not path:
                continue
            h = self.file_hashes.get(path) or self._compute_file_hash(Path(path))
            self.file_hashes[path] = h
            files_meta.append({
                "path": path,
                "file_hash": h,
                "include": bool(f.get("include", True)),
            })
        websites_meta = []
        for w in websites_list:
            url = w.get("url")
            if not url:
                continue
            h = self.file_hashes.get(url) or self._compute_url_hash(url)
            self.file_hashes[url] = h
            websites_meta.append({
                "url": url,
                "file_hash": h,
                "include": bool(w.get("include", True)),
            })

        img_prompt = (payload.get("image_prompt") or "").strip()
        if img_prompt == IMAGE_PROMPT_PLACEHOLDER:
            img_prompt = ""

        record = {
            "title": title,
            "description": (payload.get("description") or "").strip(),
            "notes": (payload.get("notes") or "").strip(),
            "concept": (payload.get("concept") or "").strip(),
            "files": files_meta,
            "websites": websites_meta,
            "saved_at": int(time.time()),
            "rephrase_variants": payload.get("rephrase_variants") or [],
            "rephrase_selected_key": payload.get("rephrase_selected_key"),
            "image_prompt": img_prompt,
        }

        entries = self._load_all_sessions()
        if exists:
            entries = [e for e in entries if (e.get("title") or "") != title]
        entries.append(record)
        self._write_all_sessions(entries)
        return record


# -----------------------------
# Concept generation helpers
# -----------------------------

def _extract_title_desc(concept_md: str, *, client: OllamaClient, model: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        prompt = (
            "Extract a concise title and a one-sentence description from the following concept.\n"
            "- Title: <= 50 chars (3-5 words).\n- Description: <= 120 chars, (one sentence) no trailing period.\n"
            "Return ONLY strict JSON with keys 'title' and 'description'.\n\nCONCEPT:\n" + concept_md
        )
        raw = client.generate(model, prompt)
        obj = _parse_json_strict(raw) or {}
        title = strip_wrapping_quotes(str(obj.get("title") or "").strip()) or None
        desc = strip_wrapping_quotes(str(obj.get("description") or "").strip()) or None
        return title, desc
    except Exception:
        return None, None


# -----------------------------
# Git helpers and PDF conversion
# -----------------------------

def _run_git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(repo_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _ensure_repo_initialized(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    if not (repo_dir / ".git").exists():
        _ = subprocess.run(["git", "init"], cwd=str(repo_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _ensure_remote_origin(repo_dir: Path, remote_url: str) -> None:
    if not remote_url:
        return
    res = _run_git(repo_dir, "remote", "get-url", "origin")
    if res.returncode == 0:
        current = (res.stdout or "").strip()
        if current != remote_url:
            _ = _run_git(repo_dir, "remote", "set-url", "origin", remote_url)
    else:
        _ = _run_git(repo_dir, "remote", "add", "origin", remote_url)


def _ensure_branch_master(repo_dir: Path) -> None:
    res = _run_git(repo_dir, "symbolic-ref", "-q", "HEAD")
    headref = (res.stdout or "").strip() if res.returncode == 0 else ""
    if not headref:
        _run_git(repo_dir, "symbolic-ref", "HEAD", "refs/heads/master")
        return
    if headref.endswith("/master"):
        return
    _run_git(repo_dir, "branch", "-M", "master")


def _slug(s: str) -> str:
    s = re.sub(r"[\s]+", "-", s.strip())
    s = re.sub(r"[^a-zA-Z0-9._-]", "-", s)
    return re.sub(r"-+", "-", s).strip("-_")


def _build_slug_map_from_sessions() -> Dict[str, Dict[str, str]]:
    engine = ConceptEngine()
    entries = engine._load_all_sessions()
    best: Dict[str, Dict[str, str]] = {}
    best_ts: Dict[str, int] = {}
    for e in entries:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        slug = _slug(title)
        ts = 0
        try:
            ts = int(e.get("saved_at") or 0)
        except Exception:
            ts = 0
        if slug not in best or ts >= best_ts.get(slug, 0):
            best[slug] = {"title": title, "description": (e.get("description") or "").strip()}
            best_ts[slug] = ts
    return best


def _write_concepts_index(repo_dir: Path) -> None:
    try:
        slug_map = _build_slug_map_from_sessions()
        items = []
        seen: Set[str] = set()
        for child in sorted(repo_dir.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith(".") or name in {".git", "node_modules"}:
                continue
            slug = name
            if slug in seen:
                continue
            seen.add(slug)
            title = slug_map.get(slug, {}).get("title") or re.sub(r"[-_]+", " ", slug).strip().title()
            desc = slug_map.get(slug, {}).get("description") or ""
            items.append((slug, title, desc))

        intro = (
            "This folder contains a library of project concepts created with the Idea -> Concept tool. "
            "Each entry links to its folder with the original concept README and related assets."
        )
        lines: List[str] = []
        lines.append("# Concepts Index")
        lines.append("")
        lines.append(intro)
        lines.append("")
        for slug, title, desc in items:
            if desc:
                lines.append(f"- [{title}](./{slug}/) - {desc}")
            else:
                lines.append(f"- [{title}](./{slug}/)")
        (repo_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    except Exception:
        pass


def _convert_markdown_to_pdf(md_file: Path, out_pdf: Path) -> Tuple[bool, Optional[Path]]:
    concept_dir = out_pdf.parent
    concept_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = IDEA_HOLE_DIR / "logs"
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
        return False, log_path

    tmp_base = IDEA_HOLE_DIR / "tmp_pdf" / concept_dir.name
    try:
        if tmp_base.exists():
            shutil.rmtree(tmp_base)
    except Exception:
        pass
    tmp_base.mkdir(parents=True, exist_ok=True)

    try:
        text = md_file.read_text(encoding="utf-8")
    except Exception as e:
        lines.append(f"read error: {e}")
        try:
            log_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass
        return False, log_path

    img_rgx = re.compile(r"!\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")
    allowed_ext = {".png", ".jpg", ".jpeg", ".pdf", ".eps"}

    def ensure_image_available(src: str) -> str:
        p = Path(src)
        if not p.is_absolute():
            p = (concept_dir / p).resolve()
        if not p.exists():
            alt = (concept_dir / Path(src).name).resolve()
            if alt.exists():
                p = alt
            else:
                lines.append(f"missing image: {src}")
                return src
        ext = p.suffix.lower()
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
        if ext == ".svg":
            out_name = p.stem + ".png"
            out_path = tmp_base / out_name
            try:
                from cairosvg import svg2png  # type: ignore
                svg2png(url=str(p), write_to=str(out_path))
                return out_name
            except Exception as e_svg_py:
                lines.append(f"cairosvg unavailable or failed: {e_svg_py}")
            try:
                tool = shutil.which("rsvg-convert")
                if tool:
                    res = subprocess.run([tool, "-f", "png", "-o", str(out_path), str(p)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    if res.returncode == 0 and out_path.exists():
                        return out_name
                    lines.append(f"rsvg-convert failed: exit {res.returncode}, {res.stdout}")
            except Exception as e_svg_cli:
                lines.append(f"rsvg-convert error: {e_svg_cli}")
            try:
                tool = shutil.which("magick") or shutil.which("convert")
                if tool:
                    res = subprocess.run([tool, str(p), str(out_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    if res.returncode == 0 and out_path.exists():
                        return out_name
                    lines.append(f"imagemagick failed: exit {res.returncode}, {res.stdout}")
            except Exception as e_im:
                lines.append(f"imagemagick error: {e_im}")

        try:
            from PIL import Image  # type: ignore
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
            out_name = p.name
            out_path = tmp_base / out_name
            try:
                shutil.copy2(str(p), str(out_path))
                return out_name
            except Exception as e2:
                lines.append(f"final copy fail: {src} ({e2})")
                return src

    def _repl(m: re.Match) -> str:
        orig = m.group(0)
        path = m.group(1)
        rep = ensure_image_available(path)
        return orig.replace(path, rep)

    mod_text = img_rgx.sub(_repl, text)

    def _preserve_extra_blank_lines(s: str) -> str:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        lines_in = s.split("\n")
        out_lines: List[str] = []
        in_fence = False
        blank_run = 0
        for ln in lines_in:
            stripped = ln.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                if blank_run > 0:
                    out_lines.append("")
                    for _ in range(blank_run - 1):
                        out_lines.append("\\vspace{1em}")
                    blank_run = 0
                out_lines.append(ln)
                in_fence = not in_fence
                continue
            if in_fence:
                if blank_run > 0:
                    out_lines.append("")
                    for _ in range(blank_run - 1):
                        out_lines.append("\\vspace{1em}")
                    blank_run = 0
                out_lines.append(ln)
                continue
            if stripped == "":
                blank_run += 1
                continue
            if blank_run > 0:
                out_lines.append("")
                for _ in range(blank_run - 1):
                    out_lines.append("\\vspace{1em}")
                blank_run = 0
            out_lines.append(ln)
        if blank_run > 0:
            out_lines.append("")
            for _ in range(blank_run - 1):
                out_lines.append("\\vspace{1em}")
        return "\n".join(out_lines)

    mod_text = _preserve_extra_blank_lines(mod_text)
    tmp_md = tmp_base / "README_pdf.md"
    tmp_md.write_text(mod_text, encoding="utf-8")

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
        except Exception as e_fallback:
            lines.append(f"fallback error: {e_fallback}")

    if not ok:
        try:
            log_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    try:
        shutil.rmtree(tmp_base)
    except Exception:
        pass

    return ok, log_path


# -----------------------------
# Image generation
# -----------------------------

def _load_sdxl_pipeline():
    try:
        import torch  # type: ignore
        from diffusers import StableDiffusionXLPipeline, DPMSolverSDEScheduler  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Diffusers/torch required for image generation: {e}")
    model_path = Path("/Volumes/SD/ML-Models/stable-diffusion-webui/models/Stable-diffusion/SDXLModels/dreamshaperXL_v21TurboDPMSDE.safetensors")
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    device = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    )
    try:
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    except Exception:
        pass
    pipe.to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe, device


def generate_image(prompt: str, output_dir: Path, title: str) -> Path:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(f"torch not available: {e}")
    pipe, device = _load_sdxl_pipeline()
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = torch.autocast(device_type=device, dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
    generator = torch.Generator(device=device) if device != "cpu" else None
    with torch.inference_mode():
        with ctx:
            res = pipe(
                prompt=prompt,
                guidance_scale=2.0,
                num_inference_steps=6,
                num_images_per_prompt=1,
                height=1024,
                width=1024,
                generator=generator,
            )
    img = res.images[0]
    slug = _slug(title or "image")
    ts = int(time.time())
    fname = f"{slug}-sdxl-{ts}.png" if slug else f"image-{ts}.png"
    out_path = output_dir / fname
    try:
        img.save(out_path)
    except Exception:
        from PIL import Image  # type: ignore
        Image.fromarray(img).save(out_path)
    return out_path


# -----------------------------
# Settings
# -----------------------------

def settings_path() -> Path:
    IDEA_HOLE_DIR.mkdir(parents=True, exist_ok=True)
    return IDEA_HOLE_DIR / "settings.json"


def load_settings() -> Dict[str, str]:
    p = settings_path()
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return {k: str(v) for k, v in obj.items() if v is not None}
    except Exception:
        return {}
    return {}


def save_settings(settings: Dict[str, str]) -> None:
    p = settings_path()
    try:
        p.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        p.write_text(json.dumps(settings), encoding="utf-8")


# -----------------------------
# Actions
# -----------------------------

def list_models() -> List[str]:
    try:
        res = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=8)
        if res.returncode != 0:
            return []
        lines = [ln.strip() for ln in (res.stdout or "").splitlines()]
        out: List[str] = []
        for ln in lines:
            if not ln or ln.lower().startswith("name"):
                continue
            name = ln.split()[0]
            if name and name not in out:
                out.append(name)
        return out
    except Exception:
        return []


def stat_paths(paths: List[str], *, expand_dirs: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for raw in paths:
        if not raw:
            continue
        p = Path(raw)
        if p.is_dir() and expand_dirs:
            for q in p.rglob("*"):
                if not q.is_file():
                    continue
                if str(q) in seen:
                    continue
                seen.add(str(q))
                out.append({
                    "name": q.name,
                    "path": str(q),
                    "type": q.suffix.lower() or "file",
                    "size": human_size(q.stat().st_size) if q.exists() else "?",
                })
        elif p.is_file():
            if str(p) in seen:
                continue
            seen.add(str(p))
            out.append({
                "name": p.name,
                "path": str(p),
                "type": p.suffix.lower() or "file",
                "size": human_size(p.stat().st_size) if p.exists() else "?",
            })
    return out


def rephrase(note: str, host: str, model: str) -> List[Dict[str, str]]:
    client = OllamaClient(host=host)
    variants: List[Dict[str, str]] = [{
        "key": "original",
        "label": "Original Note",
        "text": note,
    }]
    for idx, lens in enumerate(REPHRASE_LENSES, start=1):
        prompt = (lens.get("prompt") or "").replace("{USER_NOTE}", note)
        raw = client.generate(model=model, prompt=prompt)
        text = sanitize_llm_text_simple(raw)
        variants.append({
            "key": lens.get("key") or f"lens_{idx}",
            "label": lens.get("label") or f"Variant {idx}",
            "text": text,
        })
    return variants


def extend(note: str, host: str, model: str) -> str:
    client = OllamaClient(host=host)
    prompt = EXTEND_PROMPT.replace("{USER_NOTE}", note)
    raw = client.generate(model=model, prompt=prompt)
    text = sanitize_llm_text_simple(raw)
    if not text.strip():
        raise RuntimeError("Empty response from model")
    return text


def generate_concept(payload: Dict[str, Any]) -> Dict[str, Any]:
    notes = (payload.get("notes") or "").strip()
    files = payload.get("files") or []
    websites = payload.get("websites") or []
    include_map = payload.get("include_map") or {}
    host = payload.get("ollama_host") or "http://localhost:11434"
    model = payload.get("model") or ""

    engine = ConceptEngine()
    records = engine.build_kb_records(files, websites)
    kb = build_kb_string(records)

    assets_lines: List[str] = []
    assets_files = [p for p in files if include_map.get(str(p), True)]
    assets_urls = [u for u in websites if include_map.get(str(u), True)]
    if assets_files:
        assets_lines.append("Files:")
        assets_lines.extend(f"- {Path(p).name}" for p in assets_files)
    if assets_urls:
        assets_lines.append("URLs:")
        assets_lines.extend(f"- {u}" for u in assets_urls)
    assets_str = "\n".join(assets_lines) or "(none)"

    prompt = (
        PROMPT_TEMPLATE
        .replace("{NOTES}", notes or "(none)")
        .replace("{KB}", kb or "(empty)")
        .replace("{ASSETS}", assets_str)
    )

    client = OllamaClient(host=host)
    concept_md = client.generate(model=model, prompt=prompt)
    concept_md = sanitize_llm_text_simple(concept_md)
    title, desc = _extract_title_desc(concept_md, client=client, model=model)
    if not desc:
        desc = ""
    if title:
        concept_md = md_heading_replace_or_insert(concept_md, title)

    if not concept_md.strip():
        raise RuntimeError("Empty response from model")

    return {
        "concept": concept_md,
        "title": title or "",
        "description": strip_wrapping_quotes(desc)[:120],
        "kb_records": len(records),
    }


def generate_image_prompt(payload: Dict[str, Any]) -> str:
    idea_text = (payload.get("idea_text") or "").strip()
    host = payload.get("ollama_host") or "http://localhost:11434"
    model = payload.get("model") or ""
    client = OllamaClient(host=host)
    return generate_image_prompt_for_idea(idea_text, client=client, model=model)


def prior_art(payload: Dict[str, Any]) -> Dict[str, Any]:
    notes = (payload.get("notes") or "").strip()
    files = payload.get("files") or []
    websites = payload.get("websites") or []
    include_map = payload.get("include_map") or {}
    host = payload.get("ollama_host") or "http://localhost:11434"
    model = payload.get("model") or ""
    searx_url = payload.get("searx_url") or None

    engine = ConceptEngine()
    records = engine.build_kb_records(files, websites)
    kb = build_kb_string(records)
    assets = [p for p in files if include_map.get(str(p), True)]

    return websearch.prior_art_search(
        ollama_host=host,
        model=model,
        notes=notes,
        kb=kb,
        assets=assets,
        searx_url=searx_url,
    )


def preview_pdf(payload: Dict[str, Any]) -> Dict[str, Any]:
    concept_text = (payload.get("concept") or "").strip()
    title = (payload.get("title") or "").strip()
    files = payload.get("files") or []
    include_map = payload.get("include_map") or {}
    if not concept_text:
        raise RuntimeError("Concept text is empty")

    slug = _slug(title or "preview")
    base = IDEA_HOLE_DIR / "preview" / f"{slug}-preview"
    try:
        if base.exists():
            shutil.rmtree(base)
    except Exception:
        pass
    base.mkdir(parents=True, exist_ok=True)

    md_path = base / "README.md"
    md_path.write_text(concept_text, encoding="utf-8")

    assets = [Path(p) for p in files if include_map.get(str(p), True)]
    for src in assets:
        try:
            dst = base / src.name
            if dst.name.lower() in {"readme.md", f"{slug}-concept.pdf".lower(), f"{slug}-preview.pdf".lower()}:
                dst = base / f"asset-{src.name}"
            shutil.copy2(src, dst)
        except Exception:
            pass

    pdf_path = base / f"{slug}-preview.pdf"
    ok, log_path = _convert_markdown_to_pdf(md_path, pdf_path)
    return {
        "ok": ok,
        "pdf_path": str(pdf_path),
        "log_path": str(log_path) if log_path else "",
    }


def push_repo(payload: Dict[str, Any]) -> Dict[str, Any]:
    title = (payload.get("title") or "").strip()
    desc = (payload.get("description") or "").strip()
    concept_text = (payload.get("concept") or "").strip()
    files = payload.get("files") or []
    include_map = payload.get("include_map") or {}
    remote = (payload.get("git_remote_url") or "").strip()
    repo_dir = Path(payload.get("repo_dir") or CONCEPTS_DIR)

    if not title or not desc:
        raise RuntimeError("Title and Description are required")
    if not concept_text:
        raise RuntimeError("Concept text is empty")

    _ensure_repo_initialized(repo_dir)
    slug = _slug(title)
    concept_dir = repo_dir / slug
    concept_dir.mkdir(parents=True, exist_ok=True)
    md_path = concept_dir / "README.md"
    md_path.write_text(concept_text, encoding="utf-8")

    assets = [Path(p) for p in files if include_map.get(str(p), True)]
    for src in assets:
        try:
            dst = concept_dir / src.name
            if dst.name.lower() in {"readme.md", f"{slug}-concept.pdf".lower()}:
                dst = concept_dir / f"asset-{src.name}"
            shutil.copy2(src, dst)
        except Exception:
            pass

    pdf_path = concept_dir / f"{slug}-concept.pdf"
    ok_pdf, log_path = _convert_markdown_to_pdf(md_path, pdf_path)

    try:
        _write_concepts_index(repo_dir)
    except Exception:
        pass

    add_res = _run_git(repo_dir, "add", ".")
    if add_res.returncode != 0:
        raise RuntimeError(add_res.stdout)
    commit_msg = f"{title} - {desc}"
    commit_res = _run_git(repo_dir, "commit", "-m", commit_msg)
    if commit_res.returncode != 0:
        if "nothing to commit" not in (commit_res.stdout or "").lower():
            raise RuntimeError(commit_res.stdout)

    _ensure_branch_master(repo_dir)
    pushed = False
    if remote:
        _ensure_remote_origin(repo_dir, remote)
        push_res = _run_git(repo_dir, "push", "-u", "origin", "master")
        if push_res.returncode != 0:
            raise RuntimeError(push_res.stdout)
        pushed = True

    return {
        "repo_dir": str(repo_dir),
        "concept_dir": str(concept_dir),
        "pdf_path": str(pdf_path),
        "pdf_ok": ok_pdf,
        "pdf_log": str(log_path) if log_path else "",
        "pushed": pushed,
    }


# -----------------------------
# JSON-RPC style entrypoint
# -----------------------------

def _read_stdin_json() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw:
        return {}
    return json.loads(raw)


def main() -> int:
    try:
        req = _read_stdin_json()
        action = req.get("action")
        payload = req.get("payload") or {}

        if not action:
            raise RuntimeError("Missing action")

        if action == "list_models":
            result = list_models()
        elif action == "stat_paths":
            result = stat_paths(payload.get("paths") or [], expand_dirs=bool(payload.get("expand_dirs")))
        elif action == "rephrase":
            result = rephrase(payload.get("note") or "", payload.get("ollama_host") or "http://localhost:11434", payload.get("model") or "")
        elif action == "extend":
            result = extend(payload.get("note") or "", payload.get("ollama_host") or "http://localhost:11434", payload.get("model") or "")
        elif action == "generate_concept":
            result = generate_concept(payload)
        elif action == "generate_image_prompt":
            result = generate_image_prompt(payload)
        elif action == "prior_art":
            result = prior_art(payload)
        elif action == "preview_pdf":
            result = preview_pdf(payload)
        elif action == "push_repo":
            result = push_repo(payload)
        elif action == "generate_image":
            prompt = payload.get("prompt") or ""
            out_dir = Path(payload.get("output_dir") or "")
            title = payload.get("title") or ""
            if not prompt or not out_dir:
                raise RuntimeError("Missing prompt or output_dir")
            out_path = generate_image(prompt, out_dir, title)
            result = {"output_path": str(out_path)}
        elif action == "load_settings":
            result = load_settings()
        elif action == "save_settings":
            save_settings(payload.get("settings") or {})
            result = {"ok": True}
        elif action == "list_sessions":
            engine = ConceptEngine()
            result = engine.list_sessions()
        elif action == "load_session":
            engine = ConceptEngine()
            result = engine.load_session(payload.get("title") or "")
        elif action == "save_session":
            engine = ConceptEngine()
            result = engine.save_session(payload.get("payload") or {}, allow_overwrite=bool(payload.get("allow_overwrite")))
        else:
            raise RuntimeError(f"Unknown action: {action}")

        out = {"ok": True, "data": result}
    except Exception as e:
        out = {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(limit=6),
        }
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
