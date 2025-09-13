#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prior-art web search utilities for the Idea → Concept GUI.

Design goals
- Pure stdlib networking (urllib) to avoid new dependencies.
- Works with a local SearXNG instance.
- Uses Ollama for both query generation and embeddings.
- Small, robust, and callable from the GUI with status callbacks.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Callable

import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse
import urllib.request
import urllib.error

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


# Default SearXNG endpoint (no trailing slash)
SEARX_DEFAULT_URL = "http://localhost:8888"


# Prompt to generate search queries from sources
SEARCH_QUERIES_PROMPT = (
    """
You are a cross-domain prior-art scout. Your only task is to EMIT SEARCH QUERIES (no explanations) that can be sent to SearXNG to discover whether the user’s idea already exists.

SOURCES
- NOTES (freeform by user):
{NOTES}

- KNOWLEDGE BASE (excerpts):
{KB}

- ASSET NAMES (filenames the user selected):
{ASSETS}

RULES
- Infer the single most likely idea from the sources above (silently).
- Return STRICT JSON ONLY with exactly this schema:
  {"queries": ["<q1>", "<q2>", "<q3>"], "lang": "<iso-639-1>"}
- Requirements:
  • Exactly 3 distinct queries; each ≤ 120 characters.
  • Use quotes for distinctive phrases when helpful.
  • Include synonyms or likely alternate names if useful.
  • Add up to 2 disambiguating negatives with a leading “-” only when needed.
  • Prefer engine-portable operators only: quotes "…", OR, -, site:, filetype:. Avoid engine-specific syntax.
  • Vary the angles across the 3 queries for broad recall:
      1) direct existence/implementation check,
      2) synonym/alternate-naming variant,
      3) precision angle (compare/alternative wording OR a domain-targeted site:/filetype: if obvious).
  • Optional: include a simple year hint like 2019..2025 only if recency clearly matters.
- Choose "lang" from the dominant language in NOTES/KB/ASSETS; default to "en" if unclear.
- No prose, no markdown/code fences, no extra keys beyond {"queries","lang"}.
"""
).strip()


# -----------------------------
# Utils
# -----------------------------

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome Safari"
)

def sanitize_llm_text_simple(s: str) -> str:
    try:
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
        s = re.sub(r"^\s*```(?:\w+)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        return s.strip()
    except Exception:
        return (s or "").strip()


def _http_post_json(url: str, payload: Dict[str, Any], *, timeout: int = 600) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8", "ignore"))


def _http_get(url: str, *, timeout: int = 20, max_bytes: int = 1_500_000) -> Tuple[str, Dict[str, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = (resp.headers.get("content-type") or "").lower()
        buf = bytearray()
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) >= max_bytes:
                break
        try:
            text = buf.decode(resp.headers.get_content_charset() or "utf-8", errors="ignore")
        except Exception:
            text = buf.decode("utf-8", errors="ignore")
        return text, {"content-type": ctype}


def _http_get_json(url: str, params: Dict[str, Any], *, timeout: int = 30) -> Dict[str, Any]:
    q = urlencode(params)
    text, _ = _http_get(f"{url}?{q}", timeout=timeout)
    try:
        return json.loads(text)
    except Exception:
        return {}


def _is_probably_html_url(url: str) -> bool:
    try:
        path = urlparse(url).path.lower()
        if not path:
            return True
        if "." not in path:
            return True
        ext = path.rsplit(".", 1)[-1]
        return ext not in {
            "pdf","jpg","jpeg","png","gif","webp","svg","mp4","mp3","mov","avi",
            "zip","gz","7z","tar","rar","woff","woff2","ttf","otf"
        }
    except Exception:
        return True


def _extract_text(html: str, *, max_len: int = 120_000) -> str:
    if not html:
        return ""
    if BeautifulSoup is None:
        # crude fallback if bs4 missing
        txt = re.sub(r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", " ", html, flags=re.S|re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()[:max_len]
    try:
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
        for tag in soup.select("script,style,noscript"):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:max_len] if len(text) > max_len else text
    except Exception:
        return ""


# -----------------------------
# Ollama helpers
# -----------------------------

def ollama_generate(host: str, model: str, prompt: str, *, timeout: int = 600) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        obj = _http_post_json(url, payload, timeout=timeout)
        return (obj.get("response") or "").strip()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error {e.code}: {e.read().decode('utf-8','ignore')}")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def ollama_embed(host: str, model: str, text: str, *, timeout: int = 120) -> List[float]:
    url = f"{host.rstrip('/')}/api/embeddings"
    payload = {"model": model, "prompt": text}
    try:
        obj = _http_post_json(url, payload, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama embeddings HTTP error {e.code}: {e.read().decode('utf-8','ignore')}")
    except Exception as e:
        raise RuntimeError(f"Ollama embeddings request failed: {e}")

    # Normalize common shapes
    if isinstance(obj, dict):
        if "error" in obj:
            raise RuntimeError(f"Ollama embeddings error: {obj.get('error')}")
        if "embedding" in obj and isinstance(obj["embedding"], list):
            return [float(x) for x in obj["embedding"] if isinstance(x, (int, float))]
        if "data" in obj and isinstance(obj["data"], list) and obj["data"]:
            em = obj["data"][0].get("embedding", [])
            if isinstance(em, list):
                return [float(x) for x in em if isinstance(x, (int, float))]
        if "embeddings" in obj and isinstance(obj["embeddings"], list) and obj["embeddings"]:
            em0 = obj["embeddings"][0]
            if isinstance(em0, dict):
                em0 = em0.get("embedding", [])
            if isinstance(em0, list):
                return [float(x) for x in em0 if isinstance(x, (int, float))]
    return []


def cosine(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = na = nb = 0.0
    for i in range(n):
        x = a[i]; y = b[i]
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# -----------------------------
# SearXNG search + fetch
# -----------------------------

def searx_search(base_url: str, query: str, *, max_results: int = 6) -> List[Dict[str, Any]]:
    base = (base_url or SEARX_DEFAULT_URL).rstrip("/")
    params = {"q": query, "format": "json", "safesearch": 1}
    data = _http_get_json(f"{base}/search", params, timeout=30)
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in data.get("results", []):
        url = item.get("url")
        title = item.get("title") or ""
        if not url or url in seen:
            continue
        if not _is_probably_html_url(url):
            continue
        seen.add(url)
        out.append({"url": url, "title": title})
        if len(out) >= max_results:
            break
    return out


def fetch_pages(urls: List[str], *, max_pages: int = 8, min_text_len: int = 500) -> Dict[str, str]:
    urls = urls[:max_pages]
    out: Dict[str, str] = {}
    for u in urls:
        try:
            html, hdrs = _http_get(u, timeout=20)
            ctype = (hdrs.get("content-type") or "").lower()
            if ctype and not ("text/html" in ctype or "application/xhtml+xml" in ctype or ctype.startswith("text/")):
                continue
            txt = _extract_text(html)
            if len(txt) >= min_text_len:
                out[u] = txt
        except Exception:
            continue
    return out


# -----------------------------
# Public API
# -----------------------------

def generate_prior_art_queries(
    *,
    ollama_host: str,
    model: str,
    notes: str,
    kb: str,
    assets: List[str],
) -> Dict[str, Any]:
    assets_str = "\n".join(f"- {Path(a).name}" for a in assets) if assets else "(none)"
    prompt = (
        SEARCH_QUERIES_PROMPT
        .replace("{NOTES}", (notes or "").strip() or "(none)")
        .replace("{KB}", (kb or "").strip() or "(empty)")
        .replace("{ASSETS}", assets_str)
    )
    raw = ollama_generate(ollama_host, model, prompt)
    raw = sanitize_llm_text_simple(raw)
    obj: Dict[str, Any]
    try:
        obj = json.loads(raw)
    except Exception:
        # try to salvage JSON object substring
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise RuntimeError("Model did not return JSON for search queries.")
        try:
            obj = json.loads(m.group(0))
        except Exception:
            raise RuntimeError("Failed to parse JSON for search queries.")

    q = obj.get("queries") if isinstance(obj, dict) else None
    lang = obj.get("lang") if isinstance(obj, dict) else None
    if not isinstance(q, list) or len(q) != 3:
        raise RuntimeError("Search query generator must return exactly 3 queries.")
    queries = [str(x).strip() for x in q if str(x).strip()]
    if len(queries) != 3:
        raise RuntimeError("Invalid queries after normalization.")
    return {"queries": queries, "lang": (str(lang).strip() or "en")}


def prior_art_search(
    *,
    ollama_host: str,
    model: str,
    notes: str,
    kb: str,
    assets: List[str],
    searx_url: Optional[str] = None,
    embed_model: str = "bge-m3:latest",
    status_cb: Optional[Callable[[str], None]] = None,
    per_query_max: int = 6,
    overall_max_urls: int = 16,
    fetch_max_pages: int = 8,
) -> Dict[str, Any]:
    """
    End-to-end prior-art search pipeline.
    Returns a dict with keys: {"queries", "lang", "results": [{url,title,score,snippet}...]}
    """
    def _status(msg: str):
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                pass

    _status("Generating search queries…")
    meta = generate_prior_art_queries(
        ollama_host=ollama_host, model=model, notes=notes, kb=kb, assets=assets
    )
    queries: List[str] = meta["queries"]
    lang: str = meta.get("lang", "en")

    _status("Searching SearXNG…")
    all_urls: List[str] = []
    seen = set()
    base = (searx_url or SEARX_DEFAULT_URL).rstrip("/")
    for q in queries:
        try:
            res = searx_search(base, q, max_results=per_query_max)
        except Exception:
            res = []
        for item in res:
            u = item.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            all_urls.append(u)
            if len(all_urls) >= overall_max_urls:
                break
        if len(all_urls) >= overall_max_urls:
            break

    if not all_urls:
        return {"queries": queries, "lang": lang, "results": []}

    _status("Fetching pages…")
    pages = fetch_pages(all_urls, max_pages=fetch_max_pages, min_text_len=500)
    if not pages:
        return {"queries": queries, "lang": lang, "results": []}

    # Build doc list and short snippets
    docs: List[Tuple[str, str]] = []
    for u in all_urls:
        if u in pages:
            docs.append((u, pages[u]))
    if not docs:
        return {"queries": queries, "lang": lang, "results": []}

    _status("Embedding and reranking…")
    # Embed all 3 queries
    q_embs: List[List[float]] = []
    for q in queries:
        try:
            q_embs.append(ollama_embed(ollama_host, embed_model, f"query: {q}"))
        except Exception:
            q_embs.append([])

    # Embed documents (short passages)
    DOC_SNIPPET_CHARS = 800
    d_embs: List[List[float]] = []
    for (_u, text) in docs:
        snippet = text.replace("\n", " ")
        if len(snippet) > DOC_SNIPPET_CHARS:
            snippet = snippet[:DOC_SNIPPET_CHARS]
        try:
            d_embs.append(ollama_embed(ollama_host, embed_model, f"passage: {snippet}"))
        except Exception:
            d_embs.append([])

    # Score each doc by max cosine across query embeddings
    results: List[Tuple[str, float]] = []  # (url, score)
    for i, (u, _t) in enumerate(docs):
        best = 0.0
        for qv in q_embs:
            if not qv or not d_embs[i]:
                continue
            c = cosine(qv, d_embs[i])
            if c > best:
                best = c
        # scale to 0..100 for user-facing
        score = max(0.0, min(100.0, (best + 1.0) * 50.0))
        results.append((u, score))

    # Sort and format output
    results.sort(key=lambda x: x[1], reverse=True)
    out: List[Dict[str, Any]] = []
    for (u, sc) in results:
        txt = pages.get(u, "").strip()
        snippet = txt[:800].replace("\n", " ") if txt else ""
        out.append({"url": u, "title": "", "score": round(sc, 1), "snippet": snippet})

    return {"queries": queries, "lang": lang, "results": out}

