import asyncio
from typing import List, Tuple, Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup
import re
import json
import traceback
import hashlib

from .ollama_client import chat as ollama_chat

# Configure your local SearXNG instance URL (no trailing slash)
SEARX_URL = "http://localhost:8888"

# ----- Utilities ----------------------------------------------------------------

def clean_text(html: str, max_len: int = 120_000) -> str:
    # Prefer lxml parser if available (significantly faster); fall back gracefully.
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup.select("script,style,noscript"):
        tag.decompose()

    # Fast text extraction
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text[:max_len] if len(text) > max_len else text

def render_recent_context(messages: Optional[List[Dict[str, str]]], char_limit: int = 1200) -> str:
    """
    Turn the last few turns into a compact excerpt:
    user: ...
    assistant: ...
    Truncate aggressively for resource-friendliness.
    """
    if not messages:
        return ""
    # Keep as-is order (oldest->newest). We assume caller already sliced last N.
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # compactify each line
        content = re.sub(r"\s+", " ", content)
        lines.append(f"{role}: {content}")
    joined = "\n".join(lines)
    if len(joined) > char_limit:
        return joined[-char_limit:]  # keep tail (most recent part)
    return joined

# ----- SearXNG search & fetching -------------------------------------------------

from urllib.parse import urlparse
import itertools

# HTTP client tuning (keep-alive pool + HTTP/2)
HTTP_LIMITS = httpx.Limits(max_keepalive_connections=32, max_connections=64)
HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}

# Fetch policy
MAX_PAGES_FETCH = 8           # lower cap of pages we’ll fetch per enrichment
FETCH_CONCURRENCY = 8         # concurrent page fetches
MAX_BYTES_PER_PAGE = 1_500_000  # ~1.5 MB read cap per page (streamed)
MIN_TEXT_LENGTH = 500         # discard useless pages early

SKIP_EXTS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp4", ".mp3", ".mov", ".avi", ".zip", ".gz", ".7z", ".tar", ".rar",
    ".woff", ".woff2", ".ttf", ".otf"
}

_PAGE_CACHE: Dict[str, str] = {}  # naive in-proc cache (speeds repeated calls)
_EMB_CACHE: Dict[str, List[float]] = {}  # NEW: embedding cache by SHA1 of snippet

def _is_probably_html_url(url: str) -> bool:
    try:
        path = urlparse(url).path.lower()
        ext = (path.rsplit(".", 1)[-1] if "." in path else "")
        return (("." not in path) or (ext and f".{ext}" not in SKIP_EXTS))
    except Exception:
        return True

async def searx_search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int = 6,
    *,
    searx_url: Optional[str] = None,
    engines: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    base_url = (searx_url or SEARX_URL).rstrip("/")
    params = {"q": query, "format": "json", "safesearch": 1}
    if engines:
        # SearXNG accepts a comma-separated 'engines' filter
        params["engines"] = ",".join(engines)

    try:
        r = await client.get(f"{base_url}/search", params=params)
        r.raise_for_status()
    except Exception as e:
        print(f"[web] searx_search error for {base_url} q='{query}': {repr(e)}")
        return []

    try:
        data = r.json()
    except Exception as e:
        print(f"[web] searx_search JSON parse failed q='{query}': {repr(e)} | content-type={r.headers.get('content-type')}")
        return []

    seen = set()
    out = []
    for item in data.get("results", []):
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"url": url, "title": item.get("title", ""), "engine": item.get("engine")})
        if len(out) >= max_results:
            break
    if not out:
        print(f"[web] searx_search returned 0 urls for q='{query}' (engines={engines})")
    return out

async def searx_search_many(
    client: httpx.AsyncClient,
    queries: List[str],
    per_query_max: int = 6,
    overall_max: int = 32,
    *,
    searx_url: Optional[str] = None,
    engines: Optional[List[str]] = None,
) -> List[str]:
    print(f"[web] searx_search_many: {len(queries)} queries -> {searx_url or SEARX_URL} engines={engines or 'default'}")
    tasks = [searx_search(client, q, max_results=per_query_max, searx_url=searx_url, engines=engines) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    urls: List[str] = []
    seen = set()
    for q, res in zip(queries, results):
        if isinstance(res, Exception):
            print(f"[web] searx_search_many task error q='{q}': {repr(res)}")
            continue
        for item in res:
            u = item.get("url")
            if not u or u in seen:
                continue
            if not _is_probably_html_url(u):
                print(f"[web] skip non-HTML url: {u}")
                continue
            seen.add(u)
            urls.append(u)
            if len(urls) >= overall_max:
                print(f"[web] searx_search_many hit overall_max={overall_max}")
                return urls
    print(f"[web] searx_search_many collected {len(urls)} urls")
    return urls

async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
    if url in _PAGE_CACHE:
        return _PAGE_CACHE[url]
    if not _is_probably_html_url(url):
        return ""

    try:
        async with client.stream("GET", url, headers=DEFAULT_HEADERS) as r:
            ctype = (r.headers.get("content-type") or "").lower()
            if ctype and not ("text/html" in ctype or "application/xhtml+xml" in ctype or ctype.startswith("text/")):
                print(f"[web] skip content-type {ctype} url={url}")
                return ""

            buf = bytearray()
            async for chunk in r.aiter_bytes():
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) >= MAX_BYTES_PER_PAGE:
                    break

            try:
                text = buf.decode(r.encoding or "utf-8", errors="ignore")
            except Exception:
                text = buf.decode("utf-8", errors="ignore")

            _PAGE_CACHE[url] = text
            return text
    except Exception as e:
        print(f"[web] fetch_page error url={url}: {repr(e)}")
        return ""

async def gather_pages(
    client: httpx.AsyncClient,
    urls: List[str],
    max_pages: int = MAX_PAGES_FETCH,
) -> Dict[str, str]:
    urls = list(itertools.islice(urls, 0, max_pages))
    sem = asyncio.Semaphore(FETCH_CONCURRENCY)
    out: Dict[str, str] = {}
    counters = {"tried": 0, "ok": 0, "too_short": 0, "empty": 0}

    async def _one(u: str):
        async with sem:
            counters["tried"] += 1
            html = await fetch_page(client, u)
            if not html:
                counters["empty"] += 1
                return
            txt = clean_text(html)
            if len(txt) < MIN_TEXT_LENGTH:
                counters["too_short"] += 1
                return
            out[u] = txt
            counters["ok"] += 1

    await asyncio.gather(*(_one(u) for u in urls))
    print(f"[web] gather_pages summary: {counters} / returned={len(out)}")
    return out

# ----- LLM helpers ---------------------------------------------------------------

async def generate_queries(prompt: str, model: str, context_excerpt: str) -> List[str]:
    """
    Single LLM call that *already* incorporates recent chat context to
    resolve ellipses/pronouns. No extra rounds.
    """
    ctx_block = f"\nRecent conversation (latest last):\n{context_excerpt}\n" if context_excerpt else ""
    ask = f"""You are a search query generator.
Given the new user message and recent conversation, produce 3 diverse, terse web search queries that best address the user's *current* intent.
Rules:
- Resolve ambiguous references using the recent conversation when present.
- No quotes, no site: operators, no overly long queries.
- Return each query on its own line.

User message:
{prompt}

{ctx_block}"""
    resp = await ollama_chat(model, [{"role": "user", "content": ask}])
    lines = [l.strip(" -\t") for l in resp.splitlines() if l.strip()]
    uniq, seen = [], set()
    for l in lines:
        k = l.lower()
        if k in seen:
            continue
        uniq.append(l)
        seen.add(k)
        if len(uniq) >= 3:
            break
    return uniq or [prompt]

async def rerank(
    prompt: str,
    docs: List[Tuple[str, str]],
    model: str,                  # kept for signature compatibility (unused here)
    context_excerpt: str,
    embed_model: str = "bge-m3:latest"  # prefer explicit tag; we will auto-fallback
) -> List[Tuple[str, str, float]]:
    """
    Embedding-based reranker (bge-m3 via Ollama) using cosine similarity.

    Robustness upgrades:
    - Try embed_model; if needed fallback to the ':latest' or untagged alias.
    - Normalize multiple possible response schemas from Ollama.
    - If the query vector is empty, retry the query alone.
    - If only the query is returned, retry passages in a second call.
    - Detailed logging of counts and dimensions.
    """
    import time
    t0 = time.perf_counter()

    # --- optional fast cosine via NumPy ---------------------------------------
    try:
        import numpy as _np  # optional
    except Exception:
        _np = None

    def _cosine(a: List[float], b: List[float]) -> float:
        if _np is not None:
            va = _np.asarray(a, dtype="float32")
            vb = _np.asarray(b, dtype="float32")
            n = min(va.size, vb.size)
            if n == 0:
                return 0.0
            va = va[:n]; vb = vb[:n]
            na = _np.linalg.norm(va); nb = _np.linalg.norm(vb)
            if na <= 0.0 or nb <= 0.0:
                return 0.0
            return float(va.dot(vb) / (na * nb))
        # pure Python fallback
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

    # --- build query + passages ------------------------------------------------
    ctx_tail = (("\ncontext: " + context_excerpt[-400:]) if context_excerpt else "")
    q_text = f"query: {prompt.strip()}{ctx_tail}"

    DOC_SNIPPET_CHARS = 800  # keep short & fast
    passages: List[str] = []
    raw_snippets: List[str] = []  # for cache keys
    for (_u, t) in docs:
        snippet = t.replace("\n", " ")
        if len(snippet) > DOC_SNIPPET_CHARS:
            snippet = snippet[:DOC_SNIPPET_CHARS]
        passages.append(f"passage: {snippet}")
        raw_snippets.append(snippet)

    # --- embedding cache -------------------------------------------------------
    emb_cache: Dict[str, List[float]] = globals().get("_EMB_CACHE", {})
    try:
        import hashlib
        keys = [hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest() for s in raw_snippets]
    except Exception:
        keys = [f"nocache-{i}" for i in range(len(raw_snippets))]

    # Prepare inputs:
    #   0 -> query; >=1 -> only passages NOT present in cache
    to_embed_inputs: List[str] = [q_text]
    pos_to_passage_idx: Dict[int, int] = {}
    p_cached = 0
    for i, (p, k) in enumerate(zip(passages, keys)):
        if k in emb_cache:
            p_cached += 1
            continue
        pos = len(to_embed_inputs)
        to_embed_inputs.append(p)
        pos_to_passage_idx[pos] = i

    # --- helpers to call Ollama embeddings ------------------------------------
    async def _embed_inputs(inputs: List[str], model_name: str) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Call Ollama /api/embeddings once per input with {"model", "prompt"}.
        Keeps order; returns [[]] for failures at a given index.
        """
        timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
        sem = asyncio.Semaphore(8)  # cap concurrency a bit

        async def _one(text: str) -> Tuple[List[float], Optional[str]]:
            payload = {"model": model_name, "prompt": text}
            try:
                async with sem:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        r = await client.post("http://localhost:11434/api/embeddings", json=payload)
                        r.raise_for_status()
                        data = r.json()
            except httpx.HTTPStatusError as e:
                return [], f"http_error:{e}"
            except Exception as e:
                return [], f"request_error:{e}"

            # normalize common shapes
            if isinstance(data, dict):
                if "error" in data:
                    return [], f"model_error:{data.get('error')}"
                if "embedding" in data and isinstance(data["embedding"], list):
                    return data["embedding"], None
                if "data" in data and isinstance(data["data"], list) and data["data"]:
                    em = data["data"][0].get("embedding", [])
                    return (em if isinstance(em, list) else []), None
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    # some servers may return {"embeddings":[vector]} even for single prompt
                    emb0 = data["embeddings"][0]
                    if isinstance(emb0, dict):
                        emb0 = emb0.get("embedding", [])
                    return (emb0 if isinstance(emb0, list) else []), None

            return [], "parse_error"

        tasks = [_one(t) for t in inputs]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        embs: List[List[float]] = []
        errs: List[str] = []
        for emb, err in results:
            embs.append(emb)
            if err:
                errs.append(err)

        meta: Dict[str, Any] = {}
        if errs:
            # light meta summary
            meta["errors"] = {k: errs.count(k) for k in set(errs)}
        return embs, meta

    # --- do the embedding calls ------------------------------------------------
    t_embed = time.perf_counter()
    inputs_all = to_embed_inputs  # [query] + uncached passages
    embeddings, meta = await _embed_inputs(inputs_all, embed_model)

    # simple model fallback if *all* returned empty
    if not any(len(e) for e in embeddings):
        tried = [embed_model]
        alt = (embed_model.split(":", 1)[0] if ":" in embed_model else embed_model + ":latest")
        if alt != embed_model:
            embeddings, meta2 = await _embed_inputs(inputs_all, alt)
            if any(len(e) for e in embeddings):
                print(f"[web] embed() recovered with fallback model {alt} (meta={meta or meta2})")
                embed_model = alt
            else:
                print(f"[web] embed() FAILED (models tried={tried + [alt]}, meta={meta or meta2})")
                return [(u, t, 0.0) for (u, t) in docs]

    # split q vs passages and update cache
    q_emb = embeddings[0] if embeddings else []
    if not q_emb:
        print("[web] embed() empty query vector — aborting rerank")
        return [(u, t, 0.0) for (u, t) in docs]

    # positions >=1 correspond to passages (only those that weren’t cached)
    for pos, emb_vec in enumerate(embeddings[1:], start=1):
        i = pos_to_passage_idx.get(pos)
        if i is None:
            continue
        if not keys[i].startswith("nocache-"):
            emb_cache[keys[i]] = emb_vec

    # build aligned passage vectors
    p_emb_list: List[List[float]] = [emb_cache.get(k, []) for k in keys]

    # logging
    q_dim = len(q_emb)
    p_dims = [len(v) for v in p_emb_list]
    print(f"[web] embed() took {time.perf_counter() - t_embed:.3f}s "
          f"(model='{embed_model}', q_dim={q_dim}, p_cached={p_cached}, "
          f"p_fresh={sum(1 for d in p_dims if d>0)}, total_p={len(p_emb_list)})")

    # --- score + rank ----------------------------------------------------------
    t_score = time.perf_counter()
    scored: List[Tuple[str, str, float]] = []
    for (u, t), p_emb in zip(docs, p_emb_list):
        cos = _cosine(q_emb, p_emb)
        score_0_100 = max(0.0, min(100.0, (cos + 1.0) * 50.0))
        scored.append((u, t, score_0_100))
    scored.sort(key=lambda x: x[2], reverse=True)
    print(f"[web] cosine scoring took {time.perf_counter() - t_score:.3f}s; rerank total {time.perf_counter() - t0:.3f}s")

    return scored

def build_enriched_prompt(user_prompt: str, ranked: List[Tuple[str, str, float]], top_k: int = 6) -> Tuple[str, List[str]]:
    """
    Build an enriched prompt using only high-quality documents.
    - Keep at most `top_k` docs with score >= MIN_SCORE.
    - If none survive, return a <websearch_context> telling the assistant that
      a web search was performed but no good results were found.
    """
    MIN_SCORE = 70.0

    # Sort defensively (should already be sorted desc)
    ranked = sorted(ranked, key=lambda x: x[2], reverse=True)

    # Strict cutoff
    selected = [(u, t, sc) for (u, t, sc) in ranked if sc >= MIN_SCORE][:top_k]

    # Debug summary
    try:
        all_scores = [sc for (_u, _t, sc) in ranked]
        sel_scores = [sc for (_u, _t, sc) in selected]
        if all_scores:
            print(f"[web] selection ≥{MIN_SCORE} → total={len(all_scores)}, selected={len(selected)}, "
                  f"top_sel={max(sel_scores) if sel_scores else 0:.1f}, "
                  f"min_sel={min(sel_scores) if sel_scores else 0:.1f}")
    except Exception:
        pass

    if not selected:
        # No “good” results → still enrich with an explicit no-results message
        parts = [
            "<websearch_context>",
            f"No suitable web results found (score < {MIN_SCORE}). "
            "Tell the user you performed a web search but couldn't find any good results. "
            "If you can answer from prior context or general knowledge, say so clearly and "
            "do not fabricate citations or URLs.",
            "</websearch_context>",
        ]
        enriched = f"{user_prompt}\n\n" + "\n".join(parts)
        return enriched, []

    # Build normal context
    sources = [u for (u, _, _) in selected]
    parts = ["<websearch_context>"]
    for i, (u, t, _) in enumerate(selected, 1):
        snippet = t.strip().replace("\n", " ")
        if len(snippet) > 1200:
            snippet = snippet[:1200]
        parts.append(f"[{i}] {snippet} (source: {u})")
    parts.append("</websearch_context>")
    parts.append("\nAnswer the user using the context when relevant.\nMake your response long enough to reflect all interesting information found. No duplicate information.\nUnless related to the topic, don't mention the source from the web search.")

    enriched = f"{user_prompt}\n\n" + "\n".join(parts)
    return enriched, sources

# ----- Public API ----------------------------------------------------------------

async def enrich_prompt(
    user_prompt: str,
    model: str,
    messages: Optional[List[Dict[str, str]]] = None,
    *,
    searx_url: Optional[str] = None,
    engines: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    import time  # local import to avoid touching global imports
    start_all = time.perf_counter()

    def _no_results_enriched(reason: str, queries: Optional[List[str]] = None) -> Tuple[str, List[str]]:
        parts = ["<websearch_context>"]
        parts.append(
            "Web search was performed but no suitable results were found. "
            "Tell the user you searched the web but couldn't find any good results. "
            "If you can still answer from prior context or general knowledge, say so clearly, "
            "and do not fabricate citations or URLs."
        )
        if queries:
            try:
                qshow = ", ".join(queries[:3])
                parts.append(f"Queries attempted: {qshow}")
            except Exception:
                pass
        if engines:
            try:
                eshow = ", ".join(engines)
                parts.append(f"Engines selected: {eshow}")
            except Exception:
                pass
        parts.append(f"(reason: {reason})")
        parts.append("</websearch_context>")
        return f"{user_prompt}\n\n" + "\n".join(parts), []

    context_excerpt = render_recent_context(messages, char_limit=1200)

    # 1) queries
    try:
        t0 = time.perf_counter()
        queries = await generate_queries(user_prompt, model=model, context_excerpt=context_excerpt)
        print(f"[web] queries: {queries} (took {time.perf_counter() - t0:.3f}s)")
    except Exception:
        print("[web] ERROR in generate_queries:\n" + traceback.format_exc())
        print(f"[web] enrich_prompt total: {time.perf_counter() - start_all:.3f}s")
        return _no_results_enriched("query_generation_failed")

    # 2) search + fetch
    try:
        print("[web] opening httpx client")
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
            http2=True,
            limits=HTTP_LIMITS,
            timeout=HTTP_TIMEOUT,
        ) as client:
            t1 = time.perf_counter()
            print("[web] calling searx_search_many() …")
            all_urls = await searx_search_many(
                client,
                queries,
                per_query_max=6,
                overall_max=MAX_PAGES_FETCH * 2,
                searx_url=searx_url,
                engines=engines,
            )
            print(f"[web] searx_search_many() -> {len(all_urls)} urls (took {time.perf_counter() - t1:.3f}s)")

            if not all_urls:
                print("[web] no URLs from SearX — emitting no-results context")
                print(f"[web] enrich_prompt total: {time.perf_counter() - start_all:.3f}s")
                return _no_results_enriched("no_urls", queries)

            t2 = time.perf_counter()
            print("[web] calling gather_pages() …")
            pages = await gather_pages(client, all_urls, max_pages=MAX_PAGES_FETCH)
            print(f"[web] gather_pages() -> {len(pages)} pages (cap={MAX_PAGES_FETCH}, took {time.perf_counter() - t2:.3f}s)")
    except Exception:
        print("[web] ERROR during search/fetch:\n" + traceback.format_exc())
        print(f"[web] enrich_prompt total: {time.perf_counter() - start_all:.3f}s")
        return _no_results_enriched("search_fetch_error")

    # 3) docs
    try:
        t3 = time.perf_counter()
        docs = [(u, pages[u]) for u in all_urls if u in pages]
        print(f"[web] docs for rerank: {len(docs)} (built in {time.perf_counter() - t3:.3f}s)")
        if not docs:
            print("[web] no docs after fetch/filter — emitting no-results context")
            print(f"[web] enrich_prompt total: {time.perf_counter() - start_all:.3f}s")
            return _no_results_enriched("no_docs_after_fetch", queries)
    except Exception:
        print("[web] ERROR building docs list:\n" + traceback.format_exc())
        print(f"[web] enrich_prompt total: {time.perf_counter() - start_all:.3f}s")
        return _no_results_enriched("docs_build_failed", queries)

    # 4) rerank
    try:
        t4 = time.perf_counter()
        print("[web] calling rerank() …")
        ranked = await rerank(user_prompt, docs, model=model, context_excerpt=context_excerpt)
        print(f"[web] rerank() -> {len(ranked)} scored docs (took {time.perf_counter() - t4:.3f}s)")
        try:
            scores_only = [sc for (_u, _t, sc) in ranked]
            if scores_only:
                scores_sorted = sorted(scores_only)
                n = len(scores_sorted)
                p50 = scores_sorted[n // 2]
                p75 = scores_sorted[int(n * 0.75) - 1 if n > 3 else n - 1]
                print(f"[web] score stats → min={scores_sorted[0]:.1f}, p50={p50:.1f}, p75={p75:.1f}, max={scores_sorted[-1]:.1f}")
                print(f"[web] top scores → {[round(s,1) for s in scores_only[:min(6,len(scores_only))]]}")
        except Exception:
            pass
    except Exception:
        print("[web] ERROR in rerank:\n" + traceback.format_exc())
        print(f"[web] enrich_prompt total: {time.perf_counter() - start_all:.3f}s")
        return _no_results_enriched("rerank_failed", queries)

    # 5) build prompt
    try:
        t5 = time.perf_counter()
        enriched = build_enriched_prompt(user_prompt, ranked, top_k=6)
        print(f"[web] build_enriched_prompt() done (took {time.perf_counter() - t5:.3f}s)")
        print(f"[web] enrich_prompt total: {time.perf_counter() - start_all:.3f}s")
        return enriched
    except Exception:
        print("[web] ERROR in build_enriched_prompt:\n" + traceback.format_exc())
        print(f"[web] enrich_prompt total: {time.perf_counter() - start_all:.3f}s")
        return _no_results_enriched("build_enriched_failed", queries)