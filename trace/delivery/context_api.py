"""
Personal Context API — the hackathon delivery surface.

Three endpoints, one demo:

    POST /v2/ingest          → upload signals, build a graph, index context
    GET  /v2/context         → retrieve the top-K personal context items for a query
    POST /v2/agent-query     → ask Gemini with AND without the user's context

The frontend at GET /v2/ is a single-page before/after demo showing the same
Gemini prompt answered with and without Trace context — the money shot.

Design notes:

  *Profile isolation*: each /v2/ingest call generates a profile_id. All
  subsequent calls scope to that profile_id. This lets multiple users hit the
  same Cloud Run instance during the demo without seeing each other's data.

  *In-memory only*: ContextStores live in a dict in the FastAPI app. Cloud Run
  with ``--min-instances 1`` keeps the instance warm for the demo. Profile
  state is intentionally non-durable; the demo flow is upload → query → done.

  *Path namespace*: ``/v2/*`` to avoid colliding with the legacy newsletter
  endpoints in ``trace/delivery/api.py``. Both can coexist in the same app.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from trace.agent.gemini_agent import GeminiContextAgent
from trace.config import get_settings
from trace.context import ContextIndexer, ContextStore
from trace.graph.builder import CuriosityGraphBuilder
from trace.graph.embedder import TopicEmbedder
from trace.graph.gemini_extractor import GeminiTopicExtractor
from trace.models import RawSignal
from trace.signals.chatgpt_export import ChatGPTExportCollector
from trace.signals.google_takeout import GoogleTakeoutCollector
from trace.signals.youtube_takeout import YouTubeWatchHistoryCollector

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/v2", tags=["context"])

# Per-profile in-memory stores. Process-local; reset on container restart.
_CONTEXT_STORES: dict[str, ContextStore] = {}
_PROFILE_META: dict[str, dict[str, Any]] = {}

# Shared embedder — model load is ~3s, so we keep a singleton.
_EMBEDDER = TopicEmbedder()


# ── Request / response schemas ────────────────────────────────────────────────


class IngestResponse(BaseModel):
    profile_id: str
    topic_count: int
    context_item_count: int
    signal_count: int
    topics: list[dict[str, Any]] = Field(default_factory=list)


class ContextResult(BaseModel):
    text: str
    score: float
    source: str
    topic: str | None = None


class ContextResponse(BaseModel):
    query: str
    profile_id: str
    total_items: int
    results: list[ContextResult]


class AgentQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    profile_id: str
    context_k: int = Field(default=10, ge=1, le=50)


class AgentQueryResponse(BaseModel):
    with_context: str
    without_context: str
    context_items_used: list[dict[str, Any]]
    context_item_count: int
    model: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _require_gemini_key() -> str:
    key = get_settings().gemini_api_key
    if not key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured. Set it in your environment.",
        )
    return key


def _get_store(profile_id: str) -> ContextStore:
    store = _CONTEXT_STORES.get(profile_id)
    if store is None:
        raise HTTPException(
            status_code=404,
            detail=f"No context found for profile '{profile_id}'. POST /v2/ingest first.",
        )
    return store


async def _collect_signals_from_upload(file: UploadFile) -> list[RawSignal]:
    """Save the upload to a tmp file and route to the right collector.

    Supported formats:
      - BrowserHistory.json (Chrome via Google Takeout)
      - conversations.json (ChatGPT export)
      - watch-history.json (YouTube via Google Takeout)
      - .zip containing any of the above
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")

    filename = (file.filename or "").lower()
    tmp_dir = get_settings().upload_dir / "v2"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4()}_{filename or 'upload.json'}"

    # Detect zip uploads and extract the first usable JSON file.
    if filename.endswith(".zip") or raw[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                target = _pick_json_from_zip(zf)
                if target is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Zip did not contain a recognisable Takeout JSON.",
                    )
                tmp_path.write_bytes(zf.read(target))
                filename = target.lower()
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail=f"Bad zip: {exc}") from exc
    else:
        tmp_path.write_bytes(raw)

    # Sniff the file: every Takeout JSON is an array. Try collectors in order
    # of specificity — YouTube watch-history first (has unique 'header' field),
    # then ChatGPT (top-level objects have 'mapping'), then Browser History.
    try:
        text_head = tmp_path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read upload: {exc}") from exc

    if "watch-history" in filename or '"YouTube"' in text_head:
        collector: Any = YouTubeWatchHistoryCollector(history_path=tmp_path)
    elif "conversations" in filename or '"mapping"' in text_head:
        collector = ChatGPTExportCollector(export_path=tmp_path)
    else:
        # Default to BrowserHistory — Google Takeout's most common export.
        collector = GoogleTakeoutCollector(history_path=tmp_path)

    try:
        return await collector.collect()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse upload as a known Takeout format: {exc}",
        ) from exc


def _pick_json_from_zip(zf: zipfile.ZipFile) -> str | None:
    """Return the first member whose name suggests a Takeout JSON we know."""
    candidates = [
        "BrowserHistory.json",
        "watch-history.json",
        "conversations.json",
    ]
    names = zf.namelist()
    for cand in candidates:
        for n in names:
            if n.endswith(cand):
                return n
    # Fallback: any .json over 1 KB.
    for n in names:
        if n.lower().endswith(".json"):
            try:
                if zf.getinfo(n).file_size > 1024:
                    return n
            except KeyError:
                continue
    return None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """Build a personal context graph from a Takeout export upload."""
    api_key = _require_gemini_key()
    settings = get_settings()

    signals = await _collect_signals_from_upload(file)
    if not signals:
        raise HTTPException(
            status_code=400,
            detail="Upload parsed successfully but produced 0 signals.",
        )
    _log.info("v2/ingest: parsed %d signals from %s", len(signals), file.filename)

    extractor = GeminiTopicExtractor(api_key=api_key, model=settings.gemini_model)
    builder = CuriosityGraphBuilder(
        extractor=extractor,
        half_life_days=settings.recency_half_life_days,
        debt_threshold_occurrences=settings.debt_occurrence_threshold,
    )
    graph = await builder.build(signals)

    store = ContextStore()
    indexer = ContextIndexer(embedder=_EMBEDDER)
    item_count = indexer.index(store, graph, signals)

    profile_id = str(uuid.uuid4())
    _CONTEXT_STORES[profile_id] = store
    _PROFILE_META[profile_id] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signal_count": len(signals),
        "filename": file.filename,
    }

    return IngestResponse(
        profile_id=profile_id,
        topic_count=len(graph.topics),
        context_item_count=item_count,
        signal_count=len(signals),
        topics=[
            {
                "name": t.name,
                "frequency": t.frequency,
                "recency_score": round(t.recency_score, 3),
                "composite_score": round(t.composite_score(), 3),
                "curiosity_type": t.curiosity_type.value,
            }
            for t in graph.top_n(min(20, max(1, len(graph.topics))))
        ],
    )


@router.get("/context", response_model=ContextResponse)
async def get_context(
    query: str,
    profile_id: str,
    k: int = 10,
) -> ContextResponse:
    """Retrieve the top-K personal context items most relevant to ``query``."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be in [1, 50].")

    store = _get_store(profile_id)
    emb_map = _EMBEDDER.encode([query])
    if not emb_map:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not available. Is sentence-transformers installed?",
        )
    q_vec = emb_map[query]
    hits = store.query(q_vec, top_k=k)

    return ContextResponse(
        query=query,
        profile_id=profile_id,
        total_items=store.size(),
        results=[
            ContextResult(
                text=item.text,
                score=round(score, 4),
                source=item.source,
                topic=item.topic_name,
            )
            for item, score in hits
        ],
    )


@router.post("/agent-query", response_model=AgentQueryResponse)
async def agent_query(body: AgentQueryRequest) -> AgentQueryResponse:
    """Ask Gemini the same question twice — bare and with Trace context."""
    api_key = _require_gemini_key()
    store = _get_store(body.profile_id)

    emb_map = _EMBEDDER.encode([body.question])
    context_items: list[tuple[Any, float]] = []
    if emb_map:
        q_vec = emb_map[body.question]
        context_items = store.query(q_vec, top_k=body.context_k)

    agent = GeminiContextAgent(
        api_key=api_key,
        model=get_settings().gemini_model,
    )
    result = await agent.query(body.question, context_items)
    return AgentQueryResponse(**result)


@router.get("/profiles/{profile_id}")
async def profile_info(profile_id: str) -> dict[str, Any]:
    """Return metadata about an indexed profile (no raw items)."""
    if profile_id not in _CONTEXT_STORES:
        raise HTTPException(status_code=404, detail="Profile not found.")
    meta = dict(_PROFILE_META.get(profile_id, {}))
    meta["profile_id"] = profile_id
    meta["context_item_count"] = _CONTEXT_STORES[profile_id].size()
    return meta


@router.get("/health")
async def health() -> dict[str, Any]:
    """Quick health probe for /v2 specifically."""
    return {
        "status": "ok",
        "gemini_configured": bool(get_settings().gemini_api_key),
        "model": get_settings().gemini_model,
        "active_profiles": len(_CONTEXT_STORES),
    }


# ── Frontend ──────────────────────────────────────────────────────────────────

_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trace — Personal Context for AI Agents</title>
<style>
:root {
  --bg: #0a0a14;
  --panel: #12122a;
  --panel-2: #1a1a34;
  --border: rgba(148, 163, 255, 0.14);
  --accent: #7c3aed;
  --accent-light: #a78bfa;
  --green: #10b981;
  --amber: #f59e0b;
  --t1: #f0f0fa;
  --t2: #a0a0c0;
  --t3: #60607a;
  --mono: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--t1);
  font-family: var(--font);
  line-height: 1.55;
  min-height: 100vh;
  padding: 2rem 1.5rem 4rem;
}
.container { max-width: 1100px; margin: 0 auto; }
header { margin-bottom: 2.5rem; }
h1 {
  font-size: 2.1rem;
  letter-spacing: -0.02em;
  font-weight: 800;
  background: linear-gradient(120deg, var(--accent-light), #22d3ee);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.subtitle { color: var(--t2); margin-top: 0.4rem; font-size: 1.02rem; }
.tag {
  display: inline-block;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-light);
  background: rgba(124, 58, 237, 0.13);
  border: 1px solid rgba(124, 58, 237, 0.32);
  padding: 0.3em 0.8em;
  border-radius: 999px;
  margin-bottom: 1rem;
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.5rem;
  margin-bottom: 1.25rem;
}
.card h2 {
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.card h2 .step {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--accent);
  color: white;
  font-size: 0.72rem;
  font-weight: 800;
}
.card p { color: var(--t2); font-size: 0.93rem; margin-bottom: 0.9rem; }
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: 10px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.18s;
  color: var(--t2);
}
.upload-zone:hover, .upload-zone.dragover {
  border-color: var(--accent-light);
  background: rgba(124, 58, 237, 0.06);
  color: var(--t1);
}
.upload-zone input[type="file"] { display: none; }
.upload-zone strong { color: var(--accent-light); }
.btn {
  background: var(--accent);
  color: white;
  border: none;
  padding: 0.65rem 1.1rem;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.92rem;
  cursor: pointer;
  transition: background 0.18s;
}
.btn:hover { background: #6d28d9; }
.btn:disabled { background: #3a3a55; cursor: not-allowed; opacity: 0.6; }
.input {
  width: 100%;
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--t1);
  padding: 0.7rem 0.85rem;
  border-radius: 8px;
  font-size: 0.95rem;
  font-family: var(--font);
}
.input:focus { outline: 1px solid var(--accent-light); border-color: var(--accent-light); }
.row { display: flex; gap: 0.6rem; align-items: center; }
.row .input { flex: 1; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.8rem; }
.chip {
  font-size: 0.78rem;
  background: var(--panel-2);
  border: 1px solid var(--border);
  padding: 0.32em 0.7em;
  border-radius: 999px;
  color: var(--t1);
  font-family: var(--mono);
}
.split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-top: 1rem;
}
@media (max-width: 720px) {
  .split { grid-template-columns: 1fr; }
}
.answer {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem;
  white-space: pre-wrap;
  font-size: 0.92rem;
  min-height: 160px;
  line-height: 1.6;
}
.answer.bare { border-color: rgba(245, 158, 11, 0.3); }
.answer.context { border-color: rgba(16, 185, 129, 0.45); }
.answer-label {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 0.55rem;
  color: var(--t3);
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.answer.bare .answer-label { color: var(--amber); }
.answer.context .answer-label { color: var(--green); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.context-used {
  margin-top: 1.1rem;
  padding-top: 0.9rem;
  border-top: 1px solid var(--border);
}
.context-used h3 {
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 0.6rem;
}
.context-item {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 0.6rem 0.8rem;
  margin-bottom: 0.4rem;
  font-size: 0.85rem;
  color: var(--t2);
  display: flex;
  gap: 0.7rem;
  align-items: flex-start;
}
.context-item .badge {
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--accent-light);
  flex-shrink: 0;
  white-space: nowrap;
}
.spinner {
  display: inline-block;
  width: 13px;
  height: 13px;
  border: 2px solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.muted { color: var(--t3); font-size: 0.85rem; font-style: italic; }
.error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.35);
  color: #fca5a5;
  padding: 0.7rem 0.9rem;
  border-radius: 8px;
  font-size: 0.88rem;
  margin-top: 0.6rem;
}
footer {
  margin-top: 3rem;
  text-align: center;
  color: var(--t3);
  font-size: 0.8rem;
}
footer code { font-family: var(--mono); color: var(--t2); }
.hidden { display: none !important; }
</style>
</head>
<body>
<div class="container">
  <header>
    <span class="tag">Personal Context · Private by architecture</span>
    <h1>Trace</h1>
    <p class="subtitle">The same question, answered by Gemini — with and without your personal context.</p>
  </header>

  <div class="card">
    <h2><span class="step">1</span> Upload a behavioural signal</h2>
    <p>Drop in a <code>BrowserHistory.json</code> from Google Takeout, a ChatGPT <code>conversations.json</code>, or a YouTube <code>watch-history.json</code>. Files are processed in memory on this server — nothing is stored beyond the running instance.</p>
    <label class="upload-zone" id="dropZone">
      <input type="file" id="fileInput" accept=".json,.zip">
      <strong>Click to upload</strong> or drag a file here
      <div style="font-size: 0.78rem; margin-top: 0.4rem; color: var(--t3);">.json or .zip · max ~50 MB</div>
    </label>
    <div id="ingestStatus" class="muted" style="margin-top: 0.9rem;"></div>
    <div id="ingestError" class="error hidden"></div>
  </div>

  <div class="card hidden" id="topicsCard">
    <h2><span class="step">2</span> Your curiosity graph</h2>
    <p>Topics Gemini extracted from your signals. Each is a unit of context any agent can now retrieve.</p>
    <div id="topics" class="chips"></div>
  </div>

  <div class="card hidden" id="askCard">
    <h2><span class="step">3</span> Ask Gemini anything</h2>
    <p>Same prompt → two Gemini calls in parallel. Left has no context. Right has your personal context.</p>
    <div class="row">
      <input id="question" class="input" type="text" placeholder="e.g. What should I read this week?" autocomplete="off">
      <button id="askBtn" class="btn">Ask</button>
    </div>
    <div class="split">
      <div class="answer bare" id="bareAnswer">
        <div class="answer-label"><span class="dot"></span>Gemini · no context</div>
        <div class="muted">Awaiting your question…</div>
      </div>
      <div class="answer context" id="ctxAnswer">
        <div class="answer-label"><span class="dot"></span>Gemini · with Trace context</div>
        <div class="muted">Awaiting your question…</div>
      </div>
    </div>
    <div class="context-used hidden" id="contextUsed">
      <h3>Context items used</h3>
      <div id="contextList"></div>
    </div>
  </div>

  <footer>
    <code>POST /v2/ingest</code> · <code>GET /v2/context</code> · <code>POST /v2/agent-query</code>
    <br><span style="opacity: 0.6;">Trace · Personal Context API for AI Agents</span>
  </footer>
</div>

<script>
const state = { profileId: null };
const $ = (id) => document.getElementById(id);

const dropZone = $('dropZone');
const fileInput = $('fileInput');
['dragover', 'dragenter'].forEach((e) => dropZone.addEventListener(e, (ev) => {
  ev.preventDefault();
  dropZone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach((e) => dropZone.addEventListener(e, () => dropZone.classList.remove('dragover')));
dropZone.addEventListener('drop', (ev) => {
  ev.preventDefault();
  if (ev.dataTransfer.files.length) {
    fileInput.files = ev.dataTransfer.files;
    upload(ev.dataTransfer.files[0]);
  }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) upload(fileInput.files[0]);
});

async function upload(file) {
  $('ingestError').classList.add('hidden');
  $('ingestStatus').innerHTML = `<span class="spinner"></span> Processing <strong>${escapeHtml(file.name)}</strong> with Gemini…`;
  const fd = new FormData();
  fd.append('file', file);
  try {
    const resp = await fetch('/v2/ingest', { method: 'POST', body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    state.profileId = data.profile_id;
    $('ingestStatus').innerHTML =
      `Indexed <strong>${data.context_item_count}</strong> context items from <strong>${data.signal_count}</strong> signals — ` +
      `found <strong>${data.topic_count}</strong> topics.`;
    renderTopics(data.topics);
    $('topicsCard').classList.remove('hidden');
    $('askCard').classList.remove('hidden');
    $('question').focus();
  } catch (e) {
    $('ingestStatus').textContent = '';
    $('ingestError').textContent = `Ingest failed: ${e.message}`;
    $('ingestError').classList.remove('hidden');
  }
}

function renderTopics(topics) {
  const el = $('topics');
  el.innerHTML = '';
  topics.forEach((t) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = `${t.name} · ${t.frequency}`;
    chip.title = `Composite score: ${t.composite_score} · ${t.curiosity_type}`;
    el.appendChild(chip);
  });
}

$('askBtn').addEventListener('click', ask);
$('question').addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });

async function ask() {
  const q = $('question').value.trim();
  if (!q || !state.profileId) return;
  $('askBtn').disabled = true;
  $('bareAnswer').innerHTML = `<div class="answer-label"><span class="dot"></span>Gemini · no context</div><div class="muted"><span class="spinner"></span> Thinking…</div>`;
  $('ctxAnswer').innerHTML = `<div class="answer-label"><span class="dot"></span>Gemini · with Trace context</div><div class="muted"><span class="spinner"></span> Thinking…</div>`;
  $('contextUsed').classList.add('hidden');
  try {
    const resp = await fetch('/v2/agent-query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, profile_id: state.profileId, context_k: 10 }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    $('bareAnswer').innerHTML = `<div class="answer-label"><span class="dot"></span>Gemini · no context</div>` + escapeHtml(data.without_context);
    $('ctxAnswer').innerHTML = `<div class="answer-label"><span class="dot"></span>Gemini · with Trace context (${data.context_item_count} items)</div>` + escapeHtml(data.with_context);
    renderContextUsed(data.context_items_used || []);
  } catch (e) {
    $('bareAnswer').innerHTML = `<div class="answer-label"><span class="dot"></span>Gemini · no context</div><div class="error">${escapeHtml(e.message)}</div>`;
    $('ctxAnswer').innerHTML = `<div class="answer-label"><span class="dot"></span>Gemini · with Trace context</div><div class="error">${escapeHtml(e.message)}</div>`;
  } finally {
    $('askBtn').disabled = false;
  }
}

function renderContextUsed(items) {
  if (!items.length) {
    $('contextUsed').classList.add('hidden');
    return;
  }
  const list = $('contextList');
  list.innerHTML = '';
  items.forEach((it) => {
    const div = document.createElement('div');
    div.className = 'context-item';
    const tag = it.topic ? `${it.source}·${it.topic}` : it.source;
    div.innerHTML = `<span class="badge">${escapeHtml(tag)} · ${it.score.toFixed(3)}</span><span>${escapeHtml(it.text)}</span>`;
    list.appendChild(div);
  });
  $('contextUsed').classList.remove('hidden');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def demo_frontend() -> str:
    return _DEMO_HTML
