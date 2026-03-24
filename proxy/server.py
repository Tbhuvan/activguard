"""
ActivGuard Proxy — FastAPI OpenAI-Compatible Vulnerability-Scanning Proxy.

Architecture overview:
    Client (Cursor / VS Code Copilot / Open WebUI)
        |  POST /v1/chat/completions  (OpenAI schema)
        v
    ActivGuard Proxy  (this module)
        |  POST /api/chat  (Ollama schema, streaming NDJSON)
        v
    Ollama backend  (local, any model)

During streaming, the proxy:
    1. Forwards each Ollama NDJSON chunk to the client as an OpenAI SSE chunk.
    2. Accumulates the assistant content in a shared buffer.
    3. Every PROBE_INTERVAL tokens, submits the buffer to the CodeBERT probe
       in a background asyncio task so token delivery is not blocked.
    4. Probe results are placed on an asyncio.Queue; the main streaming loop
       drains the queue between SSE flushes and injects violation events.
    5. After the stream completes, optionally runs Bandit on the full output.
    6. Sets the X-ActivGuard-Result response header to VIOLATION or VERIFIED.

Graceful degradation:
    If the probe fails to load (weights missing, torch not installed, Ollama
    down) the proxy transparently passes all requests through without scanning.
    No 5xx error is surfaced to the client; a warning is logged instead.

Security:
    CORS is configured permissively (all origins) to allow browser-based tools.
    Input validation is performed by Pydantic on all request bodies.
    Ollama backend URL is never derived from user input.

Reference: OpenAI Chat Completions API specification,
    https://platform.openai.com/docs/api-reference/chat
Reference: Ollama /api/chat specification,
    https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from proxy.config import (
    AUTH_CONTEXT_PATH,
    BANDIT_SECOND_PASS,
    DETECTOR_TAG,
    HF_PROBE_ENABLED,
    HF_PROBE_MODEL,
    HF_PROBE_WEIGHTS,
    OLLAMA_BASE,
    PROBE_INTERVAL,
    PROBE_WEIGHTS,
    STOP_ON_VIOLATION,
    THRESHOLD,
)
from proxy.violation_event import ViolationEvent, infer_cwe_hint

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation when unavailable
# ---------------------------------------------------------------------------

try:
    from probe.universal_streaming_probe import UniversalStreamingProbe
    _PROBE_MODULE_AVAILABLE = True
except ImportError:
    _PROBE_MODULE_AVAILABLE = False

try:
    from run_baselines import run_bandit
    _BANDIT_AVAILABLE = True
except ImportError:
    _BANDIT_AVAILABLE = False

try:
    from rag.semantic_rag import SecurityRAG
    _RAG_MODULE_AVAILABLE = True
except ImportError:
    _RAG_MODULE_AVAILABLE = False

try:
    from probe.hf_hidden_probe import HFHiddenProbe
    _HF_PROBE_MODULE_AVAILABLE = True
except ImportError:
    _HF_PROBE_MODULE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level probe + RAG state (singletons, loaded once at startup)
# ---------------------------------------------------------------------------

_probe_loaded: bool = False
_probe_weights_path: str = PROBE_WEIGHTS
_probe_threshold: float = THRESHOLD
_probe_interval: int = PROBE_INTERVAL
_stop_on_violation: bool = STOP_ON_VIOLATION


def _load_probe_eagerly() -> bool:
    """Attempt to pre-load CodeBERT at startup and return True on success.

    Imports are deferred until this function so that the FastAPI app can be
    imported without triggering the heavy torch/transformers download.

    Returns:
        bool: True if the probe is ready; False if it degraded gracefully.
    """
    if not _PROBE_MODULE_AVAILABLE:
        logger.warning(
            "probe.universal_streaming_probe not importable — "
            "L1-activation scanning disabled (torch/transformers missing?)."
        )
        return False

    weights = Path(_probe_weights_path)
    if not weights.exists():
        logger.warning(
            "Probe weights not found at %s — L1-activation scanning disabled.",
            weights,
        )
        return False

    try:
        # Instantiate with a placeholder model name; the probe only uses it
        # for logging — CodeBERT itself is the actual encoder.
        _ProbeClass = UniversalStreamingProbe  # noqa: N806 — alias for clarity
        probe = _ProbeClass(
            gen_model="proxy-passthrough",
            probe_weights_path=str(weights),
            threshold=_probe_threshold,
            probe_interval=_probe_interval,
        )
        # Stash on the module so _run_probe_async can reuse it.
        globals()["_probe_instance"] = probe
        logger.info("CodeBERT probe loaded successfully (%s).", DETECTOR_TAG)
        return True
    except Exception as exc:  # broad — intentional startup degradation
        logger.warning("Probe load failed (%s) — L1 scanning disabled.", exc)
        return False


_probe_instance: UniversalStreamingProbe | None = None  # type: ignore[type-arg]

# ---------------------------------------------------------------------------
# L2 RAG state
# ---------------------------------------------------------------------------

_rag_loaded: bool = False
_rag_instance: Any = None  # SecurityRAG | None
_RAG_CONFIDENCE_THRESHOLD: float = 0.1   # ChromaDB match confidence to fire L2

# ---------------------------------------------------------------------------
# HF residual stream probe state (true L1 — Qwen2.5-Coder's own hidden states)
# ---------------------------------------------------------------------------

_hf_probe_loaded: bool = False
_hf_probe_instance: Any = None   # HFHiddenProbe | None


def _load_hf_probe_eagerly() -> bool:
    """Attempt to load the HF residual stream probe at startup.

    Derives the weights path from the model slug if HF_PROBE_WEIGHTS is empty.
    Degrades gracefully if weights are missing or torch/transformers are absent.

    Returns:
        bool: True if the HF probe is ready; False on graceful degradation.
    """
    global _hf_probe_instance
    if not HF_PROBE_ENABLED:
        logger.info("HF probe disabled (ACTIVGUARD_HF_PROBE_ENABLED=0).")
        return False
    if not _HF_PROBE_MODULE_AVAILABLE:
        logger.warning("probe.hf_hidden_probe not importable — HF L1 disabled.")
        return False

    slug = HF_PROBE_MODEL.lower().split("/")[-1].replace(".", "-").replace("_", "-")
    weights_path = HF_PROBE_WEIGHTS or f".activguard/hf_probe_{slug}.pkl"

    if not Path(weights_path).exists():
        logger.warning(
            "HF probe weights not found at %s — run scripts/train_hf_probe.py. "
            "HF L1 disabled.",
            weights_path,
        )
        return False

    try:
        probe = HFHiddenProbe(
            model_name=HF_PROBE_MODEL,
            weights_path=weights_path,
            threshold=THRESHOLD,
        )
        _hf_probe_instance = probe
        logger.info(
            "HF residual stream probe loaded: model=%s layer=%d dim=%d",
            HF_PROBE_MODEL, probe.layer, probe._hidden_dim,
        )
        return True
    except Exception as exc:
        logger.warning("HF probe load failed (%s) — HF L1 disabled.", exc)
        return False


def _load_rag_eagerly() -> bool:
    """Attempt to pre-load the SecurityRAG ChromaDB collection on startup.

    If ``AUTH_CONTEXT_PATH`` is set, also extracts the project's auth model
    via ``AuthExtractor`` and indexes it into the ``auth_model`` collection.
    This enables project-specific IDOR detection without retraining the probe.

    Returns:
        bool: True if the RAG layer is ready; False if it degraded gracefully.
    """
    global _rag_instance
    if not _RAG_MODULE_AVAILABLE:
        logger.warning("rag.semantic_rag not importable — L2-RAG scanning disabled.")
        return False
    try:
        rag = SecurityRAG()
        _rag_instance = rag
        logger.info("SecurityRAG loaded (L2-RAG ready).")
    except Exception as exc:
        logger.warning("RAG load failed (%s) — L2-RAG scanning disabled.", exc)
        return False

    if AUTH_CONTEXT_PATH:
        try:
            logger.info("Indexing auth model from %s ...", AUTH_CONTEXT_PATH)
            rag.add_project_context(AUTH_CONTEXT_PATH)
        except FileNotFoundError:
            logger.warning(
                "AUTH_CONTEXT_PATH=%s does not exist — skipping auth indexing.",
                AUTH_CONTEXT_PATH,
            )
        except Exception as exc:
            logger.warning("Auth model indexing failed: %s", exc)
    return True


# ---------------------------------------------------------------------------
# Lifespan: load probe + RAG once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan handler — loads the CodeBERT probe and SecurityRAG on startup."""
    global _probe_loaded, _rag_loaded, _hf_probe_loaded
    logger.info("ActivGuard proxy starting — loading probe + RAG...")
    _probe_loaded    = _load_probe_eagerly()
    _hf_probe_loaded = _load_hf_probe_eagerly()
    _rag_loaded      = _load_rag_eagerly()
    logger.info(
        "Startup complete: L1-codebert=%s  L1-hf=%s  L2-RAG=%s  L3-bandit=%s",
        _probe_loaded, _hf_probe_loaded, _rag_loaded, _BANDIT_AVAILABLE,
    )
    yield
    logger.info("ActivGuard proxy shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="ActivGuard Proxy",
    description=(
        "OpenAI-compatible proxy that scans LLM-generated code for "
        "vulnerabilities using a CodeBERT activation probe (L1) and "
        "Bandit static analysis (L3)."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in an OpenAI chat completions request."""

    role: str = Field(..., description="One of: system, user, assistant, tool.")
    content: str | list[dict[str, Any]] = Field(
        ..., description="Message content (string or multipart list)."
    )


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completions request body."""

    model: str = Field(..., description="Ollama model name, e.g. 'dolphin3:8b'.")
    messages: list[ChatMessage] = Field(..., description="Conversation history.")
    stream: bool = Field(False, description="Enable SSE streaming.")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1)
    stop: str | list[str] | None = Field(None)
    # Extra fields are silently ignored — forward-compatibility.
    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Probe helpers (async wrappers around synchronous CodeBERT calls)
# ---------------------------------------------------------------------------


async def _run_probe_async(text: str) -> float:
    """Run the CodeBERT probe in a thread pool to avoid blocking the event loop.

    Args:
        text: Accumulated partial output to score.

    Returns:
        float: P(vulnerable) in [0, 1].  Returns 0.0 on any failure.
    """
    global _probe_instance
    if _probe_instance is None or not text.strip():
        return 0.0
    # Don't probe preamble — wait for actual code signals
    _CODE_SIGNALS = ("def ", "class ", "import ", "cursor.", "execute(",
                     "subprocess.", "pickle.", "requests.get", "requests.post",
                     "open(", "os.system(", "eval(", "yaml.load(", "```python",
                     ".format(", "f\"SELECT", "f'SELECT")
    if not any(sig in text for sig in _CODE_SIGNALS):
        return 0.0
    loop = asyncio.get_running_loop()
    try:
        score: float = await loop.run_in_executor(
            None, _probe_instance._probe_text, text  # noqa: SLF001
        )
        return score
    except Exception as exc:
        logger.debug("Probe call failed (degraded): %s", exc)
        return 0.0


async def _run_hf_probe_async(text: str) -> float:
    """Run the HF residual stream probe in a thread pool.

    Uses Qwen2.5-Coder's own hidden states to score the accumulated output,
    providing model-specific vulnerability signatures (vs. CodeBERT's universal
    encoding).  Slower than CodeBERT (~200ms vs ~10ms) so only fires when code
    signals are present.

    Args:
        text: Accumulated partial LLM output to score.

    Returns:
        float: P(vulnerable) ∈ [0, 1].  Returns 0.0 on failure.
    """
    if not _hf_probe_loaded or _hf_probe_instance is None or not text.strip():
        return 0.0
    _CODE_SIGNALS = ("def ", "class ", "import ", "cursor.", "execute(",
                     "subprocess.", "pickle.", "requests.get", "requests.post",
                     "open(", "os.system(", "eval(", "yaml.load(", "```python",
                     ".format(", "f\"SELECT", "f'SELECT")
    if not any(sig in text for sig in _CODE_SIGNALS):
        return 0.0
    loop = asyncio.get_running_loop()
    try:
        score: float = await loop.run_in_executor(
            None, _hf_probe_instance.score, text
        )
        return score
    except Exception as exc:
        logger.debug("HF probe call failed (degraded): %s", exc)
        return 0.0


async def _run_bandit_async(code: str) -> dict[str, Any]:
    """Run Bandit in a thread pool to avoid blocking the event loop.

    Args:
        code: Complete Python source code string.

    Returns:
        dict: Bandit result dict {flagged, n_findings, findings, error}.
            Returns a safe default on failure.
    """
    if not _BANDIT_AVAILABLE:
        return {"flagged": False, "n_findings": 0, "findings": [], "error": "bandit not available"}
    loop = asyncio.get_running_loop()
    try:
        result: dict[str, Any] = await loop.run_in_executor(None, run_bandit, code)
        return result
    except Exception as exc:
        logger.debug("Bandit call failed: %s", exc)
        return {"flagged": False, "n_findings": 0, "findings": [], "error": str(exc)}


async def _run_rag_async(text: str) -> tuple[float, str]:
    """Run the L2 SecurityRAG query in a thread pool.

    Called only when L1-activation fires mid-stream (L1-gated per architecture).
    Queries the ChromaDB anti-pattern collection for the top matching pattern
    and returns its confidence score.

    Args:
        text: Partial LLM output at the moment L1 flagged.

    Returns:
        Tuple of (max_confidence, matched_pattern_id).
        Returns (0.0, "") on failure or if RAG layer is unavailable.
    """
    if not _rag_loaded or _rag_instance is None or not text.strip():
        return 0.0, ""

    def _query() -> tuple[float, str]:
        try:
            result = _rag_instance.query(text, n_results=1)
            if not result:
                return 0.0, ""
            # SecurityRAG.query() returns a flat dict with 'safe', 'confidence', 'patterns_matched'
            conf = float(result.get("confidence", 0.0))
            safe = bool(result.get("safe", True))
            # Fire if not safe OR confidence above threshold
            effective_conf = conf if not safe else 0.0
            patterns = result.get("patterns_matched", [])
            pid = patterns[0] if patterns else ""
            return effective_conf, str(pid)
        except Exception as exc:
            logger.debug("RAG query failed: %s", exc)
            return 0.0, ""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _query)


# ---------------------------------------------------------------------------
# Ollama client helpers
# ---------------------------------------------------------------------------


def _messages_to_ollama(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert OpenAI message list to Ollama /api/chat message list.

    Args:
        messages: Parsed OpenAI ChatMessage objects.

    Returns:
        list[dict]: Ollama-format messages with role and content keys.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else (
            " ".join(
                part.get("text", "") for part in msg.content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        )
        result.append({"role": msg.role, "content": content})
    return result


def _build_ollama_options(req: ChatCompletionRequest) -> dict[str, Any]:
    """Build the Ollama options dict from OpenAI request parameters.

    Args:
        req: Parsed ChatCompletionRequest.

    Returns:
        dict: Ollama options dict (empty keys omitted).
    """
    opts: dict[str, Any] = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.max_tokens is not None:
        opts["num_predict"] = req.max_tokens
    return opts


# ---------------------------------------------------------------------------
# SSE chunk factories
# ---------------------------------------------------------------------------


def _make_content_chunk(
    completion_id: str,
    model: str,
    content: str,
    role: str | None = None,
) -> str:
    """Build an OpenAI SSE data line carrying a content delta.

    Args:
        completion_id: Unique completion identifier string.
        model: Model name to embed in the chunk.
        content: Token text for this delta.
        role: If provided (first chunk only), include in delta.

    Returns:
        str: "data: {...}\\n\\n" formatted SSE line.
    """
    delta: dict[str, Any] = {"content": content}
    if role is not None:
        delta["role"] = role
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _make_stop_chunk(completion_id: str, model: str) -> str:
    """Build the OpenAI SSE stop chunk (finish_reason=stop).

    Args:
        completion_id: Unique completion identifier string.
        model: Model name to embed in the chunk.

    Returns:
        str: "data: {...}\\n\\n" formatted SSE line.
    """
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _make_violation_chunk(
    completion_id: str,
    model: str,
    event: ViolationEvent,
) -> str:
    """Build the ActivGuard violation SSE chunk.

    This is a non-standard extension to the OpenAI SSE format.  It carries
    the 'activguard' key alongside the standard fields so that compliant
    OpenAI clients ignore it gracefully while ActivGuard-aware clients can
    surface the warning.

    Args:
        completion_id: Unique completion identifier string.
        model: Model name.
        event: ViolationEvent carrying confidence, layer, and CWE hint.

    Returns:
        str: "data: {...}\\n\\n" formatted SSE line with activguard payload.
    """
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }
    payload.update(event.to_sse_dict())
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Core streaming generator
# ---------------------------------------------------------------------------


async def _stream_with_probe(
    req: ChatCompletionRequest,
    completion_id: str,
    threshold: float,
    probe_interval: int,
    stop_on_violation: bool,
) -> AsyncGenerator[str, None]:
    """Async generator that proxies Ollama and applies the two-pass probe.

    The generator:
      - Opens a streaming connection to Ollama /api/chat via httpx.
      - Parses each NDJSON line into an OpenAI SSE chunk and yields it.
      - Every probe_interval tokens, schedules a CodeBERT probe task.
      - Drains completed probe tasks between SSE yields and injects
        violation chunks when threshold is exceeded.
      - After stream end, runs Bandit and emits a final violation chunk
        if Bandit fires.

    Args:
        req: Parsed ChatCompletionRequest.
        completion_id: Unique completion identifier for this request.
        threshold: P(vulnerable) boundary for the CodeBERT probe.
        probe_interval: Tokens between probe calls.
        stop_on_violation: If True, stop yielding content on first violation.

    Yields:
        str: SSE data lines (including the terminal "data: [DONE]\\n\\n").
    """
    ollama_messages = _messages_to_ollama(req.messages)
    ollama_options = _build_ollama_options(req)
    ollama_payload: dict[str, Any] = {
        "model": req.model,
        "messages": ollama_messages,
        "stream": True,
    }
    if ollama_options:
        ollama_payload["options"] = ollama_options
    if req.stop:
        ollama_payload["stop"] = req.stop if isinstance(req.stop, list) else [req.stop]

    # Mutable state shared across the generator body
    buffer: list[str] = [""]        # index 0 = accumulated content
    token_count: list[int] = [0]
    violations: list[ViolationEvent] = []
    first_chunk: list[bool] = [True]
    violation_emitted: list[bool] = [False]

    # asyncio.Queue for L1 probe results
    probe_result_queue: asyncio.Queue[tuple[int, float, str]] = asyncio.Queue()
    # asyncio.Queue for L2 RAG results (token_index, confidence, pattern_id, snapshot)
    rag_result_queue: asyncio.Queue[tuple[int, float, str, str]] = asyncio.Queue()
    pending_probe_tasks: set[asyncio.Task[None]] = set()

    async def _schedule_probe(snapshot: str, snap_token: int) -> None:
        """Background task: run L1 probe and enqueue result."""
        score = await _run_probe_async(snapshot)
        await probe_result_queue.put((snap_token, score, snapshot))

    async def _schedule_rag(snapshot: str, snap_token: int) -> None:
        """Background task: run L2 RAG query and enqueue result.

        Only called when L1 fires (L1-gated per architecture invariant).
        """
        conf, pattern_id = await _run_rag_async(snapshot)
        await rag_result_queue.put((snap_token, conf, pattern_id, snapshot))

    def _drain_probe_queue() -> list[ViolationEvent]:
        """Drain all completed L1 probe results and return new ViolationEvents.

        When L1 fires, also schedules the L2 RAG task (L1-gating).
        """
        new_events: list[ViolationEvent] = []
        while not probe_result_queue.empty():
            try:
                tok_idx, score, snap = probe_result_queue.get_nowait()
                logger.debug("L1 probe result: token=%d score=%.3f", tok_idx, score)
                if score >= threshold:
                    cwe = infer_cwe_hint(snap)
                    event = ViolationEvent(
                        token_index=tok_idx,
                        confidence=score,
                        layer="L1-activation",
                        cwe_hint=cwe,
                        partial_output=snap,
                    )
                    new_events.append(event)
                    logger.info(
                        "L1-activation VIOLATION: token=%d score=%.3f cwe=%s — scheduling L2",
                        tok_idx, score, cwe,
                    )
                    # L1-gated L2: schedule RAG query on the partial output
                    if _rag_loaded:
                        rag_task = asyncio.create_task(_schedule_rag(snap, tok_idx))
                        pending_probe_tasks.add(rag_task)
                        rag_task.add_done_callback(pending_probe_tasks.discard)
            except asyncio.QueueEmpty:
                break
        return new_events

    def _drain_rag_queue() -> list[ViolationEvent]:
        """Drain all completed L2 RAG results and return new ViolationEvents."""
        new_events: list[ViolationEvent] = []
        while not rag_result_queue.empty():
            try:
                tok_idx, conf, pattern_id, snap = rag_result_queue.get_nowait()
                logger.debug(
                    "L2 RAG result: token=%d conf=%.3f pattern=%s",
                    tok_idx, conf, pattern_id,
                )
                if conf >= _RAG_CONFIDENCE_THRESHOLD:
                    cwe = infer_cwe_hint(snap)
                    event = ViolationEvent(
                        token_index=tok_idx,
                        confidence=conf,
                        layer="L2-rag",
                        cwe_hint=cwe,
                        partial_output=snap,
                    )
                    new_events.append(event)
                    logger.info(
                        "L2-RAG VIOLATION: token=%d conf=%.3f pattern=%s cwe=%s",
                        tok_idx, conf, pattern_id, cwe,
                    )
            except asyncio.QueueEmpty:
                break
        return new_events

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE}/api/chat",
                json=ollama_payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Ollama error: {body.decode(errors='replace')[:512]}",
                    )

                async for raw_line in response.aiter_lines():
                    if not raw_line.strip():
                        continue

                    # Parse Ollama NDJSON chunk
                    try:
                        chunk_data: dict[str, Any] = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    message = chunk_data.get("message", {})
                    content_piece: str = message.get("content", "")
                    done: bool = chunk_data.get("done", False)

                    # Drain any completed L1 probe tasks before emitting content
                    new_violations = _drain_probe_queue()
                    for v in new_violations:
                        violations.append(v)
                        violation_emitted[0] = True
                        yield _make_violation_chunk(completion_id, req.model, v)

                    # Drain any completed L2 RAG tasks
                    new_rag_violations = _drain_rag_queue()
                    for v in new_rag_violations:
                        violations.append(v)
                        violation_emitted[0] = True
                        yield _make_violation_chunk(completion_id, req.model, v)

                    # If stop_on_violation is set and we already flagged, stop
                    # emitting content (but keep consuming Ollama to close conn).
                    if stop_on_violation and violations:
                        if done:
                            break
                        continue

                    # Emit content delta
                    if content_piece:
                        role_arg: str | None = "assistant" if first_chunk[0] else None
                        first_chunk[0] = False
                        buffer[0] += content_piece
                        token_count[0] += 1
                        yield _make_content_chunk(
                            completion_id, req.model, content_piece, role=role_arg
                        )

                        # Schedule a probe every probe_interval tokens
                        if token_count[0] % probe_interval == 0:
                            snapshot = buffer[0]
                            task = asyncio.create_task(
                                _schedule_probe(snapshot, token_count[0])
                            )
                            pending_probe_tasks.add(task)
                            task.add_done_callback(pending_probe_tasks.discard)

                    if done:
                        break

        # Wait for any in-flight probe tasks to complete before Bandit pass
        if pending_probe_tasks:
            await asyncio.gather(*pending_probe_tasks, return_exceptions=True)

        # Final drain after all tasks finish — L1 then L2
        for v in _drain_probe_queue():
            violations.append(v)
            yield _make_violation_chunk(completion_id, req.model, v)
        for v in _drain_rag_queue():
            violations.append(v)
            yield _make_violation_chunk(completion_id, req.model, v)

        # -----------------------------------------------------------------
        # L3-bandit second pass on full output
        # -----------------------------------------------------------------
        full_output = buffer[0]
        if BANDIT_SECOND_PASS and full_output.strip():
            bandit_result = await _run_bandit_async(full_output)
            if bandit_result.get("flagged"):
                cwe = infer_cwe_hint(full_output)
                bandit_event = ViolationEvent(
                    token_index=token_count[0],
                    confidence=1.0,
                    layer="L3-bandit",
                    cwe_hint=cwe,
                    partial_output=full_output,
                    bandit_findings=bandit_result.get("findings", []),
                )
                violations.append(bandit_event)
                logger.info(
                    "L3-bandit VIOLATION: %d findings, cwe=%s",
                    bandit_result.get("n_findings", 0),
                    cwe,
                )
                yield _make_violation_chunk(completion_id, req.model, bandit_event)

    except HTTPException:
        raise
    except (httpx.ConnectError, httpx.ReadTimeout) as exc:
        logger.error("Ollama connection error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Ollama backend unreachable: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error in streaming proxy: %s", exc)
        raise HTTPException(status_code=500, detail="Proxy internal error") from exc
    finally:
        # Cancel any remaining probe tasks on early exit
        for task in pending_probe_tasks:
            task.cancel()

    yield _make_stop_chunk(completion_id, req.model)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------


async def _complete_non_streaming(
    req: ChatCompletionRequest,
    completion_id: str,
    threshold: float,
) -> tuple[str, list[ViolationEvent]]:
    """Fetch complete Ollama response and probe the full output.

    Args:
        req: Parsed ChatCompletionRequest.
        completion_id: Unique completion identifier.
        threshold: P(vulnerable) boundary.

    Returns:
        Tuple of (full_content_string, list_of_ViolationEvents).

    Raises:
        HTTPException: On Ollama connectivity or protocol errors.
    """
    ollama_messages = _messages_to_ollama(req.messages)
    ollama_options = _build_ollama_options(req)
    ollama_payload: dict[str, Any] = {
        "model": req.model,
        "messages": ollama_messages,
        "stream": False,
    }
    if ollama_options:
        ollama_payload["options"] = ollama_options

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json=ollama_payload,
            )
    except (httpx.ConnectError, httpx.ReadTimeout) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama backend unreachable: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Ollama error: {response.text[:512]}",
        )

    try:
        data: dict[str, Any] = response.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Ollama returned invalid JSON") from exc

    content: str = data.get("message", {}).get("content", "")
    violations: list[ViolationEvent] = []

    # L1: Run CodeBERT probe on full output
    if content.strip():
        score = await _run_probe_async(content)
        if score >= threshold:
            cwe = infer_cwe_hint(content)
            violations.append(
                ViolationEvent(
                    token_index=0,
                    confidence=score,
                    layer="L1-activation",
                    cwe_hint=cwe,
                    partial_output=content,
                )
            )
            # L1-gated L2: only query RAG if L1 fired
            if _rag_loaded:
                rag_conf, rag_pid = await _run_rag_async(content)
                if rag_conf >= _RAG_CONFIDENCE_THRESHOLD:
                    violations.append(
                        ViolationEvent(
                            token_index=0,
                            confidence=rag_conf,
                            layer="L2-rag",
                            cwe_hint=cwe,
                            partial_output=content,
                        )
                    )
                    logger.info("L2-RAG VIOLATION (non-streaming): conf=%.3f pattern=%s", rag_conf, rag_pid)

    # L3: Bandit second pass
    if BANDIT_SECOND_PASS and content.strip():
        bandit_result = await _run_bandit_async(content)
        if bandit_result.get("flagged"):
            cwe = infer_cwe_hint(content)
            violations.append(
                ViolationEvent(
                    token_index=0,
                    confidence=1.0,
                    layer="L3-bandit",
                    cwe_hint=cwe,
                    partial_output=content,
                    bandit_findings=bandit_result.get("findings", []),
                )
            )

    return content, violations


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    """Return proxy liveness and layer readiness.

    Returns:
        JSON object with status, per-layer readiness flags, and detector identifier.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "layers": {
                "L1_codebert": _probe_loaded,
                "L1_hf_residual_stream": _hf_probe_loaded,
                "L2_rag": _rag_loaded,
                "L3_bandit": _BANDIT_AVAILABLE,
            },
            "model": DETECTOR_TAG,
            "hf_model": HF_PROBE_MODEL if _hf_probe_loaded else None,
        }
    )


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """Return available Ollama models in OpenAI /v1/models format.

    Queries the Ollama /api/tags endpoint and reformats the response to match
    the OpenAI model listing schema so that client tools discover models
    correctly.

    Returns:
        JSONResponse: OpenAI-compatible model list, or an error payload if
            Ollama is unreachable.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE}/api/tags")
        response.raise_for_status()
        tags_data: dict[str, Any] = response.json()
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
        logger.warning("Could not reach Ollama for model list: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Ollama unreachable: {exc}", "type": "proxy_error"}},
        )

    models_raw: list[dict[str, Any]] = tags_data.get("models", [])
    openai_models = [
        {
            "id": m.get("name", "unknown"),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ollama",
        }
        for m in models_raw
    ]
    return JSONResponse(content={"object": "list", "data": openai_models})


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    """OpenAI-compatible chat completions endpoint with vulnerability scanning.

    Accepts a standard OpenAI chat completions request, forwards it to Ollama,
    and scans the generated content using the CodeBERT probe (L1-activation)
    and optionally Bandit (L3-bandit).

    Streaming mode:
        Returns a StreamingResponse with Server-Sent Events.  Violation events
        are injected as extra SSE chunks with an 'activguard' key.  The final
        SSE response includes the X-ActivGuard-Result header.

    Non-streaming mode:
        Returns a standard JSON completion.  If violations are detected, a
        warning block is prepended to the assistant message content.  The
        X-ActivGuard-Result header is set on the JSON response.

    Args:
        request: Raw FastAPI Request (used to parse the JSON body with full
            error handling before constructing the Pydantic model).

    Returns:
        StreamingResponse for stream=True requests, JSONResponse otherwise.

    Raises:
        HTTPException 422: On invalid request body.
        HTTPException 502: When Ollama is unreachable.
        HTTPException 500: On unexpected proxy errors.
    """
    # Parse and validate request body
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON body: {exc}") from exc

    try:
        req = ChatCompletionRequest.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Request validation failed: {exc}") from exc

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    threshold = _probe_threshold
    probe_interval = _probe_interval
    stop_on_violation = _stop_on_violation

    if req.stream:
        # --- Streaming path ---
        # We need the violation list to set the response header, but with
        # StreamingResponse the headers must be set before the body starts.
        # We work around this by always setting X-ActivGuard-Result to
        # "SCANNING" initially; ActivGuard-aware clients should read the
        # violation SSE chunks instead of relying solely on the header.
        # The violation chunks carry authoritative detection results.
        async def event_generator() -> AsyncGenerator[bytes, None]:
            async for chunk in _stream_with_probe(
                req,
                completion_id,
                threshold,
                probe_interval,
                stop_on_violation,
            ):
                yield chunk.encode("utf-8")

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-ActivGuard-Result": "SCANNING",
                "X-ActivGuard-Detector": DETECTOR_TAG,
            },
        )

    # --- Non-streaming path ---
    content, violations = await _complete_non_streaming(req, completion_id, threshold)

    result_label = "VIOLATION" if violations else "VERIFIED"
    peak_confidence: float = max((v.confidence for v in violations), default=0.0)

    warning_prefix = ""
    if violations:
        cwe_hints = ", ".join(
            sorted({v.cwe_hint for v in violations if v.cwe_hint != "CWE-unknown"})
            or ["CWE-unknown"]
        )
        layers = ", ".join(sorted({v.layer for v in violations}))
        warning_prefix = (
            f"[ACTIVGUARD VIOLATION — confidence={peak_confidence:.2f} "
            f"layer={layers} cwe={cwe_hints}]\n\n"
        )

    response_body: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": warning_prefix + content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,   # Ollama does not return token counts here
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }

    return JSONResponse(
        content=response_body,
        headers={
            "X-ActivGuard-Result": result_label,
            "X-ActivGuard-Confidence": str(round(peak_confidence, 4)),
            "X-ActivGuard-Detector": DETECTOR_TAG,
        },
    )


# ---------------------------------------------------------------------------
# Runtime configuration injection (called from __main__.py)
# ---------------------------------------------------------------------------


def configure(
    probe_weights: str | None = None,
    threshold: float | None = None,
    probe_interval: int | None = None,
    stop_on_violation: bool | None = None,
) -> None:
    """Override module-level configuration before the app starts serving.

    This must be called BEFORE the first request is processed (i.e., before
    uvicorn begins serving).  Calling it after startup has no effect on the
    already-loaded probe instance.

    Args:
        probe_weights: Path to probe weights .pkl file.
        threshold: P(vulnerable) decision boundary override.
        probe_interval: Token interval between probe calls.
        stop_on_violation: Whether to halt streaming on first violation.
    """
    global _probe_weights_path, _probe_threshold, _probe_interval, _stop_on_violation
    if probe_weights is not None:
        _probe_weights_path = probe_weights
    if threshold is not None:
        _probe_threshold = threshold
    if probe_interval is not None:
        _probe_interval = probe_interval
    if stop_on_violation is not None:
        _stop_on_violation = stop_on_violation
