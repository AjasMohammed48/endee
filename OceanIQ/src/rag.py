"""
rag.py — OceanIQ answer generation
  🤖 AI Summary  → phi3:mini via Ollama (best quality for 8GB RAM)
  🔍 Semantic    → Extractive summary (no LLM)
"""

import requests
import json

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"   # 2.3GB — best Q&A quality for 8GB RAM


# ─────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────
def _ollama_status() -> dict:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=4)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        has_model = any("phi3" in m.lower() for m in models)
        return {"running": True, "has_model": has_model, "models": models, "error": None}
    except requests.exceptions.ConnectionError:
        return {"running": False, "has_model": False, "models": [], "error": "connection_refused"}
    except Exception as e:
        return {"running": False, "has_model": False, "models": [], "error": str(e)}


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────
def _build_context(local_results: list, web_results: dict) -> str:
    """
    phi3:mini has a 4096 token window — we can give it richer context than TinyLlama.
    Budget ~1200 chars for context so there's room for a solid answer.
    """
    parts = []

    for r in local_results[:3]:
        text = (r.get("meta") or {}).get("text", "").strip()
        title = (r.get("meta") or {}).get("title", "")
        if text:
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
            chunk = ". ".join(sentences[:3]) + "."
            parts.append(f"• {title}: {chunk}" if title else f"• {chunk}")

    for item in (web_results.get("wikipedia") or [])[:2]:
        snippet = (item.get("summary") or item.get("abstract") or "").strip()
        title = item.get("title", "")
        if snippet:
            sentences = [s.strip() for s in snippet.split(".") if len(s.strip()) > 20]
            chunk = ". ".join(sentences[:3]) + "."
            parts.append(f"• {title}: {chunk}" if title else f"• {chunk}")

    for item in (web_results.get("semantic_scholar") or [])[:2]:
        snippet = (item.get("abstract") or item.get("summary") or "").strip()
        title = item.get("title", "")
        if snippet:
            sentences = [s.strip() for s in snippet.split(".") if len(s.strip()) > 20]
            chunk = ". ".join(sentences[:2]) + "."
            parts.append(f"• {title}: {chunk}" if title else f"• {chunk}")

    for item in (web_results.get("pubmed") or [])[:1]:
        snippet = (item.get("abstract") or item.get("summary") or "").strip()
        title = item.get("title", "")
        if snippet:
            sentences = [s.strip() for s in snippet.split(".") if len(s.strip()) > 20]
            chunk = ". ".join(sentences[:2]) + "."
            parts.append(f"• {title}: {chunk}" if title else f"• {chunk}")

    context = "\n".join(parts)
    return context[:1200]


# ─────────────────────────────────────────────
# PHI3 CALL  (streaming for reliability on CPU)
# ─────────────────────────────────────────────
def _call_phi3(context: str, query: str) -> tuple:
    """
    phi3:mini uses a simple <|user|> / <|end|> / <|assistant|> format.
    stream=True reads token by token — reliable on CPU.
    """
    prompt = (
        "<|user|>\n"
        "You are an expert ocean science research assistant. "
        "Using ONLY the context below, write a clear and informative answer (3-5 sentences). "
        "Do not make up facts. If the context doesn't fully answer the question, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature":    0.3,
            "num_predict":    280,
            "top_p":          0.9,
            "top_k":          40,
            "repeat_penalty": 1.1,
            "stop": ["<|end|>", "<|user|>", "<|system|>"],
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=150, stream=True)
        resp.raise_for_status()

        tokens = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            tokens.append(chunk.get("response", ""))
            if chunk.get("done", False):
                break

        answer = "".join(tokens).strip()

        # Clean leaked prompt tokens
        for tag in ["<|assistant|>", "<|user|>", "<|end|>", "<|system|>"]:
            answer = answer.replace(tag, "").strip()

        return (answer, None) if len(answer) > 20 else (None, f"Short response: '{answer}'")

    except requests.exceptions.ConnectionError:
        return None, "connection_refused"
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# EXTRACTIVE FALLBACK
# ─────────────────────────────────────────────
def _extractive_summary(local_results: list, web_results: dict) -> str:
    sentences = []

    for r in local_results[:3]:
        text = (r.get("meta") or {}).get("text", "").strip()
        if text:
            first = text.split(".")[0].strip()
            if len(first) > 30:
                sentences.append(first + ".")

    for item in (web_results.get("wikipedia") or [])[:2]:
        snippet = (item.get("summary") or item.get("abstract") or "").strip()
        if snippet:
            first = snippet.split(".")[0].strip()
            if len(first) > 30:
                sentences.append(first + ".")

    for item in (web_results.get("semantic_scholar") or [])[:1]:
        snippet = (item.get("abstract") or "").strip()
        if snippet:
            first = snippet.split(".")[0].strip()
            if len(first) > 30:
                sentences.append(first + ".")

    return " ".join(sentences[:4])


def _error_html(title: str, detail: str, fallback: str) -> str:
    return (
        f"<div style='background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.3);"
        f"border-radius:8px;padding:12px 14px;margin-bottom:12px;font-size:13px;'>"
        f"<span style='color:#f5a623;font-weight:700;'>⚠️ {title}</span><br>"
        f"<span style='color:#8499b8;'>{detail}</span>"
        f"</div>"
        + (f"<i style='color:#5a6e8a;font-size:12px;'>Showing extracted summary instead:</i>"
           f"<br><br>{fallback}" if fallback else "")
    )


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def build_ai_summary(query: str, local_results: list, web_results: dict) -> str:
    """🤖 AI Summary — phi3:mini answers using retrieved context."""

    context = _build_context(local_results, web_results)
    fallback = _extractive_summary(local_results, web_results)

    if not context:
        return (
            "No relevant documents found to answer this question. "
            "Try different keywords or make sure Endee has indexed data."
        )

    answer, error = _call_phi3(context, query)

    if answer:
        return answer

    # Diagnose
    if error == "connection_refused":
        return _error_html(
            "Ollama is not running",
            "Open a terminal and run: <code>ollama serve</code>",
            fallback,
        )

    if error == "timeout":
        return _error_html(
            "phi3:mini timed out",
            "It may still be loading into RAM. Wait 30 seconds and try again.",
            fallback,
        )

    status = _ollama_status()
    if status["running"] and not status["has_model"]:
        installed = ", ".join(status["models"]) or "none"
        return _error_html(
            "phi3:mini is not installed",
            f"Run: <code>ollama pull phi3:mini</code> &nbsp;|&nbsp; "
            f"Installed: {installed}",
            fallback,
        )

    if error:
        return _error_html("phi3:mini error", str(error), fallback)

    return fallback if fallback else "Could not generate an answer. Try rephrasing."


def build_semantic_only_summary(query: str, local_results: list) -> str:
    """🔍 Semantic Search — extractive summary, no LLM."""
    if not local_results:
        return f"No matching documents found for <b>{query}</b>. Try different keywords."

    lines = []
    for r in local_results[:4]:
        meta  = r.get("meta") or {}
        title = meta.get("title", "")
        text  = meta.get("text", "").strip()
        if text:
            sentence = text.split(".")[0].strip()
            sentence = sentence if len(sentence) >= 40 else text[:200]
            entry = f"<strong>{title}</strong> — {sentence}." if title else sentence + "."
            lines.append(entry)

    return "<br><br>".join(lines) if lines else f"No text content found for <b>{query}</b>."