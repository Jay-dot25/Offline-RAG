"""
llm.py — Offline LLM via Ollama REST API  (v3 — conversation memory)

Switches from /api/generate (stateless) to /api/chat (multi-turn).
History is passed as a messages list — Ollama handles the context window.
"""

import httpx
import json

OLLAMA_BASE   = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
MAX_HISTORY   = 6     # Keep last 6 turns (3 Q+A pairs) — safe for 4GB VRAM
MAX_CTX       = 4096  # Context window tokens

# ── System prompt ─────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are an intelligent offline assistant with memory of this conversation.

Rules:
- Answer using ONLY the provided document context. Do not use prior knowledge.
- You CAN reference previous answers in this conversation to answer follow-ups.
- If the context doesn't contain enough information, say so clearly.
- Be concise and accurate. Mention the source filename when quoting.
- For comparisons or summaries, use clear sections."""

CONTEXT_TEMPLATE = """Relevant document context:

{context}

---
Question: {question}"""


# ── Ollama helpers ────────────────────────────────────────────────────────────

def check_ollama() -> tuple[bool, list]:
    try:
        resp  = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, models
    except Exception as e:
        return False, str(e)


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(chunks: list[dict], max_chars: int = 3000) -> str:
    parts = []
    total = 0
    for c in chunks:
        entry = f"[Source: {c['source']} | Score: {c.get('rerank_score', c.get('score','?'))}]\n{c['text']}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n---\n\n".join(parts)


# ── Message history utilities ─────────────────────────────────────────────────

def trim_history(history: list[dict], max_turns: int = MAX_HISTORY) -> list[dict]:
    """
    Keep only the last max_turns messages (each turn = 1 user + 1 assistant).
    Always keeps pairs intact so the conversation stays coherent.
    """
    if len(history) <= max_turns:
        return history
    # Trim from front, keep pairs (user+assistant)
    trim = len(history) - max_turns
    # Round up to even to keep pairs
    trim = trim + (trim % 2)
    return history[trim:]


# ── Main generation (multi-turn) ──────────────────────────────────────────────

def generate(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Generate an answer given question, retrieved chunks, and conversation history.

    Args:
        question: Current user question
        chunks:   Retrieved chunks from hybrid_search()
        history:  List of {"role": "user"/"assistant", "content": "..."} dicts
                  from previous turns. Pass None or [] for first question.
        model:    Ollama model name

    Returns:
        {answer, model, sources, num_sources, updated_history}
    """
    history = history or []

    if not chunks:
        answer = (
            "No relevant documents found. Please upload files first, "
            "then ask your question."
        )
        updated = history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {
            "answer": answer, "model": model,
            "sources": [], "num_sources": 0,
            "updated_history": trim_history(updated),
        }

    context = build_context(chunks)
    user_msg = CONTEXT_TEMPLATE.format(context=context, question=question)

    # Build message list: system + trimmed history + current user message
    messages = (
        [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        + trim_history(history)
        + [{"role": "user", "content": user_msg}]
    )

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model":    model,
                "messages": messages,
                "stream":   False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                    "num_ctx":     MAX_CTX,
                    "top_p":       0.9,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json()["message"]["content"].strip()

    except httpx.ConnectError:
        answer = (
            "⚠️ Cannot connect to Ollama.\n"
            f"  Run: ollama run {model}"
        )
    except Exception as e:
        answer = f"⚠️ LLM error: {str(e)}"

    sources = list({c["source"] for c in chunks})

    # Append this turn to history (store original question, not context-padded version)
    updated_history = trim_history(
        history
        + [{"role": "user",      "content": question},
           {"role": "assistant", "content": answer}]
    )

    return {
        "answer":          answer,
        "model":           model,
        "sources":         sources,
        "num_sources":     len(sources),
        "updated_history": updated_history,
    }


# ── Streaming version ─────────────────────────────────────────────────────────

def generate_streaming(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
):
    """
    Streaming generator — yields (token, updated_history) tuples.
    updated_history is None on each token except the final one.
    """
    history = history or []

    if not chunks:
        msg = "No relevant documents found. Please upload files first."
        yield msg, trim_history(
            history + [{"role": "user",      "content": question},
                        {"role": "assistant", "content": msg}]
        )
        return

    context  = build_context(chunks)
    user_msg = CONTEXT_TEMPLATE.format(context=context, question=question)

    messages = (
        [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        + trim_history(history)
        + [{"role": "user", "content": user_msg}]
    )

    full_answer = ""
    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model, "messages": messages, "stream": True,
                "options": {"temperature": 0.1, "num_predict": 512, "num_ctx": MAX_CTX},
            },
            timeout=120,
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                data  = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    full_answer += token
                    yield token, None
                if data.get("done"):
                    break

    except httpx.ConnectError:
        err = f"\n\n⚠️ Ollama not running. Start: ollama run {model}"
        yield err, None
        full_answer += err
    except Exception as e:
        err = f"\n\n⚠️ Error: {e}"
        yield err, None
        full_answer += err

    updated = trim_history(
        history + [{"role": "user",      "content": question},
                   {"role": "assistant", "content": full_answer}]
    )
    yield "", updated
