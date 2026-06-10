"""
ui.py — Gradio frontend v3
- gr.Chatbot for multi-turn conversation with memory
- Source highlighting rendered as HTML
- Annotated answer with inline citations
"""

import gradio as gr
import httpx
from pathlib import Path

API_BASE = "http://localhost:8000"


# ── API helpers ───────────────────────────────────────────────────────────────

def api_health():
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        d = r.json()
        ollama = "Running" if d["ollama_running"] else "Offline"
        ollama_color = "#4ade80" if d["ollama_running"] else "#f87171"
        chunks = d["vectorstore"]["total_chunks"]
        files  = d["vectorstore"]["total_files"]
        vocab  = d.get("bm25", {}).get("vocab_size", "—")
        return (
            f'<span style="color:{ollama_color};font-weight:600;">LLM {ollama}</span>'
            f'&emsp;·&emsp;{files} files&emsp;·&emsp;{chunks} chunks&emsp;·&emsp;BM25 vocab {vocab}'
        )
    except Exception as e:
        return f'<span style="color:#f87171;">Backend unreachable — {e}</span>'


def upload_file(file_obj):
    if file_obj is None:
        return "No file selected.", refresh_doc_list()
    filepath = file_obj.name
    filename = Path(filepath).name
    try:
        with open(filepath, "rb") as f:
            r = httpx.post(f"{API_BASE}/upload",
                           files={"file": (filename, f)}, timeout=300)
        if r.status_code == 200:
            d = r.json()
            msg = (f"**{filename}** is already indexed."
                   if d["status"] == "already_indexed"
                   else f"**{filename}** indexed — {d['chunks_added']} chunks added.")
        else:
            msg = f"Error: {r.json().get('detail', r.text)}"
    except Exception as e:
        msg = f"Error: {e}"
    return msg, refresh_doc_list()


def refresh_doc_list():
    try:
        r = httpx.get(f"{API_BASE}/documents", timeout=5)
        docs = r.json()
        if not docs:
            return "_No documents indexed._"
        rows = ["| File | Type | Chunks |", "| --- | --- | --- |"]
        for d in docs:
            rows.append(f"| {d['filename']} | {d['file_type']} | {d['chunks']} |")
        return "\n".join(rows)
    except:
        return "_Could not fetch document list._"


def delete_doc(filename):
    if not filename.strip():
        return "Enter a filename.", refresh_doc_list()
    try:
        r = httpx.delete(f"{API_BASE}/documents/{filename.strip()}", timeout=10)
        if r.status_code == 200:
            return f"Deleted `{filename}`", refresh_doc_list()
        return f"Error: {r.json().get('detail','Not found')}", refresh_doc_list()
    except Exception as e:
        return f"Error: {e}", refresh_doc_list()


# ── Chat handler ──────────────────────────────────────────────────────────────

def chat(
    user_message: str,
    chat_history: list,
    api_history: list,
    n_results: int,
    source_filter: str,
    use_reranker: bool,
):
    if not user_message.strip():
        yield chat_history, api_history, "", api_health()
        return

    source = source_filter.strip() or None
    chat_history = chat_history + [[user_message, None]]
    yield chat_history, api_history, "Retrieving relevant chunks…", api_health()

    try:
        r = httpx.post(
            f"{API_BASE}/query",
            json={
                "question":      user_message,
                "n_results":     int(n_results),
                "source_filter": source,
                "use_reranker":  use_reranker,
                "history":       api_history,
            },
            timeout=180,
        )

        if r.status_code == 200:
            d = r.json()
            answer    = d["annotated_answer"] or d["answer"]
            mode      = d.get("retrieval_mode", "hybrid")
            sources   = d["sources"]
            chunks    = d["retrieved_chunks"]
            new_hist  = d["updated_history"]

            chat_history[-1][1] = answer

            src_header = (
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                f'color:#6b7280;letter-spacing:0.04em;margin-bottom:12px;padding-bottom:10px;'
                f'border-bottom:1px solid #1f2937;">'
                f'MODE &nbsp;<span style="color:#d1d5db;">{mode.upper()}</span>'
                f'&emsp;&ensp;SOURCES &nbsp;'
                + "&ensp;".join(f'<span style="color:#e5e7eb;">{s}</span>' for s in sources)
                + f'</div>'
            )
            chunks_html = src_header + "\n".join(
                c.get("highlighted_html", f"<div>{c['text'][:300]}</div>")
                for c in chunks
            )

            yield chat_history, new_hist, chunks_html, api_health()

        else:
            err = r.json().get("detail", r.text)
            chat_history[-1][1] = f"Error: {err}"
            yield chat_history, api_history, "", api_health()

    except Exception as e:
        chat_history[-1][1] = f"Connection error: {e}"
        yield chat_history, api_history, "", api_health()


def clear_chat():
    return [], [], "", "Conversation cleared."


def debug_search(question, n_results):
    if not question.strip():
        return "Enter a query."
    try:
        r = httpx.post(
            f"{API_BASE}/debug/search",
            json={"question": question, "n_results": int(n_results)},
            timeout=30,
        )
        d = r.json()
        lines = [f"### Query: `{d['query']}`\n"]
        for stage, key in [
            ("Vector top-5",    "vector_top5"),
            ("BM25 top-5",      "bm25_top5"),
            ("RRF top-5",       "rrf_top5"),
            ("Reranked top-5",  "reranked_top5"),
        ]:
            lines.append(f"**{stage}**")
            for item in d.get(key, []):
                score_val = next((v for v in item.values() if isinstance(v, float)), 0)
                lines.append(f"- `{item['source']}` — {round(score_val, 4)}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ── UI layout ─────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
.contain {
    background: #0a0a0a !important;
    color: #d1d5db !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 1440px !important;
    padding: 0 24px !important;
}

/* ── Page header ─────────────────────────────────────────── */
#page-header {
    padding: 36px 0 28px;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 28px;
}
#page-header h1 {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #f9fafb;
    margin: 0 0 4px;
}
#page-header p {
    font-size: 13px;
    color: #6b7280;
    margin: 0;
    letter-spacing: 0.01em;
}

/* ── Tabs ────────────────────────────────────────────────── */
.tabs > .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #1a1a1a !important;
    gap: 0 !important;
    padding: 0 !important;
}
.tabs > .tab-nav > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #6b7280 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    text-transform: uppercase !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.tabs > .tab-nav > button.selected {
    color: #f9fafb !important;
    border-bottom-color: #f9fafb !important;
}
.tabs > .tab-nav > button:hover:not(.selected) {
    color: #9ca3af !important;
}
.tabitem {
    padding: 24px 0 0 !important;
    background: transparent !important;
}

/* ── Chatbot ─────────────────────────────────────────────── */
.chatbot-wrap .wrap {
    background: #111111 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 10px !important;
}
.chatbot-wrap .message.user > div {
    background: #1a1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #e5e7eb !important;
    border-radius: 8px 8px 2px 8px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
.chatbot-wrap .message.bot > div {
    background: #131313 !important;
    border: 1px solid #1f2937 !important;
    color: #d1d5db !important;
    border-radius: 8px 8px 8px 2px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* ── Inputs ──────────────────────────────────────────────── */
input[type="text"],
input[type="number"],
textarea {
    background: #111111 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
    color: #e5e7eb !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.15s !important;
}
input[type="text"]:focus,
textarea:focus {
    border-color: #374151 !important;
    outline: none !important;
    box-shadow: none !important;
}
::placeholder { color: #4b5563 !important; }

/* ── Buttons ─────────────────────────────────────────────── */
button.primary {
    background: #f9fafb !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 10px 22px !important;
    transition: background 0.15s !important;
}
button.primary:hover { background: #e5e7eb !important; }

button.secondary {
    background: transparent !important;
    color: #9ca3af !important;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: border-color 0.15s, color 0.15s !important;
}
button.secondary:hover {
    border-color: #374151 !important;
    color: #d1d5db !important;
}

button.stop {
    background: transparent !important;
    color: #f87171 !important;
    border: 1px solid #7f1d1d !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: border-color 0.15s !important;
}
button.stop:hover { border-color: #ef4444 !important; }

/* ── Slider ──────────────────────────────────────────────── */
.gr-slider input[type="range"] {
    accent-color: #4b5563 !important;
}

/* ── Checkbox ────────────────────────────────────────────── */
input[type="checkbox"] {
    accent-color: #f9fafb !important;
}

/* ── Labels ──────────────────────────────────────────────── */
label span,
.gr-form label {
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #6b7280 !important;
}

/* ── Markdown content ────────────────────────────────────── */
.prose, .gr-markdown, .gr-markdown p {
    color: #9ca3af !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
.gr-markdown h2 {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #e5e7eb !important;
    letter-spacing: -0.01em !important;
    margin-top: 24px !important;
}
.gr-markdown h3 {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #d1d5db !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
.gr-markdown code {
    background: #1a1a1a !important;
    border: 1px solid #1f2937 !important;
    color: #a5b4fc !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
}
.gr-markdown pre {
    background: #111111 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

/* ── Section labels ──────────────────────────────────────── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4b5563;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a1a1a;
}

/* ── Status bar ──────────────────────────────────────────── */
#status-bar {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: #6b7280 !important;
    padding: 10px 14px !important;
    background: #111111 !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
    margin-bottom: 16px !important;
}

/* ── Sources panel ───────────────────────────────────────── */
#sources-panel {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 10px !important;
    padding: 16px !important;
    min-height: 200px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    color: #6b7280 !important;
    line-height: 1.6 !important;
}

/* ── Upload area ─────────────────────────────────────────── */
.file-preview {
    background: #111111 !important;
    border: 1px dashed #1f2937 !important;
    border-radius: 10px !important;
    color: #6b7280 !important;
}

/* ── Table styling ───────────────────────────────────────── */
.gr-markdown table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 13px !important;
}
.gr-markdown th {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #6b7280 !important;
    border-bottom: 1px solid #1f2937 !important;
    padding: 8px 12px !important;
    text-align: left !important;
}
.gr-markdown td {
    color: #9ca3af !important;
    border-bottom: 1px solid #111111 !important;
    padding: 8px 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Dividers ────────────────────────────────────────────── */
hr { border-color: #1a1a1a !important; margin: 20px 0 !important; }

/* ── Scrollbars ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #374151; }
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    ),
    css=CUSTOM_CSS,
    title="RAG — Document Intelligence"
) as demo:

    # ── Page header ───────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="page-header">
        <h1>Document Intelligence</h1>
        <p>Offline retrieval-augmented generation with hybrid search and conversation memory</p>
    </div>
    """)

    # ── Persistent state ──────────────────────────────────────────────────────
    api_history_state = gr.State([])

    with gr.Tab("Chat"):
        with gr.Row(equal_height=False):

            # ── Left: chat ────────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="",
                    height=440,
                    bubble_full_width=False,
                    show_copy_button=True,
                    elem_classes=["chatbot-wrap"],
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about your documents",
                        show_label=False,
                        scale=5,
                        lines=1,
                    )
                    send_btn  = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                with gr.Row():
                    n_results      = gr.Slider(1, 10, value=5, step=1, label="Chunks")
                    source_filter  = gr.Textbox(label="Filter by file", placeholder="All files")
                    reranker_check = gr.Checkbox(value=True, label="Reranker")

            # ── Right: status + sources ───────────────────────────────────────
            with gr.Column(scale=2):
                status_md = gr.HTML(value=api_health(), elem_id="status-bar")
                gr.Button("Refresh status", variant="secondary").click(
                    fn=api_health, outputs=status_md
                )
                gr.HTML('<div class="section-label" style="margin-top:20px;">Source highlights</div>')
                sources_html = gr.HTML(
                    value='<div style="color:#374151;font-size:13px;">Relevant passages will appear here after a query.</div>',
                    elem_id="sources-panel",
                )

        # ── Event wiring ──────────────────────────────────────────────────────
        send_inputs  = [msg_box, chatbot, api_history_state,
                        n_results, source_filter, reranker_check]
        send_outputs = [chatbot, api_history_state, sources_html, status_md]

        send_btn.click(fn=chat, inputs=send_inputs, outputs=send_outputs).then(
            fn=lambda: "", outputs=msg_box
        )
        msg_box.submit(fn=chat, inputs=send_inputs, outputs=send_outputs).then(
            fn=lambda: "", outputs=msg_box
        )
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, api_history_state, sources_html, status_md],
        )

    with gr.Tab("Retrieval Debug"):
        gr.Markdown("Inspect each retrieval stage for a given query.")
        with gr.Row():
            dbg_q = gr.Textbox(label="Query", placeholder="Enter a question")
            dbg_n = gr.Slider(1, 10, value=5, step=1, label="Top-N")
        gr.Button("Run", variant="primary").click(
            fn=debug_search, inputs=[dbg_q, dbg_n],
            outputs=gr.Markdown()
        )

    with gr.Tab("Documents"):
        gr.Markdown(
            "Supported formats: PDF, DOCX, PNG, JPG, BMP, TIFF, MP3, MP4, WAV, M4A, OGG"
        )
        with gr.Row():
            with gr.Column():
                file_input    = gr.File(label="Select file")
                upload_btn    = gr.Button("Index document", variant="primary")
                upload_result = gr.Markdown()
            with gr.Column():
                gr.HTML('<div class="section-label">Indexed documents</div>')
                doc_list = gr.Markdown(value=refresh_doc_list())
                gr.Button("Refresh", variant="secondary").click(
                    fn=refresh_doc_list, outputs=doc_list
                )
                with gr.Row():
                    del_input = gr.Textbox(label="Filename")
                    del_btn   = gr.Button("Remove", variant="stop")
                del_result = gr.Markdown()

        upload_btn.click(
            fn=upload_file, inputs=[file_input],
            outputs=[upload_result, doc_list]
        )
        del_btn.click(
            fn=delete_doc, inputs=[del_input],
            outputs=[del_result, doc_list]
        )

    with gr.Tab("System"):
        gr.Markdown("""
## Pipeline — v3

**Conversation memory**

Uses Ollama `/api/chat` (multi-turn) instead of `/api/generate` (stateless). Retains the last 6 messages (3 Q+A turns) in context. Click **Clear** on the Chat tab to reset memory between topics.

**Source highlighting**

Each retrieved chunk is scanned sentence by sentence. Sentences sharing significant content words with the answer are highlighted. The answer receives inline source citations. This makes the exact grounding of each answer visible.

## Retrieval flow

```
User message + history
        |
        v
Hybrid search  (Vector + BM25 + RRF + Reranker)
        |
        v
llama3.2:3b  <--  system prompt + history + chunks
        |
        v
Answer --> highlight_sources() --> highlighted HTML
Answer --> highlight_answer_citations() --> annotated answer
        |
        v
Chat UI  (memory persists until Clear)
```
        """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)