"""
api.py — FastAPI backend v3 (conversation memory + source highlighting)
"""

import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

import ingest
import vectorstore
import bm25_index
import retrieval
import llm as llm_module
import highlight as hl

app = FastAPI(title="Multimodal Offline RAG", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                  allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Models ────────────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    role: str       # "user" | "assistant"
    content: str

class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    source_filter: str | None = None
    use_reranker: bool = True
    history: list[HistoryItem] = []     # conversation history from client


class QueryResponse(BaseModel):
    answer: str
    annotated_answer: str               # answer with inline [source] citations
    sources: list[str]
    num_sources: int
    model: str
    retrieved_chunks: list[dict]        # chunks with highlighted_html added
    retrieval_mode: str
    updated_history: list[dict]         # history to send back to client


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    ollama_ok, models = llm_module.check_ollama()
    return {
        "status": "ok",
        "ollama_running": ollama_ok,
        "available_models": models if ollama_ok else [],
        "vectorstore": vectorstore.get_stats(),
        "bm25": bm25_index.get_bm25().stats(),
    }


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ingest.SUPPORTED_FORMATS:
        raise HTTPException(400, f"Unsupported: {ext}")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result     = ingest.process_file(str(save_path))
        vec_added  = vectorstore.add_document(result["chunks"], result["filename"], result["file_type"])
        bm25_added = bm25_index.get_bm25().add_documents(result["chunks"], result["filename"], result["file_type"])
        already    = vec_added == 0 and bm25_added == 0
        return {
            "filename":    result["filename"],
            "file_type":   result["file_type"],
            "num_chunks":  result["num_chunks"],
            "chunks_added": max(vec_added, bm25_added),
            "status":      "already_indexed" if already else "indexed",
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # Convert Pydantic history items → plain dicts for llm.py
    history_dicts = [{"role": h.role, "content": h.content} for h in req.history]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    chunks = retrieval.hybrid_search(
        query=req.question,
        n_results=req.n_results,
        source_filter=req.source_filter,
        use_reranker=req.use_reranker,
    )

    # ── Generation (with history) ─────────────────────────────────────────────
    result = llm_module.generate(
        question=req.question,
        chunks=chunks,
        history=history_dicts,
    )

    # ── Highlighting ──────────────────────────────────────────────────────────
    highlighted_chunks  = hl.highlight_sources(result["answer"], chunks)
    annotated_answer    = hl.highlight_answer_citations(result["answer"], chunks)

    return QueryResponse(
        answer=result["answer"],
        annotated_answer=annotated_answer,
        sources=result["sources"],
        num_sources=result["num_sources"],
        model=result["model"],
        retrieved_chunks=highlighted_chunks,
        retrieval_mode="hybrid+reranker" if req.use_reranker else "hybrid",
        updated_history=result["updated_history"],
    )


# ── Debug ─────────────────────────────────────────────────────────────────────

@app.post("/debug/search")
def debug_search(req: QueryRequest):
    return retrieval.search_debug(req.question, n=req.n_results)


# ── Document management ───────────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    return vectorstore.list_documents()


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    v = vectorstore.delete_document(filename)
    b = bm25_index.get_bm25().delete_document(filename)
    if v == 0 and b == 0:
        raise HTTPException(404, f"{filename} not found")
    p = UPLOAD_DIR / filename
    if p.exists():
        p.unlink()
    return {"filename": filename, "vec_deleted": v, "bm25_deleted": b, "status": "deleted"}


@app.get("/stats")
def get_stats():
    return {"vectorstore": vectorstore.get_stats(), "bm25": bm25_index.get_bm25().stats()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
