"""
vectorstore.py — ChromaDB vector store with offline sentence-transformer embeddings
Uses all-MiniLM-L6-v2 (~90MB, runs on CPU, fast)
"""

import uuid
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
PERSIST_DIR = "./vectordb"          # ChromaDB saves here between sessions
COLLECTION_NAME = "rag_documents"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 90MB, offline, runs on CPU


# ── Singleton client + collection ────────────────────────────────────────────
_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        print(f"[VectorStore] Initializing ChromaDB at {PERSIST_DIR}")
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL,
            device="cpu",           # CPU keeps VRAM free for the LLM
        )
        _client = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        print(f"[VectorStore] Collection ready — {_collection.count()} docs indexed")
    return _collection


# ── Write operations ─────────────────────────────────────────────────────────

def add_document(chunks: list[str], filename: str, file_type: str) -> int:
    """
    Add chunks from a single file into the vector store.
    Returns number of chunks added.
    """
    collection = _get_collection()

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": filename, "file_type": file_type, "chunk_index": i}
        for i, _ in enumerate(chunks)
    ]

    # ChromaDB deduplication: skip chunks already from this file
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        print(f"[VectorStore] {filename} already indexed ({len(existing['ids'])} chunks) — skipping")
        return 0

    # Add in batches of 100 (safe for memory)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i : i + batch_size],
            ids=ids[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    print(f"[VectorStore] Added {len(chunks)} chunks from {filename}")
    return len(chunks)


def delete_document(filename: str) -> int:
    """Remove all chunks belonging to a file."""
    collection = _get_collection()
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"[VectorStore] Deleted {len(existing['ids'])} chunks for {filename}")
        return len(existing["ids"])
    return 0


# ── Read operations ──────────────────────────────────────────────────────────

def query(
    query_text: str,
    n_results: int = 5,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Semantic search over the vector store.
    Returns top-k chunks with source metadata and similarity scores.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    where = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "file_type": meta["file_type"],
            "chunk_index": meta["chunk_index"],
            "score": round(1 - dist, 4),   # Convert cosine distance → similarity
        })

    return hits


def list_documents() -> list[dict]:
    """Return unique files currently indexed."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    seen = {}
    for m in all_meta:
        src = m["source"]
        if src not in seen:
            seen[src] = {"filename": src, "file_type": m["file_type"], "chunks": 0}
        seen[src]["chunks"] += 1

    return list(seen.values())


def get_stats() -> dict:
    """Quick stats for the UI."""
    collection = _get_collection()
    docs = list_documents()
    return {
        "total_chunks": collection.count(),
        "total_files": len(docs),
        "files": docs,
    }
