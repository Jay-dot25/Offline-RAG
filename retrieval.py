"""
retrieval.py — Hybrid search pipeline

Stage 1: Parallel retrieval
  ├── Vector search (ChromaDB + MiniLM) — semantic similarity
  └── BM25 search (keyword index)        — exact term matching

Stage 2: Reciprocal Rank Fusion (RRF)
  Merges both ranked lists into a single unified ranking.
  Formula: score = Σ 1/(k + rank_i), k=60 (standard constant)
  Works without score normalization — just needs ranks.

Stage 3: Cross-encoder reranker (optional, ~80MB model)
  Takes top-N RRF results and re-scores each (query, chunk) pair
  using a fine-tuned relevance model. Much more accurate than
  bi-encoder embeddings for final selection.
  Model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations
import vectorstore
import bm25_index

# ── Reranker (loaded lazily) ─────────────────────────────────────────────────
_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("[Reranker] Loading cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)...")
        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",       # CPU — keep VRAM for LLM
            max_length=512,
        )
        print("[Reranker] Ready.")
    return _reranker


# ── RRF Fusion ───────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion.

    Each hit gets score = 1/(k + rank) from each list it appears in.
    Hits in both lists get both contributions — they naturally rank higher.
    k=60 is the standard value from the original RRF paper (Cormack 2009).

    Uses (source, chunk_text[:80]) as deduplication key — safe across both indexes.
    """
    scores: dict[str, float] = {}
    registry: dict[str, dict] = {}  # key → full chunk dict

    def key(hit: dict) -> str:
        return f"{hit['source']}::{hit['text'][:80]}"

    for rank, hit in enumerate(vector_hits):
        k_ = key(hit)
        scores[k_] = scores.get(k_, 0.0) + 1.0 / (k + rank + 1)
        if k_ not in registry:
            registry[k_] = {**hit, "from_vector": True, "from_bm25": False}
        else:
            registry[k_]["from_vector"] = True

    for rank, hit in enumerate(bm25_hits):
        k_ = key(hit)
        scores[k_] = scores.get(k_, 0.0) + 1.0 / (k + rank + 1)
        if k_ not in registry:
            registry[k_] = {**hit, "from_vector": False, "from_bm25": True}
        else:
            registry[k_]["from_bm25"] = True

    # Sort by RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for k_, rrf_score in ranked:
        hit = registry[k_].copy()
        hit["rrf_score"] = round(rrf_score, 6)
        # Carry forward original scores where available
        if "score" not in hit:
            hit["score"] = round(rrf_score, 4)
        results.append(hit)

    return results


# ── Cross-encoder reranker ────────────────────────────────────────────────────

def rerank(query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-score (query, chunk) pairs with a cross-encoder.
    Much more accurate than bi-encoder + BM25 for final top-k selection.

    Input:  RRF-merged hits (can be 10-20)
    Output: top_n hits re-sorted by cross-encoder relevance score
    """
    if not hits:
        return hits

    reranker = _get_reranker()

    pairs = [(query, hit["text"]) for hit in hits]
    scores = reranker.predict(pairs, show_progress_bar=False)

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = round(float(score), 4)

    reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# ── Main entry point ──────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    n_results: int = 5,
    source_filter: str | None = None,
    use_reranker: bool = True,
    vector_k: int = 10,     # Retrieve more than needed before reranking
    bm25_k: int = 10,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline.

    1. Vector search  → top vector_k chunks
    2. BM25 search    → top bm25_k chunks
    3. RRF fusion     → single merged + ranked list
    4. Cross-encoder  → re-score top-(n_results * 2) → return top n_results

    Args:
        query:         Natural language question
        n_results:     Final number of chunks to return to LLM
        source_filter: Restrict to a single file (optional)
        use_reranker:  Set False for faster responses (skips cross-encoder)
        vector_k:      How many to fetch from vector DB before fusion
        bm25_k:        How many to fetch from BM25 before fusion
    """

    # ── Stage 1: parallel retrieval ──────────────────────────────────────────
    vector_hits = vectorstore.query(
        query_text=query,
        n_results=vector_k,
        source_filter=source_filter,
    )

    bm25_hits = bm25_index.get_bm25().query(
        query_text=query,
        n_results=bm25_k,
        source_filter=source_filter,
    )

    # ── Stage 2: RRF fusion ───────────────────────────────────────────────────
    fused = reciprocal_rank_fusion(vector_hits, bm25_hits)

    if not fused:
        return []

    # ── Stage 3: reranking ────────────────────────────────────────────────────
    if use_reranker and len(fused) > 0:
        # Feed reranker 2x more candidates than final n for better selection
        candidates = fused[: n_results * 2]
        final = rerank(query, candidates, top_n=n_results)
    else:
        final = fused[:n_results]

    # ── Tag retrieval method for transparency ─────────────────────────────────
    for hit in final:
        methods = []
        if hit.get("from_vector"):
            methods.append("vector")
        if hit.get("from_bm25"):
            methods.append("bm25")
        hit["retrieval_method"] = "+".join(methods) if methods else "unknown"

    return final


# ── Quick diagnostics ─────────────────────────────────────────────────────────

def search_debug(query: str, n: int = 5) -> dict:
    """
    Returns all intermediate results for debugging/evaluation.
    Useful during demo to show judges how retrieval works.
    """
    vector_hits = vectorstore.query(query, n_results=10)
    bm25_hits   = bm25_index.get_bm25().query(query, n_results=10)
    fused       = reciprocal_rank_fusion(vector_hits, bm25_hits)
    reranked    = rerank(query, fused[:n*2], top_n=n) if fused else []

    return {
        "query": query,
        "vector_top5":  [{"source": h["source"], "score": h.get("score")} for h in vector_hits[:5]],
        "bm25_top5":    [{"source": h["source"], "bm25_score": h.get("bm25_score")} for h in bm25_hits[:5]],
        "rrf_top5":     [{"source": h["source"], "rrf_score": h.get("rrf_score")} for h in fused[:5]],
        "reranked_top5":[{"source": h["source"], "rerank_score": h.get("rerank_score")} for h in reranked[:5]],
    }
