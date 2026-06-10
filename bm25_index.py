"""
bm25_index.py — BM25 keyword index with JSON persistence
Runs alongside ChromaDB for hybrid search.

Why BM25 alongside vectors?
- Vectors: great for semantic similarity ("what causes heart disease" ↔ "cardiac risk factors")
- BM25:    great for exact matches (product codes, names, acronyms, rare terms)
- Together (RRF): catches both — significant retrieval quality improvement
"""

import json
import re
import math
from pathlib import Path
from collections import defaultdict

BM25_STORE = "./vectordb/bm25_store.json"

# ── Tokenizer ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple but effective tokenizer: lowercase, split on non-alphanumeric."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    # Remove single-char tokens and very common stopwords
    stopwords = {"a","an","the","is","in","it","of","to","and","or","for",
                 "on","at","by","be","as","we","he","she","they","this","that"}
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


# ── BM25 parameters ──────────────────────────────────────────────────────────
K1 = 1.5    # Term frequency saturation (1.2–2.0 typical)
B  = 0.75   # Length normalization (0 = none, 1 = full)


# ── Persistent store ─────────────────────────────────────────────────────────

class BM25Index:
    """
    Disk-backed BM25 index.
    Stores documents as a flat list with source metadata.
    Rebuilds inverted index in memory on load (fast for <100k chunks).
    """

    def __init__(self, store_path: str = BM25_STORE):
        self.store_path = store_path
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)

        # Persistent state (serialized to JSON)
        self.docs: list[dict] = []        # [{id, text, source, file_type, tokens}]

        # In-memory index (rebuilt from docs on load)
        self.df: dict[str, int] = {}      # document frequency per term
        self.avgdl: float = 0.0           # average document length

        self._load()
        self._rebuild_index()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if Path(self.store_path).exists():
            with open(self.store_path) as f:
                data = json.load(f)
                self.docs = data.get("docs", [])
            print(f"[BM25] Loaded {len(self.docs)} chunks from disk")
        else:
            self.docs = []

    def _save(self):
        with open(self.store_path, "w") as f:
            json.dump({"docs": self.docs}, f)

    def _rebuild_index(self):
        """Build inverted index from self.docs. Called on init and after mutations."""
        self.df = defaultdict(int)
        total_len = 0

        for doc in self.docs:
            tokens = set(doc["tokens"])   # unique terms per doc for DF
            for t in tokens:
                self.df[t] += 1
            total_len += len(doc["tokens"])

        self.avgdl = total_len / len(self.docs) if self.docs else 1.0

    # ── Write operations ─────────────────────────────────────────────────────

    def add_documents(self, chunks: list[str], filename: str, file_type: str):
        """Add chunks from a file. Skip if already indexed."""
        # Check for duplicates
        existing = {d["source"] for d in self.docs}
        if filename in existing:
            print(f"[BM25] {filename} already in index — skipping")
            return 0

        new_docs = []
        for i, text in enumerate(chunks):
            new_docs.append({
                "id": f"{filename}::{i}",
                "text": text,
                "source": filename,
                "file_type": file_type,
                "tokens": tokenize(text),
            })

        self.docs.extend(new_docs)
        self._rebuild_index()
        self._save()
        print(f"[BM25] Added {len(new_docs)} chunks from {filename}")
        return len(new_docs)

    def delete_document(self, filename: str) -> int:
        before = len(self.docs)
        self.docs = [d for d in self.docs if d["source"] != filename]
        removed = before - len(self.docs)
        if removed:
            self._rebuild_index()
            self._save()
            print(f"[BM25] Removed {removed} chunks for {filename}")
        return removed

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _bm25_score(self, doc_tokens: list[str], query_tokens: list[str]) -> float:
        """Compute BM25 score for one document against a query."""
        N = len(self.docs)
        dl = len(doc_tokens)
        score = 0.0

        # Term frequency in this document
        tf_map: dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            tf_map[t] += 1

        for term in query_tokens:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue

            df = self.df.get(term, 0)
            if df == 0:
                continue

            # IDF (Robertson-Sparck Jones)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            # TF component with length normalization
            tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / self.avgdl))

            score += idf * tf_norm

        return score

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, query_text: str, n_results: int = 10,
              source_filter: str | None = None) -> list[dict]:
        """
        BM25 keyword search.
        Returns top-n chunks sorted by BM25 score (descending).
        """
        if not self.docs:
            return []

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        candidates = self.docs
        if source_filter:
            candidates = [d for d in candidates if d["source"] == source_filter]

        scored = []
        for doc in candidates:
            score = self._bm25_score(doc["tokens"], query_tokens)
            if score > 0:
                scored.append({
                    "text": doc["text"],
                    "source": doc["source"],
                    "file_type": doc["file_type"],
                    "bm25_score": round(score, 4),
                    "id": doc["id"],
                })

        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored[:n_results]

    def stats(self) -> dict:
        sources = {}
        for d in self.docs:
            src = d["source"]
            sources[src] = sources.get(src, 0) + 1
        return {
            "total_chunks": len(self.docs),
            "total_files": len(sources),
            "vocab_size": len(self.df),
            "avg_doc_length": round(self.avgdl, 1),
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_bm25 = None

def get_bm25() -> BM25Index:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Index()
    return _bm25
