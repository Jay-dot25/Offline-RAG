"""
highlight.py — Source sentence highlighting

For each retrieved chunk, identifies which sentences the LLM likely
drew from when generating the answer, then wraps them in HTML <mark> tags.

Approach: word-overlap scoring between answer tokens and chunk sentences.
- Fast (no extra model calls)
- Surprisingly accurate — the LLM tends to paraphrase source sentences,
  so shared content words are a reliable signal
- Threshold tuned to ~40% overlap to reduce false positives

Returns chunks with an additional `highlighted_html` field ready for
direct rendering in the Gradio UI.
"""

import re
from collections import Counter


# ── Text utilities ────────────────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","is","in","it","of","to","and","or","for","on","at","by",
    "be","as","we","he","she","they","this","that","was","are","were","with",
    "has","have","had","not","but","from","its","their","which","who","will",
    "can","may","also","been","than","then","so","if","do","does","did","up",
    "i","you","your","my","our","his","her","these","those","into","about",
    "would","should","could","each","more","all","any","some","no","use",
}

def _content_words(text: str) -> Counter:
    """Extract meaningful words (lowercase, no stopwords, length ≥ 3)."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return Counter(w for w in words if len(w) >= 3 and w not in STOPWORDS)


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    Handles abbreviations roughly — good enough for highlighting purposes.
    """
    # Split on .  !  ? followed by whitespace and a capital letter
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(])", text.strip())
    # Also split on newlines used as sentence separators
    sentences = []
    for seg in raw:
        for line in seg.split("\n"):
            line = line.strip()
            if len(line) > 15:  # skip very short fragments
                sentences.append(line)
    return sentences if sentences else [text]


def _overlap_score(answer_words: Counter, sentence: str) -> float:
    """
    Fraction of a sentence's content words that also appear in the answer.
    Score in [0, 1].  Higher = more likely the answer drew from this sentence.
    """
    sent_words = _content_words(sentence)
    if not sent_words:
        return 0.0
    matched = sum(min(sent_words[w], answer_words[w]) for w in sent_words if w in answer_words)
    return matched / sum(sent_words.values())


# ── HTML escaping ─────────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


# ── Main highlighting function ────────────────────────────────────────────────

def highlight_sources(
    answer: str,
    chunks: list[dict],
    threshold: float = 0.35,
) -> list[dict]:
    """
    For each chunk, produce `highlighted_html`:
    - Sentences with overlap score ≥ threshold are wrapped in <mark>
    - Other sentences rendered as plain text
    - Source + score badge shown above each chunk

    Args:
        answer:    The LLM-generated answer text
        chunks:    Retrieved chunks from hybrid_search()
        threshold: Overlap score [0–1] above which a sentence is highlighted.
                   0.35 works well — catches strong matches, avoids noise.

    Returns:
        Same chunks list with `highlighted_html` added to each dict.
    """
    answer_words = _content_words(answer)

    result = []
    for chunk in chunks:
        text = chunk["text"]
        sentences = _split_sentences(text)

        html_parts = []
        any_highlighted = False

        for sent in sentences:
            score = _overlap_score(answer_words, sent)
            escaped = _escape(sent)

            if score >= threshold:
                # Highlight — darker amber mark, visible in both light/dark mode
                html_parts.append(
                    f'<mark style="background:rgba(239,159,39,0.35);'
                    f'border-radius:3px;padding:1px 3px;" '
                    f'title="overlap score: {score:.2f}">{escaped}</mark>'
                )
                any_highlighted = True
            else:
                html_parts.append(escaped)

        body = " ".join(html_parts)

        # Header badge
        source   = _escape(chunk.get("source", "unknown"))
        method   = chunk.get("retrieval_method", "")
        rr_score = chunk.get("rerank_score", None)
        rrf      = chunk.get("rrf_score", None)

        score_parts = []
        if rr_score is not None:
            score_parts.append(f"rerank: {rr_score:.3f}")
        if rrf is not None:
            score_parts.append(f"rrf: {rrf:.4f}")
        score_str = " &nbsp;·&nbsp; ".join(score_parts)

        highlight_indicator = (
            ' &nbsp;<span style="color:#EF9F27;font-weight:500">● highlighted</span>'
            if any_highlighted else ""
        )

        header = (
            f'<div style="font-size:11px;opacity:0.65;margin-bottom:4px;">'
            f'<strong>{source}</strong> &nbsp;·&nbsp; '
            f'<code>{method}</code> &nbsp;·&nbsp; {score_str}'
            f'{highlight_indicator}</div>'
        )

        chunk_copy = dict(chunk)
        chunk_copy["highlighted_html"] = (
            f'<div style="border-left:3px solid rgba(128,128,128,0.25);'
            f'padding:8px 12px;margin-bottom:12px;font-size:13px;line-height:1.6;">'
            f'{header}{body}</div>'
        )
        result.append(chunk_copy)

    return result


# ── Answer sentence highlighting ──────────────────────────────────────────────

def highlight_answer_citations(answer: str, chunks: list[dict]) -> str:
    """
    Optionally: annotate the answer itself with [Source: filename] tags
    after sentences that have high overlap with a specific chunk.
    Returns markdown-formatted answer with inline citations.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return answer

    annotated = []
    for sent in sentences:
        sent_words = _content_words(sent)
        best_score = 0.0
        best_source = None

        for chunk in chunks:
            chunk_words = _content_words(chunk["text"])
            if not chunk_words:
                continue
            matched = sum(min(sent_words[w], chunk_words[w])
                          for w in sent_words if w in chunk_words)
            score = matched / max(sum(sent_words.values()), 1)
            if score > best_score:
                best_score = score
                best_source = chunk.get("source", "")

        if best_score >= 0.4 and best_source:
            # Append citation inline
            name = best_source.rsplit(".", 1)[0]  # drop extension for brevity
            annotated.append(f"{sent} *[{name}]*")
        else:
            annotated.append(sent)

    return " ".join(annotated)
