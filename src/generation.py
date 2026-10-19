# src/generation.py
"""
Robust offline answer extractor for CRAG-style QA.
- Uses spaCy NER (person/org/gpe/date/cardinal) when available.
- Uses question-type heuristics (who/when/where/how many/which/what).
- Scans the top retrieved chunks (sorted by score) and ranks candidates.
- Falls back to regex heuristics if spaCy is not installed.
"""

import re
from typing import List, Tuple, Optional
import math

# Try to import spaCy; if not available, we'll fallback to regex heuristics.
try:
    import spacy

    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

# Patterns
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NUMBER_RE = re.compile(r"\b\d{1,6}\b")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


def _qtype(query: str) -> str:
    q = query.lower().strip()
    if q.startswith("who") or " who " in q:
        return "who"
    if q.startswith("when") or "year" in q or "when" in q:
        return "when"
    if q.startswith("where") or " country" in q or "city" in q or "located" in q:
        return "where"
    if q.startswith("how many") or "how many" in q or "number of" in q:
        return "how_many"
    if q.startswith("which") or q.startswith("what"):
        # rough: treat `who`/`where`/`when` earlier. remaining -> what/which
        return "what"
    # fallback
    return "what"


def _candidates_from_spacy(text: str):
    doc = _NLP(text)
    cands = []
    for ent in doc.ents:
        # keep common types
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "NORP", "EVENT", "PRODUCT"):
            cands.append((ent.text.strip(), ent.label_))
    # also fallback: proper noun chunks
    for chunk in doc.noun_chunks:
        # ignore too short pronouns etc
        if len(chunk.text.split()) <= 4:
            cands.append((chunk.text.strip(), "NP"))
    return cands


def _candidates_from_regex(text: str):
    cands = []
    # entity-like multiword capitalized phrases
    for m in ENTITY_RE.findall(text):
        cands.append((m.strip(), "ENTITY"))
    # years
    for m in YEAR_RE.findall(text):
        cands.append((m[0], "YEAR"))  # YEAR_RE groups first digits; we'll handle properly later
    # numbers
    for m in NUMBER_RE.findall(text):
        cands.append((m.strip(), "NUMBER"))
    return cands


def _score_candidate(candidate: str, ent_type: str, query_type: str, passage_score: float, query: str) -> float:
    """
    Heuristic scoring:
      - prefer types that match query_type (who -> PERSON, when -> DATE/YEAR, where -> GPE/LOC)
      - prefer candidates found in higher-scoring passages (passage_score from FAISS)
      - prefer multi-word names for persons
      - small boost if candidate tokens overlap query tokens (rare)
    """
    score = 0.0

    # base from passage similarity (already cosine-like, typically in [0,1])
    score += max(0.0, passage_score)

    # type match bonuses
    qt = query_type
    et = ent_type.lower() if ent_type else ""

    if qt == "who" and et in ("person", "nor p", "np", "entity"):
        score += 2.0
    if qt == "when" and (et in ("date", "year", "cardinal") or YEAR_RE.search(candidate)):
        score += 2.0
    if qt == "where" and et in ("gpe", "loc", "org", "np"):
        score += 1.8
    if qt == "how_many" and (et in ("cardinal", "number") or NUMBER_RE.search(candidate)):
        score += 2.2

    # prefer longer multi-token person names (reduce single token noisy picks)
    tokens = candidate.split()
    if len(tokens) >= 2 and et in ("person", "np", "entity"):
        score += 0.9

    # small overlap boost if candidate shares words with query (e.g., "United States" ~ "United States")
    qset = set([t.lower() for t in re.findall(r"\w+", query) if len(t) > 2])
    cset = set([t.lower() for t in tokens if len(t) > 2])
    overlap = len(qset & cset)
    if overlap:
        score += 0.4 * overlap

    # length penalty: too long phrases less likely
    if len(tokens) > 5:
        score -= 0.6

    return score


def generate_answer(query: str, retrieved: List[Tuple[str, float]], top_chunks_to_scan: int = 5) -> Tuple[str, List[dict]]:
    """
    Main function expected by pipelines:
      - query: user query string
      - retrieved: list of (chunk_text, passage_score) typically top-k from retriever
      - returns: (answer_str, sources_list)
    """

    if not retrieved:
        return "", []

    # ensure sorted by passage score descending
    retrieved = sorted(retrieved, key=lambda x: -x[1])

    query_type = _qtype(query)

    candidates = []  # list of (candidate_text, ent_type, passage_score, query_type)

    # scan top-N chunks (safe default 5)
    for chunk_text, passage_score in retrieved[:top_chunks_to_scan]:
        if _NLP:
            ents = _candidates_from_spacy(chunk_text)
        else:
            ents = _candidates_from_regex(chunk_text)

        for cand_text, ent_type in ents:
            # normalize cand_text whitespace
            cand = " ".join(cand_text.split())
            candidates.append((cand, ent_type, passage_score))

    # deduplicate candidates (keep highest passage_score for each)
    best_for_text = {}
    for cand, ent_type, pscore in candidates:
        if cand not in best_for_text or pscore > best_for_text[cand][1]:
            best_for_text[cand] = (ent_type, pscore)

    scored = []
    for cand_text, (ent_type, pscore) in best_for_text.items():
        s = _score_candidate(cand_text, ent_type, query_type, pscore, query)
        scored.append((cand_text, ent_type, pscore, s))

    # if we found candidates, pick the best by score
    if scored:
        scored.sort(key=lambda x: -x[3])
        top = scored[0]
        answer = top[0]
        # sources: top_k chunks returned for provenance
        sources = [{"text": c[:500], "score": s} for c, s in retrieved[:max(3, min(len(retrieved), 5))]]
        return answer, sources

    # fallback heuristics (try directed patterns and year)
    # look through chunks for "directed by X" etc
    for chunk_text, _ in retrieved[:top_chunks_to_scan]:
        m = re.search(r"directed by\s+([A-Z][\w\-\s']{2,60})", chunk_text, flags=re.I)
        if m:
            return m.group(1).strip(" ,."), [{"text": c[:500], "score": s} for c, s in retrieved[:3]]

    # year fallback
    for chunk_text, _ in retrieved[:top_chunks_to_scan]:
        m = YEAR_RE.search(chunk_text)
        if m:
            return m.group(0), [{"text": c[:500], "score": s} for c, s in retrieved[:3]]

    # final fallback: return first 3-gram from top chunk
    fallback = " ".join(retrieved[0][0].split()[:3])
    return fallback, [{"text": c[:500], "score": s} for c, s in retrieved[:3]]