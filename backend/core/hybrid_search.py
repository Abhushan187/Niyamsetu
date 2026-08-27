# backend/core/hybrid_search.py
# ─────────────────────────────────────────────────────────
# Hybrid retrieval: dense vectors + sparse lexical, fused by RRF.
#
# Sits ALONGSIDE core.vectorstore.search(), which is untouched. Nothing
# in the existing query path changes until a caller opts in by calling
# hybrid_search() instead, so the two can be compared on equal terms.
#
# ── WHY ──────────────────────────────────────────────────
# Dense retrieval matches on meaning and is blind to exact tokens. In a
# corpus of government resolutions that is the wrong blindness: the
# discriminating parts of a GR are precisely the literal strings — a GR
# number, a rule citation, a date, a class range, a section heading.
# Two GRs about the same subject differ by their numbers, and a single
# embedding of a 2,400-character chunk cannot keep one number separable
# from another.
#
# BM25 has the opposite bias: it finds the literal token and cannot tell
# a paraphrase from an unrelated sentence. Fusing the two lets each
# cover the other's failure mode.
#
# ── WHY RRF RATHER THAN SCORE BLENDING ───────────────────
# A weighted sum of a cosine similarity and a BM25 score needs the two
# to be on a comparable scale, and they are not: cosine is bounded and
# clusters tightly (this corpus sits in a 0.6-0.85 band), BM25 is
# unbounded and depends on corpus statistics that shift every time a
# document is added. Any fixed blend weight calibrated today is wrong
# after the next upload.
#
# Reciprocal Rank Fusion uses only the ORDER each retriever produced, so
# it needs no calibration, no score normalisation, and no per-corpus
# tuning. A chunk both retrievers rank highly wins; a chunk one ranks
# first and the other ignores still places well. That is the behaviour
# wanted here.
#
# ── CORPUS-AGNOSTIC BY CONSTRUCTION ──────────────────────
# Everything below is built from whatever the FAISS index holds at call
# time. There are no filenames, no document names, no subject strings,
# no regexes for any particular GR's layout. Tokenisation is
# script-agnostic Unicode. A corpus of entirely different documents, in
# a different language, gets the same treatment.
# ─────────────────────────────────────────────────────────

import threading
import unicodedata
from typing import Optional

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from config import settings
from core.keyword_match import normalize
from core.vectorstore import load_store

# Default RRF constant. 60 is the value from the original RRF paper
# (Cormack et al. 2009) and the de-facto default since; it is large
# enough that the top few ranks are not winner-take-all, small enough
# that rank 1 still clearly beats rank 20.
RRF_K = 60

# How deep to go into each retriever before fusing. Fusing only the top
# TOP_K of each list would throw away exactly the information RRF exists
# to use: a chunk that dense ranks 40th and lexical ranks 2nd is the
# case worth rescuing, and it is invisible if both lists are cut at 12.
#
# A multiple of top_k rather than a constant, so raising TOP_K widens the
# pool with it. Clamped to the corpus size by _pool_size().
POOL_MULTIPLIER = 10
POOL_MINIMUM = 50

# Punctuation allowed INSIDE a token. Government reference numbers —
# "मराग्रं-2025/प्र.क्र.58/ई-1093601" — are single identifiers, not seven
# words, and splitting them destroys the most discriminating string in the
# document. A general character set, not a pattern for any particular
# numbering scheme.
_INTRA_TOKEN = "./-"

# Unicode general categories for combining marks: non-spacing (Mn),
# spacing-combining (Mc), enclosing (Me).
#
# These MUST be treated as word characters, and Python's \w does not.
# Devanagari carries its vowels as marks — the ि of निर्णय is Mc and the
# ् is Mn — so a regex built on \w splits every Marathi word at every
# matra ("निर्णय" -> "न", "र", "णय"). That silently reduces BM25 to
# matching consonant fragments. Hence the explicit scanner below rather
# than a \w-based pattern.
_MARK_CATEGORIES = frozenset(("Mn", "Mc", "Me"))

# Whether to fold Devanagari combining marks away before indexing.
#
# OCR of Marathi is unreliable exactly on the matras: the same word comes
# back as शासन in one scan, झासन or शासि in another. Comparing consonant
# skeletons makes those one term, at the cost of also merging genuinely
# different words that share a skeleton.
#
# Which way that trade lands is an empirical question about the corpus,
# not something to assert — see tests/test_hybrid_search.py, which
# measures both settings on the same queries.
FOLD_MARKS = True


def _is_word_char(ch: str) -> bool:
    """Alphanumeric in any script, or a combining mark."""
    return ch.isalnum() or unicodedata.category(ch) in _MARK_CATEGORIES


def tokenize(text: str, fold_marks: bool = None) -> list:
    """
    Splits text into lexical terms, script-agnostically.

    A token is a maximal run of word characters, optionally joined by
    intra-token punctuation, with that punctuation trimmed off the ends.
    Nothing here knows what script it is reading.

    Args:
        text       : any string; None and "" give []
        fold_marks : strip combining marks from each term.
                     Defaults to the module's FOLD_MARKS.

    Returns:
        List of lower-cased terms. Empty for text with no word characters.
    """
    if fold_marks is None:
        fold_marks = FOLD_MARKS

    tokens, current = [], []

    def flush():
        if not current:
            return
        term = "".join(current).strip(_INTRA_TOKEN)
        current.clear()
        if not term:
            return
        if fold_marks:
            term = normalize(term)
        term = term.lower()
        if term:
            tokens.append(term)

    for ch in (text or ""):
        if _is_word_char(ch):
            current.append(ch)
        elif ch in _INTRA_TOKEN and current:
            current.append(ch)
        else:
            flush()
    flush()

    return tokens


class LexicalIndex:
    """
    BM25 over the STORED chunk text.

    Deliberately page_content, not the embedding input built by
    vectorstore.build_embedding_text(). Those differ: the embedding input
    strips boilerplate and prepends a synthetic identity line. The stored
    text is what the document actually says, and lexical matching should
    answer "does this chunk contain the words asked about", which is a
    question about the document — not about how its vector was shaped.

    Built from a list of Documents, with no knowledge of where they came
    from. Pure and API-free: constructing and querying this never touches
    a network, so it is fully testable without a GPU.
    """

    def __init__(self, documents: list):
        self.documents = list(documents)
        corpus = [tokenize(d.page_content) for d in self.documents]
        # A chunk that tokenises to nothing (a scan that OCR'd to
        # punctuation, say) would make BM25Okapi divide by a zero average
        # length. Give it a single sentinel term no query can produce.
        self._corpus = [c if c else ["\x00empty"] for c in corpus]
        self.bm25 = BM25Okapi(self._corpus)

    def __len__(self) -> int:
        return len(self.documents)

    def rank(self, query: str) -> list:
        """
        Scores every chunk against the query.

        Returns:
            List of (position, score) over ALL chunks, best first, where
            position indexes into self.documents. Ties keep corpus order,
            so the output is stable across calls.

            A query whose terms appear nowhere scores every chunk 0.0 and
            still returns a full list — the caller decides what to do
            with an all-zero ranking rather than getting a silent [].
        """
        terms = tokenize(query)
        if not terms:
            return [(i, 0.0) for i in range(len(self.documents))]
        scores = self.bm25.get_scores(terms)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(i, float(scores[i])) for i in order]


# ── Index cache ───────────────────────────────────────────
# Rebuilding BM25 over the corpus on every question is wasted work, but a
# cache that outlived an embed run would score against documents that no
# longer exist. Keyed on the identity of the loaded FAISS store and its
# vector count, both of which change when the index is rebuilt or when
# documents are added or deleted.
_cache_lock = threading.Lock()
_cached_index: Optional["LexicalIndex"] = None
_cached_key = None


def _store_documents(db) -> list:
    """
    Every Document currently in the FAISS store, in index order.

    Reads through index_to_docstore_id so the order matches the vector
    positions, which keeps the lexical and dense sides talking about the
    same corpus even where the docstore's own dict order differs.
    """
    docs = []
    for i in range(db.index.ntotal):
        doc_id = db.index_to_docstore_id.get(i)
        if doc_id is None:
            continue
        doc = db.docstore.search(doc_id)
        if isinstance(doc, Document):
            docs.append(doc)
    return docs


def get_lexical_index(db=None) -> Optional["LexicalIndex"]:
    """
    The BM25 index for the current corpus, built on first use and reused
    until the vector store changes underneath it.

    Returns None when there is no vector store yet, matching what
    vectorstore.search() does in the same situation.
    """
    global _cached_index, _cached_key

    if db is None:
        db = load_store()
    if db is None:
        return None

    key = (id(db), db.index.ntotal)
    with _cache_lock:
        if _cached_index is not None and _cached_key == key:
            return _cached_index
        index = LexicalIndex(_store_documents(db))
        _cached_index, _cached_key = index, key
        return index


def clear_lexical_cache():
    """Drops the cached BM25 index. Mirrors vectorstore.clear_cache()."""
    global _cached_index, _cached_key
    with _cache_lock:
        _cached_index, _cached_key = None, None


# ── Fusion ────────────────────────────────────────────────

def reciprocal_rank_fusion(rankings: list, k: int = RRF_K) -> list:
    """
    Combines several ranked lists into one.

        score(d) = sum over lists of  1 / (k + rank(d))

    with rank 1-based, and a list contributing nothing for a document it
    did not rank at all.

    Args:
        rankings : list of ranked lists. Each is a sequence of keys
                   (anything hashable), best first.
        k        : the RRF constant. Larger flattens the contribution
                   curve; smaller lets the very top ranks dominate.

    Returns:
        List of (key, score), best first. Ties are broken by the best
        rank the key reached in any input list, and then by the key
        itself where keys are mutually comparable — so the result does
        not depend on dict ordering, nor on which order the caller
        happened to pass the input lists in. RRF is symmetric in its
        inputs and the output should be too.

        Where keys are not comparable (mixed types), the final fallback
        is first appearance, which is still deterministic for a given
        input.

    Pure: no I/O, no API, no global state. This is the whole of the
    fusion logic, and it can be tested on synthetic rankings alone.
    """
    scores, best_rank, first_seen = {}, {}, {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in best_rank or rank < best_rank[key]:
                best_rank[key] = rank
            first_seen.setdefault(key, len(first_seen))

    try:
        order = sorted(scores, key=lambda x: (-scores[x], best_rank[x], x))
    except TypeError:
        order = sorted(scores, key=lambda x: (-scores[x], best_rank[x], first_seen[x]))

    return [(key, scores[key]) for key in order]


# ── Public search entry points ────────────────────────────

def _pool_size(top_k: int, corpus_size: int) -> int:
    """How deep to read each retriever before fusing."""
    return min(max(top_k * POOL_MULTIPLIER, POOL_MINIMUM), corpus_size)


def lexical_search(query: str, top_k: int = None) -> list:
    """
    BM25-only retrieval. The sparse counterpart to vectorstore.search().

    Args:
        query : the user's question
        top_k : how many chunks to return (defaults to settings.TOP_K)

    Returns:
        List of (Document, bm25_score), best first. Empty list when the
        vector store is not ready.

    Makes no embedding call, so this half of the system works with no
    GPU and no Ollama running.
    """
    if top_k is None:
        top_k = settings.TOP_K

    index = get_lexical_index()
    if index is None or len(index) == 0:
        return []

    return [(index.documents[i], score)
            for i, score in index.rank(query)[:top_k]]


def hybrid_search(query: str, top_k: int = None, k: int = RRF_K,
                  pool: int = None) -> list:
    """
    Dense and lexical retrieval, run independently and fused by RRF.

    Args:
        query : the user's question
        top_k : how many chunks to return (defaults to settings.TOP_K)
        k     : RRF constant (default 60)
        pool  : how deep to read each retriever before fusing; defaults
                to _pool_size(). Larger costs nothing but CPU.

    Returns:
        List of Documents, best first — the same shape
        vectorstore.search() returns, so a caller can swap one for the
        other without touching anything downstream.

        Degrades rather than failing: if the dense side errors, the
        lexical ranking is returned on its own. Empty list only when
        neither side produced anything.

    Requires an embedding call for the dense half, exactly as
    vectorstore.search() does. The lexical half does not.
    """
    if top_k is None:
        top_k = settings.TOP_K

    db = load_store()
    if db is None:
        return []

    index = get_lexical_index(db)
    if index is None or len(index) == 0:
        return []

    depth = pool if pool is not None else _pool_size(top_k, len(index))

    # Chunks are keyed by their position in the corpus, so the two
    # retrievers agree on identity without relying on Document being
    # hashable or on page_content being unique.
    position = {id(doc): i for i, doc in enumerate(index.documents)}

    dense_ranking = []
    try:
        for doc in db.similarity_search(query, k=depth):
            pos = position.get(id(doc))
            if pos is None:
                # FAISS handed back a Document object the docstore did
                # not give us; fall back to matching on content.
                pos = next((i for i, d in enumerate(index.documents)
                            if d.page_content == doc.page_content), None)
            if pos is not None:
                dense_ranking.append(pos)
    except Exception as e:
        print(f"⚠️  hybrid_search: dense side failed, using lexical only — {e}")

    lexical_ranking = [i for i, _ in index.rank(query)[:depth]]

    rankings = [r for r in (dense_ranking, lexical_ranking) if r]
    if not rankings:
        return []

    fused = reciprocal_rank_fusion(rankings, k=k)
    return [index.documents[pos] for pos, _ in fused[:top_k]]
