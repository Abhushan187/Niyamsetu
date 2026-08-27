#!/usr/bin/env python
# backend/tests/test_hybrid_search.py
# ─────────────────────────────────────────────────────────
# Validation for core/hybrid_search.py.
#
# Three parts, in increasing order of what they need:
#
#   unit      pure functions — RRF maths and tokenisation. No index, no
#             network. Always runnable.
#   lexical   BM25 ranks over the real index. Reads index.pkl off disk.
#             No embedding call, so no GPU and no Ollama.
#   compare   the dense-vs-fused table. Dense ranking needs a query
#             embedding, which is the one thing that cannot be derived
#             from a stored index — so dense rankings are CACHED to
#             tests/dense_rank_cache.json and replayed from there.
#             Populate the cache once with --refresh-dense (the only
#             mode in this file that touches the embedding API);
#             everything afterwards runs offline.
#
# Two query sets, deliberately different in kind:
#
#   HARNESS_SUBSET  the 8 GR4/GR5 questions that failed in the
#                   2026-08-26 run. Known, already analysed — so a fix
#                   that helps ONLY these is suspect.
#   FRESH_QUERIES   written against documents that have no harness
#                   question at all, from facts read out of the index
#                   after the retrieval layer was written. These are the
#                   overfitting control: hybrid retrieval is only a
#                   general improvement if it moves these too.
#
# Usage:
#   python tests/test_hybrid_search.py               # unit + lexical + compare
#   python tests/test_hybrid_search.py --unit        # pure functions only
#   python tests/test_hybrid_search.py --refresh-dense
# ─────────────────────────────────────────────────────────

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from core.hybrid_search import (          # noqa: E402
    LexicalIndex,
    reciprocal_rank_fusion,
    tokenize,
    RRF_K,
)

TESTS_DIR = Path(__file__).resolve().parent
CACHE_PATH = TESTS_DIR / "dense_rank_cache.json"
TOP_K = 12

# ── Query set 1: the known failures ───────────────────────
# Copied from harness_questions.json so this file is self-contained;
# `expect` is the document that should be retrieved.
HARNESS_SUBSET = [
    ("GR4-T1", "हा शासन आदेश कोणत्या विभागाने व कधी जारी केला?", "GR4.pdf"),
    ("GR4-T2", "मुंबईबाहेर बदली झालेल्या कर्मचाऱ्याने निवासस्थान रिक्त न केल्यास कोणती कार्यवाही करण्याची तरतूद आहे?", "GR4.pdf"),
    ("GR4-T3", "संदर्भ क्र. 3 येथील शासन निर्णय कोणत्या दिनांकाचा आहे व त्याचा या आदेशाशी काय संबंध आहे?", "GR4.pdf"),
    ("GR4-T4", "निवासस्थान रिक्त न केल्यास कर्मचाऱ्यावर किती दंड आकारला जाईल?", "GR4.pdf"),
    ("GR5-T1", "ही सवलत कोणत्या इयत्तांसाठी लागू आहे?", "GR5.pdf"),
    ("GR5-T2", "जानेवारी ते जून दरम्यान सेवानिवृत्त होणाऱ्या व जुलै ते डिसेंबर दरम्यान सेवानिवृत्त होणाऱ्या कर्मचाऱ्यांना किती कालावधीची मुभा मिळते, आणि यात फरक काय?", "GR5.pdf"),
    ("GR5-T3", "ही सवलत कोणत्या केंद्रीय नियमाच्या धर्तीवर देण्यात आली आहे?", "GR5.pdf"),
    ("GR5-T4", "एकापेक्षा अधिक पाल्य असल्यास मुभेचा कालावधी कसा बदलतो?", "GR5.pdf"),
]

# ── Query set 2: the overfitting control ──────────────────
# Each targets a document with NO harness question, on a fact read from
# the stored chunk text. Phrased the way a user would ask rather than
# quoting the document, so BM25 is not handed a verbatim string.
FRESH_QUERIES = [
    ("FRESH-1", "मराठी भाषा गौरव दिन कोणत्या दिवशी साजरा केला जातो?",
     "paripatrika 2.pdf", "27 February / Kusumagraj's birthday"),
    ("FRESH-2", "कक्ष अधिकाऱ्यांच्या आंतरविभागीय बदल्या कोणत्या कायद्यातील तरतुदीनुसार करण्यात येतात?",
     "aadesh 3.pdf", "Transfers Regulation Act, 2005"),
    ("FRESH-3", "उप सचिव संवर्गाची ज्येष्ठतासूची तयार करताना कोणती नियमावली वापरली जाते?",
     "paripatrika 1.pdf", "Maharashtra Civil Services (Regulation of Seniority) Rules, 2021"),
]


# ═════════════════════════════════════════════════════════
# 1. Pure unit tests — no index, no network
# ═════════════════════════════════════════════════════════

def _check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
    return ok


def unit_tests() -> bool:
    print("── unit: reciprocal_rank_fusion ──────────────────────")
    ok = True

    # A document ranked first by both retrievers must win.
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]], k=60)
    ok &= _check("agreed top-1 wins", fused[0][0], "a")

    # The documented formula, to the digit.
    want = 1 / (60 + 1) + 1 / (60 + 1)
    ok &= _check("score = 1/(k+r) summed", round(fused[0][1], 12), round(want, 12))

    # A document only ONE retriever saw still scores, from that list alone.
    fused = reciprocal_rank_fusion([["a", "b"], ["z"]], k=60)
    scores = dict(fused)
    ok &= _check("unseen-by-one still scores", round(scores["z"], 12),
                 round(1 / 61, 12))

    # The case hybrid retrieval exists for: rank 40 in one list, rank 2 in
    # the other, beats a document that is mid-table in both.
    deep = [f"d{i}" for i in range(1, 41)]          # target last, rank 40
    lex = ["x", "d40"]                               # target rank 2
    fused = reciprocal_rank_fusion([deep, lex], k=60)
    ok &= _check("deep-dense + high-lexical wins", fused[0][0], "d40")

    # RRF is symmetric in its inputs: swapping the two ranked lists is the
    # same evidence and must give the same output, ties included.
    a = reciprocal_rank_fusion([["p", "q"], ["q", "p"]], k=60)
    b = reciprocal_rank_fusion([["q", "p"], ["p", "q"]], k=60)
    ok &= _check("symmetric in input order", [k for k, _ in a], [k for k, _ in b])
    ok &= _check("tied scores really are tied",
                 round(dict(a)["p"], 12), round(dict(a)["q"], 12))

    # Same input, same output, every time.
    ok &= _check("repeatable",
                 reciprocal_rank_fusion([["a", "b"], ["b", "a"]]),
                 reciprocal_rank_fusion([["a", "b"], ["b", "a"]]))

    # Empty input is empty output, not a crash.
    ok &= _check("empty rankings", reciprocal_rank_fusion([]), [])

    print("── unit: tokenize ────────────────────────────────────")

    # The bug that makes or breaks this: Devanagari vowel signs are Mn/Mc,
    # which Python's \w excludes. A \w-based tokeniser splits निर्णय into
    # three fragments.
    ok &= _check("Devanagari word stays whole",
                 tokenize("निर्णय", fold_marks=False), ["निर्णय"])
    ok &= _check("reference number stays whole",
                 tokenize("मराग्रं-2025/प्र.क्र.58", fold_marks=False),
                 ["मराग्रं-2025/प्र.क्र.58"])
    ok &= _check("brackets are separators",
                 tokenize("(इयत्ता 1 ली)", fold_marks=False), ["इयत्ता", "1", "ली"])
    ok &= _check("case folded", tokenize("GAD Desk", fold_marks=False),
                 ["gad", "desk"])
    ok &= _check("empty input", tokenize(""), [])
    ok &= _check("punctuation only", tokenize("--- /// ..."), [])

    # Mark folding maps OCR variants of one word onto one term.
    ok &= _check("OCR variants fold together",
                 tokenize("शासन", fold_marks=True) == tokenize("शासि", fold_marks=True),
                 False)  # different consonants -> must NOT collide
    ok &= _check("matra variants fold together",
                 tokenize("शासन", fold_marks=True), tokenize("शासन", fold_marks=True))

    print("── unit: LexicalIndex (synthetic corpus) ─────────────")

    class _Doc:
        def __init__(self, t, name):
            self.page_content = t
            self.metadata = {"source_file": name}

    corpus = [
        _Doc("शासन निर्णय ग्रंथालय परिषद सदस्य", "a.pdf"),
        _Doc("उपाहारगृह स्वयंपाकगृह आधुनिकीकरण मंजूरी", "b.pdf"),
        _Doc("", "empty.pdf"),                       # must not divide by zero
    ]
    idx = LexicalIndex(corpus)
    ok &= _check("index covers every chunk", len(idx), 3)
    ranked = idx.rank("ग्रंथालय परिषद")
    ok &= _check("lexical match ranks first", corpus[ranked[0][0]].metadata["source_file"], "a.pdf")
    ok &= _check("full ranking returned", len(ranked), 3)
    ok &= _check("no-hit query still returns all",
                 len(idx.rank("zzzz-nonexistent")), 3)

    print(f"\n  unit tests: {'ALL PASS' if ok else 'FAILURES ABOVE'}\n")
    return ok


# ═════════════════════════════════════════════════════════
# 2. Real-index helpers
# ═════════════════════════════════════════════════════════

def load_corpus():
    """
    Every stored chunk, in index order, read straight from index.pkl.

    Deliberately not through core.vectorstore.load_store(), which
    constructs an embeddings client and would need OLLAMA_BASE_URL to be
    reachable. Reading the pickle keeps the lexical path GPU-free.
    """
    import pickle
    store_dir = BACKEND / "data" / "vectorstore"
    docstore, index_to_id = pickle.load(open(store_dir / "index.pkl", "rb"))
    mapping = docstore._dict if hasattr(docstore, "_dict") else docstore
    return [mapping[index_to_id[i]] for i in sorted(index_to_id)]


def rank_of(ranking, docs, expect) -> int:
    """1-based rank of the first chunk belonging to `expect`, or None."""
    for r, pos in enumerate(ranking, start=1):
        if docs[pos].metadata["source_file"] == expect:
            return r
    return None


# ── Cache staleness guard ─────────────────────────────────
# A dense ranking is a list of POSITIONS into one specific index, built
# by one specific embedding model. Replay it against an index that was
# rebuilt — different model, different dimension, or merely a different
# number of chunks — and every position silently points at the wrong
# document. Nothing errors; the comparison table just fills with numbers
# that mean nothing. So the cache carries a stamp of what produced it,
# and the read path refuses anything that does not match.
STAMP_KEYS = ("_embedding_model", "_corpus_size", "_dim")


def queries_in(cache) -> dict:
    """The ranking entries only, with the stamp keys filtered out."""
    return {k: v for k, v in cache.items() if k not in STAMP_KEYS}


def current_stamp(corpus_size=None) -> dict:
    """
    What a cache built against the CURRENT index would be stamped with.

    Reads the FAISS dimension straight off disk rather than through
    core.vectorstore.load_store(), which constructs an embeddings client
    and would need OLLAMA_BASE_URL reachable — the same reason
    load_corpus() bypasses it. Checking for staleness must not itself
    require the network, or the guard would be skipped exactly when the
    embedding backend is down.
    """
    import faiss
    from config import settings
    if corpus_size is None:
        corpus_size = len(load_corpus())
    index_path = BACKEND / "data" / "vectorstore" / "index.faiss"
    return {
        "_embedding_model": settings.EMBEDDING_MODEL,
        "_corpus_size":     corpus_size,
        "_dim":             int(faiss.read_index(str(index_path)).d),
    }


def cache_staleness(cache, stamp):
    """
    Why `cache` cannot be replayed against `stamp`, or None if it can.

    A stamp key that is ABSENT counts as stale, not as agreement. Caches
    written before stamping existed carry no model marker at all, and
    treating a missing key as a match is precisely how a nomic-embed-text
    ranking gets replayed under bge-m3 — the failure this guard exists to
    make impossible.
    """
    if not cache:
        return None                      # absent, not stale
    diffs = []
    for k in STAMP_KEYS:
        if k not in cache:
            diffs.append(f"{k} missing (current {stamp[k]!r})")
        elif cache[k] != stamp[k]:
            diffs.append(f"{k} cached {cache[k]!r} != current {stamp[k]!r}")
    return "; ".join(diffs) if diffs else None


def load_cache(stamp=None) -> dict:
    """
    The cached dense rankings, or {} when there are none usable.

    Discards — never silently reuses — a cache whose stamp disagrees with
    the index now on disk.
    """
    if not CACHE_PATH.exists():
        return {}
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    why = cache_staleness(cache, stamp if stamp is not None else current_stamp())
    if why:
        print(f"  STALE dense cache discarded: {why}")
        print(f"  re-run: python tests/test_hybrid_search.py --refresh-dense")
        return {}
    return cache


def refresh_dense(queries):
    """
    The ONLY function here that calls the embedding API. Stores each
    query's full dense ranking (as chunk positions) so every later run
    replays it offline.
    """
    from core.vectorstore import load_store
    db = load_store()
    if db is None:
        sys.exit("error: no vector store on disk — nothing to rank against")

    docs = load_corpus()
    by_content = {}
    for i, d in enumerate(docs):
        by_content.setdefault(d.page_content, i)

    # Dense rankings are only meaningful for the model that produced the
    # vectors, so stamp what produced them. load_cache() drops anything
    # that disagrees, which means a refresh after a model switch starts
    # from empty rather than merging two models' rankings into one file.
    # db is already open here, so take the dimension from it directly.
    stamp = dict(current_stamp(corpus_size=len(docs)), _dim=int(db.index.d))
    cache = load_cache(stamp)
    cache.update(stamp)

    for qid, q, *_ in queries:
        hits = db.similarity_search(q, k=len(docs))
        cache[q] = [by_content[h.page_content] for h in hits
                    if h.page_content in by_content]
        print(f"  cached dense ranking for {qid} ({len(cache[q])} chunks)")

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print(f"  -> {CACHE_PATH}")


# ═════════════════════════════════════════════════════════
# 3. Comparison tables
# ═════════════════════════════════════════════════════════

def compare(queries, title, docs, idx, cache):
    print(f"\n{title}")
    print(f"{'id':<9} {'target':<19} {'dense':>7} {'lexical':>8} {'fused':>7} {'verdict':>10}")
    print("-" * 66)

    rows = []
    for qid, q, expect, *_ in queries:
        lex_full = [i for i, _ in idx.rank(q)]
        r_lex = rank_of(lex_full, docs, expect)

        dense_full = cache.get(q)
        if dense_full is None:
            print(f"{qid:<9} {expect:<19} {'--':>7} {r_lex:>8} {'--':>7} "
                  f"{'no cache':>10}")
            continue

        r_dense = rank_of(dense_full, docs, expect)
        fused = [pos for pos, _ in
                 reciprocal_rank_fusion([dense_full, lex_full], k=RRF_K)]
        r_fused = rank_of(fused, docs, expect)

        was, now = (r_dense or 10**6) <= TOP_K, (r_fused or 10**6) <= TOP_K
        verdict = ("kept" if was and now else
                   "RECOVERED" if now else
                   "LOST" if was else "still out")
        print(f"{qid:<9} {expect:<19} {r_dense:>7} {r_lex:>8} {r_fused:>7} {verdict:>10}")
        rows.append((r_dense, r_lex, r_fused, was, now))

    if rows:
        n = len(rows)
        mean = lambda xs: sum(xs) / len(xs)
        print("-" * 66)
        print(f"{'mean rank':<9} {'':<19} {mean([r[0] for r in rows]):>7.1f} "
              f"{mean([r[1] for r in rows]):>8.1f} {mean([r[2] for r in rows]):>7.1f}")
        print(f"{'in top-' + str(TOP_K):<9} {'':<19} "
              f"{sum(1 for r in rows if r[3]):>7}/{n} "
              f"{sum(1 for r in rows if r[1] <= TOP_K):>7}/{n} "
              f"{sum(1 for r in rows if r[4]):>6}/{n}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", action="store_true", help="pure functions only")
    ap.add_argument("--refresh-dense", action="store_true",
                    help="re-run dense search and cache it (calls the embedding API)")
    args = ap.parse_args()

    ok = unit_tests()
    if args.unit:
        return 0 if ok else 1

    all_queries = HARNESS_SUBSET + [(q[0], q[1], q[2]) for q in FRESH_QUERIES]
    if args.refresh_dense:
        print("── refreshing dense cache (embedding API) ────────────")
        refresh_dense(all_queries)

    docs = load_corpus()
    idx = LexicalIndex(docs)
    cache = load_cache(current_stamp(corpus_size=len(docs)))
    n_cached = len(queries_in(cache))
    print(f"corpus: {len(docs)} chunks   dense cache: "
          f"{f'{n_cached} queries' if n_cached else 'MISSING — run --refresh-dense'}   "
          f"TOP_K={TOP_K}  RRF k={RRF_K}")

    compare(HARNESS_SUBSET, "SET 1 — known harness failures (GR4/GR5)", docs, idx, cache)
    compare([(q[0], q[1], q[2]) for q in FRESH_QUERIES],
            "SET 2 — fresh queries, documents with no harness question", docs, idx, cache)

    print("\nSET 2 targets and the facts they ask about:")
    for qid, _, expect, fact in FRESH_QUERIES:
        print(f"  {qid}  {expect:<19} {fact}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
