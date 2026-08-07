# backend/scripts/analyze_corpus.py
# ─────────────────────────────────────────────────────────
# Corpus-wide structural analysis of every GR PDF in grdocs/.
#
# Read-only with respect to the pipeline: it IMPORTS the existing
# extraction logic (core/ocr.py) and the existing relationship
# patterns (core/gr_graph.py) rather than reimplementing either, so
# results always track whatever those files currently do. No pipeline
# file is modified.
#
# Usage:
#   python scripts/analyze_corpus.py
#   python scripts/analyze_corpus.py --limit 5           # quick smoke test
#   python scripts/analyze_corpus.py --out other.csv
#
# Output: backend/data/corpus_analysis.csv (UTF-8 with BOM so Excel
#         on Windows renders Devanagari correctly), plus a printed
#         summary.
#
# Requires Tesseract on PATH only if the corpus contains scanned pages.
# ─────────────────────────────────────────────────────────

import sys
import os
import re
import csv
import argparse
from pathlib import Path
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config.py declares MONGODB_URL and JWT_SECRET as required with no
# defaults, so importing anything under core/ without a .env raises a
# ValidationError. Neither is used by this script — it never touches
# Mongo — so placeholders keep the import chain alive.
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("JWT_SECRET", "corpus-analysis-only")

from config import settings
from core.ocr import load_pdf_with_ocr_fallback
from core.language import DEVANAGARI_PATTERN, detect_language
from core.gr_graph import (
    SUPERSEDES_PATTERNS,
    AMENDS_PATTERNS,
    REFERS_PATTERNS,
)


# ══════════════════════════════════════════════════════════════════
# Document type detection
# ══════════════════════════════════════════════════════════════════
# Order matters. These keywords are not mutually exclusive: almost every
# Maharashtra GR quotes "शासन निर्णय" inside its वाचा/संदर्भ block
# regardless of what the document itself is, so the most generic type is
# tested LAST. A corrigendum is tested first because it typically cites
# the GR it corrects.
#
# Two spellings of "corrigendum" are accepted (शुध्दीपत्रक / शुद्धीपत्रक)
# because both appear in practice and OCR flips the conjunct either way.
DOC_TYPE_PATTERNS = [
    ("Corrigendum (शुध्दीपत्रक)", [r"शुध्दीपत्रक", r"शुद्धीपत्रक", r"\bcorrigendum\b"]),
    ("Circular (शासन परिपत्रक)", [r"शासन\s*परिपत्रक", r"\bgovernment\s+circular\b"]),
    ("Order (शासन आदेश)",        [r"शासन\s*आदेश",     r"\bgovernment\s+order\b"]),
    ("Resolution (शासन निर्णय)", [r"शासन\s*निर्णय",   r"\bgovernment\s+resolution\b"]),
]

# The title sits in the header of page 1. Searching the whole document
# would match the reference block and misclassify nearly everything, so
# we scan a header window first and only fall back to the full text.
TITLE_WINDOW_CHARS = 1500


def detect_doc_type(page_one_text: str, full_text: str) -> str:
    for label, patterns in DOC_TYPE_PATTERNS:
        for pat in patterns:
            if re.search(pat, page_one_text[:TITLE_WINDOW_CHARS], re.IGNORECASE | re.UNICODE):
                return label

    # Fallback — title not found in the header window (common when OCR
    # garbles the letterhead). Accept a match anywhere in the document.
    for label, patterns in DOC_TYPE_PATTERNS:
        for pat in patterns:
            if re.search(pat, full_text, re.IGNORECASE | re.UNICODE):
                return label + " [body-match]"

    return "Unknown"


# ══════════════════════════════════════════════════════════════════
# वाचा / संदर्भ reference-block parsing
# ══════════════════════════════════════════════════════════════════
# Maharashtra GRs open with a "वाचा :-" or "संदर्भ :-" block listing the
# prior documents the GR builds on, as a numbered list:
#
#   वाचा :-
#     १) शासन निर्णय क्रमांक ... दिनांक १५ मार्च २०२३
#     २) शासन परिपत्रक क्रमांक ... दिनांक ०२ जून २०२३
#
# Strategy: locate the marker, slice forward to the first terminator
# (or a hard char cap), then count line-initial numbered items.
REF_BLOCK_START = re.compile(r"(वाचा|संदर्भ|संदर्भाधीन)\s*[:：\-–—]*", re.UNICODE)

# The block ends where the GR's own body begins.
REF_BLOCK_END = re.compile(
    r"(प्रस्तावना|प्रस्तावाना|शासन\s*निर्णय\s*[:：\-–—]|"
    r"शासन\s*आदेश\s*[:：\-–—]|महाराष्ट्र\s*शासन|\bpreamble\b)",
    re.IGNORECASE | re.UNICODE,
)

REF_BLOCK_MAX_CHARS = 2500

# A numbered item: optional bracket, then Latin (0-9) or Devanagari
# (०-९, U+0966-U+096F) digits, then a closing delimiter.
NUMBERED_ITEM = re.compile(
    r"^[ \t]*[\(\[]?\s*([0-9]+|[०-९]+)\s*[\)\]\.।:]",
    re.MULTILINE | re.UNICODE,
)


def count_references(full_text: str) -> tuple:
    """
    Returns (reference_count, block_found).

    Counts numbered list items inside the वाचा/संदर्भ block only —
    numbered lists elsewhere in the GR body are not references.
    """
    match = REF_BLOCK_START.search(full_text)
    if not match:
        return 0, False

    start = match.end()
    window = full_text[start:start + REF_BLOCK_MAX_CHARS]

    end_match = REF_BLOCK_END.search(window)
    if end_match:
        window = window[:end_match.start()]

    return len(NUMBERED_ITEM.findall(window)), True


# ══════════════════════════════════════════════════════════════════
# Relationship pattern matching
# ══════════════════════════════════════════════════════════════════
# Patterns are imported from core/gr_graph.py, not copied, so this
# report reflects exactly what the live graph builder would detect.
#
# NOTE: gr_graph.py defines SUPERSEDES / AMENDS / REFERS_TO only.
# There is no EXTENSION pattern list (मुदतवाढ, "extension of time"),
# so extension language is NOT measured here — adding a list to
# gr_graph.py would be a pipeline change, which is out of scope.

def count_pattern_hits(text: str, patterns: list) -> int:
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, re.IGNORECASE | re.UNICODE))
    return total


# ══════════════════════════════════════════════════════════════════
# Table detection heuristic
# ══════════════════════════════════════════════════════════════════
# A line looks tabular if EITHER:
#   (a) it has >= 3 whitespace-separated tokens and >= 2 are numeric
#       (Latin or Devanagari digits, decimals, percentages, currency), or
#   (b) it contains >= 2 column gutters (runs of 3+ spaces)
# A document is judged to contain a table when enough such lines cluster
# together — an isolated numeric line is usually a date or an amount.
NUMERIC_TOKEN = re.compile(r"^[०-९0-9][०-९0-9,.\-/%₹]*$", re.UNICODE)
GUTTER = re.compile(r"\s{3,}")

MIN_TABULAR_LINES = 3      # total tabular lines required
MIN_CONSECUTIVE = 2        # ...at least this many back to back


def analyze_tables(full_text: str) -> tuple:
    """Returns (has_table, tabular_line_count, longest_consecutive_run)."""
    tabular = 0
    run = 0
    longest_run = 0

    for line in full_text.splitlines():
        stripped = line.strip()
        if not stripped:
            run = 0
            continue

        tokens = stripped.split()
        numeric_tokens = sum(1 for t in tokens if NUMERIC_TOKEN.match(t))

        looks_tabular = (
            (len(tokens) >= 3 and numeric_tokens >= 2)
            or len(GUTTER.findall(line)) >= 2
        )

        if looks_tabular:
            tabular += 1
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0

    has_table = tabular >= MIN_TABULAR_LINES and longest_run >= MIN_CONSECUTIVE
    return has_table, tabular, longest_run


# ══════════════════════════════════════════════════════════════════
# Language mix
# ══════════════════════════════════════════════════════════════════
# Same formula as core/language.py detect_language(): Devanagari chars
# over total non-whitespace chars. DEVANAGARI_PATTERN is imported so the
# two can never drift apart.

def devanagari_ratio(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    devanagari = len(DEVANAGARI_PATTERN.findall(text))
    total = len([c for c in text if c.strip() and not c.isspace()])
    if total == 0:
        return 0.0
    return round(devanagari / total, 4)


# ══════════════════════════════════════════════════════════════════
# Per-file analysis
# ══════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "filename", "pages", "total_chars", "ocr_pages", "ocr_failed_pages",
    "doc_type", "ref_block_found", "ref_count",
    "matches_supersedes", "matches_amends", "matches_refers_to",
    "supersedes_hits", "amends_hits", "refers_to_hits",
    "has_table", "tabular_lines", "longest_tabular_run",
    "devanagari_ratio", "language", "error",
]


def analyze_pdf(pdf_path: Path) -> dict:
    row = {k: "" for k in CSV_FIELDS}
    row["filename"] = pdf_path.name

    try:
        pages = load_pdf_with_ocr_fallback(str(pdf_path))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row

    page_texts = [p.page_content for p in pages]
    full_text = "\n".join(page_texts)
    page_one = page_texts[0] if page_texts else ""

    row["pages"] = len(pages)
    row["total_chars"] = len(full_text)
    row["ocr_pages"] = sum(1 for p in pages if p.metadata.get("ocr_used"))
    row["ocr_failed_pages"] = sum(1 for p in pages if p.metadata.get("ocr_failed"))

    row["doc_type"] = detect_doc_type(page_one, full_text)

    ref_count, block_found = count_references(full_text)
    row["ref_count"] = ref_count
    row["ref_block_found"] = block_found

    sup = count_pattern_hits(full_text, SUPERSEDES_PATTERNS)
    amd = count_pattern_hits(full_text, AMENDS_PATTERNS)
    ref = count_pattern_hits(full_text, REFERS_PATTERNS)
    row["supersedes_hits"] = sup
    row["amends_hits"] = amd
    row["refers_to_hits"] = ref
    row["matches_supersedes"] = sup > 0
    row["matches_amends"] = amd > 0
    row["matches_refers_to"] = ref > 0

    has_table, tabular_lines, longest_run = analyze_tables(full_text)
    row["has_table"] = has_table
    row["tabular_lines"] = tabular_lines
    row["longest_tabular_run"] = longest_run

    row["devanagari_ratio"] = devanagari_ratio(full_text)
    row["language"] = detect_language(full_text)

    return row


# ══════════════════════════════════════════════════════════════════
# Summary reporting
# ══════════════════════════════════════════════════════════════════

def pct(n: int, total: int) -> str:
    return f"{round((n / total) * 100)}%" if total else "—"


def print_summary(rows: list):
    ok = [r for r in rows if not r["error"]]
    failed = [r for r in rows if r["error"]]
    total = len(ok)

    print("\n" + "=" * 68)
    print("  CORPUS ANALYSIS SUMMARY")
    print("=" * 68)
    print(f"  PDFs found     : {len(rows)}")
    print(f"  Analyzed OK    : {total}")
    print(f"  Failed to load : {len(failed)}")

    if failed:
        for r in failed:
            print(f"      ✕ {r['filename']}: {r['error']}")

    if not total:
        print("=" * 68)
        return

    print("\n  ── Document type ────────────────────────────────────────")
    for doc_type, n in Counter(r["doc_type"] for r in ok).most_common():
        print(f"    {doc_type:<34} {n:>4}  ({pct(n, total)})")

    print("\n  ── Relationship language (patterns from gr_graph.py) ────")
    for label, key in [
        ("supersedes", "matches_supersedes"),
        ("amends", "matches_amends"),
        ("refers_to", "matches_refers_to"),
    ]:
        n = sum(1 for r in ok if r[key])
        print(f"    {label:<34} {n:>4}  ({pct(n, total)})")
    none_matched = sum(
        1 for r in ok
        if not (r["matches_supersedes"] or r["matches_amends"] or r["matches_refers_to"])
    )
    print(f"    {'no relationship detected':<34} {none_matched:>4}  ({pct(none_matched, total)})")
    print("    note: no EXTENSION (मुदतवाढ) pattern list exists in gr_graph.py")

    print("\n  ── Structure ────────────────────────────────────────────")
    with_tables = sum(1 for r in ok if r["has_table"])
    print(f"    {'contains a table':<34} {with_tables:>4}  ({pct(with_tables, total)})")
    with_refs = sum(1 for r in ok if r["ref_block_found"])
    print(f"    {'वाचा/संदर्भ block found':<34} {with_refs:>4}  ({pct(with_refs, total)})")

    print("\n  ── Averages ─────────────────────────────────────────────")
    avg_chars = sum(r["total_chars"] for r in ok) / total
    avg_pages = sum(r["pages"] for r in ok) / total
    avg_refs = sum(r["ref_count"] for r in ok) / total
    print(f"    avg characters per document      {avg_chars:>10,.0f}")
    print(f"    avg pages per document           {avg_pages:>10.1f}")
    print(f"    avg references in वाचा block     {avg_refs:>10.1f}")

    # Directly relevant to the 12,000-char truncation cap in summarizer.py
    over_cap = sum(1 for r in ok if r["total_chars"] > 12000)
    print(f"\n    documents over the 12,000-char summarizer cap: "
          f"{over_cap} ({pct(over_cap, total)})")

    print("\n  ── Language mix ─────────────────────────────────────────")
    for lang, n in Counter(r["language"] for r in ok).most_common():
        print(f"    {lang:<34} {n:>4}  ({pct(n, total)})")
    avg_dev = sum(r["devanagari_ratio"] for r in ok) / total
    print(f"    avg Devanagari ratio             {avg_dev:>10.3f}")

    print("\n  ── OCR ──────────────────────────────────────────────────")
    ocr_docs = sum(1 for r in ok if r["ocr_pages"])
    total_ocr_pages = sum(r["ocr_pages"] for r in ok)
    total_failed = sum(r["ocr_failed_pages"] for r in ok)
    print(f"    {'documents needing OCR':<34} {ocr_docs:>4}  ({pct(ocr_docs, total)})")
    print(f"    {'total pages OCR-ed':<34} {total_ocr_pages:>4}")
    if total_failed:
        print(f"    ⚠️  OCR returned nothing on {total_failed} page(s) — "
              f"is Tesseract installed with the Marathi pack?")
    print("=" * 68 + "\n")


# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze the GR corpus structurally.")
    parser.add_argument("--dir", default=None, help="override grdocs directory")
    parser.add_argument("--out", default=None, help="override output CSV path")
    parser.add_argument("--limit", type=int, default=0, help="analyze only the first N PDFs")
    args = parser.parse_args()

    src_dir = Path(args.dir) if args.dir else settings.GRDOCS_PATH
    out_path = Path(args.out) if args.out else Path(settings.DATA_DIR) / "corpus_analysis.csv"

    if not src_dir.exists():
        print(f"Directory does not exist: {src_dir.resolve()}")
        sys.exit(1)

    pdfs = sorted(src_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[:args.limit]

    if not pdfs:
        print(f"No PDFs found in {src_dir.resolve()}")
        sys.exit(1)

    print(f"Analyzing {len(pdfs)} PDF(s) from {src_dir.resolve()}\n")

    rows = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i}/{len(pdfs)}] {pdf.name}")
        rows.append(analyze_pdf(pdf))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel on Windows renders Devanagari instead of mojibake
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ CSV written: {out_path.resolve()}")
    print_summary(rows)


if __name__ == "__main__":
    main()
