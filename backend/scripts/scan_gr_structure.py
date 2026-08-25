#!/usr/bin/env python
# backend/scripts/scan_gr_structure.py
# ─────────────────────────────────────────────────────────
# Standalone structural scanner for GR PDFs. No LLM, no FastAPI.
#
# Reads every PDF in a folder using the SAME loader the real pipeline
# uses (core.ocr.load_pdf_with_ocr_fallback), then reports, per document:
#
#   * which candidate section-header keywords appear, and in what order
#   * the first N characters, to eyeball the genre without opening the PDF
#   * whether a "प्रत" section appears near the end of the document
#
# It REPORTS ONLY. It does not classify, score, or guess the genre —
# that judgement is left to whoever reads the output.
#
# Usage:
#   python scripts/scan_gr_structure.py
#   python scripts/scan_gr_structure.py --folder data/grdocs --format json
#   python scripts/scan_gr_structure.py --out /tmp/scan.csv --tail-pct 30
#
# ── WHY THERE ARE TWO SETS OF MATCH COLUMNS ──────────────
# GR PDFs embed legacy non-Unicode Marathi fonts, so extraction corrupts
# vowel signs and literal matching finds almost nothing. Each keyword is
# therefore reported twice — exact_* (literal, sparse but certain) and
# fuzzy_* (mark-stripped, catches corrupted spellings, can false-positive).
# They are never merged into one verdict; every fuzzy hit ships with the
# snippet it matched so you can confirm or reject it by eye.
#
# The matching itself lives in core/keyword_match.py and is shared — this
# script only decides WHICH keywords to look for and how to report them.
# See that module's header for the full rationale.
# ─────────────────────────────────────────────────────────

import argparse
import csv
import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.ocr import load_pdf_with_ocr_fallback
from core.keyword_match import (
    match_keywords,
    SECTION_KEYWORDS,
    FOOTER_KEYWORD,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_MIN_FUZZY_LEN,
)


# ── Candidate section headers ─────────────────────────────
# Same list, same order as before — it now lives in core/keyword_match.py
# so that core/vectorstore.py chunks on exactly the markers this script
# reports on. Extend it there; nothing here needs to change.
#
# Order in this list does not matter; report order is by the position
# each keyword is found at in the document.
KEYWORDS = list(SECTION_KEYWORDS)

# The keyword whose position near the end of the document is reported
# separately (distribution list on a GR usually trails the signature).
# Note this strips to a 3-character skeleton, below the default
# --min-fuzzy-len, so tail_fuzzy_* is empty unless that guard is lowered.
TAIL_KEYWORD = FOOTER_KEYWORD


def scan_document(pdf_path: Path, args) -> dict:
    """
    Extracts one document and reports its structural signals.
    Never raises — extraction failures are recorded in `extract_error`.
    """
    row = {
        "filename":      pdf_path.name,
        "pages":         0,
        "total_chars":   0,
        "extract_error": "",
        "preview":       "",
    }

    try:
        documents = load_pdf_with_ocr_fallback(str(pdf_path))
        text = "\n".join(d.page_content for d in documents)
        row["pages"] = len(documents)
    except Exception as e:
        row["extract_error"] = f"{type(e).__name__}: {e}"
        return row

    row["total_chars"] = len(text)
    # Newlines flattened so the preview stays on one CSV line.
    row["preview"] = text[:args.preview_chars].replace("\n", " ").strip()

    if not text.strip():
        row["extract_error"] = "no text extracted"
        return row

    # All matching is delegated to core.keyword_match — this script owns
    # only the keyword list and the reporting shape.
    report = match_keywords(
        text,
        KEYWORDS,
        threshold=args.fuzzy_threshold,
        min_len=args.min_fuzzy_len,
    )

    for kw in KEYWORDS:
        em = report.exact.get(kw)
        row[f"exact_count__{kw}"] = em.count    if em else 0
        row[f"exact_pos__{kw}"]   = em.position if em else ""

        fm = report.fuzzy.get(kw)
        row[f"fuzzy_pos__{kw}"]     = fm.position if fm else ""
        row[f"fuzzy_score__{kw}"]   = fm.score    if fm else ""
        row[f"fuzzy_snippet__{kw}"] = fm.snippet  if fm else ""

    # ── order of appearance ───────────────────────────────
    row["exact_found_count"] = len(report.exact_order)
    row["exact_order"]       = " > ".join(report.exact_order)
    row["fuzzy_found_count"] = len(report.fuzzy_order)
    row["fuzzy_order"]       = " > ".join(report.fuzzy_order)

    # ── TAIL_KEYWORD near the end? ────────────────────────
    # "near the end" = within the last --tail-pct of the character stream.
    # For exact, the LAST occurrence is used (position_ratio defaults to
    # use_last=True), since a distribution list is what we are looking for
    # and the word can also appear mid-document.
    for kind in ("exact", "fuzzy"):
        ratio = report.position_ratio(TAIL_KEYWORD, kind=kind)
        row[f"tail_{kind}_found"]     = ratio is not None
        row[f"tail_{kind}_pos_ratio"] = ratio if ratio is not None else ""
        row[f"tail_{kind}_near_end"]  = report.is_near_end(
            TAIL_KEYWORD, tail_pct=args.tail_pct, kind=kind
        )

    return row


def build_fieldnames() -> list:
    """Stable column order — summary first, then per-keyword detail."""
    cols = [
        "filename", "pages", "total_chars", "extract_error", "preview",
        "exact_found_count", "exact_order",
        "fuzzy_found_count", "fuzzy_order",
        "tail_exact_found", "tail_exact_pos_ratio", "tail_exact_near_end",
        "tail_fuzzy_found", "tail_fuzzy_pos_ratio", "tail_fuzzy_near_end",
    ]
    for kw in KEYWORDS:
        cols += [f"exact_count__{kw}", f"exact_pos__{kw}"]
    for kw in KEYWORDS:
        cols += [f"fuzzy_pos__{kw}", f"fuzzy_score__{kw}", f"fuzzy_snippet__{kw}"]
    return cols


def main():
    ap = argparse.ArgumentParser(
        description="Report structural keyword signals for GR PDFs. Reports only — no genre classification.",
    )
    ap.add_argument("--folder", type=Path, default=settings.GRDOCS_PATH,
                    help="folder of PDFs to scan (default: settings.GRDOCS_PATH)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default: gr_structure_scan.<ext> in CWD)")
    ap.add_argument("--format", choices=["csv", "json"], default="csv")
    ap.add_argument("--preview-chars", type=int, default=200,
                    help="how many leading characters to report (default: 200)")
    ap.add_argument("--tail-pct", type=float, default=25.0,
                    help="'near the end' means the last N%% of characters (default: 25)")
    ap.add_argument("--fuzzy-threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD,
                    help="minimum difflib ratio for a fuzzy hit, 0-1 "
                         f"(default: {DEFAULT_FUZZY_THRESHOLD})")
    ap.add_argument("--min-fuzzy-len", type=int, default=DEFAULT_MIN_FUZZY_LEN,
                    help="skip fuzzy matching for keywords whose mark-stripped form is "
                         f"shorter than this (default: {DEFAULT_MIN_FUZZY_LEN}; lowering "
                         "it enables fuzzy matching of short keywords such as the tail "
                         "keyword, at the cost of false positives)")
    args = ap.parse_args()

    folder = args.folder
    if not folder.is_dir():
        print(f"error: not a folder: {folder}", file=sys.stderr)
        return 2

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"error: no PDFs found in {folder}", file=sys.stderr)
        return 2

    out = args.out or Path.cwd() / f"gr_structure_scan.{args.format}"

    print(f"scanning {len(pdfs)} PDF(s) in {folder}")
    print(f"keywords ({len(KEYWORDS)}): {', '.join(KEYWORDS)}")
    print(f"tail window: last {args.tail_pct}%  |  fuzzy threshold: {args.fuzzy_threshold}"
          f"  |  min fuzzy len: {args.min_fuzzy_len}\n")

    rows = []
    for n, pdf in enumerate(pdfs, 1):
        print(f"[{n}/{len(pdfs)}] {pdf.name} ... ", end="", flush=True)
        row = scan_document(pdf, args)
        rows.append(row)
        if row["extract_error"]:
            print(f"ERROR: {row['extract_error']}")
        else:
            print(f"{row['total_chars']} chars, "
                  f"exact={row['exact_found_count']}/{len(KEYWORDS)} "
                  f"fuzzy={row['fuzzy_found_count']}/{len(KEYWORDS)}")

    fieldnames = build_fieldnames()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        # utf-8-sig so Excel opens the Devanagari columns correctly.
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
    else:
        payload = {
            "scanned_folder":  str(folder),
            "keywords":        KEYWORDS,
            "tail_keyword":    TAIL_KEYWORD,
            "tail_pct":        args.tail_pct,
            "fuzzy_threshold": args.fuzzy_threshold,
            "min_fuzzy_len":   args.min_fuzzy_len,
            "preview_chars":   args.preview_chars,
            "note": ("exact_* is literal substring matching. fuzzy_* strips Devanagari "
                     "combining marks to survive legacy-font extraction corruption and "
                     "may yield false positives — check fuzzy_snippet__* before trusting "
                     "a hit. No genre classification is performed."),
            "documents":       [{k: row.get(k, "") for k in fieldnames} for row in rows],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {len(rows)} row(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
