# backend/core/ocr.py
# ─────────────────────────────────────────────────────────
# OCR fallback for scanned PDFs.
#
# Two problems, both solved here:
#
#   1. Scanned GRs (photos of pages, no embedded text) return
#      empty strings from PyPDFLoader.
#   2. GRs that DO have a text layer can still return unusable
#      Marathi. The Devanagari in this corpus is authored with
#      subset "Sakal Marathi" fonts whose ToUnicode CMap is
#      broken: U+094D VIRAMA is never declared, so every conjunct
#      collapses, and several distinct glyphs all claim to be
#      U+0930 RA, so the ि matra extracts as र. Measured on
#      GR1.pdf: "निर्णय" -> "रनणणय", "सदस्य" -> "सदसय",
#      "परिषद" -> "पररषद". Verified identical under pypdf,
#      PyMuPDF, pdfplumber and pdfminer.six, so it is the file's
#      own CMap, not a library or normalization issue — NFC/NFD/
#      NFKC/NFKD are all exact no-ops on it.
#
# Both cases are handled by converting the page to an image and
# running Tesseract OCR (English + Marathi) to produce real text.
#
# Called by:
#   core/vectorstore.py → embed_all_pdfs() uses this instead
#                          of raw PyPDFLoader when pages look scanned
# ─────────────────────────────────────────────────────────

import sys
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPopplerTimeoutError
import pytesseract

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Below this many real characters, a page is treated as scanned/image-only
MIN_REAL_TEXT_CHARS = 20

# Devanagari block. Any codepoint in this range in the text layer means the
# page carries Marathi, and therefore that its text layer cannot be trusted
# (see the CMap defect described in the header).
DEVANAGARI_START = 0x0900
DEVANAGARI_END   = 0x097F

# Devanagari digits ०-९ (U+0966-U+096F) -> ASCII 0-9.
# Tesseract reads the numerals in a Marathi page as Devanagari digits, so
# "GR क्रमांक ... 2025" comes back as "२०२५". Users type GR numbers, dates
# and संकेतांक in ASCII, so the stored text has to match that to be findable.
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Tesseract language codes: eng = English, mar = Marathi (Devanagari)
# Requires the Marathi language pack installed alongside Tesseract itself
OCR_LANGUAGES = "eng+mar"

# Seconds to let poppler rasterize ONE page before giving up on it.
#
# Without this, convert_from_path waits forever. Observed on this machine:
# a pdftoppm rendering page 1 of a 2-page GR sat for 42 minutes at 0.22s
# of CPU and zero I/O, with the Python parent blocked in communicate() and
# its shell already gone — an orphan that had to be killed by PID. Nothing
# downstream can recover from that: an embed job hangs at the OCR stage
# and never reaches the Ollama call, so EMBED_TIMEOUT_SECONDS in
# core/vectorstore.py never gets to fire.
#
# pdf2image passes this to Popen.communicate(timeout=...) and, on
# TimeoutExpired, kills the poppler process before raising. So the timeout
# also reaps the child rather than leaving the orphan described above.
#
# Sized per page and deliberately generous: a 300-dpi render of one page
# takes 1-3 seconds here, so 90s is a 30-90x margin. It is a deadlock
# detector, not a performance budget — a page that legitimately needs
# more than a minute and a half does not exist in this corpus, and if one
# ever does, raising this is the correct fix rather than lowering it.
OCR_PAGE_TIMEOUT_SECONDS = 90

# Seconds to let Tesseract recognise ONE page before giving up on it.
#
# The companion to OCR_PAGE_TIMEOUT_SECONDS, and the larger share of the
# work: of the ~3.7s a page takes here, the poppler render is sub-second
# and Tesseract is the rest. 60s is therefore roughly a 20x margin —
# again a hang detector, not a performance budget.
#
# pytesseract passes this to Popen.communicate(timeout=...) and, on
# TimeoutExpired, terminates then kills the tesseract process before
# raising, so this reaps the child as the poppler timeout does.
#
# NOTE the two timeouts are sequential within one page, so the worst case
# for a single page is their sum, ~150s. That bounds a page, not a job: a
# document that wedges on every page still takes pages x 150s to fail.
OCR_TESSERACT_TIMEOUT_SECONDS = 60

# pytesseract has no dedicated timeout exception — timeout_manager raises
# a bare RuntimeError carrying exactly this string. Matching on the
# message is unpleasant but it is the only signal available, and the
# alternative (treating every RuntimeError as a timeout) would mislabel
# TesseractError, which is a real OCR failure and also a RuntimeError.
# If a pytesseract upgrade changes this wording the guard below stops
# recognising timeouts and they degrade to the generic failure report —
# noisier, but never wrong.
TESSERACT_TIMEOUT_MESSAGE = "Tesseract process timeout"


def _page_looks_scanned(text: str) -> bool:
    """
    Heuristic: if a page's extracted text is empty or near-empty,
    it's almost certainly a scanned image with no real text layer.
    """
    return len(text.strip()) < MIN_REAL_TEXT_CHARS


def _contains_devanagari(text: str) -> bool:
    """
    True if the text layer contains any Devanagari codepoint.

    Used as the OCR trigger for Marathi pages. Deliberately ANY, not a
    ratio: a page with a corrupted Marathi paragraph and a lot of English
    still needs OCR, and the corrupted extraction is itself made of
    Devanagari letters, so it always trips this.
    """
    return any(DEVANAGARI_START <= ord(ch) <= DEVANAGARI_END for ch in text)


def _normalize_devanagari_digits(text: str) -> str:
    """
    Rewrites Devanagari numerals ०-९ as ASCII 0-9.

    Applied to OCR output only. Digits are the one thing OCR gets
    "right" in a way that hurts: the source PDF's own text layer
    renders them as ASCII, users type them as ASCII, and GR numbers,
    dates and संकेतांक are exactly what gets searched numerically.

    Touches nothing but the ten digit codepoints — letters, matras and
    punctuation pass through untouched.
    """
    return text.translate(DEVANAGARI_DIGITS)


def _page_needs_ocr(text: str) -> tuple[bool, str]:
    """
    Decides whether a page must be re-read with OCR.

    Two independent reasons, checked in order:
        "devanagari" : the text layer contains Marathi. It is corrupted
                       by the font defect described at the top of this
                       file, no matter how much of it there is, so the
                       text layer is discarded outright.
        "scanned"    : the text layer is empty or near-empty — the
                       original reason this module existed.

    Returns:
        (needs_ocr, reason) — reason is "" when no OCR is needed.
    """
    if _contains_devanagari(text):
        return True, "devanagari"
    if _page_looks_scanned(text):
        return True, "scanned"
    return False, ""


def _ocr_single_page(pdf_path: str, page_number: int) -> str:
    """
    Converts one specific PDF page to an image, then runs Tesseract OCR.
    page_number is 1-indexed (matches pdf2image's first_page/last_page).

    Returns:
        Recognized text string (may contain OCR errors — this is expected)
    """
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=300,  # higher DPI improves OCR accuracy, especially for Devanagari
            # Bounds the poppler subprocess. See OCR_PAGE_TIMEOUT_SECONDS —
            # without it a wedged pdftoppm hangs the whole embed job.
            timeout=OCR_PAGE_TIMEOUT_SECONDS,
        )
        if not images:
            return ""

        # Bounds the tesseract subprocess, for the same reason the render
        # above is bounded. Default is timeout=0, which means "wait forever".
        text = pytesseract.image_to_string(
            images[0],
            lang=OCR_LANGUAGES,
            timeout=OCR_TESSERACT_TIMEOUT_SECONDS,
        )
        # ASCII-ise numerals before the text goes anywhere else, so every
        # consumer (chunking, embedding, citations, summaries) sees the same
        # digit form the user will type.
        return _normalize_devanagari_digits(text.strip())

    except PDFPopplerTimeoutError:
        # Reported separately from every other OCR failure, and before the
        # generic handler that would otherwise swallow it — this one is an
        # infrastructure fault, not a page the OCR could not read, and the
        # two want completely different responses from whoever reads the
        # log. Named with the file so a hung page is findable in a run that
        # touched fifty documents.
        print(f"  ⛔ OCR TIMEOUT on {Path(pdf_path).name} page {page_number} — "
              f"poppler exceeded {OCR_PAGE_TIMEOUT_SECONDS}s and was killed. "
              f"This page is being skipped; the rest of the job continues.",
              flush=True)
        return ""

    except RuntimeError as e:
        # Tesseract's timeout arrives as a bare RuntimeError, so the
        # message is what separates it from TesseractError and any other
        # RuntimeError — see TESSERACT_TIMEOUT_MESSAGE. Anything that is
        # not the timeout is reported exactly as the generic handler
        # below would report it, rather than being mislabelled.
        if str(e) != TESSERACT_TIMEOUT_MESSAGE:
            print(f"  ⚠️ OCR failed on page {page_number}: {e}")
            return ""

        print(f"  ⛔ OCR TIMEOUT on {Path(pdf_path).name} page {page_number} — "
              f"tesseract exceeded {OCR_TESSERACT_TIMEOUT_SECONDS}s and was killed. "
              f"This page is being skipped; the rest of the job continues.",
              flush=True)
        return ""

    except Exception as e:
        print(f"  ⚠️ OCR failed on page {page_number}: {e}")
        return ""


def load_pdf_with_ocr_fallback(pdf_path: str) -> list[Document]:
    """
    Loads a PDF page by page. For each page:
        - reads the text layer first (fast)
        - if that page contains Devanagari, DISCARDS the text layer and
          uses OCR instead — the layer is corrupted for Marathi in this
          corpus, and a high character count is not evidence it is sound
        - if that page is empty or near-empty, falls back to OCR as before
        - otherwise (pure Latin/English) keeps the text layer untouched

    This is a drop-in replacement for PyPDFLoader(pdf_path).load() —
    returns the same list[Document] shape, so nothing downstream
    (chunking, embedding, citations) needs to change.

    Returns:
        List of Document objects, one per page, each with
        page_content and metadata including "page", "ocr_used" (bool)
        and, when OCR ran, "ocr_reason" ("devanagari" or "scanned")
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    result_pages = []
    ocr_page_count = 0
    reason_counts  = {"devanagari": 0, "scanned": 0}

    for page in pages:
        real_text = page.page_content
        needs_ocr, reason = _page_needs_ocr(real_text)

        if needs_ocr:
            page_number = page.metadata.get("page", 0) + 1  # PyPDFLoader is 0-indexed
            ocr_text = _ocr_single_page(pdf_path, page_number)

            if ocr_text:
                ocr_page_count += 1
                reason_counts[reason] += 1
                # The text layer is dropped entirely for this page — for a
                # Devanagari page it is the corrupted text, and keeping any
                # of it would put the broken spellings back into the index.
                new_doc = Document(
                    page_content=ocr_text,
                    metadata={**page.metadata, "ocr_used": True, "ocr_reason": reason},
                )
                result_pages.append(new_doc)
            else:
                # OCR produced nothing. Keep whatever the text layer held
                # rather than dropping the page: for a scanned page that is
                # the empty string this branch always kept, and for a
                # Devanagari page corrupted text still beats no text at all
                # while keeping page numbering intact. Flagged loudly so a
                # page that silently kept the bad extraction is findable.
                page.metadata["ocr_used"]   = True
                page.metadata["ocr_reason"] = reason
                page.metadata["ocr_failed"] = True
                if reason == "devanagari":
                    print(f"  ⚠️ OCR failed on {Path(pdf_path).name} page {page_number} — "
                          f"falling back to the CORRUPTED text layer for this page",
                          flush=True)
                result_pages.append(page)
        else:
            page.metadata["ocr_used"] = False
            result_pages.append(page)

    if ocr_page_count > 0:
        print(f"  🔍 OCR applied to {ocr_page_count}/{len(pages)} page(s) in "
              f"{Path(pdf_path).name} "
              f"(devanagari: {reason_counts['devanagari']}, scanned: {reason_counts['scanned']})",
              flush=True)

    return result_pages