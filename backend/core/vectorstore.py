# backend/core/vectorstore.py
# ─────────────────────────────────────────────────────────
# FAISS vector store — embedding and search.
#
# Two main operations:
#   1. embed_all_pdfs()  → reads PDFs, creates embeddings, saves FAISS index
#   2. search()          → finds the most relevant chunks for a query
#
# Called by:
#   api/embed.py  → triggers embed_all_pdfs()
#   core/rag.py   → calls search() before sending to LLM
# ─────────────────────────────────────────────────────────

import sys
import os
import shutil
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# LangChain PDF loader — reads PDF pages into Document objects
from langchain_community.document_loaders import PyPDFLoader

# Splits large documents into smaller overlapping chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS vector store wrapper from LangChain
from langchain_community.vectorstores import FAISS

# Ollama embeddings — converts text to vectors using nomic-embed-text
from langchain_ollama import OllamaEmbeddings

# LangChain Document type — represents one chunk of text with metadata
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Fuzzy Devanagari matching — how the chunker finds section headers in
# text whose vowel signs the legacy-font extraction corrupted.
from core.keyword_match import (
    match_keywords,
    normalize,
    SECTION_KEYWORDS,
    FOOTER_KEYWORD,
)


# ── Embedding diagnostics ─────────────────────────────────
# Added because an embed job can sit in Ollama for an unbounded time with
# NOTHING printed: the progress callback only fires while PDFs are being
# loaded, and the embedding itself is a single blocking call. Without the
# heartbeat below, a hang and a slow run look identical in the terminal.


# Seconds to wait on a single Ollama embed request before giving up.
# ollama.Client defaults to timeout=None, i.e. wait forever — a genuinely
# hung request then blocks the worker thread permanently, _run_embedding_job's
# finally never runs, and _embed_state["running"] stays True until the server
# is restarted. A bounded wait converts that into a normal, reportable failure.
#
# Sized for the whole bulk request, not per chunk: embed_documents() posts
# every chunk in ONE call, so this budget covers all of them together. Raise
# it if a legitimately large corpus starts tripping it.
EMBED_TIMEOUT_SECONDS = 180.0

# Chunks sent to Ollama per HTTP request. One request for the whole corpus
# (266 chunks / ~168k chars) exceeds what this hardware answers inside the
# timeout, so the work is split. This changes ONLY the request size — the
# chunks themselves, their text and their metadata are untouched, and the
# resulting index is the same as a single call would have produced.
#
# The timeout above applies per request, so it is now a per-batch budget.
EMBED_BATCH_SIZE = 25


def _ts() -> str:
    """UTC HH:MM:SS stamp for diagnostic lines."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _log(message: str) -> None:
    """
    Prints a diagnostic line that cannot take the caller down with it.

    stdout here is whatever console the server was started from. On Windows
    that is often cp1252, where printing an emoji raises UnicodeEncodeError.
    The embed path can absorb that — it is already inside a broad try. The
    deletion path cannot: its prints come AFTER the index has been written,
    so an exception there would report a failure for work that actually
    succeeded, and abort the remaining cleanup steps in core/purge.py while
    the vectors are already gone.

    Destructive operations must not depend on the console codepage, so the
    message is re-encoded to ASCII if the terminal cannot take it, and
    dropped entirely rather than raised if even that fails.
    """
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        try:
            print(message.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass
    except Exception:
        pass


class _Heartbeat:
    """
    Prints an elapsed-time line every `interval` seconds while a blocking
    call is in flight, then stops.

    Diagnostic only — it observes, it does not interrupt, time out, or
    cancel anything. A daemon thread so it can never keep the process
    alive on its own.

    Usage:
        with _Heartbeat("embedding 214 chunks"):
            blocking_call()
    """

    def __init__(self, label: str, interval: float = 15.0):
        self.label    = label
        self.interval = interval
        self._stop    = threading.Event()
        self._thread  = None
        self.started  = None

    def _tick(self):
        while not self._stop.wait(self.interval):
            waited = time.time() - self.started
            print(f"  [{_ts()}] ... still waiting on {self.label} "
                  f"— {waited:.0f}s elapsed, no response yet", flush=True)

    def __enter__(self):
        self.started = time.time()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return False


# ── Module-level cache ────────────────────────────────────
# The FAISS index is loaded once and kept in memory.
# Without this, every query would reload the index from disk
# which takes 5-10 seconds and kills performance.
_vector_store: Optional[FAISS] = None


def get_embeddings() -> OllamaEmbeddings:
    """
    Creates an Ollama embeddings instance.
    Uses nomic-embed-text model — good multilingual support.
    Called internally — not used outside this file.
    """
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        # Forwarded to the underlying httpx client inside ollama.Client.
        # Without this the client inherits timeout=None and waits forever.
        client_kwargs={"timeout": EMBED_TIMEOUT_SECONDS},
    )


def is_ready() -> bool:
    """
    Checks if the FAISS index exists on disk.
    Used by the frontend to show "Vector Store Ready" status.

    Returns:
        True if index.faiss exists, False otherwise
    """
    index_file = settings.VECTORSTORE_PATH / "index.faiss"
    return index_file.exists()


def load_store() -> Optional[FAISS]:
    """
    Loads FAISS index from disk into memory.
    Uses module-level cache so it only loads once per session.

    Returns:
        FAISS instance if index exists, None otherwise
    """
    global _vector_store

    # Return cached version if already loaded
    if _vector_store is not None:
        return _vector_store

    # Check if index exists on disk
    if not is_ready():
        return None

    try:
        embeddings = get_embeddings()
        _vector_store = FAISS.load_local(
            str(settings.VECTORSTORE_PATH),
            embeddings,
            # This flag is required by LangChain for security
            # We set it True because we trust our own saved index
            allow_dangerous_deserialization=True,
        )
        return _vector_store
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return None


def clear_cache():
    """
    Clears the in-memory cache.
    Called after embed_all_pdfs() so the next query
    loads the fresh index instead of the old cached one.
    """
    global _vector_store
    _vector_store = None


# ── Index maintenance ─────────────────────────────────────
# Everything below exists to support deleting a document from the index
# without re-embedding the corpus. None of it touches how chunks are made,
# embedded or retrieved — it only edits an index that already exists.


def _load_store_from_disk() -> Optional[FAISS]:
    """
    Loads the index straight from disk, bypassing the module cache.

    Maintenance must never operate on `_vector_store`: that object is the
    one live queries are reading through. Mutating it in place would let a
    concurrent search see a half-deleted index, and a failed save would
    leave memory and disk permanently disagreeing. Editing a private copy
    and swapping the files means the worst case is "the change did not
    happen", never "the index is now wrong".

    Note this needs no working Ollama connection. FAISS.load_local() only
    stores the embeddings object for later use; it does not call it. So
    deletion keeps working when Ollama is down — which a rebuild-based
    delete would not.
    """
    if not is_ready():
        return None

    return FAISS.load_local(
        str(settings.VECTORSTORE_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _save_store_to_disk(store: FAISS) -> None:
    """
    Writes `store` over the on-disk index as close to atomically as two
    files allow, then clears the cache.

    save_local() writes index.faiss and index.pkl separately and truncates
    each in place. Crashing between them leaves vectors and docstore out of
    sync — an index that loads without complaint and returns wrong
    documents, which is the exact silent corruption this whole feature is
    meant to avoid. So: write to a temp folder first, keep a backup of the
    current pair, swap both, and roll back if the swap fails partway.
    """
    settings.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    tmp_dir = settings.VECTORSTORE_PATH.parent / f".vectorstore_tmp_{os.getpid()}"
    bak_dir = settings.VECTORSTORE_PATH.parent / f".vectorstore_bak_{os.getpid()}"
    for d in (tmp_dir, bak_dir):
        shutil.rmtree(d, ignore_errors=True)

    names = ("index.faiss", "index.pkl")

    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(tmp_dir))

        # Back up whatever is currently live so a failed swap can be undone.
        bak_dir.mkdir(parents=True, exist_ok=True)
        backed_up = []
        for name in names:
            current = settings.VECTORSTORE_PATH / name
            if current.exists():
                shutil.copy2(current, bak_dir / name)
                backed_up.append(name)

        try:
            for name in names:
                os.replace(tmp_dir / name, settings.VECTORSTORE_PATH / name)
        except Exception:
            # Put the previous pair back so we never leave a mismatched index.
            for name in backed_up:
                try:
                    shutil.copy2(bak_dir / name, settings.VECTORSTORE_PATH / name)
                except Exception:
                    pass
            raise

    finally:
        for d in (tmp_dir, bak_dir):
            shutil.rmtree(d, ignore_errors=True)

    # Next query must reload rather than serve the pre-edit copy.
    clear_cache()


def _drop_index_files() -> None:
    """
    Removes the index files entirely and clears the cache.

    Used when the last indexed document is deleted. An index with zero
    vectors still satisfies is_ready(), so leaving it behind would show
    "vector store ready" over an empty store and let queries run against
    nothing. Removing the files puts the system back in its genuine
    "not embedded yet" state.
    """
    for name in ("index.faiss", "index.pkl"):
        target = settings.VECTORSTORE_PATH / name
        if target.exists():
            target.unlink()
    clear_cache()


def _ids_for_sources(store: FAISS, filenames: set) -> list:
    """
    Returns the docstore ids of every chunk that came from `filenames`.

    Chunks carry the originating PDF in metadata["source_file"], set during
    loading in embed_all_pdfs(). That is the only link between a document
    and its vectors, so it is what deletion keys on.
    """
    return [
        doc_id
        for doc_id, doc in store.docstore._dict.items()
        if doc.metadata.get("source_file") in filenames
    ]


def list_indexed_sources() -> dict:
    """
    Returns {source_file: chunk_count} for everything currently in the index.

    Used by the integrity check to spot chunks whose PDF is gone — the
    orphan case that silently poisons retrieval.
    """
    try:
        store = _load_store_from_disk()
    except Exception as e:
        _log(f"❌ Could not read vector store: {e}")
        return {}

    if store is None:
        return {}

    counts = {}
    for doc in store.docstore._dict.values():
        name = doc.metadata.get("source_file", "<unknown>")
        counts[name] = counts.get(name, 0) + 1
    return counts


def delete_from_index(filenames: list) -> dict:
    """
    Removes every chunk belonging to `filenames` from the FAISS index.

    In-place removal, not a rebuild. Which one is correct comes down to what
    FAISS itself supports: this index is an IndexFlatL2, a flat store of
    vectors, and flat indexes implement remove_ids(). LangChain's
    FAISS.delete() uses it and then renumbers index_to_docstore_id so the
    positional mapping stays dense and correct. There is no approximate
    structure (IVF/HNSW) here whose internals a deletion could leave
    inconsistent, and no training state to invalidate.

    A rebuild would also be correct, but strictly worse: it re-embeds every
    remaining document through Ollama, taking minutes and able to fail
    halfway, to reproduce vectors that are already on disk and already
    right. It would also make deletion impossible whenever Ollama is down.
    In-place removal touches only the entries being deleted.

    Returns:
        dict with success, removed (chunk count), remaining, message
    """
    try:
        store = _load_store_from_disk()
    except Exception as e:
        return {
            "success":   False,
            "removed":   0,
            "remaining": None,
            "message":   f"Could not load vector store: {type(e).__name__}: {e}",
        }

    if store is None:
        # Nothing embedded yet — there is nothing to orphan, so this is a
        # success with no work, not a failure.
        return {
            "success":   True,
            "removed":   0,
            "remaining": 0,
            "message":   "No vector store on disk — nothing to remove.",
        }

    targets = set(filenames)
    ids     = _ids_for_sources(store, targets)

    if not ids:
        return {
            "success":   True,
            "removed":   0,
            "remaining": len(store.docstore._dict),
            "message":   "No chunks in the index for those documents.",
        }

    try:
        store.delete(ids)
    except Exception as e:
        return {
            "success":   False,
            "removed":   0,
            "remaining": None,
            "message":   f"FAISS delete failed: {type(e).__name__}: {e}",
        }

    remaining = len(store.docstore._dict)

    # The docstore and the FAISS index must agree, or retrieval maps
    # positions onto the wrong documents. Verify before writing anything.
    if store.index.ntotal != remaining:
        return {
            "success":   False,
            "removed":   0,
            "remaining": None,
            "message":   (
                f"Refusing to save: index has {store.index.ntotal} vectors but "
                f"docstore has {remaining} documents. Nothing was written; the "
                f"index on disk is unchanged."
            ),
        }

    try:
        if remaining == 0:
            _drop_index_files()
        else:
            _save_store_to_disk(store)
    except Exception as e:
        return {
            "success":   False,
            "removed":   0,
            "remaining": None,
            "message":   f"Could not save updated index: {type(e).__name__}: {e}",
        }

    _log(f"  [{_ts()}] 🧹 Removed {len(ids)} chunk(s) for {len(targets)} document(s); "
         f"{remaining} chunk(s) remain")

    return {
        "success":   True,
        "removed":   len(ids),
        "remaining": remaining,
        "message":   f"Removed {len(ids)} chunk(s); {remaining} remain in the index.",
    }


# ── Genre-aware section chunking ──────────────────────────
# WHY THIS REPLACED FIXED-SIZE SPLITTING
#
# RecursiveCharacterTextSplitter cuts at CHUNK_SIZE characters wherever a
# separator happens to fall. On a GR that boundary lands mid-decision, and
# a fact stated once at the top of a section and elaborated 600 characters
# later ends up in two different chunks. Retrieval then returns one of
# them and the LLM answers from half the fact — a confirmed failure on
# GR1.pdf, where the member TOTAL and the member BREAKDOWN never appeared
# in the same chunk no matter how TOP_K or the prompt were tuned.
#
# A GR is not unstructured prose. It is a sequence of named sections
# (संदर्भ / प्रस्तावना / शासन निर्णय / ...) and the section, not an
# arbitrary character count, is the unit a question is actually about. So
# the section becomes the chunk.
#
# Section detection is delegated to core.keyword_match, which is fuzzy
# because these PDFs embed legacy non-Unicode fonts and literal matching
# finds almost nothing — see that module's header.

# A section longer than this is split again by the ordinary splitter.
# Not a target, a safety valve: a genuinely long शासन निर्णय is better
# kept whole, but an unbounded chunk starves the context window and
# degrades embedding quality. Every trip is logged, because how often it
# fires is the measure of whether this ceiling is set right.
SECTION_MAX_CHARS = 2500

# Characters of the PREVIOUS section carried into the start of each
# section, snapped outward to a line boundary.
#
# Same idea as CHUNK_OVERLAP, and load-bearing rather than cosmetic. GR
# sections are not independent: प्रस्तावना states the matter under
# consideration and शासन निर्णय resolves it, so the sentence naming a
# total ("सहा सदस्यांचा समावेश") routinely sits in the last line of
# प्रस्तावना while the itemised breakdown it refers to sits in शासन
# निर्णय. Butt-joining the sections would reproduce the exact split this
# change exists to fix, one boundary to the left.
SECTION_OVERLAP_CHARS = settings.CHUNK_OVERLAP

# A footer marker earlier than this fraction of the document is not the
# distribution list — "प्रत" is a prefix of ordinary words (प्रति,
# प्रत्येक), so position is what separates the trailing block from a
# false positive in the body.
FOOTER_MIN_POSITION_RATIO = 0.5

# What may follow a marker for it to be a section HEADER rather than a
# mention of the same words mid-sentence.
#
# A header is punctuated ("संदर्भ :-1)", "वाचा.-", "प्रति,") or stands
# alone on its line. The letterhead reference number that opens every GR
# — "शासन परिपत्रक क्रमांकः उसज्ये 1126/..." — is neither: the marker is
# followed by another WORD. That distinction is the whole test, and it is
# what keeps the letterhead from being mistaken for the section it names.
# ASCII punctuation is used because it survives the legacy-font
# corruption that mangles the Devanagari around it.
HEADER_PUNCTUATION = ":-.,;–—"

# Pages are joined with this before sectioning. Sections routinely span a
# page break, so structure has to be found on the whole document; a
# single-character join keeps offsets easy to map back to pages.
_PAGE_JOIN = "\n"


def _oversize_splitter() -> RecursiveCharacterTextSplitter:
    """
    The splitter used on a section too long to embed whole.

    Deliberately NOT the CHUNK_SIZE splitter. Cutting a 5,400-character
    शासन निर्णय at 800 characters would hand back the fixed-size chunks
    this change exists to replace — the section boundary would be
    respected and then immediately thrown away. Cutting at
    SECTION_MAX_CHARS keeps the pieces as close to whole-section size as
    the ceiling allows.

    The no-marker fallback in _split_document_into_sections keeps using
    the CHUNK_SIZE splitter, because that path is meant to reproduce the
    old behaviour exactly.

    Separators and overlap match the CHUNK_SIZE splitter's, so a split
    section still breaks at paragraph then sentence then word, never
    mid-character.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=SECTION_MAX_CHARS,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "।", ". ", " ", ""],
    )


def _line_start(text: str, pos: int) -> int:
    """Start of the line containing `pos`."""
    nl = text.rfind("\n", 0, pos)
    return 0 if nl == -1 else nl + 1


def _is_line_start(text: str, pos: int) -> bool:
    """True if only whitespace separates `pos` from the start of its line."""
    return not text[_line_start(text, pos):pos].strip()


def _is_header(text: str, pos: int, length: int) -> bool:
    """
    True if the marker occupying text[pos:pos+length] opens a section,
    rather than merely being those words inside a sentence.

    Almost every occurrence is the latter. On GR1.pdf "शासन निर्णय"
    occurs five times — the letterhead reference number, twice inside the
    संदर्भ citation list, the actual section header, and the closing
    sentence about the digital signature. Only the fourth is a boundary,
    and taking the first (which is what a single best-match position
    gives you) would put the whole decision in the wrong section.

    Two conditions, both cheap and both robust to font corruption:
        the marker starts its line — a header owns its line; and
        nothing but punctuation follows it on that line.

    Combining marks are skipped before the punctuation test because
    extraction sprinkles stray matras onto header lines ("शासन आदेशा" for
    a standalone "शासन आदेश"), and a visarga where a colon was typed is
    the same character class. Neither should cost a real header.
    """
    if not _is_line_start(text, pos):
        return False

    line_end = text.find("\n", pos)
    after    = text[pos + length:line_end if line_end != -1 else len(text)]

    after = normalize(after).lstrip()
    return after == "" or after[0] in HEADER_PUNCTUATION


def _footer_cut(text: str, report) -> Optional[int]:
    """
    Offset where the प्रत distribution list begins, or None.

    The footer is pure noise in every genre seen so far — a list of the
    offices a copy went to. It is dropped before chunking rather than
    after, so it can neither become a chunk of its own nor pad out the
    tail of a real section.

    Only exact matches count. The fuzzy path skips "प्रत" anyway (it
    strips to three characters, under DEFAULT_MIN_FUZZY_LEN), and cutting
    a live document on a fuzzy guess is the wrong trade for a destructive
    operation.
    """
    match = report.exact.get(FOOTER_KEYWORD)
    if match is None or not report.text_length:
        return None

    # Latest header-like occurrence in the tail — the list is the last
    # structural thing in the document.
    for pos in sorted(match.positions, reverse=True):
        if pos / report.text_length < FOOTER_MIN_POSITION_RATIO:
            break
        if _is_header(text, pos, len(FOOTER_KEYWORD)):
            return _line_start(text, pos)
    return None


def _section_boundaries(text: str, report) -> list:
    """
    Header-like marker positions, left to right, as [(offset, marker)].

    One boundary per marker: a GR names each of its sections once, and
    allowing repeats would shatter a section on every in-body mention of
    its own name. The first occurrence that passes _is_header wins.

    Exact positions are used when present — they are certain, and there
    are all of them. Fuzzy contributes one more candidate, which is what
    rescues documents whose headers were mangled past literal matching;
    its match length comes from the snippet, since a corrupted header is
    not the same length as the correctly-spelled keyword.
    """
    boundaries = []

    for marker in SECTION_KEYWORDS:
        if marker == FOOTER_KEYWORD:
            continue  # terminates the body, does not open a section

        exact = report.exact.get(marker)
        fuzzy = report.fuzzy.get(marker)

        candidates = [(pos, len(marker)) for pos in (exact.positions if exact else ())]
        if fuzzy is not None:
            candidates.append((fuzzy.position, len(fuzzy.snippet)))

        for pos, length in sorted(set(candidates)):
            if _is_header(text, pos, length):
                boundaries.append((_line_start(text, pos), marker))
                break

    boundaries.sort()

    # Two markers resolving to the same line is one boundary, not two —
    # keep the first and drop the empty section that would follow.
    deduped = []
    for pos, marker in boundaries:
        if deduped and pos == deduped[-1][0]:
            continue
        deduped.append((pos, marker))
    return deduped


def _page_index_for_offset(offsets: list, pos: int) -> int:
    """
    Index of the page containing character `pos`, given each page's start
    offset in the joined text. Chunks can span a page break now, so
    citations report the page a chunk STARTS on.
    """
    idx = 0
    for i, start in enumerate(offsets):
        if start <= pos:
            idx = i
        else:
            break
    return idx


def _split_document_into_sections(pages: list, splitter, source_file: str) -> list:
    """
    Chunks one PDF's pages by structural section.

    Args:
        pages       : the Documents load_pdf_with_ocr_fallback returned, in order
        splitter    : the fixed-size splitter, used only as a fallback
        source_file : filename, for logging

    Returns:
        list[Document]. Every chunk carries metadata["section_type"] — the
        marker it came from, or "__header__" for the block above the first
        marker — plus metadata["chunking"], so a later analysis can tell
        sectioned chunks from fallback ones without re-deriving anything.

    Never raises on structure it does not recognise: a document with no
    detectable markers takes the fixed-size path unchanged rather than
    being skipped.
    """
    text = _PAGE_JOIN.join(p.page_content for p in pages)

    # Start offset of each page inside `text`, for mapping chunks back.
    offsets, running = [], 0
    for p in pages:
        offsets.append(running)
        running += len(p.page_content) + len(_PAGE_JOIN)

    if not text.strip():
        return []

    report = match_keywords(text, SECTION_KEYWORDS)

    # ── Footer ────────────────────────────────────────────
    cut = _footer_cut(text, report)
    if cut is not None:
        print(f"  [{_ts()}]    section: {source_file} — dropped प्रत footer "
              f"({len(text) - cut} chars from offset {cut})", flush=True)
        text = text[:cut]

    boundaries = _section_boundaries(text, report)

    # ── Fallback: no structure found ──────────────────────
    if not boundaries:
        # Unchanged fixed-size behaviour, on the same per-page Documents
        # the old code split — minus the footer, which is noise in every
        # genre and is dropped unconditionally.
        kept = []
        for i, page in enumerate(pages):
            body = page.page_content
            if cut is not None:
                if offsets[i] >= cut:
                    continue
                body = body[:cut - offsets[i]]
            if not body.strip():
                continue
            kept.append(Document(page_content=body, metadata=dict(page.metadata)))

        chunks = splitter.split_documents(kept)
        for c in chunks:
            c.metadata["section_type"] = "__none__"
            c.metadata["chunking"]     = "fallback"
        print(f"  [{_ts()}]    section: {source_file} — NO markers detected, "
              f"fixed-size fallback ({len(chunks)} chunks)", flush=True)
        return chunks

    # ── Section spans ─────────────────────────────────────
    spans = []
    if boundaries[0][0] > 0:
        # Letterhead, subject line, GR number. Small, and it is what
        # "which GR is this" questions actually match on.
        spans.append((0, boundaries[0][0], "__header__"))

    for i, (start, marker) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        spans.append((start, end, marker))

    chunks   = []
    oversize = 0

    for index, (start, end, marker) in enumerate(spans):
        # Carry the tail of the previous section in, snapped out to a line
        # boundary so the overlap never begins mid-word.
        ctx_start = start
        if start > 0 and SECTION_OVERLAP_CHARS > 0:
            ctx_start = _line_start(text, max(0, start - SECTION_OVERLAP_CHARS))

        body = text[ctx_start:end].strip()
        if not body:
            continue

        page_idx = _page_index_for_offset(offsets, start)
        base     = dict(pages[page_idx].metadata)
        base.update({
            "section_type":  marker,
            "section_index": index,
            "chunking":      "section",
            "section_chars": len(body),
        })

        if len(body) <= SECTION_MAX_CHARS:
            chunks.append(Document(page_content=body, metadata=base))
            continue

        # ── Oversize fallback ─────────────────────────────
        oversize += 1
        parts = _oversize_splitter().split_text(body)
        print(f"  [{_ts()}]    section: {source_file} — section '{marker}' is "
              f"{len(body)} chars (> {SECTION_MAX_CHARS}), split into "
              f"{len(parts)} parts", flush=True)

        # Each part gets the page it actually starts on, not the section's.
        # A section long enough to trip the ceiling usually spans several
        # pages, and stamping them all with the first one would send every
        # citation for the tail of the section to the wrong page.
        cursor = 0
        for n, part in enumerate(parts, 1):
            at = body.find(part, cursor)
            if at == -1:          # splitter normalised whitespace somewhere
                at = cursor
            cursor = at + 1

            meta = dict(base)
            meta.update(pages[_page_index_for_offset(offsets, ctx_start + at)].metadata)
            meta.update({
                "section_type":  marker,
                "section_index": index,
                "section_chars": len(body),
                "chunking":      "section_split",
                "section_part":  n,
                "section_parts": len(parts),
            })
            chunks.append(Document(page_content=part, metadata=meta))

    print(f"  [{_ts()}]    section: {source_file} — {len(spans)} section(s) "
          f"[{', '.join(m for _, _, m in spans)}] -> {len(chunks)} chunk(s)"
          + (f", {oversize} oversize" if oversize else ""), flush=True)

    return chunks


def split_documents_by_section(all_documents: list, splitter) -> list:
    """
    Section-chunks a whole load, grouping the flat page list back into the
    documents it came from.

    Structure has to be found on the full document, not page by page: a
    section routinely opens on one page and ends on the next, and a
    per-page view would cut every one of them at the page break — the
    same defect as a fixed character count, just with a different ruler.

    A document that fails to section for any reason falls back to
    fixed-size splitting rather than dropping out of the index.
    """
    grouped = {}
    for doc in all_documents:
        grouped.setdefault(doc.metadata.get("source_file", "<unknown>"), []).append(doc)

    chunks = []
    for source_file, pages in grouped.items():
        try:
            chunks.extend(_split_document_into_sections(pages, splitter, source_file))
        except Exception as e:
            # Chunking must not be able to lose a document. Anything
            # unexpected here degrades to the old behaviour, loudly.
            print(f"  [{_ts()}]    section: {source_file} — sectioning FAILED "
                  f"({type(e).__name__}: {e}); fixed-size fallback", flush=True)
            fallback = splitter.split_documents(pages)
            for c in fallback:
                c.metadata["section_type"] = "__error__"
                c.metadata["chunking"]     = "fallback"
            chunks.extend(fallback)

    return chunks


async def embed_all_pdfs(progress_callback=None, filenames: Optional[list] = None) -> dict:
    """
    Main embedding function — processes PDFs in the GRDOCS folder.

    Steps:
        1. Find the PDFs to embed in grdocs/ folder
        2. Load each PDF into pages using PyPDFLoader
        3. Split pages into chunks using RecursiveCharacterTextSplitter
        4. Generate embeddings for each chunk via Ollama
        5. Save FAISS index to disk
        6. Clear cache so next query uses fresh index

    Args:
        progress_callback : optional function called with (filename, current, total)
                           Used to stream progress to the frontend
        filenames         : optional list of bare PDF filenames to embed. None
                           (the default) means "every PDF in grdocs/" and takes
                           the original full-rebuild path, byte for byte.

    Scoped mode (filenames given) differs ONLY in which files are loaded and
    how the result is written to disk. Chunking, the batched Ollama calls and
    the OCR fallback are the exact same code either way — a scoped run
    produces the same vectors for those files that a full run would.

    Writing is where the two modes must differ. FAISS.save_local() serialises
    whatever store it is called on, so saving a store built from a subset
    would silently discard every other document's vectors. Scoped mode
    therefore MERGES: it loads the existing index, drops any prior chunks for
    the files being re-embedded (so a re-embed replaces rather than
    duplicates), adds the fresh ones, and saves the combined store.

    Returns:
        dict with success, message, total_chunks, failed_files
    """
    scoped = filenames is not None

    if scoped:
        # Resolve each requested name inside grdocs/ — reject anything that
        # is not a bare filename, so a caller cannot reach outside the folder.
        pdf_files = []
        missing   = []
        for name in filenames:
            if not name or Path(name).name != name:
                return {
                    "success": False,
                    "message": f"Invalid filename: {name!r}",
                    "total_chunks": 0,
                    "failed_files": [],
                }
            candidate = settings.GRDOCS_PATH / name
            if candidate.exists():
                pdf_files.append(candidate)
            else:
                missing.append(name)

        if missing:
            return {
                "success": False,
                "message": f"Not found in grdocs folder: {', '.join(missing)}",
                "total_chunks": 0,
                "failed_files": missing,
            }

        if not pdf_files:
            return {
                "success": False,
                "message": "No documents selected to embed.",
                "total_chunks": 0,
                "failed_files": [],
            }
    else:
        # Find all PDFs
        pdf_files = list(settings.GRDOCS_PATH.glob("*.pdf"))

        if not pdf_files:
            return {
                "success": False,
                "message": "No PDF files found in grdocs folder. Upload some GRs first.",
                "total_chunks": 0,
                "failed_files": [],
            }

    all_documents = []
    failed_files   = []

    # ── Step 1 & 2: Load each PDF ─────────────────────────
    for i, pdf_path in enumerate(pdf_files):
        if progress_callback:
            progress_callback(pdf_path.name, i + 1, len(pdf_files))

        try:
            from core.ocr import load_pdf_with_ocr_fallback
            pages = load_pdf_with_ocr_fallback(str(pdf_path))

            # Add filename to each page's metadata
            # This is what powers citations later —
            # we know which file and which page each chunk came from
            for page in pages:
                page.metadata["source_file"] = pdf_path.name

            all_documents.extend(pages)
            print(f"  [{_ts()}] ✅ Loaded {i + 1}/{len(pdf_files)}: {pdf_path.name} "
                  f"({len(pages)} pages)", flush=True)

        except Exception as e:
            failed_files.append(pdf_path.name)
            print(f"  [{_ts()}] ❌ Failed to load {pdf_path.name}: {e}", flush=True)

    if not all_documents:
        return {
            "success": False,
            "message": "All PDFs failed to load. Check if files are valid PDFs.",
            "total_chunks": 0,
            "failed_files": failed_files,
        }

    # ── Step 3: Split into chunks ─────────────────────────
    # Chunking is by structural SECTION now, not by character count — see
    # the "Genre-aware section chunking" block above for why.
    #
    # The fixed-size splitter below still exists and is still configured
    # exactly as before. It is simply no longer the primary path: it now
    # handles documents where no section marker could be detected at all,
    # and sections too long to embed whole. Its settings are untouched so
    # those fallbacks behave the way the old pipeline did.
    #
    # RecursiveCharacterTextSplitter tries to split at:
    # paragraphs → sentences → words → characters (in that order)
    # This preserves meaning better than hard character cuts
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,         # max characters per chunk
        chunk_overlap=settings.CHUNK_OVERLAP,   # overlap prevents losing context at boundaries
        separators=["\n\n", "\n", "।", ". ", " ", ""],  # । is Devanagari sentence end
    )

    chunks = split_documents_by_section(all_documents, splitter)

    # ── DIAGNOSTIC: chunk inventory ───────────────────────
    # Printed before the embed call so that if it hangs, the terminal
    # already shows exactly what was handed to Ollama.
    sizes = sorted(len(c.page_content) for c in chunks)
    total_chars = sum(sizes)
    print(f"  [{_ts()}] 📄 Total chunks created: {len(chunks)}", flush=True)
    if sizes:
        biggest = max(chunks, key=lambda c: len(c.page_content))
        print(f"  [{_ts()}]    chunk chars: total={total_chars} min={sizes[0]} "
              f"median={sizes[len(sizes) // 2]} max={sizes[-1]}", flush=True)
        print(f"  [{_ts()}]    largest chunk: {len(biggest.page_content)} chars from "
              f"{biggest.metadata.get('source_file', '?')} "
              f"p{biggest.metadata.get('page', '?')}", flush=True)

    # ── Step 4 & 5: Embed and save ────────────────────────
    try:
        embeddings = get_embeddings()

        # FAISS.from_documents() does two things:
        #   - calls Ollama to get vector for each chunk
        #   - stores all vectors in a FAISS index
        # This is the slow step — takes 2-5 min depending on doc count
        # ── Batched embedding ─────────────────────────────
        # embed_documents() posts its whole argument list in ONE request, so
        # batching here is what bounds the request size. Each batch is a
        # separate HTTP call and therefore gets its own EMBED_TIMEOUT_SECONDS
        # budget rather than sharing one across the entire job.
        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        n_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

        print(f"  [{_ts()}] 🔄 EMBED START — {len(chunks)} chunks, {total_chars} chars "
              f"→ model={settings.EMBEDDING_MODEL} at {settings.OLLAMA_BASE_URL}", flush=True)
        print(f"  [{_ts()}]    {n_batches} batch(es) of up to {EMBED_BATCH_SIZE} chunks; "
              f"timeout={EMBED_TIMEOUT_SECONDS:.0f}s per batch", flush=True)

        all_embeddings = []
        embed_started  = time.time()

        for b in range(n_batches):
            lo = b * EMBED_BATCH_SIZE
            hi = min(lo + EMBED_BATCH_SIZE, len(chunks))
            batch_texts = texts[lo:hi]
            batch_chars = sum(len(t) for t in batch_texts)

            print(f"  [{_ts()}]    |- batch {b + 1}/{n_batches} START — "
                  f"{len(batch_texts)} chunks [{lo}..{hi - 1}], {batch_chars} chars",
                  flush=True)

            batch_started = time.time()
            try:
                with _Heartbeat(f"batch {b + 1}/{n_batches} ({len(batch_texts)} chunks)"):
                    batch_vectors = embeddings.embed_documents(batch_texts)
            except Exception as e:
                # Stop at the FIRST failure. Returning here means no index is
                # built and none is saved, so the previous index on disk stays
                # intact rather than being replaced by a partial one.
                batch_elapsed = time.time() - batch_started
                kind = "timed out" if isinstance(e, httpx.TimeoutException) else "failed"
                print(f"  [{_ts()}]    +- batch {b + 1}/{n_batches} {kind.upper()} after "
                      f"{batch_elapsed:.1f}s — {type(e).__name__}: {e}", flush=True)
                print(f"  [{_ts()}] ❌ EMBED ABORTED — {b} of {n_batches} batch(es) had "
                      f"completed; nothing was saved", flush=True)
                return {
                    "success": False,
                    "message": (
                        f"Embedding {kind} on batch {b + 1}/{n_batches} "
                        f"(chunks {lo}-{hi - 1} of {len(chunks)}, {batch_chars} chars) "
                        f"after {batch_elapsed:.0f}s: {type(e).__name__}: {e}. "
                        f"{b} batch(es) completed before the failure. No partial index "
                        f"was written — the existing vector store is unchanged."
                    ),
                    "total_chunks": 0,
                    "failed_files": failed_files,
                }

            # A short/long return would silently misalign vectors against their
            # texts, producing an index that looks fine and retrieves nonsense.
            if len(batch_vectors) != len(batch_texts):
                print(f"  [{_ts()}]    +- batch {b + 1}/{n_batches} RETURNED "
                      f"{len(batch_vectors)} vectors for {len(batch_texts)} chunks",
                      flush=True)
                print(f"  [{_ts()}] ❌ EMBED ABORTED — vector/chunk count mismatch; "
                      f"nothing was saved", flush=True)
                return {
                    "success": False,
                    "message": (
                        f"Embedding returned {len(batch_vectors)} vectors for "
                        f"{len(batch_texts)} chunks on batch {b + 1}/{n_batches} "
                        f"(chunks {lo}-{hi - 1}). Aborted to avoid building an index "
                        f"with misaligned vectors. No partial index was written."
                    ),
                    "total_chunks": 0,
                    "failed_files": failed_files,
                }

            all_embeddings.extend(batch_vectors)
            batch_elapsed = time.time() - batch_started
            print(f"  [{_ts()}]    +- batch {b + 1}/{n_batches} DONE — "
                  f"{len(batch_texts)} chunks in {batch_elapsed:.1f}s "
                  f"({batch_chars / batch_elapsed:.0f} chars/sec), "
                  f"{len(all_embeddings)}/{len(chunks)} total", flush=True)

        embed_elapsed = time.time() - embed_started
        rate = (total_chars / embed_elapsed) if embed_elapsed else 0
        print(f"  [{_ts()}] ✅ EMBED DONE — {len(chunks)} chunks in {n_batches} batch(es), "
              f"{embed_elapsed:.1f}s total ({rate:.0f} chars/sec)", flush=True)

        # Same index as FAISS.from_documents() would have built — the vectors
        # were simply fetched in batches instead of in one request.
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, all_embeddings)),
            embedding=embeddings,
            metadatas=metadatas,
        )

        if scoped:
            # Merge into the index already on disk instead of replacing it.
            existing = _load_store_from_disk()
            if existing is not None:
                embedded_names = {f.name for f in pdf_files}
                stale_ids = _ids_for_sources(existing, embedded_names)
                if stale_ids:
                    # A re-embed of an already-indexed file must REPLACE its
                    # chunks. Without this the old vectors survive alongside
                    # the new ones and retrieval starts returning the same
                    # passage twice, from two generations of the same document.
                    existing.delete(stale_ids)
                    _log(f"  [{_ts()}] 🧹 Replaced {len(stale_ids)} existing chunk(s) "
                         f"for {len(embedded_names)} re-embedded file(s)")
                existing.merge_from(vector_store)
                vector_store = existing

        # Save index to disk so it survives app restarts
        _save_store_to_disk(vector_store)
        print(f"  [{_ts()}] 💾 Vector store saved to: {settings.VECTORSTORE_PATH}", flush=True)

        # ── Step 6: Clear cache ───────────────────────────
        # Force next search() call to reload fresh index
        clear_cache()

        msg = f"Successfully embedded {len(pdf_files) - len(failed_files)} PDFs — {len(chunks)} chunks created."
        if failed_files:
            msg += f" Failed: {', '.join(failed_files)}"

        return {
            "success":      True,
            "message":      msg,
            "total_chunks": len(chunks),
            "failed_files": failed_files,
        }

    except httpx.TimeoutException as e:
        # Per-batch timeouts are handled inside the loop above, which reports
        # the failing batch and its chunk range. This stays as a backstop for a
        # timeout raised outside the loop (e.g. during save_local).
        print(f"  [{_ts()}] ❌ EMBED TIMED OUT — {type(e).__name__}: {e}", flush=True)
        return {
            "success":      False,
            "message":      (
                f"Embedding timed out outside the batch loop: {type(e).__name__}: {e}. "
                f"Check that {settings.EMBEDDING_MODEL} is pulled and Ollama is running."
            ),
            "total_chunks": 0,
            "failed_files": failed_files,
        }

    except Exception as e:
        # Timestamped and typed — a bare str(e) is often empty and tells
        # you nothing about where in the embed/save sequence it died.
        print(f"  [{_ts()}] ❌ EMBED FAILED — {type(e).__name__}: {e}", flush=True)
        return {
            "success":      False,
            "message":      f"Embedding failed: {type(e).__name__}: {e}. Is Ollama running?",
            "total_chunks": 0,
            "failed_files": failed_files,
        }


def search(query: str, top_k: int = None) -> list[Document]:
    """
    Searches the FAISS index for chunks most relevant to the query.
    Uses FAISS built-in similarity search — no manual vector loops.

    Args:
        query : the user's question
        top_k : number of chunks to retrieve (defaults to settings.TOP_K)

    Returns:
        List of Document objects with page_content and metadata
        Empty list if vector store not ready or search fails

    Each Document looks like:
        Document(
            page_content = "...relevant text from the GR...",
            metadata = {
                "source_file": "GR_2024_transfer.pdf",
                "page": 3,
            }
        )
    """
    if top_k is None:
        top_k = settings.TOP_K

    db = load_store()

    if db is None:
        # Vector store not ready — return empty list
        # rag.py handles this gracefully
        return []

    try:
        docs = db.similarity_search(query, k=top_k)
        return docs
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []


def cosine_search_with_scores(query: str, top_k: int = 5) -> list[tuple]:
    """
    Like search() but also returns similarity scores.
    Used on the Search page to show relevance scores to users.

    Returns:
        List of (Document, score) tuples
        Score is between 0 and 1 — higher means more relevant
        (FAISS returns L2 distance, we convert to similarity)
    """
    db = load_store()

    if db is None:
        return []

    try:
        # Returns list of (Document, distance) tuples
        # Lower distance = more similar in FAISS
        results_with_scores = db.similarity_search_with_score(query, k=top_k)

        # Convert L2 distance to similarity score between 0 and 1
        # Formula: similarity = 1 / (1 + distance)
        converted = []
        for doc, distance in results_with_scores:
            similarity = round(1 / (1 + float(distance)), 4)
            converted.append((doc, similarity))

        return converted

    except Exception as e:
        print(f"❌ Scored search failed: {e}")
        return []