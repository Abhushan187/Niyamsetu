# backend/core/summarizer.py
# ─────────────────────────────────────────────────────────
# GR Summary and Metadata Extraction.
#
# Two LLM calls per document:
#   1. Metadata extraction → returns structured JSON
#      (GR number, department, date, subject, signatory)
#   2. Summary generation  → returns readable paragraphs
#      (purpose, provisions, beneficiaries, deadlines)
#
# Output saved as both JSON and TXT files in summaries/ folder.
# Called by:
#   api/summary.py → triggers this on admin request
# ─────────────────────────────────────────────────────────

import sys
import os
import json
import asyncio
import traceback
from pathlib import Path
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.language import clean_text, truncate_for_context
from core.trace import trace


def get_llm() -> OllamaLLM:
    """
    OllamaLLM (not ChatOllama) — used here because summary prompts
    work better with the completion-style interface.
    temperature=0.1 allows slight creativity for readable summaries
    while staying grounded in the document.
    """
    return OllamaLLM(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
        # Same pinned context as the chat path (config.py: LLM_NUM_CTX).
        # Sharing one context size matters here: Ollama keys the loaded
        # llama-server on it, so a summary at a different size would evict
        # and reload the model the chat path is already using.
        num_ctx=settings.LLM_NUM_CTX,
        keep_alive=settings.LLM_KEEP_ALIVE,
    )


def _load_pdf_text(pdf_path: str, stage_label: str = "") -> str:
    """
    Loads all text from a PDF file.
    Cleans and truncates to fit within LLM context window.

    Args:
        pdf_path    : full path to the PDF file
        stage_label : diagnostic only — names the caller ("metadata" /
                      "summary") so the stage prints below can be told
                      apart, since this runs twice per document.

    Returns:
        Cleaned text string, max 12000 characters
    """
    from core.ocr import load_pdf_with_ocr_fallback

    # ── DIAGNOSTIC ────────────────────────────────────────
    # Character counts at each transform, so a document that ends up
    # producing an empty or truncated prompt can be traced to the exact
    # stage that dropped the text. flush=True because uvicorn buffers
    # stdout — without it these can appear after the HTTP access logs.
    tag = f"[{Path(pdf_path).name}|{stage_label or 'unlabelled'}]"

    documents = load_pdf_with_ocr_fallback(pdf_path)
    raw_chars = sum(len(doc.page_content) for doc in documents)
    print(f"{tag} STAGE 1 pdf loaded      : {len(documents)} page(s), {raw_chars} chars", flush=True)

    # Join all pages into one text block
    full_text = "\n".join(doc.page_content for doc in documents)
    print(f"{tag} STAGE 2 pages joined    : {len(full_text)} chars", flush=True)

    # Clean whitespace and normalize line breaks
    full_text = clean_text(full_text)
    print(f"{tag} STAGE 3 after clean_text: {len(full_text)} chars", flush=True)

    # Truncate to fit LLM context window safely
    full_text = truncate_for_context(full_text, max_chars=12000)
    print(f"{tag} STAGE 4 after truncate  : {len(full_text)} chars", flush=True)

    preview = full_text[:120].replace("\n", " ")
    print(f"{tag} STAGE 5 sent to llm     : {len(full_text)} chars | preview: {preview!r}", flush=True)
    if not full_text.strip():
        print(f"{tag} STAGE 5 WARNING: text is empty — the LLM will receive a blank document", flush=True)

    return full_text


async def extract_metadata(pdf_path: str) -> dict:
    """
    Extracts structured metadata from a GR document using LLM.

    Asks the LLM to return valid JSON with these fields:
        gr_number         : the GR reference number
        department        : issuing department name
        issue_date        : date the GR was issued
        subject           : subject/title of the GR
        applicable_region : which region/district this applies to
        signatory         : name and designation of signing authority

    Args:
        pdf_path : path to the PDF

    Returns:
        dict of extracted metadata fields
        Falls back to {"raw_output": "..."} if JSON parsing fails
    """
    # _load_pdf_text() is fully synchronous: PyPDFLoader, pdf2image and
    # Tesseract OCR (core/ocr.py) all block, for seconds to minutes on a
    # scanned document. Awaiting it on the event loop stalls every other
    # request. Offloaded to a worker thread — the function itself, and the
    # text it produces, are unchanged.
    full_text = await asyncio.to_thread(_load_pdf_text, pdf_path, stage_label="metadata")
    llm       = get_llm()
    parser    = StrOutputParser()

    metadata_prompt = PromptTemplate.from_template("""
Extract the following information from this Government Resolution document.
Return ONLY valid JSON — no explanation, no markdown, no backticks.
If a field is not found, use null.

Required JSON format:
{{
  "gr_number": "...",
  "department": "...",
  "issue_date": "...",
  "subject": "...",
  "applicable_region": "...",
  "signatory": "..."
}}

Document:
{text}
""")

    # chain.invoke() is the SYNCHRONOUS LangChain entrypoint — it drives
    # OllamaLLM through ollama.Client (blocking httpx), so it pins the event
    # loop for the whole generation. Same class of bug fixed in api/embed.py.
    # Offloaded to a worker thread; the chain and prompt are unchanged.
    chain = metadata_prompt | llm | parser
    # Separates PDF load/OCR (the STAGE lines above) from generation. If the
    # terminal stops after STAGE 5 the document is still being read; if it
    # stops here, Ollama is generating.
    trace(
        "LLM      start  metadata",
        f"{Path(pdf_path).name} model={settings.LLM_MODEL} text={len(full_text)} chars",
    )
    raw   = await asyncio.to_thread(chain.invoke, {"text": full_text})
    trace("LLM      done   metadata", f"{Path(pdf_path).name} {len(raw)} chars returned")

    # Try to parse as JSON
    # LLMs sometimes add extra text before/after — we strip it
    try:
        # Remove any markdown code fences if present
        cleaned = raw.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        metadata = json.loads(cleaned)
        return metadata

    except json.JSONDecodeError:
        # If JSON parsing fails, return raw output
        # Frontend handles this gracefully
        return {"raw_output": raw.strip()}


async def generate_summary(pdf_path: str) -> str:
    """
    Generates a structured human-readable summary of a GR document.

    Summary includes:
        1. Purpose          — why this GR was issued
        2. Key Provisions   — what rules/changes it introduces
        3. Beneficiaries    — who is affected
        4. Financial Impact — any monetary implications
        5. Implementation   — how/when it takes effect
        6. Deadlines        — any important dates

    Args:
        pdf_path : path to the PDF

    Returns:
        Summary as a formatted text string
    """
    # Synchronous PDF load + OCR — see extract_metadata() above.
    full_text = await asyncio.to_thread(_load_pdf_text, pdf_path, stage_label="summary")
    llm       = get_llm()
    parser    = StrOutputParser()

    summary_prompt = PromptTemplate.from_template("""
You are an expert analyst of Maharashtra Government Resolution documents.
Provide a clear, structured summary of this Government Resolution.

Include these sections (use the exact headings):

1. PURPOSE
What is the reason this resolution was issued?

2. KEY PROVISIONS
What are the main rules, decisions, or changes introduced?

3. BENEFICIARIES / TARGET GROUP
Who does this resolution apply to or benefit?

4. FINANCIAL IMPLICATIONS
Are there any monetary allocations, grants, or financial impacts?

5. IMPLEMENTATION
How and when does this resolution take effect?

6. IMPORTANT DATES / DEADLINES
List any specific dates mentioned.

Keep each section concise — 2 to 4 sentences maximum.
If information for a section is not available, write "Not specified."

Document:
{text}
""")

    # Synchronous blocking LLM call — see extract_metadata() above.
    chain   = summary_prompt | llm | parser
    # Second generation of the pair — see extract_metadata() above.
    trace(
        "LLM      start  summary",
        f"{Path(pdf_path).name} model={settings.LLM_MODEL} text={len(full_text)} chars",
    )
    summary = await asyncio.to_thread(chain.invoke, {"text": full_text})
    trace("LLM      done   summary", f"{Path(pdf_path).name} {len(summary)} chars returned")
    return summary.strip()


async def process_gr(pdf_path: str, progress_callback=None) -> dict:
    """
    Full summary pipeline for one GR document.
    Runs metadata extraction and summary generation,
    then saves both as files in the summaries/ folder.

    Args:
        pdf_path          : path to the PDF
        progress_callback : optional function called with status string
                           e.g. progress_callback("Extracting metadata...")

    Returns:
        dict with:
            success    : bool
            message    : status message
            metadata   : extracted metadata dict
            summary    : summary text string
            json_path  : path to saved JSON file
            txt_path   : path to saved TXT file
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        return {
            "success": False,
            "message": f"File not found: {pdf_path.name}",
            "metadata": {},
            "summary": "",
        }

    try:
        # ── Step 1: Extract metadata ──────────────────────
        if progress_callback:
            progress_callback("Extracting metadata...")

        metadata = await extract_metadata(str(pdf_path))

        # ── Step 2: Generate summary ──────────────────────
        if progress_callback:
            progress_callback("Generating summary...")

        summary = await generate_summary(str(pdf_path))

        # ── Step 3: Save outputs ──────────────────────────
        if progress_callback:
            progress_callback("Saving report...")

        settings.SUMMARIES_PATH.mkdir(parents=True, exist_ok=True)
        base_name = pdf_path.stem

        # Save as JSON — for programmatic use
        result_data = {
            "filename":     pdf_path.name,
            "processed_at": str(datetime.now(timezone.utc)),
            "metadata":     metadata,
            "summary":      summary,
        }

        json_path = settings.SUMMARIES_PATH / f"{base_name}_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

        # Save as TXT — for human reading and download
        txt_path = settings.SUMMARIES_PATH / f"{base_name}_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"GR DOCUMENT: {pdf_path.name}\n")
            f.write(f"Processed: {result_data['processed_at']}\n")
            f.write("=" * 60 + "\n\n")
            f.write("METADATA\n")
            f.write("-" * 40 + "\n")
            f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
            f.write("\n\n" + "=" * 60 + "\n\n")
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(summary)

        return {
            "success":   True,
            "message":   "Summary generated successfully.",
            "metadata":  metadata,
            "summary":   summary,
            "json_path": str(json_path),
            "txt_path":  str(txt_path),
        }

    except Exception as e:
        # DIAGNOSTIC — this except is where the traceback used to die.
        # Callers only ever saw str(e), which loses the exception type and
        # the frame it came from. Print the full trace before collapsing it
        # into the return dict.
        print(f"\n{'=' * 60}", flush=True)
        print(f"process_gr FAILED for {pdf_path.name}", flush=True)
        print(f"  {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        print(f"{'=' * 60}\n", flush=True)

        return {
            "success": False,
            # Include the exception type — "" from a bare str(e) is a common
            # and completely unreadable failure message.
            "message": f"Summary generation failed: {type(e).__name__}: {e}",
            "metadata": {},
            "summary": "",
        }


def list_summaries() -> list:
    """
    Returns all previously generated summaries.
    Used on the admin Summaries page to show past reports.

    Returns:
        List of dicts with summary info, newest first
    """
    if not settings.SUMMARIES_PATH.exists():
        return []

    summaries = []

    for json_file in sorted(
        settings.SUMMARIES_PATH.glob("*_summary.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,           # newest first
    ):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("metadata", {})

            summaries.append({
                "filename":     data.get("filename", json_file.name),
                "processed_at": data.get("processed_at", ""),
                "subject":      meta.get("subject", "N/A"),
                "department":   meta.get("department", "N/A"),
                "gr_number":    meta.get("gr_number", "N/A"),
                "txt_path":     str(json_file).replace("_summary.json", "_summary.txt"),
                "json_path":    str(json_file),
            })

        except Exception:
            # Skip corrupted files silently
            pass

    return summaries