# backend/api/summary.py
# ─────────────────────────────────────────────────────────
# GR Summary generation endpoints.
#
# Endpoints:
#   POST /api/summary/generate      → generate summary for one GR
#   GET  /api/summary/list          → list all past summaries
#   GET  /api/summary/download/{filename} → download summary TXT file
#
# Admin only — summary generation is a heavy LLM operation.
# Uses BackgroundTasks like embed.py — returns immediately,
# runs in background, frontend polls for completion.
# ─────────────────────────────────────────────────────────

import sys
import os
import traceback
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_admin_user, get_current_user
from core.summarizer import process_gr, list_summaries
from core.trace import trace
from db.gr_meta import get_all_gr_metadata
from config import settings

router = APIRouter(prefix="/api/summary", tags=["Summary"])


def _safe_path(base: Path, filename: str) -> Path:
    """
    Resolves `filename` inside `base`, rejecting path traversal.

    Two gates:
      1. Path(filename).name strips any directory component, so a mismatch
         means the input was not a bare filename. Note that FastAPI
         percent-decodes path params AFTER routing, so a request for
         "..%2F..%2F.env" arrives here already decoded as "../../.env"
         and is caught here.
      2. Even a name that passes gate 1 must still resolve to somewhere
         inside `base` — covers symlinks and platform-specific quirks.

    Raises:
        HTTPException 400 if the filename is unsafe.
    """
    if not filename or Path(filename).name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    resolved = (base / filename).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    return resolved


# ── Summary job state tracker ─────────────────────────────
# Same pattern as embed.py — track background job state in memory
_summary_state = {
    "running":      False,
    "last_status":  "idle",       # idle | running | done | failed
    "last_message": "Not started",
    "current_step": "",
    "filename":     "",
    "result":       None,         # stores the result when done
}


class GenerateRequest(BaseModel):
    """What the frontend sends to trigger summary generation."""
    filename: str   # e.g. "GR_2024_transfer.pdf"


class BatchRequest(BaseModel):
    """
    Optional body for POST /generate-batch.

    filenames=None (or no body at all) keeps the original behaviour:
    summarize every embedded document that has no summary yet. A list
    scopes the run to exactly those documents, including ones that already
    have a summary — selecting a document is an explicit instruction to
    (re)summarize it, so it is not filtered against the pending list.
    """
    filenames: list[str] | None = None


# ── Batch summary job state tracker ───────────────────────
# Same in-memory pattern as _summary_state and embed.py's _embed_state.
_batch_state = {
    "running":       False,
    "last_status":   "idle",      # idle | running | done | failed
    "last_message":  "Not started",
    "progress":      0,           # 0-100
    "total_files":   0,
    "completed":     0,           # exact count done so far — used for "X/Y" UI
    "current_file":  "",
    "failed_files":  [],       # list[str] — filenames only, kept for existing consumers
    "failures":      [],       # list[{filename, error}] — per-document reason
}


async def get_pending_summary_files() -> list:
    """
    Returns filenames of GRs that are embedded but have no summary yet.
    Compares gr_metadata (embedded=True) against existing summary JSON files.
    This is what powers the "X documents need summarizing" banner.
    """
    all_metadata = await get_all_gr_metadata()
    existing_summaries = list_summaries()
    summarized_filenames = {s["filename"] for s in existing_summaries}

    pending = [
        m["filename"] for m in all_metadata
        if m.get("embedded") and m["filename"] not in summarized_filenames
    ]
    return pending


async def _run_summary_job(pdf_path: str, filename: str):
    """
    Background job — runs process_gr() and updates state.
    Called via BackgroundTasks so endpoint returns immediately.
    """
    global _summary_state

    def on_progress(step: str):
        """Called by process_gr() at each step."""
        _summary_state["current_step"] = step
        _summary_state["last_message"] = step

    result = await process_gr(
        pdf_path=pdf_path,
        progress_callback=on_progress,
    )

    if result["success"]:
        _summary_state["running"]      = False
        _summary_state["last_status"]  = "done"
        _summary_state["last_message"] = "Summary generated successfully."
        _summary_state["result"]       = {
            "metadata": result["metadata"],
            "summary":  result["summary"],
            "txt_path": result.get("txt_path", ""),
        }
    else:
        _summary_state["running"]      = False
        _summary_state["last_status"]  = "failed"
        _summary_state["last_message"] = result["message"]
        _summary_state["result"]       = None


@router.post("/generate")
async def generate_summary(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    admin: dict = Depends(get_admin_user),
):
    """
    Starts summary generation for a specific GR PDF.
    Admin only.

    Returns immediately — frontend polls /status for completion.
    Result is stored in _summary_state["result"] when done.
    """
    global _summary_state

    # ── Arrival trace ─────────────────────────────────────
    # Before the running-check and the filesystem probe, so even a rejected
    # request is visible the instant it lands.
    trace(
        "REQUEST  POST /api/summary/generate",
        f"admin={admin['username']} file={request.filename!r}",
    )

    # Don't start if already running
    if _summary_state["running"]:
        return {
            "success": False,
            "message": "Summary generation already running.",
            "state":   _summary_state,
        }

    # Validate the filename before touching the filesystem. Unlike the
    # download route, this name arrives in a JSON body rather than a URL
    # path param, so Starlette's [^/]+ converter never sees it and plain
    # forward slashes pass straight through — "../../.env" would reach
    # the .exists() check below as-is, turning it into an existence
    # oracle for arbitrary paths (and, for any PDF-parseable file, a
    # content disclosure once _run_summary_job summarizes it).
    pdf_path = _safe_path(settings.GRDOCS_PATH, request.filename)
    if not pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{request.filename}' not found in grdocs folder.",
        )

    # Reset state
    _summary_state.update({
        "running":      True,
        "last_status":  "running",
        "last_message": f"Starting summary for {request.filename}...",
        "current_step": "Initializing",
        "filename":     request.filename,
        "result":       None,
    })

    # Run in background
    background_tasks.add_task(_run_summary_job, str(pdf_path), request.filename)

    return {
        "success": True,
        "message": f"Summary generation started for '{request.filename}'. Poll /api/summary/status.",
        "state":   _summary_state,
    }


@router.get("/status")
async def get_summary_status(
    admin: dict = Depends(get_admin_user),
):
    """
    Returns current summary generation status.
    Frontend polls this every 3 seconds while job runs.
    When last_status is "done", result contains metadata + summary.
    """
    return {
        "success": True,
        "state":   _summary_state,
    }


@router.get("/pending")
async def get_pending_summaries(
    current_user: dict = Depends(get_current_user),
):
    """
    Returns list of embedded GRs that don't have a summary yet.
    Used by KnowledgeBase.jsx to show the persistent
    "N documents need summarizing" banner.
    """
    pending = await get_pending_summary_files()
    return {
        "success": True,
        "pending": pending,
        "total":   len(pending),
    }


async def _run_batch_summary_job(filenames: list):
    """
    Background job — runs process_gr() for each pending file in sequence.
    Sequential (not parallel) because each summary is a heavy LLM call —
    running them in parallel would overload Ollama on modest hardware.
    """
    global _batch_state

    _batch_state["running"]      = True
    _batch_state["last_status"]  = "running"
    _batch_state["total_files"]  = len(filenames)
    _batch_state["completed"]    = 0
    _batch_state["failed_files"] = []
    _batch_state["failures"]     = []
    _batch_state["progress"]     = 0

    for i, filename in enumerate(filenames, 1):
        _batch_state["current_file"] = filename
        _batch_state["last_message"] = f"Summarizing {filename} ({i}/{len(filenames)})"

        pdf_path = settings.GRDOCS_PATH / filename
        print(f"\n─── batch {i}/{len(filenames)}: {filename} ───", flush=True)

        try:
            result = await process_gr(str(pdf_path))
            if not result["success"]:
                # process_gr caught the exception itself and collapsed it
                # into a message — this, not the except below, is the normal
                # failure path. The message used to be dropped on the floor.
                reason = result.get("message", "process_gr returned success=False with no message")
                print(f"❌ FAILED (handled): {filename}\n   {reason}", flush=True)
                _batch_state["failed_files"].append(filename)
                _batch_state["failures"].append({"filename": filename, "error": reason})
        except Exception as e:
            # Anything process_gr did not catch — traceback is still live here.
            reason = f"{type(e).__name__}: {e}"
            print(f"❌ FAILED (uncaught): {filename}\n   {reason}", flush=True)
            traceback.print_exc()
            _batch_state["failed_files"].append(filename)
            _batch_state["failures"].append({"filename": filename, "error": reason})

        _batch_state["completed"] = i
        _batch_state["progress"]  = round((i / len(filenames)) * 100)

    _batch_state["running"]     = False
    _batch_state["last_status"] = "failed" if _batch_state["failed_files"] else "done"
    fail_note = (
        f" {len(_batch_state['failed_files'])} failed."
        if _batch_state["failed_files"] else ""
    )
    _batch_state["last_message"] = f"Done. {len(filenames)} document(s) processed.{fail_note}"


@router.post("/generate-batch")
async def generate_batch_summaries(
    background_tasks: BackgroundTasks,
    request: BatchRequest | None = None,
    admin: dict = Depends(get_admin_user),
):
    """
    Starts batch summary generation.
    Admin only. Triggered by the "Summarize All" / "Summarize Selected" button.

    Body is optional. With no body this summarizes every embedded GR that
    has no summary yet — the original, unchanged behaviour. With a
    filenames list it summarizes exactly those documents.

    Returns immediately — frontend polls /batch-status for progress.
    """
    global _batch_state

    selected = request.filenames if request else None

    # ── Arrival trace ─────────────────────────────────────
    trace(
        "REQUEST  POST /api/summary/generate-batch",
        f"admin={admin['username']} scope="
        f"{'all pending' if selected is None else f'{len(selected)} selected'}",
    )

    if _batch_state["running"]:
        return {
            "success": False,
            "message": "Batch summarization already running.",
            "state":   _batch_state,
        }

    if selected is not None:
        if not selected:
            return {
                "success": False,
                "message": "No documents selected to summarize.",
                "state":   _batch_state,
            }

        # Validate every name before the job starts. _run_batch_summary_job
        # builds paths from these directly, and it runs in the background
        # where a rejection would only ever surface via polling.
        missing = []
        for name in selected:
            path = _safe_path(settings.GRDOCS_PATH, name)
            if not path.exists():
                missing.append(name)

        if missing:
            return {
                "success": False,
                "message": f"Not found in grdocs folder: {', '.join(missing)}",
                "state":   _batch_state,
            }

        targets = selected
    else:
        targets = await get_pending_summary_files()
        if not targets:
            return {
                "success": False,
                "message": "No documents need summarizing.",
                "state":   _batch_state,
            }

    _batch_state.update({
        "running":      True,
        "last_status":  "running",
        "last_message": f"Starting batch summary for {len(targets)} document(s)...",
        "progress":     0,
        "total_files":  len(targets),
        "current_file": "",
        "failed_files": [],
        "failures":     [],
    })

    background_tasks.add_task(_run_batch_summary_job, targets)

    return {
        "success": True,
        "message": f"Batch summarization started for {len(targets)} document(s).",
        "state":   _batch_state,
    }


@router.get("/batch-status")
async def get_batch_status(
    admin: dict = Depends(get_admin_user),
):
    """Returns current batch summarization progress. Polled every 3s."""
    return {
        "success": True,
        "state":   _batch_state,
    }


@router.get("/list")
async def get_summaries_list(
    current_user: dict = Depends(get_current_user),
):
    """
    Returns list of all previously generated summaries.
    Used to populate the past summaries tab on the Summary page.
    """
    summaries = list_summaries()
    return {
        "success":   True,
        "summaries": summaries,
        "total":     len(summaries),
    }


@router.get("/download/{filename}")
async def download_summary(
    filename: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Downloads a summary TXT file.
    Frontend triggers this when admin clicks Download button.

    Args:
        filename : the TXT filename e.g. "GR_2024_transfer_summary.txt"
    """
    # Validate before touching the filesystem — `filename` comes straight
    # from the URL. Without this, any logged-in user could read arbitrary
    # files, including backend/.env and its JWT_SECRET.
    file_path = _safe_path(settings.SUMMARIES_PATH, filename)

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Summary file '{filename}' not found.",
        )

    # FileResponse streams the file directly to the browser
    # media_type forces browser to download rather than display
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="text/plain",
    )