# backend/api/embed.py
# ─────────────────────────────────────────────────────────
# Embedding endpoints — triggers FAISS vector store creation.
#
# Endpoints:
#   POST /api/embed/start   → start embedding in background (admin only)
#   GET  /api/embed/status  → check if vector store is ready
#
# Why BackgroundTasks?
#   Embedding takes 2-5 minutes depending on number of PDFs.
#   If we ran it synchronously the HTTP request would timeout.
#   BackgroundTasks lets us return "started" immediately,
#   then embedding runs in the background.
#   Frontend polls /status every few seconds to check progress.
# ─────────────────────────────────────────────────────────

import sys
import os
import asyncio
from datetime import datetime, timezone

from pathlib import Path

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_admin_user
from core.vectorstore import embed_all_pdfs, is_ready
from core.gr_graph import build_graph
from db.gr_meta import get_all_gr_metadata, mark_as_embedded
from config import settings

router = APIRouter(prefix="/api/embed", tags=["Embedding"])


class EmbedRequest(BaseModel):
    """
    Optional body for POST /start.

    filenames=None (or no body at all) keeps the original behaviour:
    embed every PDF in grdocs/ and rebuild the index from scratch. A list
    scopes the run to those documents and merges them into the existing
    index instead.
    """
    filenames: list[str] | None = None

# ── Embedding state tracker ───────────────────────────────
# Stored in memory — tracks whether embedding is currently running.
# Simple dict instead of a database because:
#   - it only needs to survive the current server session
#   - embedding runs once, not continuously
#   - no need to persist across restarts
_embed_state = {
    "running":      False,
    "last_status":  "idle",         # idle | running | done | failed
    "last_message": "Not started",
    "total_chunks": 0,
    "started_at":   None,
    "finished_at":  None,
    "current_file": "",
    "progress":     0,              # 0-100 percentage
    "total_files":  0,
}


async def _run_embedding_job(filenames: list = None):
    """
    The actual embedding job — runs in background.
    Updates _embed_state as it progresses so frontend can poll status.

    Args:
        filenames : optional list of PDFs to embed. None means all of them.
                    When given, embed_all_pdfs() merges into the existing
                    index rather than rebuilding it, so the documents that
                    were not selected keep their vectors.

    Steps:
        1. Embed the selected PDFs into FAISS
        2. Mark each PDF as embedded in MongoDB
        3. Build GR relationship graph
        4. Update final state

    Wrapped in try/except/finally: if ANY step throws an unhandled
    exception, "running" must still reset to False, otherwise the
    frontend sees a permanently stuck progress bar and /start refuses
    to start a new job ("already running") forever, requiring a
    server restart to recover.

    The try covers the ENTIRE body — including the initial glob — because
    anything that escapes before the try would hang the job in exactly the
    way this guard exists to prevent.

    Step 1 runs on a worker thread so the event loop stays free and
    GET /api/embed/status keeps answering while the job runs.
    """
    global _embed_state

    try:
        _embed_state["running"]     = True
        _embed_state["last_status"] = "running"
        _embed_state["started_at"]  = str(datetime.now(timezone.utc))
        _embed_state["progress"]    = 0

        # Count total files for progress tracking.
        # Step 2 below marks these as embedded, so this list must be the
        # files the job actually processed — not every PDF on disk. A
        # scoped run that marked the whole folder embedded would flag
        # documents that were never touched.
        if filenames is None:
            pdf_files = list(settings.GRDOCS_PATH.glob("*.pdf"))
        else:
            pdf_files = [settings.GRDOCS_PATH / name for name in filenames]
        _embed_state["total_files"] = len(pdf_files)

        def on_progress(filename: str, current: int, total: int):
            """
            Called by embed_all_pdfs() after each PDF is processed.

            Runs on the worker thread, not the event loop. Each statement
            is a single dict item assignment, which is atomic under the
            GIL, so the /status poller can never observe a torn value.
            """
            _embed_state["current_file"] = filename
            _embed_state["progress"]     = round((current / total) * 80)  # 0-80%
            # NOTE: this callback fires only while PDFs are being LOADED.
            # It is never called during the embed call itself, so progress
            # necessarily parks at 80% / "N/N" for the whole embedding
            # phase. A frozen bar there is expected and says nothing about
            # whether Ollama is working — watch the EMBED START/heartbeat/
            # EMBED DONE lines in the server terminal instead.
            _embed_state["last_message"] = f"Loading {filename} ({current}/{total})"

        # ── Step 1: Embed PDFs ────────────────────────────
        # embed_all_pdfs() is declared `async def` but contains no `await`:
        # PDF loading, Tesseract OCR and FAISS.from_documents() are all
        # fully synchronous and can run for minutes to hours. Awaiting it
        # directly pins the event loop for that entire duration, so
        # GET /api/embed/status — polled every 3s by the frontend — cannot
        # respond, the progress bar freezes at 0%, and every other request
        # to the API hangs behind it.
        #
        # Moving it to a worker thread keeps the loop free. Note it is a
        # coroutine function, so calling it merely builds a coroutine
        # object; asyncio.run() drives that coroutine to completion on the
        # worker thread's own event loop. embed_all_pdfs' internals are
        # unchanged — only the invocation moved.
        def _embed_on_worker_thread():
            return asyncio.run(embed_all_pdfs(
                progress_callback=on_progress,
                filenames=filenames,
            ))

        result = await asyncio.to_thread(_embed_on_worker_thread)

        if not result["success"]:
            _embed_state["last_status"]  = "failed"
            _embed_state["last_message"] = result["message"]
            return

        _embed_state["total_chunks"] = result["total_chunks"]
        _embed_state["progress"]     = 85
        _embed_state["last_message"] = "Updating document records..."

        # ── Step 2: Mark files as embedded in MongoDB ─────
        for pdf_file in pdf_files:
            if pdf_file.name not in result.get("failed_files", []):
                await mark_as_embedded(pdf_file.name)

        _embed_state["progress"]     = 90
        _embed_state["last_message"] = "Building GR relationship graph..."

        # ── Step 3: Build GR graph ────────────────────────
        # NOT moved to a worker thread: build_graph() awaits Motor
        # (gr_graph.replace_one), and a Motor client is bound to the event
        # loop that created it — driving it on a second loop in another
        # thread would break. It does still block the loop while scanning
        # PDFs, so /status can stall during this step. Fixing that means
        # splitting build_graph() into a sync scan plus an async write,
        # which is a change to its internals and out of scope here.
        # Always rebuilds from every PDF on disk, including after a scoped
        # embed. That is correct and intentional: relationships are edges
        # BETWEEN documents, so a GR that was not re-embedded can still gain
        # or lose an edge because a selected one changed.
        graph_result = await build_graph()

        # ── Step 4: Mark complete ─────────────────────────
        _embed_state["last_status"]  = "done"
        _embed_state["progress"]     = 100
        _embed_state["last_message"] = (
            f"Done. {result['total_chunks']} chunks embedded. "
            f"Graph: {graph_result.get('nodes', 0)} GRs, "
            f"{graph_result.get('edges', 0)} relationships."
        )

    except Exception as e:
        # Catch anything unhandled from steps 1-3 (Mongo write failure,
        # PDF read error in build_graph, etc.) so the job never hangs silently.
        _embed_state["last_status"]  = "failed"
        _embed_state["last_message"] = f"Embedding job crashed: {str(e)}"
        print(f"❌ Embedding job crashed: {e}")

    finally:
        # This ALWAYS runs, success or failure — guarantees the UI
        # never gets stuck showing "running" forever.
        _embed_state["running"]     = False
        _embed_state["finished_at"] = str(datetime.now(timezone.utc))


@router.post("/start")
async def start_embedding(
    background_tasks: BackgroundTasks,
    request: EmbedRequest | None = None,
    admin: dict = Depends(get_admin_user),
):
    """
    Starts the embedding pipeline in the background.
    Returns immediately — frontend polls /status for progress.
    Admin only.

    Body is optional. With no body (or filenames=null) this embeds every
    PDF and rebuilds the index — the original, unchanged behaviour. With
    a filenames list it embeds only those and merges them into the
    existing index.

    If embedding is already running, returns current status
    without starting a new job.
    """
    global _embed_state

    # Don't start if already running
    if _embed_state["running"]:
        return {
            "success": False,
            "message": "Embedding is already running. Check /status for progress.",
            "state":   _embed_state,
        }

    selected = request.filenames if request else None

    if selected is not None:
        if not selected:
            return {
                "success": False,
                "message": "No documents selected to embed.",
                "state":   _embed_state,
            }

        # Validate every name before starting. The job runs in the
        # background, so a bad name caught later would surface only as a
        # failed status the caller has to poll for.
        missing = []
        for name in selected:
            if not name or Path(name).name != name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid filename: {name!r}",
                )
            if not (settings.GRDOCS_PATH / name).exists():
                missing.append(name)

        if missing:
            return {
                "success": False,
                "message": f"Not found in grdocs folder: {', '.join(missing)}",
                "state":   _embed_state,
            }

        pdf_files = selected
        scope_note = f"{len(selected)} selected PDF(s)"
    else:
        # Check there are PDFs to embed
        pdf_files = [f.name for f in settings.GRDOCS_PATH.glob("*.pdf")]
        if not pdf_files:
            return {
                "success": False,
                "message": "No PDF files found. Upload some GR documents first.",
                "state":   _embed_state,
            }
        scope_note = f"{len(pdf_files)} PDF(s)"

    # Reset state for new run
    _embed_state.update({
        "running":      True,
        "last_status":  "running",
        "last_message": f"Starting embedding pipeline for {scope_note}...",
        "total_chunks": 0,
        "current_file": "",
        "progress":     0,
        "total_files":  len(pdf_files),
    })

    # Add to background tasks — runs after this function returns
    background_tasks.add_task(_run_embedding_job, selected)

    return {
        "success": True,
        "message": f"Embedding started for {scope_note}. Poll /api/embed/status for progress.",
        "state":   _embed_state,
    }


@router.get("/status")
async def get_embed_status(
    admin: dict = Depends(get_admin_user),
):
    """
    Returns current embedding status.
    Frontend polls this every 3 seconds while embedding runs.

    Response includes:
        running      : bool — is embedding currently in progress
        last_status  : "idle" | "running" | "done" | "failed"
        progress     : 0-100 percentage
        last_message : human readable status message
        vector_ready : bool — is the FAISS index ready to use
    """
    return {
        "success":      True,
        "state":        _embed_state,
        "vector_ready": is_ready(),
    }