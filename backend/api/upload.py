# backend/api/upload.py
# ─────────────────────────────────────────────────────────
# GR Document Upload endpoints.
#
# Endpoints:
#   POST /api/upload            → upload a GR PDF file (admin only)
#   GET  /api/upload/list       → list all uploaded GR files
#   GET  /api/upload/integrity  → report orphaned/partial state (admin only)
#   POST /api/upload/delete     → delete several GR files (admin only)
#   DELETE /api/upload/{filename} → delete one GR file (admin only)
#
# Both delete routes go through core/purge.py, which removes the PDF, its
# FAISS chunks, its Mongo record, its summary and its graph entries as one
# ordered operation and then verifies nothing was left behind.
#
# Files are saved to backend/data/grdocs/
# Metadata is saved to MongoDB gr_metadata collection
# ─────────────────────────────────────────────────────────

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_admin_user, get_current_user
from db.gr_meta import save_gr_metadata, get_all_gr_metadata, delete_gr_metadata, get_gr_stats
from core.purge import purge_documents, check_integrity
from config import settings


class DeleteRequest(BaseModel):
    """Body of POST /api/upload/delete — the selected documents."""
    filenames: list[str]

router = APIRouter(prefix="/api/upload", tags=["Upload"])

# Only these file types are accepted
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}
# Max file size — 50MB
MAX_FILE_SIZE_MB = 50


def _safe_path(base: Path, filename: str) -> Path:
    """
    Resolves `filename` inside `base`, rejecting path traversal.

    Two gates:
      1. Path(filename).name strips any directory component, so a mismatch
         means the input was not a bare filename. Note that FastAPI
         percent-decodes path params AFTER routing, so a request for
         "..%2F..%2Fconfig.py" arrives here already decoded as
         "../../config.py" and is caught here.
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


@router.post("/")
async def upload_gr_file(
    file: UploadFile = File(...),
    admin: dict = Depends(get_admin_user),
):
    """
    Upload a GR document (PDF/DOC/DOCX).
    Admin only.

    The frontend sends this as multipart/form-data.
    FastAPI handles the parsing automatically via UploadFile.

    Steps:
        1. Validate file extension
        2. Validate file size
        3. Save file to grdocs/ folder
        4. Save metadata to MongoDB
        5. Return confirmation
    """
    # ── Step 0: Sanitize the client-supplied filename ─────
    # file.filename comes from the multipart Content-Disposition header
    # and is fully attacker-controlled — "../../../evil.pdf" would
    # otherwise be written outside GRDOCS_PATH. .name strips any
    # directory component; everything below uses safe_name, never
    # file.filename.
    safe_name = Path(file.filename or "").name
    if not safe_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    # ── Step 1: Validate extension ────────────────────────
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{suffix}' not allowed. Only PDF, DOC, DOCX accepted.",
        )

    # ── Step 2: Read file and check size ──────────────────
    contents = await file.read()
    size_mb   = len(contents) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f}MB). Maximum allowed is {MAX_FILE_SIZE_MB}MB.",
        )

    size_kb = round(len(contents) / 1024, 1)

    # ── Step 3: Save to disk ──────────────────────────────
    settings.GRDOCS_PATH.mkdir(parents=True, exist_ok=True)
    destination = _safe_path(settings.GRDOCS_PATH, safe_name)

    # If file already exists add timestamp suffix to avoid overwrite
    if destination.exists():
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem      = Path(safe_name).stem
        dest_name = f"{stem}_{ts}{suffix}"
        destination = _safe_path(settings.GRDOCS_PATH, dest_name)
    else:
        dest_name = safe_name

    with open(destination, "wb") as f:
        f.write(contents)

    # ── Step 4: Count pages (PDF only) ───────────────────
    page_count = 0
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader     = PdfReader(str(destination))
            page_count = len(reader.pages)
        except Exception:
            page_count = 0     # not critical if page count fails

    # ── Step 5: Save metadata to MongoDB ─────────────────
    await save_gr_metadata(
        filename=dest_name,
        uploaded_by=admin["username"],
        file_size_kb=size_kb,
        page_count=page_count,
    )

    return {
        "success":    True,
        "message":    f"File '{dest_name}' uploaded successfully.",
        "filename":   dest_name,
        "size_kb":    size_kb,
        "page_count": page_count,
    }


@router.get("/list")
async def list_uploaded_files(
    current_user: dict = Depends(get_current_user),
):
    """
    Returns list of all uploaded GR files with metadata.
    Available to all logged-in users (not admin only)
    so the embed page and chat page can show available GRs.
    """
    records = await get_all_gr_metadata()

    # Also check which files actually exist on disk
    # MongoDB record might exist but file could have been deleted
    verified = []
    for record in records:
        file_path = settings.GRDOCS_PATH / record["filename"]
        record["exists_on_disk"] = file_path.exists()
        verified.append(record)

    return {
        "success": True,
        "files":   verified,
        "total":   len(verified),
    }


@router.get("/stats")
async def get_upload_stats(
    admin: dict = Depends(get_admin_user),
):
    """
    Returns summary stats for admin dashboard.
    Admin only.
    """
    stats = await get_gr_stats()
    return {"success": True, **stats}


@router.get("/integrity")
async def get_integrity_report(
    admin: dict = Depends(get_admin_user),
):
    """
    Cross-checks disk, the FAISS index and MongoDB, and reports every
    disagreement between them. Read-only — it changes nothing.

    This is what makes a partial-deletion state visible instead of silent.
    The category that matters most is orphaned_chunks: vectors whose PDF is
    gone are still retrievable and will still be cited, with nothing in the
    UI to say the source no longer exists.

    Admin only.
    """
    report = await check_integrity()
    return {"success": True, **report}


@router.post("/delete")
async def delete_selected_files(
    request: DeleteRequest,
    admin: dict = Depends(get_admin_user),
):
    """
    Deletes several GR documents and every trace of them.
    Admin only. Used by the Knowledge Base checkbox selection.

    Runs as ONE purge rather than a loop of single deletes, so the FAISS
    index is loaded, edited and written exactly once no matter how many
    documents are selected. A loop would rewrite the index N times and give
    N separate chances to fail partway through the batch.
    """
    if not request.filenames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents selected.",
        )

    # Reject unsafe names up front, with the same gate the single-file
    # route uses, so a bad name never reaches the filesystem.
    for name in request.filenames:
        _safe_path(settings.GRDOCS_PATH, name)

    result = await purge_documents(request.filenames)
    return result


@router.delete("/{filename}")
async def delete_uploaded_file(
    filename: str,
    admin: dict = Depends(get_admin_user),
):
    """
    Deletes one GR document and every trace of it.
    Admin only.

    Removes, in this order: FAISS chunks, the embedded flag, graph entries,
    summary files, the Mongo record, then the PDF. The order is what
    guarantees a crash cannot leave vectors behind for a file that is gone —
    see core/purge.py for why that direction and not the other.
    """
    # Validate before anything else — `filename` comes straight from the URL,
    # so without this "..%2F..%2Fconfig.py" would delete arbitrary files.
    _safe_path(settings.GRDOCS_PATH, filename)

    result = await purge_documents([filename])
    return result