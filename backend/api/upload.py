# backend/api/upload.py
# ─────────────────────────────────────────────────────────
# GR Document Upload endpoints.
#
# Endpoints:
#   POST /api/upload        → upload a GR PDF file (admin only)
#   GET  /api/upload/list   → list all uploaded GR files
#   DELETE /api/upload/{filename} → delete a GR file (admin only)
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_admin_user, get_current_user
from db.gr_meta import save_gr_metadata, get_all_gr_metadata, delete_gr_metadata, get_gr_stats
from config import settings

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


@router.delete("/{filename}")
async def delete_uploaded_file(
    filename: str,
    admin: dict = Depends(get_admin_user),
):
    # Validate before unlink() — `filename` comes straight from the URL,
    # so without this "..%2F..%2Fconfig.py" would delete arbitrary files.
    file_path = _safe_path(settings.GRDOCS_PATH, filename)

    if file_path.exists():
        file_path.unlink()

    result = await delete_gr_metadata(filename)

    # Also remove any generated summary files for this GR —
    # otherwise Summaries page keeps showing a summary for a GR that no longer exists
    base_name = Path(filename).stem
    for ext in ("_summary.json", "_summary.txt"):
        # Derived names are validated too: base_name comes from the same
        # untrusted input, so it gets the same containment check.
        summary_file = _safe_path(settings.SUMMARIES_PATH, f"{base_name}{ext}")
        if summary_file.exists():
            summary_file.unlink()

    return {
        "success": True,
        "message": f"File '{filename}' deleted.",
    }