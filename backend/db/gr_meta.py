# backend/db/gr_meta.py
# ─────────────────────────────────────────────────────────
# All MongoDB operations related to GR document metadata.
# When a PDF is uploaded → record created here.
# When embedding runs → record updated here.
# Admin dashboard reads from here for stats and file listings.
# ─────────────────────────────────────────────────────────

import sys
import os
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.mongo import get_db


async def save_gr_metadata(
    filename: str,
    uploaded_by: str,
    file_size_kb: float,
    page_count: int = 0,
) -> dict:
    """
    Creates a metadata record when a GR PDF is uploaded.

    Args:
        filename     : original filename e.g. "GR_2024_transfer.pdf"
        uploaded_by  : username of the admin who uploaded it
        file_size_kb : file size in kilobytes
        page_count   : number of pages (0 if not yet counted)

    Returns:
        The created document as a dict
    """
    database = get_db()

    # Check if this filename already exists
    existing = await database.gr_metadata.find_one({"filename": filename})
    if existing:
        # Return existing record — don't create duplicate
        existing.pop("_id", None)
        return existing

    doc = {
        "filename":      filename,
        "uploaded_by":   uploaded_by,
        "file_size_kb":  file_size_kb,
        "page_count":    page_count,
        # embedded = False means this PDF hasn't been added to vector store yet
        # Changes to True after embed step runs
        "embedded":      False,
        "uploaded_at":   datetime.now(timezone.utc),
        "embedded_at":   None,          # filled in when embedding completes
    }

    await database.gr_metadata.insert_one(doc)
    doc.pop("_id", None)               # remove MongoDB internal id before returning
    return doc


async def mark_as_embedded(filename: str) -> None:
    """
    Updates a GR record to show it has been embedded into the vector store.
    Called after embed step completes successfully.

    Args:
        filename : the PDF filename to mark as embedded
    """
    database = get_db()

    await database.gr_metadata.update_one(
        {"filename": filename},                     # find this document
        {"$set": {                                  # update these fields
            "embedded":    True,
            "embedded_at": datetime.now(timezone.utc),
        }}
    )


async def mark_as_not_embedded(filename: str) -> None:
    """
    Clears the embedded flag on a GR record.

    Used as a crash guard during deletion. The vectors are removed from
    FAISS before the metadata record itself is deleted, and those are two
    separate writes. If the process dies between them, this call has
    already run, so the record left behind says "not embedded" — which is
    the truth. Without it the survivor would claim embedded=True while
    having no chunks in the index, and the Knowledge Base would show a
    green "Embedded" badge for a document the retriever cannot see.

    Args:
        filename : the PDF filename to mark as no longer embedded
    """
    database = get_db()

    await database.gr_metadata.update_one(
        {"filename": filename},
        {"$set": {
            "embedded":    False,
            "embedded_at": None,
        }}
    )


async def get_all_gr_metadata() -> list:
    """
    Returns metadata for all uploaded GR documents.
    Used on admin Upload page and Dashboard.

    Returns:
        List of metadata dicts, newest first
    """
    database = get_db()

    cursor = database.gr_metadata.find(
        {},
        {"_id": 0}                      # exclude internal MongoDB _id
    ).sort("uploaded_at", -1)           # newest first

    records = await cursor.to_list(length=500)
    return records


async def get_gr_stats() -> dict:
    """
    Summary counts for the admin dashboard stat cards.

    Returns dict:
        total_uploaded : total GR documents uploaded
        total_embedded : how many have been embedded into vector store
        total_pages    : sum of all page counts across all GRs
    """
    database = get_db()

    total_uploaded = await database.gr_metadata.count_documents({})
    total_embedded = await database.gr_metadata.count_documents({"embedded": True})

    # Sum up all page counts using aggregation
    pipeline = [
        {"$group": {"_id": None, "total_pages": {"$sum": "$page_count"}}}
    ]
    result = await database.gr_metadata.aggregate(pipeline).to_list(length=1)
    total_pages = result[0]["total_pages"] if result else 0

    return {
        "total_uploaded": total_uploaded,
        "total_embedded": total_embedded,
        "total_pages":    total_pages,
    }


async def delete_gr_metadata(filename: str) -> dict:
    """
    Removes a GR metadata record from MongoDB.
    Called if a file is deleted from disk.

    Args:
        filename : the PDF filename to remove

    Returns:
        dict with success status and message
    """
    database = get_db()

    result = await database.gr_metadata.delete_one({"filename": filename})

    if result.deleted_count == 0:
        return {"success": False, "message": "Record not found."}

    return {"success": True, "message": f"Metadata for '{filename}' deleted."}