# backend/core/purge.py
# ─────────────────────────────────────────────────────────
# Complete removal of a GR document from every place it exists.
#
# A GR leaves traces in five separate stores:
#   1. the PDF on disk           (data/grdocs/<name>.pdf)
#   2. chunks in the FAISS index (data/vectorstore/)
#   3. the metadata record       (MongoDB gr_metadata)
#   4. the generated summary     (data/summaries/<stem>_summary.{json,txt})
#   5. graph nodes and edges     (MongoDB gr_graph, via core/gr_graph.py)
#
# Deleting from some but not all of these is worse than not deleting at
# all, so the ordering below is chosen deliberately. See purge_documents().
#
# Called by:
#   api/upload.py → DELETE /api/upload/{filename} and POST /api/upload/delete
# ─────────────────────────────────────────────────────────

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.vectorstore import delete_from_index, list_indexed_sources
from core.gr_graph import remove_from_graph
from db.gr_meta import (
    get_all_gr_metadata,
    delete_gr_metadata,
    mark_as_not_embedded,
)


def _safe_name(filename: str) -> str:
    """
    Rejects anything that is not a bare filename.

    Every path this module builds is derived from caller-supplied text, and
    it deletes files. Path(filename).name strips directory components, so a
    mismatch means the input tried to carry one.

    Raises:
        ValueError if the filename is unsafe.
    """
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Invalid filename: {filename!r}")
    return filename


def _contained(base: Path, filename: str) -> Path:
    """
    Resolves `filename` inside `base` and confirms it stayed there.

    Second gate after _safe_name — covers symlinks and platform quirks that
    a name-only check cannot see.

    Raises:
        ValueError if the resolved path escapes `base`.
    """
    resolved = (base / filename).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"Invalid filename: {filename!r}")
    return resolved


def summary_paths(filename: str) -> list:
    """
    The summary artefacts for one GR: the JSON and the TXT.

    Both are named from the PDF's stem by core/summarizer.py, so they are
    derived here the same way.
    """
    stem = Path(_safe_name(filename)).stem
    return [
        _contained(settings.SUMMARIES_PATH, f"{stem}{ext}")
        for ext in ("_summary.json", "_summary.txt")
    ]


async def purge_documents(filenames: list) -> dict:
    """
    Removes `filenames` and every trace of them, or removes nothing.

    ── Why this order ────────────────────────────────────
    The five stores cannot be written in one transaction, so the ordering is
    chosen so that any crash leaves the *harmless* inconsistency rather than
    the dangerous one. The two failure shapes are not symmetric:

      file gone, vectors remain
          The retriever still matches those chunks and still cites the
          document. Answers keep quoting text from a GR that no longer
          exists, and nothing in the UI reveals it. This is silent
          corruption — exactly what must never happen.

      vectors gone, file remains
          The document simply stops being searchable until the next embed,
          and shows up in the Knowledge Base as "Pending". Visible,
          harmless, fixed by re-embedding.

    So the vectors go FIRST and the file goes LAST, and every intermediate
    step moves from "more authoritative" to "less":

        1. FAISS chunks      — the only store that can corrupt answers
        2. embedded=False    — crash guard: a surviving record now tells
                               the truth about having no chunks
        3. graph nodes/edges — stops citation clicks resolving to it
        4. summary files     — stops the Summaries page listing it
        5. Mongo record      — it disappears from the document list
        6. PDF on disk       — last, because everything above keys off it

    ── Why step 1 aborts the whole operation ─────────────
    If FAISS removal fails, nothing else is touched at all. The document
    stays fully present and fully consistent, and the caller sees a plain
    error. The alternative — carrying on and deleting the file anyway —
    manufactures the exact orphaned-chunk state this function exists to
    prevent.

    Args:
        filenames : bare PDF filenames, e.g. ["GR1.3.pdf"]

    Returns:
        dict with success, deleted, failed, steps, verification
    """
    # Validate every name before touching anything. A bad name in position 3
    # must not leave the first two half-deleted.
    try:
        names = [_safe_name(f) for f in filenames]
    except ValueError as e:
        return {
            "success":  False,
            "deleted":  [],
            "failed":   list(filenames),
            "message":  str(e),
            "steps":    {},
        }

    if not names:
        return {
            "success": False,
            "deleted": [],
            "failed":  [],
            "message": "No documents specified.",
            "steps":   {},
        }

    steps = {}

    # ── Step 1: FAISS chunks ──────────────────────────────
    index_result   = delete_from_index(names)
    steps["index"] = index_result

    if not index_result["success"]:
        # Hard stop — see docstring. Nothing else has been modified.
        return {
            "success": False,
            "deleted": [],
            "failed":  names,
            "message": (
                f"Aborted before deleting anything: the vector index could not be "
                f"updated ({index_result['message']}). The documents are untouched "
                f"and the system is still consistent."
            ),
            "steps":   steps,
        }

    # ── Step 2: crash guard ───────────────────────────────
    # The chunks are gone as of step 1. Until step 5 removes the record,
    # that record must not keep claiming the document is embedded.
    for name in names:
        try:
            await mark_as_not_embedded(name)
        except Exception as e:
            print(f"⚠️ Could not clear embedded flag for {name}: {e}")

    # ── Step 3: graph ─────────────────────────────────────
    try:
        steps["graph"] = await remove_from_graph(names)
    except Exception as e:
        # Non-fatal: a stale node is cosmetic and the next build_graph()
        # rewrites the graph from disk anyway. Recorded, not raised.
        steps["graph"] = {
            "success": False,
            "message": f"Graph cleanup failed: {type(e).__name__}: {e}",
        }

    # ── Step 4: summary files ─────────────────────────────
    removed_summaries = []
    summary_errors    = []
    for name in names:
        for path in summary_paths(name):
            if path.exists():
                try:
                    path.unlink()
                    removed_summaries.append(path.name)
                except Exception as e:
                    summary_errors.append(f"{path.name}: {e}")

    steps["summaries"] = {
        "success": not summary_errors,
        "removed": removed_summaries,
        "errors":  summary_errors,
        "message": f"Removed {len(removed_summaries)} summary file(s).",
    }

    # ── Step 5: Mongo metadata ────────────────────────────
    removed_records = []
    record_errors   = []
    for name in names:
        try:
            result = await delete_gr_metadata(name)
            # "Record not found" is fine — the goal is absence, and a file
            # uploaded before metadata existed has nothing to delete. Only
            # count the ones that actually had a record.
            if result.get("success"):
                removed_records.append(name)
        except Exception as e:
            record_errors.append(f"{name}: {type(e).__name__}: {e}")

    steps["metadata"] = {
        "success": not record_errors,
        "removed": removed_records,
        "errors":  record_errors,
        "message": f"Removed {len(removed_records)} metadata record(s).",
    }

    # ── Step 6: the PDF itself ────────────────────────────
    deleted    = []
    failed     = []
    file_errors = []
    for name in names:
        try:
            path = _contained(settings.GRDOCS_PATH, name)
            if path.exists():
                path.unlink()
            deleted.append(name)
        except Exception as e:
            failed.append(name)
            file_errors.append(f"{name}: {type(e).__name__}: {e}")

    steps["file"] = {
        "success": not file_errors,
        "removed": deleted,
        "errors":  file_errors,
        "message": f"Deleted {len(deleted)} file(s) from disk.",
    }

    # ── Verification ──────────────────────────────────────
    # Do not report success on the strength of having made the calls —
    # go back and look. This is what turns "we ran the deletes" into
    # "no trace of these documents remains".
    verification = await verify_absent(names)

    success = bool(verification["clean"]) and not failed

    message = (
        f"Deleted {len(deleted)} document(s). "
        f"{index_result['removed']} chunk(s) removed from the vector index, "
        f"{len(removed_records)} metadata record(s), "
        f"{len(removed_summaries)} summary file(s)."
    )
    if not verification["clean"]:
        message += (
            f" ⚠️ Verification found leftover traces: "
            f"{'; '.join(verification['leftovers'])}"
        )
    if file_errors:
        message += f" File errors: {'; '.join(file_errors)}"

    return {
        "success":      success,
        "deleted":      deleted,
        "failed":       failed,
        "message":      message,
        "steps":        steps,
        "verification": verification,
    }


async def verify_absent(filenames: list) -> dict:
    """
    Re-reads all five stores and reports any surviving trace of `filenames`.

    Called at the end of purge_documents() so success is asserted from
    observed state rather than from the delete calls having returned.

    Returns:
        dict with clean (bool) and leftovers (list of human-readable strings)
    """
    leftovers = []
    targets   = set(filenames)

    # 1. PDFs on disk
    for name in filenames:
        try:
            if (settings.GRDOCS_PATH / name).exists():
                leftovers.append(f"{name}: file still on disk")
        except Exception as e:
            leftovers.append(f"{name}: could not check disk ({e})")

    # 2. FAISS chunks
    try:
        indexed = list_indexed_sources()
        for name in filenames:
            if indexed.get(name):
                leftovers.append(f"{name}: {indexed[name]} chunk(s) still in the index")
    except Exception as e:
        leftovers.append(f"could not read vector index ({e})")

    # 3. Mongo metadata
    try:
        records = await get_all_gr_metadata()
        for record in records:
            if record.get("filename") in targets:
                leftovers.append(f"{record['filename']}: metadata record still present")
    except Exception as e:
        leftovers.append(f"could not read metadata ({e})")

    # 4. summary files
    for name in filenames:
        try:
            for path in summary_paths(name):
                if path.exists():
                    leftovers.append(f"{name}: summary file {path.name} still present")
        except Exception as e:
            leftovers.append(f"{name}: could not check summaries ({e})")

    # 5. graph nodes
    try:
        from core.gr_graph import get_graph
        graph = await get_graph()
        stems = {Path(n).stem for n in filenames}
        for node in graph.get("nodes", []):
            if node.get("filename") in targets or node.get("id") in stems:
                leftovers.append(f"{node.get('id')}: graph node still present")
    except Exception as e:
        leftovers.append(f"could not read graph ({e})")

    return {"clean": not leftovers, "leftovers": leftovers}


async def check_integrity() -> dict:
    """
    Cross-checks the three stores that must agree and reports every
    mismatch, without changing anything.

    This is the diagnostic that makes a partial-deletion state visible
    rather than silent. It answers, for the whole corpus at once, the
    question purge_documents() answers for one delete.

    Categories:
        orphaned_chunks   : indexed chunks whose PDF is gone. The dangerous
                            one — retrieval will still cite these.
        orphaned_metadata : Mongo records whose PDF is gone.
        untracked_files   : PDFs on disk with no metadata record.
        stale_embedded    : records flagged embedded=True with no chunks in
                            the index (or the reverse).

    Returns:
        dict with clean (bool), the four categories, and counts
    """
    try:
        disk_files = {p.name for p in settings.GRDOCS_PATH.glob("*.pdf")}
    except Exception:
        disk_files = set()

    try:
        indexed = list_indexed_sources()
    except Exception as e:
        indexed = {}
        print(f"⚠️ Integrity check could not read the index: {e}")

    try:
        records = await get_all_gr_metadata()
    except Exception as e:
        records = []
        print(f"⚠️ Integrity check could not read metadata: {e}")

    record_names = {r.get("filename") for r in records}

    orphaned_chunks = [
        {"filename": name, "chunks": count}
        for name, count in sorted(indexed.items())
        if name not in disk_files
    ]

    orphaned_metadata = sorted(record_names - disk_files - {None})

    untracked_files = sorted(disk_files - record_names)

    stale_embedded = []
    for record in records:
        name = record.get("filename")
        if name not in disk_files:
            continue                       # already reported as orphaned metadata
        chunks = indexed.get(name, 0)
        if record.get("embedded") and chunks == 0:
            stale_embedded.append({
                "filename": name,
                "issue":    "flagged embedded but has no chunks in the index",
            })
        elif not record.get("embedded") and chunks > 0:
            stale_embedded.append({
                "filename": name,
                "issue":    f"flagged not-embedded but has {chunks} chunk(s) in the index",
            })

    clean = not (orphaned_chunks or orphaned_metadata or untracked_files or stale_embedded)

    return {
        "clean":             clean,
        "orphaned_chunks":   orphaned_chunks,
        "orphaned_metadata": orphaned_metadata,
        "untracked_files":   untracked_files,
        "stale_embedded":    stale_embedded,
        "totals": {
            "files_on_disk":   len(disk_files),
            "metadata_records": len(records),
            "indexed_sources": len(indexed),
            "indexed_chunks":  sum(indexed.values()),
        },
    }
