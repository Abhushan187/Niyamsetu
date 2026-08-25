# backend/core/gr_graph.py
# ─────────────────────────────────────────────────────────
# GR Supersession and Amendment Graph Detection.
#
# What it does:
#   Scans GR text for phrases like "supersedes GR dated..."
#   or "रद्द करण्यात येत आहे" (Marathi: "is being cancelled")
#   and builds a relationship graph between GR documents.
#
# Why it matters:
#   Without this, if GR-2019 says "10 days leave" and
#   GR-2023 supersedes it with "12 days leave", a naive
#   RAG system might return the outdated GR-2019 answer.
#   This graph lets the system know GR-2023 is the valid one.
#
# Called by:
#   api/embed.py → runs build_graph() after embedding completes
#   api/graph.py → returns graph data to frontend for visualization
# ─────────────────────────────────────────────────────────

import sys
import os
import re
from pathlib import Path
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from db.mongo import get_db


# ── Relationship detection patterns ──────────────────────
# Each pattern is a regex that detects a specific relationship type.
# We scan the first few pages of each GR for these phrases.

SUPERSEDES_PATTERNS = [
    # English patterns
    r"supersedes?\s+(?:the\s+)?(?:GR|Government\s+Resolution|circular|order)[\s\w\-/\.,]+?dated",
    r"in\s+supersession\s+of[\s\w\-/\.,]+",
    r"this\s+(?:order|resolution|circular)\s+supersedes",
    r"(?:hereby\s+)?cancels?\s+(?:the\s+)?(?:GR|Government\s+Resolution|circular)",
    r"stands?\s+cancelled",
    r"(?:hereby\s+)?replaces?\s+(?:the\s+)?(?:GR|Government\s+Resolution)",
    r"previous\s+(?:GR|order|circular)\s+(?:is\s+)?hereby\s+(?:cancelled|superseded)",

    # Marathi patterns — common phrases in Maharashtra GRs
    r"रद्द\s+करण्यात\s+येत",          # "is being cancelled"
    r"अधिक्रमित\s+करण्यात\s+येत",     # "is being superseded"
    r"रद्दबातल\s+करण्यात\s+येत",      # "is being annulled"
    r"निरस्त\s+करण्यात\s+येत",        # "is being revoked"
]

AMENDS_PATTERNS = [
    # English patterns
    r"amends?\s+(?:the\s+)?(?:GR|Government\s+Resolution|circular)",
    r"partial\s+(?:modification|amendment)",
    r"modified\s+to\s+read\s+as",
    r"amendment\s+to\s+(?:the\s+)?(?:GR|Government\s+Resolution)",
    r"(?:hereby\s+)?modifies?\s+(?:the\s+)?(?:GR|Government\s+Resolution)",

    # Marathi patterns
    r"दुरुस्ती\s+करण्यात\s+येत",      # "amendment is being made"
    r"सुधारणा\s+करण्यात\s+येत",       # "modification is being made"
]

REFERS_PATTERNS = [
    # English patterns
    r"(?:with\s+)?reference\s+to\s+(?:GR|Government\s+Resolution|circular)",
    r"vide\s+(?:GR|Government\s+Resolution|circular)",
    r"as\s+per\s+(?:GR|Government\s+Resolution|circular)",
    r"in\s+pursuance\s+of\s+(?:GR|Government\s+Resolution)",
    r"referred\s+to\s+in\s+(?:GR|Government\s+Resolution)",

    # Marathi patterns
    r"संदर्भ\s+क्र",                   # "Reference No."
    r"उपरोक्त\s+शासन\s+निर्णय",       # "above mentioned GR"
    r"संदर्भाधीन\s+शासन\s+निर्णय",    # "referenced GR"
]


def _detect_relationships(text: str, source_gr_id: str) -> list:
    """
    Scans text for relationship phrases and returns detected relationships.

    Args:
        text         : the GR document text (first few pages)
        source_gr_id : identifier of the GR being scanned (filename stem)

    Returns:
        List of relationship dicts:
        [
            {
                "source":       "GR_2024_transfer",
                "relation":     "supersedes",
                "target_hint":  "the matched text snippet",
                "snippet":      "...full context around the match..."
            },
            ...
        ]
    """
    relationships = []

    def scan_patterns(patterns: list, relation_type: str):
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.UNICODE)
            for match in matches:
                # Get surrounding context — 100 chars before and after match
                start   = max(0, match.start() - 50)
                end     = min(len(text), match.end() + 100)
                snippet = text[start:end].strip()

                relationships.append({
                    "source":      source_gr_id,
                    "relation":    relation_type,
                    "target_hint": match.group(0)[:120],
                    "snippet":     snippet[:300],
                })

    scan_patterns(SUPERSEDES_PATTERNS, "supersedes")
    scan_patterns(AMENDS_PATTERNS,     "amends")
    scan_patterns(REFERS_PATTERNS,     "refers_to")

    return relationships


def _match_target_to_known_gr(
    hint: str,
    all_gr_ids: list,
    current_id: str,
) -> str:
    """
    Tries to match a detected relationship hint to a known GR filename.

    For example if we detect "supersedes GR dated 15 March 2023"
    we try to find which uploaded GR file matches that date.

    Args:
        hint       : the text snippet that triggered detection
        all_gr_ids : list of all known GR filename stems
        current_id : the source GR (to avoid self-reference)

    Returns:
        Matched GR id string, or "Unknown" if no match found
    """
    hint_lower = hint.lower()

    for gr_id in all_gr_ids:
        if gr_id == current_id:
            continue   # skip self

        # Check if the GR filename or parts of it appear in the hint
        gr_lower = gr_id.lower()

        # Try matching by filename stem
        if gr_lower in hint_lower:
            return gr_id

        # Try matching by year — if GR filename contains a year
        # and the hint mentions that year, it's likely the same GR
        year_match = re.search(r'(20\d{2})', gr_id)
        if year_match:
            year = year_match.group(1)
            if year in hint_lower:
                return gr_id

    return "Unknown"


async def build_graph() -> dict:
    """
    Main graph building function — called after embedding completes.

    Steps:
        1. Find all PDFs in grdocs/ folder
        2. For each PDF, load first 6 pages (relationship info is usually at top)
        3. Scan text for supersession/amendment/reference patterns
        4. Save nodes and edges to MongoDB
        5. Return summary stats

    Returns:
        dict with success, nodes count, edges count
    """
    database = get_db()
    pdf_files = list(settings.GRDOCS_PATH.glob("*.pdf"))

    if not pdf_files:
        return {"success": False, "message": "No PDFs found.", "nodes": 0, "edges": 0}

    # Build list of all GR identifiers (filename without extension)
    all_gr_ids = [f.stem for f in pdf_files]

    nodes = []
    edges = []

    for pdf_file in pdf_files:
        gr_id = pdf_file.stem

        # Add this GR as a node in the graph
        nodes.append({
            "id":       gr_id,
            "label":    gr_id,
            "filename": pdf_file.name,
        })

        try:
            # Only load first 6 pages — relationship declarations
            # are almost always in the preamble of a GR
            loader = PyPDFLoader(str(pdf_file))
            pages  = loader.load()
            first_pages = pages[:6]
            text = " ".join(p.page_content for p in first_pages)

            # Detect relationships in this GR's text
            relationships = _detect_relationships(text, gr_id)

            for rel in relationships:
                # Try to match the detected hint to a known GR file
                target = _match_target_to_known_gr(
                    rel["target_hint"],
                    all_gr_ids,
                    gr_id,
                )

                edges.append({
                    "source":   gr_id,
                    "target":   target,       # "Unknown" if no match
                    "relation": rel["relation"],
                    "snippet":  rel["snippet"],
                })

        except Exception as e:
            print(f"  ⚠️ Could not scan {pdf_file.name} for relationships: {e}")

    # Save to MongoDB — replace previous graph entirely
    # upsert=True means: insert if not exists, update if exists
    graph_doc = {
        "nodes":      nodes,
        "edges":      edges,
        "built_at":   datetime.now(timezone.utc),
        "pdf_count":  len(pdf_files),
    }

    await database.gr_graph.replace_one(
        {"_id": "main_graph"},              # always use same document id
        {"_id": "main_graph", **graph_doc},
        upsert=True,
    )

    known_edges   = [e for e in edges if e["target"] != "Unknown"]
    unknown_edges = [e for e in edges if e["target"] == "Unknown"]

    print(f"  🕸️ Graph built: {len(nodes)} nodes, {len(edges)} edges "
          f"({len(known_edges)} matched, {len(unknown_edges)} unmatched)")

    return {
        "success":        True,
        "message":        f"Graph built with {len(nodes)} GRs and {len(edges)} relationships detected.",
        "nodes":          len(nodes),
        "edges":          len(edges),
        "matched_edges":  len(known_edges),
    }


async def get_graph() -> dict:
    """
    Retrieves the stored graph from MongoDB.
    Called by api/graph.py to send to frontend.

    Returns:
        dict with nodes list and edges list, or empty graph if not built yet
    """
    database = get_db()
    graph    = await database.gr_graph.find_one({"_id": "main_graph"})

    if not graph:
        return {"nodes": [], "edges": [], "built": False}

    return {
        "nodes":  graph.get("nodes", []),
        "edges":  graph.get("edges", []),
        "built":  True,
        "built_at": str(graph.get("built_at", "")),
    }


async def remove_from_graph(filenames: list) -> dict:
    """
    Removes deleted GRs from the stored relationship graph.

    build_graph() rewrites the whole graph from whatever PDFs are on disk,
    so it would eventually drop these on its own. That is not good enough
    here: the graph is read by api/graph.py on every citation click, and
    between a delete and the next embed it would keep offering nodes for
    documents that no longer exist and edges that resolve to nothing.

    Two different removals are needed, because a GR appears in the graph in
    two roles:

      - as a NODE, and as the SOURCE of edges it declared. Both describe a
        document that is gone, so both are dropped outright.

      - as the TARGET of some *other* GR's edge. That edge belongs to a
        document that still exists and still genuinely says "supersedes
        ...". Deleting it would erase a real statement from a surviving
        document. Instead the target is reset to "Unknown", which is the
        same value build_graph() assigns when it cannot resolve a hint —
        so the edge stays visible and honest, just unresolved.

    Args:
        filenames : bare PDF filenames being deleted, e.g. ["GR1.3.pdf"]

    Returns:
        dict with success, nodes_removed, edges_removed, edges_unlinked
    """
    database = get_db()
    graph    = await database.gr_graph.find_one({"_id": "main_graph"})

    if not graph:
        return {
            "success":        True,
            "nodes_removed":  0,
            "edges_removed":  0,
            "edges_unlinked": 0,
            "message":        "No graph stored — nothing to remove.",
        }

    targets = set(filenames)
    # Nodes are keyed by filename stem, edges reference that same stem.
    stems   = {Path(name).stem for name in targets}

    old_nodes = graph.get("nodes", [])
    old_edges = graph.get("edges", [])

    # Match on filename where the node carries one, falling back to the id,
    # so nodes written before "filename" existed are still cleaned up.
    nodes = [
        n for n in old_nodes
        if n.get("filename") not in targets and n.get("id") not in stems
    ]

    edges          = [e for e in old_edges if e.get("source") not in stems]
    edges_unlinked = 0
    for edge in edges:
        if edge.get("target") in stems:
            edge["target"] = "Unknown"
            edges_unlinked += 1

    await database.gr_graph.update_one(
        {"_id": "main_graph"},
        {"$set": {
            "nodes":     nodes,
            "edges":     edges,
            "pdf_count": len(nodes),
        }},
    )

    result = {
        "success":        True,
        "nodes_removed":  len(old_nodes) - len(nodes),
        "edges_removed":  len(old_edges) - len(edges),
        "edges_unlinked": edges_unlinked,
    }
    result["message"] = (
        f"Graph: removed {result['nodes_removed']} node(s) and "
        f"{result['edges_removed']} edge(s), unlinked {edges_unlinked} incoming edge(s)."
    )
    return result


async def get_relationships_for_gr(gr_id: str) -> list:
    """
    Returns all edges where this GR is source or target.
    Used when user clicks a node in the graph visualization.

    Args:
        gr_id : the GR filename stem to look up

    Returns:
        List of edge dicts related to this GR
    """
    database = get_db()
    graph    = await database.gr_graph.find_one({"_id": "main_graph"})

    if not graph:
        return []

    edges = graph.get("edges", [])

    # Return edges where this GR is either the source or target
    related = [
        e for e in edges
        if e["source"] == gr_id or e["target"] == gr_id
    ]

    return related