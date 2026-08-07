# backend/api/graph.py
# ─────────────────────────────────────────────────────────
# GR Relationship Graph endpoints.
#
# Endpoints:
#   GET /api/graph/          → full graph (nodes + edges)
#   GET /api/graph/{gr_id}   → relationships for one specific GR
#
# Data comes from MongoDB gr_graph collection.
# Built automatically when admin runs embedding.
# Frontend uses this data to render the graph visualization.
# ─────────────────────────────────────────────────────────

import sys
import os

from fastapi import APIRouter, Depends

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_current_user
from core.gr_graph import get_graph, get_relationships_for_gr

router = APIRouter(prefix="/api/graph", tags=["Graph"])


@router.get("/")
async def fetch_graph(
    current_user: dict = Depends(get_current_user),
):
    """
    Returns the full GR relationship graph.
    Available to all logged-in users — not admin only.
    Both users and admins can view the graph.

    Response shape:
        {
            "success": true,
            "built": true,
            "built_at": "2025-07-05T...",
            "nodes": [
                {"id": "GR_2024_transfer", "label": "GR_2024_transfer", "filename": "GR_2024_transfer.pdf"},
                ...
            ],
            "edges": [
                {
                    "source": "GR_2024_transfer",
                    "target": "GR_2021_transfer",
                    "relation": "supersedes",
                    "snippet": "...text evidence..."
                },
                ...
            ],
            "stats": {
                "total_nodes": 5,
                "total_edges": 3,
                "supersedes_count": 2,
                "amends_count": 1,
                "extends_count": 1,
                "refers_to_count": 0
            }
        }
    """
    graph = await get_graph()

    # Compute edge type breakdown for frontend stats display
    edges = graph.get("edges", [])
    stats = {
        "total_nodes":     len(graph.get("nodes", [])),
        "total_edges":     len(edges),
        "supersedes_count": sum(1 for e in edges if e["relation"] == "supersedes"),
        "amends_count":     sum(1 for e in edges if e["relation"] == "amends"),
        "extends_count":    sum(1 for e in edges if e["relation"] == "extends"),
        "refers_to_count":  sum(1 for e in edges if e["relation"] == "refers_to"),
        "unknown_targets":  sum(1 for e in edges if e.get("target") == "Unknown"),
    }

    return {
        "success":  True,
        "built":    graph.get("built", False),
        "built_at": graph.get("built_at", ""),
        "nodes":    graph.get("nodes", []),
        "edges":    edges,
        "stats":    stats,
    }


@router.get("/{gr_id}")
async def fetch_gr_relationships(
    gr_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Returns all relationships for a specific GR.
    Called when user clicks a node in the graph visualization.

    Args:
        gr_id : the GR filename stem e.g. "GR_2024_transfer"

    Response shape:
        {
            "success": true,
            "gr_id": "GR_2024_transfer",
            "relationships": [
                {
                    "source": "GR_2024_transfer",
                    "target": "GR_2021_transfer",
                    "relation": "supersedes",
                    "snippet": "...evidence text..."
                }
            ],
            "total": 1
        }
    """
    relationships = await get_relationships_for_gr(gr_id)

    return {
        "success":       True,
        "gr_id":         gr_id,
        "relationships": relationships,
        "total":         len(relationships),
    }