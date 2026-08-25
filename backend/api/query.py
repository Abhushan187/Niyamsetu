# backend/api/query.py
# ─────────────────────────────────────────────────────────
# Chat query endpoint — the core of the application.
#
# Endpoints:
#   POST /api/query/chat    → main RAG query with history
#   POST /api/query/search  → cosine similarity search with scores
#   GET  /api/query/health  → check if Ollama + vector store are ready
#
# Every chat message flows through /chat:
#   frontend → POST /api/query/chat → rag.query() → LLM → response
# ─────────────────────────────────────────────────────────

import sys
import os
import asyncio
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_current_user
from core.rag import query as rag_query
from core.trace import trace
from core.vectorstore import cosine_search_with_scores, is_ready
from db.logs import save_query_log
from config import settings

router = APIRouter(prefix="/api/query", tags=["Query"])


# ── Request / Response schemas ────────────────────────────

class ChatMessage(BaseModel):
    """One turn in the conversation history."""
    role: str       # "user" or "assistant"
    content: str    # the message text


class ChatRequest(BaseModel):
    """
    What the frontend sends for every chat message.

    query        : the current question
    history      : previous turns for context (can be empty list)
    language     : force language, or omit for auto-detect
    top_k        : how many chunks to retrieve (optional)
    """
    query:    str
    history:  list[ChatMessage] = []
    language: Optional[str]     = None
    top_k:    Optional[int]     = None


class SearchRequest(BaseModel):
    """What the frontend sends for cosine similarity search."""
    query:       str
    top_k:       int = 5
    filter_file: Optional[str] = None   # filter results to one file


# ── Routes ────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Main RAG chat endpoint.
    Called by the frontend on every message send.

    Flow:
        1. Convert history to plain dicts for rag.query()
        2. Run RAG pipeline — retrieve chunks → build prompt → LLM
        3. Save query log to MongoDB
        4. Return answer + citations + timing

    Protected — requires valid JWT token.
    Any logged-in user can query (not admin-only).
    """
    # ── Arrival trace ─────────────────────────────────────
    # First thing in the handler, before any work: uvicorn's access log
    # only prints once the response is sent, so without this the terminal
    # shows nothing at all while a slow query is in flight.
    trace(
        "REQUEST  POST /api/query/chat",
        f"user={current_user['username']} "
        f"history={len(request.history)} turn(s) "
        f"query={request.query[:80]!r}",
    )

    # Convert Pydantic ChatMessage objects to plain dicts
    # rag.query() expects list of {"role": ..., "content": ...}
    history_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in request.history
    ]

    # ── Run RAG pipeline ──────────────────────────────────
    result = await rag_query(
        user_query=request.query,
        chat_history=history_dicts,
        top_k=request.top_k,
        language=request.language,
    )

    # ── Save to query log ─────────────────────────────────
    # Save regardless of success — failed queries are useful for debugging
    await save_query_log(
        username=current_user["username"],
        query=request.query,
        answer=result["answer"],
        citations=result["citations"],
        language=result["language"],
        elapsed_sec=result["elapsed_sec"],
        was_successful=result["success"],
    )

    # Return the full result to frontend
    return {
        "success":     result["success"],
        "answer":      result["answer"],
        "citations":   result["citations"],
        "language":    result["language"],
        "elapsed_sec": result["elapsed_sec"],
    }


@router.post("/search")
async def similarity_search(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Cosine similarity search endpoint.
    Returns ranked results with similarity scores — no LLM involved.
    Used on the Search page.

    Different from /chat because:
        - No LLM call — just vector similarity
        - Returns multiple results with scores
        - User can see exactly which chunks matched and how well
    """
    if not is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not ready. Ask admin to embed documents first.",
        )

    # Get results with similarity scores.
    # cosine_search_with_scores() is synchronous — it embeds the query via a
    # blocking Ollama HTTP call (and loads the FAISS index from disk on the
    # first call), which pins the event loop and stalls every other request.
    # Same offload as core/rag.py:194; search behaviour is unchanged.
    results_with_scores = await asyncio.to_thread(
        cosine_search_with_scores,
        query=request.query,
        top_k=request.top_k,
    )

    # Filter by filename if requested
    if request.filter_file:
        results_with_scores = [
            (doc, score) for doc, score in results_with_scores
            if doc.metadata.get("source_file", "") == request.filter_file
        ]

    # Format results for frontend
    formatted = []
    for doc, score in results_with_scores:
        raw_page = doc.metadata.get("page", None)
        page_display = str(int(raw_page) + 1) if raw_page is not None else "?"

        formatted.append({
            "score":   score,
            "file":    doc.metadata.get("source_file", "Unknown"),
            "page":    page_display,
            "preview": doc.page_content[:400].strip(),
        })

    return {
        "success": True,
        "query":   request.query,
        "results": formatted,
        "total":   len(formatted),
    }


@router.get("/health")
async def system_health(
    current_user: dict = Depends(get_current_user),
):
    """
    Returns system readiness status.
    Frontend calls this on load to show status indicators.

    Checks:
        - Is FAISS vector store ready?
        - Is Ollama reachable at localhost:11434?
    """
    # Check Ollama
    ollama_online = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(settings.OLLAMA_BASE_URL)
            ollama_online = response.status_code == 200
    except Exception:
        ollama_online = False

    vector_ready = is_ready()

    return {
        "success":      True,
        "vector_ready": vector_ready,
        "ollama_online": ollama_online,
        "llm_model":    settings.LLM_MODEL,
        "embed_model":  settings.EMBEDDING_MODEL,
    }