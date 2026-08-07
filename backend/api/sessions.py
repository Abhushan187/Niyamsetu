# backend/api/sessions.py
# ─────────────────────────────────────────────────────────
# Chat session endpoints.
#
# Endpoints:
#   GET    /api/sessions/         → list user's sessions
#   POST   /api/sessions/         → create new session
#   GET    /api/sessions/{id}     → load full session with messages
#   DELETE /api/sessions/{id}     → delete a session
# ─────────────────────────────────────────────────────────

import sys
import os

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.router import get_current_user
from db.sessions import (
    create_session,
    get_sessions_by_user,
    get_session_by_id,
    delete_session,
    append_message,
    set_session_pinned,
)

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"


@router.get("/")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    """Returns all sessions for logged-in user, newest first."""
    sessions = await get_sessions_by_user(current_user["username"])
    # Convert datetime objects to strings
    for s in sessions:
        if "updated_at" in s:
            s["updated_at"] = str(s["updated_at"])
        if "created_at" in s:
            s["created_at"] = str(s["created_at"])
    return {"success": True, "sessions": sessions}


@router.post("/")
async def create_new_session(
    request: CreateSessionRequest,
    current_user: dict = Depends(get_current_user),
):
    """Creates a new empty chat session."""
    session = await create_session(
        username=current_user["username"],
        title=request.title,
    )
    if "updated_at" in session:
        session["updated_at"] = str(session["updated_at"])
    if "created_at" in session:
        session["created_at"] = str(session["created_at"])
    return {"success": True, "session": session}


@router.get("/{session_id}")
async def load_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Loads a full session with all messages."""
    session = await get_session_by_id(session_id, current_user["username"])
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    if "updated_at" in session:
        session["updated_at"] = str(session["updated_at"])
    if "created_at" in session:
        session["created_at"] = str(session["created_at"])
    return {"success": True, "session": session}


class AppendMessageRequest(BaseModel):
    user_msg:      dict
    assistant_msg: dict


@router.post("/{session_id}/messages")
async def append_to_session(
    session_id: str,
    request: AppendMessageRequest,
    current_user: dict = Depends(get_current_user),
):
    """Appends a user + assistant message pair to a session."""
    success = await append_message(
        session_id=session_id,
        username=current_user["username"],
        user_msg=request.user_msg,
        assistant_msg=request.assistant_msg,
    )
    return {"success": success}

class RenameSessionRequest(BaseModel):
    title: str


@router.patch("/{session_id}")
async def rename_session(
    session_id: str,
    request: RenameSessionRequest,
    current_user: dict = Depends(get_current_user),
):
    """Renames a session title."""
    from db.mongo import get_db
    from bson import ObjectId
    database = get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session id.")
    await database.chat_sessions.update_one(
        {"_id": oid, "username": current_user["username"]},
        {"$set": {"title": request.title.strip()}}
    )
    return {"success": True}

class PinRequest(BaseModel):
    pinned: bool


@router.patch("/{session_id}/pin")
async def toggle_pin(
    session_id: str,
    request: PinRequest,
    current_user: dict = Depends(get_current_user),
):
    """Pins or unpins a session."""
    updated = await set_session_pinned(session_id, current_user["username"], request.pinned)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return {"success": True, "pinned": request.pinned}


@router.delete("/{session_id}")
async def remove_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Deletes a session."""
    deleted = await delete_session(session_id, current_user["username"])
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    return {"success": True, "message": "Session deleted."}