# backend/db/sessions.py
# ─────────────────────────────────────────────────────────
# Chat session storage in MongoDB.
# Each session = one conversation with title + messages.
# Users can have multiple sessions, each independently loaded.
# ─────────────────────────────────────────────────────────

import sys
import os
from datetime import datetime, timezone
from bson import ObjectId

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.mongo import get_db


def _serialize(doc: dict) -> dict:
    """Convert MongoDB ObjectId to string for JSON serialization."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def create_session(username: str, title: str = "New Chat") -> dict:
    """
    Creates a new empty chat session.
    Called when user clicks '+ New Chat'.

    Returns the created session with its id.
    """
    database = get_db()

    session = {
        "username":   username,
        "title":      title,
        # messages = list of {role, content, citations, elapsed_sec, language}
        "messages":   [],
        "pinned":     False,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    result = await database.chat_sessions.insert_one(session)
    session["_id"] = str(result.inserted_id)
    return session


async def get_sessions_by_user(username: str) -> list:
    """
    Returns all chat sessions for a user, newest first.
    Used to populate the session list in the Chat sidebar.
    Only returns id + title + updated_at — not full messages.
    Keeps the list lightweight.
    """
    database = get_db()

    cursor = database.chat_sessions.find(
        {"username": username},
        # Only return these fields — messages excluded for performance
        {"_id": 1, "title": 1, "updated_at": 1, "created_at": 1, "pinned": 1}
    ).sort([("pinned", -1), ("updated_at", -1)]).limit(50)

    sessions = await cursor.to_list(length=50)
    return [_serialize(s) for s in sessions]


async def get_session_by_id(session_id: str, username: str) -> dict | None:
    """
    Returns a full session including all messages.
    Called when user clicks a past session to load it.
    Verifies the session belongs to the requesting user.
    """
    database = get_db()

    try:
        oid = ObjectId(session_id)
    except Exception:
        return None

    session = await database.chat_sessions.find_one(
        {"_id": oid, "username": username}
    )
    return _serialize(session) if session else None


async def append_message(
    session_id: str,
    username: str,
    user_msg: dict,
    assistant_msg: dict,
) -> bool:
    """
    Appends a user + assistant message pair to a session.
    Called after every successful chat query.
    Also updates the session title if it's still 'New Chat'.

    Verifies ownership before reading or writing — matching
    get_session_by_id / delete_session / set_session_pinned.

    Args:
        session_id    : the session to update
        username      : owner of the session — both queries filter on this
        user_msg      : {role: 'user', content: str}
        assistant_msg : {role: 'assistant', content, citations, elapsed_sec, language}

    Returns:
        True if updated successfully, False otherwise
        (including when the session does not exist or belongs to someone else)
    """
    database = get_db()

    try:
        oid = ObjectId(session_id)
    except Exception:
        return False

    # Auto-title: use first 50 chars of first user message
    # Only updates title if it's still the default 'New Chat'
    session = await database.chat_sessions.find_one(
        {"_id": oid, "username": username}
    )

    # Not found, or owned by another user — refuse rather than
    # dereferencing None (which previously raised AttributeError → 500)
    if session is None:
        return False

    new_title = session.get("title", "New Chat")
    if new_title == "New Chat" and user_msg.get("content"):
        new_title = user_msg["content"][:50]

    result = await database.chat_sessions.update_one(
        {"_id": oid, "username": username},
        {
            "$push": {
                "messages": {"$each": [user_msg, assistant_msg]}
            },
            "$set": {
                "title":      new_title,
                "updated_at": datetime.now(timezone.utc),
            }
        }
    )
    return result.modified_count > 0


async def delete_session(session_id: str, username: str) -> bool:
    """
    Deletes a chat session.
    Verifies ownership before deleting.
    """
    database = get_db()

    try:
        oid = ObjectId(session_id)
    except Exception:
        return False

    result = await database.chat_sessions.delete_one(
        {"_id": oid, "username": username}
    )
    return result.deleted_count > 0

async def set_session_pinned(session_id: str, username: str, pinned: bool) -> bool:
    """
    Sets or clears the pinned flag on a session.
    Verifies ownership before updating.
    """
    database = get_db()

    try:
        oid = ObjectId(session_id)
    except Exception:
        return False

    result = await database.chat_sessions.update_one(
        {"_id": oid, "username": username},
        {"$set": {"pinned": pinned}}
    )
    return result.modified_count > 0