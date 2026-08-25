# backend/main.py
# ─────────────────────────────────────────────────────────
# FastAPI application entry point.
#
# This file does 4 things:
#   1. Creates the FastAPI app instance
#   2. Adds CORS middleware so React frontend can call the API
#   3. Registers all routers (auth, upload, embed, query, etc.)
#   4. Connects to MongoDB and seeds default users on startup
#
# Run with:
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# API docs auto-generated at:
#   http://localhost:8000/docs
# ─────────────────────────────────────────────────────────

import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from db.mongo import connect_db, close_db
from db.users import seed_default_users

# Import all routers
from auth.router import router as auth_router
from api.upload  import router as upload_router
from api.embed   import router as embed_router
from api.query   import router as query_router
from api.graph   import router as graph_router
from api.summary import router as summary_router
from api.logs     import router as logs_router
from api.sessions import router as sessions_router

# ── Create FastAPI app ────────────────────────────────────
app = FastAPI(
    title="Niyamsetu API",
    description="Context-Aware Conversational Platform for Maharashtra GR Documents",
    version="1.0.0",
    # Docs available at /docs — useful during development
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ───────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Browsers block requests from one origin (localhost:5173 React)
# to another origin (localhost:8000 FastAPI) by default.
# This middleware tells the browser: "it's okay, allow these origins."
#
# The allowed origins come from CORS_ALLOWED_ORIGINS in .env — see that
# setting in config.py. It is an explicit list, never "*", because
# allow_credentials below is True and the two together tell Starlette to
# reflect any caller's Origin back as allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,   # explicit list from .env, never "*"
    allow_credentials=True,       # allow cookies and auth headers
    allow_methods=["*"],          # allow GET, POST, DELETE, etc.
    allow_headers=["*"],          # allow Authorization header etc.
)

# ── Register all routers ──────────────────────────────────
# Each router handles a group of related endpoints.
# The prefix is already set inside each router file.
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(embed_router)
app.include_router(query_router)
app.include_router(graph_router)
app.include_router(summary_router)
app.include_router(logs_router)
app.include_router(sessions_router)


# ── Startup and shutdown events ───────────────────────────

@app.on_event("startup")
async def on_startup():
    """
    Runs once when the server starts.
    Order matters — connect DB before seeding users.
    """
    print("\n🚀 Niyamsetu API starting...")

    # Connect to MongoDB Atlas
    await connect_db()

    # Create default admin and user accounts if DB is empty
    await seed_default_users()

    print("✅ Niyamsetu API ready\n")


@app.on_event("shutdown")
async def on_shutdown():
    """
    Runs once when the server shuts down (Ctrl+C).
    Cleanly closes MongoDB connection.
    """
    await close_db()
    print("👋 Niyamsetu API shut down.")


# ── Root endpoint ─────────────────────────────────────────

@app.get("/")
async def root():
    """
    Health check endpoint.
    Visiting http://localhost:8000 in browser shows this.
    """
    return {
        "app":     "Niyamsetu API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "http://localhost:8000/docs",
    }