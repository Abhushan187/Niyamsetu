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
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import dotenv_values
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from rate_limit import limiter
from db.mongo import connect_db, close_db
from db.users import seed_default_users

# Import all routers
from auth.router import router as auth_router
from api.upload  import router as upload_router
from api.embed   import router as embed_router
from api.query   import router as query_router
from api.graph   import router as graph_router
from api.meta    import router as meta_router
from api.files   import router as files_router
from api.summary import router as summary_router
from api.logs     import router as logs_router
from api.sessions import router as sessions_router

# ── Startup and shutdown ──────────────────────────────
# A single lifespan context manager replaces the deprecated @app.on_event
# handlers: everything before the yield runs once at startup, everything
# after it runs once at shutdown. FastAPI takes it as a constructor
# argument, so it has to be defined above the app.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup runs before the first request is served, shutdown after the
    last one. Order matters — connect DB before seeding users.
    """
    print("\n[Niyamsetu API starting...]")

    # Connect to MongoDB Atlas
    await connect_db()

    # Create default admin and user accounts if DB is empty
    await seed_default_users()

    print("[OK] Niyamsetu API ready\n")

    yield

    # Runs once when the server shuts down (Ctrl+C).
    # Cleanly closes MongoDB connection.
    await close_db()
    print("👋 Niyamsetu API shut down.")


# ── Create FastAPI app ────────────────────────────────────
app = FastAPI(
    title="Niyamsetu API",
    description="Context-Aware Conversational Platform for Maharashtra GR Documents",
    version="1.0.0",
    # Docs available at /docs — useful during development
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Rate limiting (slowapi) ───────────────────────────────
# Only the login route is decorated; see auth/router.py.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _cors_origins() -> list[str]:
    """
    Allowed origins from CORS_ORIGINS (comma separated). Read from the
    process environment first, then from backend/.env. When unset, fall
    back to the origins allowed before (settings.CORS_ORIGINS, which comes
    from CORS_ALLOWED_ORIGINS, default http://localhost:5173).
    """
    raw = os.environ.get("CORS_ORIGINS")
    if raw is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        raw = dotenv_values(env_path).get("CORS_ORIGINS")
    if raw is None:
        return settings.CORS_ORIGINS
    return [o.strip() for o in raw.split(",") if o.strip()]

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
    allow_origins=_cors_origins(),   # explicit list from CORS_ORIGINS, never "*"
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
app.include_router(meta_router)
app.include_router(files_router)
app.include_router(summary_router)
app.include_router(logs_router)
app.include_router(sessions_router)


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


@app.get("/health")
async def health():
    """Unauthenticated liveness check for the frontend. No DB access."""
    return {"ok": True}