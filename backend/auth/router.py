# backend/auth/router.py
# ─────────────────────────────────────────────────────────
# Authentication routes.
#
# Endpoints:
#   POST /api/auth/login   → returns JWT token
#   GET  /api/auth/me      → returns current user info
#   POST /api/auth/logout  → client-side logout (clears token)
#
# How auth works in FastAPI:
#   Protected routes use Depends(get_current_user)
#   This tells FastAPI: "run get_current_user() first,
#   inject the result, reject the request if it fails"
# ─────────────────────────────────────────────────────────

import sys
import os

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth.models import (
    LoginRequest,
    LoginResponse,
    UserResponse,
    MessageResponse,
    CreateUserRequest,
)
from auth.jwt_handler import create_token, verify_token
from db.users import authenticate_user, create_user, delete_user, list_all_users
from rate_limit import limiter

# ── Router setup ──────────────────────────────────────────
# prefix means all routes here start with /api/auth
# tags groups them together in the auto-generated API docs
router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# ── OAuth2 scheme ─────────────────────────────────────────
# Tells FastAPI where to find the token in requests
# tokenUrl is the login endpoint
# FastAPI uses this to show an "Authorize" button in /docs
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Dependency — extracts and verifies the current user from JWT token.
    Used by ALL protected routes via Depends(get_current_user).

    FastAPI calls this automatically before the route function runs.
    If token is missing or invalid → returns 401 Unauthorized.
    If token is valid → injects user dict into the route function.

    Args:
        token : JWT string extracted from Authorization header by FastAPI

    Returns:
        dict with username and role
    """
    return verify_token(token)


async def get_admin_user(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    Dependency — same as get_current_user but also checks admin role.
    Used by admin-only routes via Depends(get_admin_user).

    If user is not admin → returns 403 Forbidden.

    Args:
        current_user : injected by get_current_user dependency

    Returns:
        dict with username and role (only if role is admin)
    """
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user


# ── Routes ────────────────────────────────────────────────

@router.post("/login", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login(request: Request, credentials: LoginRequest):
    """
    Login endpoint — validates credentials and returns JWT token.

    Flow:
        1. Frontend sends { username, password }
        2. We check credentials against MongoDB
        3. If valid → create JWT token → return it
        4. If invalid → return 401

    The frontend stores the returned token and sends it
    in the Authorization header for all future requests:
        Authorization: Bearer <token>
    """
    # Authenticate against MongoDB
    user = await authenticate_user(credentials.username, credentials.password)

    if not user:
        # Use 401 not 404 — don't reveal whether username exists
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create JWT token with username and role embedded
    token = create_token({
        "username": user["username"],
        "role":     user["role"],
    })

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        username=user["username"],
        name=user["name"],
        role=user["role"],
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Returns the currently logged-in user's info.
    Frontend calls this on page load to restore session.

    Protected — requires valid JWT token.
    """
    return UserResponse(
        username=current_user["username"],
        name=current_user.get("name", current_user["username"]),
        role=current_user["role"],
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout endpoint.

    JWT tokens are stateless — the server can't invalidate them.
    So logout is handled client-side — the frontend simply
    deletes the stored token.

    This endpoint exists so the frontend has a clean API to call,
    and so we can log the logout event in future.
    """
    return MessageResponse(message="Logged out successfully.")


# ── Admin-only user management routes ─────────────────────

@router.get("/users", response_model=list[UserResponse])
async def get_users(admin: dict = Depends(get_admin_user)):
    """
    Returns list of all users.
    Admin only.
    """
    users = await list_all_users()
    return [UserResponse(**u) for u in users]


@router.post("/users", response_model=MessageResponse)
async def create_new_user(
    request: CreateUserRequest,
    admin: dict = Depends(get_admin_user),
):
    """
    Creates a new user account.
    Admin only.
    """
    result = await create_user(
        username=request.username,
        password=request.password,
        name=request.name,
        role=request.role,
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )

    return MessageResponse(message=result["message"])


@router.delete("/users/{username}", response_model=MessageResponse)
async def delete_existing_user(
    username: str,
    admin: dict = Depends(get_admin_user),
):
    """
    Deletes a user account.
    Admin only. Cannot delete default admin.
    """
    result = await delete_user(username)

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )

    return MessageResponse(message=result["message"])