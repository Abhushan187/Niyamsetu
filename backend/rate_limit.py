# backend/rate_limit.py
# ─────────────────────────────────────────────────────────
# Shared slowapi Limiter. Lives in its own module so both main.py
# (which registers it on the app) and auth/router.py (which decorates
# the login route) can import it without a circular import.
#
# Behind the Cloudflare tunnel every request reaches uvicorn from
# 127.0.0.1, so keying on the peer address alone would make all
# visitors share one limit. cloudflared puts the real visitor address
# in CF-Connecting-IP. That header is trusted ONLY when the direct peer
# is loopback (the local tunnel or the Vite proxy); a remote client
# connecting directly could otherwise set it to anything and dodge the
# limit. This guarantee holds only while the backend stays bound to
# 127.0.0.1 — never bind it to 0.0.0.0 when this is deployed.
# ─────────────────────────────────────────────────────────

from slowapi import Limiter

_LOOPBACK = {"127.0.0.1", "::1"}


def key_func(request) -> str:
    direct = request.client.host if request.client else "unknown"
    if direct in _LOOPBACK:
        cf_ip = (request.headers.get("CF-Connecting-IP") or "").strip()
        if cf_ip:
            return cf_ip
    return direct


limiter = Limiter(key_func=key_func)
