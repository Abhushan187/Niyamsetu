# backend/core/trace.py
# ─────────────────────────────────────────────────────────
# Timestamped stdout tracing for long-running request paths.
#
# Why this exists:
#   uvicorn only prints its access log line AFTER a request finishes.
#   On a CPU-bound Ollama box a single /api/query/chat can take minutes,
#   during which the terminal is completely silent — indistinguishable
#   from a server that has hung or died. These traces mark the moments
#   uvicorn cannot: request arrival, and the start of the LLM call.
#
# Purely diagnostic — nothing here affects request handling.
#
# flush=True because uvicorn buffers stdout; without it these lines can
# surface late, in the wrong order relative to the access log.
# ─────────────────────────────────────────────────────────

from datetime import datetime


def trace(event: str, detail: str = "") -> None:
    """
    Prints one timestamped diagnostic line.

    Args:
        event  : short label, e.g. "REQUEST  POST /api/query/chat"
        detail : optional extra context appended after a separator

    Example output:
        [2026-08-24 13:33:38.412] REQUEST  POST /api/query/chat | user=admin
    """
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] {event}"
    if detail:
        line += f" | {detail}"
    print(line, flush=True)
