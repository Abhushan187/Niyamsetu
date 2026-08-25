#!/usr/bin/env python
# backend/tests/run_harness.py
# ─────────────────────────────────────────────────────────
# RAG answer-capture harness. NOT a unittest suite, and
# deliberately separate from json_test_runner.py, which only
# understands binary pass/fail raised from assertions.
#
# What this does:
#   reads tests/harness_questions.json
#   → asks each question against the LIVE POST /api/query/chat
#   → records the system's actual answer + citations next to the
#     original question, gr_file, tier and gold_answer
#   → writes tests/harness_results_<timestamp>.json
#
# What this deliberately does NOT do:
#   ANY grading. It never compares `answer` to `gold_answer`, never
#   scores, never marks pass/fail. gold_answer is copied through
#   untouched so a human — or a validated LLM judge — can grade in a
#   separate later step. Adding "is the answer right?" logic here
#   would prejudge exactly the thing the harness exists to measure.
#
# Requires the backend to already be running:
#   uvicorn main:app
#
# Usage:
#   python tests/run_harness.py
#   python tests/run_harness.py --validate            # check the file, ask nothing
#   python tests/run_harness.py --limit 3             # smoke test on 3 questions
#   python tests/run_harness.py --base-url http://localhost:8000
# ─────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

TESTS_DIR = Path(__file__).resolve().parent

# Recognised tier values. Unknown tiers are reported but NOT rejected —
# extending the taxonomy should not require editing this script.
KNOWN_TIERS = {"direct_lookup", "synthesis", "cross_reference", "negative_trap"}

# Fields every question must carry. `id` is filled in positionally if absent.
REQUIRED_FIELDS = ("question", "gr_file", "tier", "gold_answer")

# Seconds between "still waiting" lines while a question is in flight.
HEARTBEAT_INTERVAL = 15.0


class _Heartbeat:
    """
    Prints an elapsed-time line every `interval` seconds while a request is
    in flight, so a slow question is distinguishable from a stuck one.

    Mirrors the heartbeat in core/vectorstore.py rather than importing it:
    that module pulls in FAISS and langchain at import time, which this
    standalone script has no reason to load.

    Observes only — it never cancels, times out, or touches the request.
    Daemon thread, so it can never hold the process open by itself.

    `ticks` counts the lines printed, which the caller uses to decide
    whether the result needs its own line.
    """

    def __init__(self, label: str, interval: float = HEARTBEAT_INTERVAL):
        self.label    = label
        self.interval = interval
        self.ticks    = 0
        self._stop    = threading.Event()
        self._thread  = None
        self.started  = None

    def _tick(self):
        while not self._stop.wait(self.interval):
            self.ticks += 1
            waited = time.time() - self.started
            # Leading newline: the caller left the cursor mid-line after
            # printing the question header with end="".
            print(f"\n       ... still waiting on {self.label} "
                  f"— {waited:.0f}s elapsed", end="", flush=True)

    def __enter__(self):
        self.started = time.time()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return False


def load_questions(path: Path) -> list:
    """
    Reads the question set.

    Accepts either shape:
        [ {...}, {...} ]                  — a bare list
        { "questions": [ {...}, ... ] }   — wrapped, any sibling keys ignored

    Each question needs: question, gr_file, tier, gold_answer.
    `id` is optional — index-based ids (Q01, Q02, ...) are assigned if missing.

    Raises:
        SystemExit with a readable message if the file is unusable.
    """
    if not path.exists():
        sys.exit(f"error: question file not found: {path}\n"
                 f"       place harness_questions.json in {TESTS_DIR}")

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"error: {path.name} is not valid JSON — {e}")

    if isinstance(raw, dict):
        questions = raw.get("questions")
        if questions is None:
            sys.exit(f"error: {path.name} is an object but has no 'questions' key")
    elif isinstance(raw, list):
        questions = raw
    else:
        sys.exit(f"error: {path.name} must be a list or an object with 'questions'")

    if not questions:
        sys.exit(f"error: {path.name} contains no questions")

    problems = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            problems.append(f"  [{i}] not an object")
            continue
        q.setdefault("id", f"Q{i + 1:02d}")
        missing = [f for f in REQUIRED_FIELDS if not str(q.get(f, "")).strip()]
        if missing:
            problems.append(f"  [{i}] id={q['id']!r} missing/empty: {', '.join(missing)}")

    if problems:
        sys.exit(f"error: {path.name} has {len(problems)} malformed question(s):\n"
                 + "\n".join(problems))

    return questions


def describe_set(questions: list) -> None:
    """Prints a breakdown of the question set — tiers, files, unknown tiers."""
    by_tier, by_file = {}, {}
    for q in questions:
        by_tier[q["tier"]] = by_tier.get(q["tier"], 0) + 1
        by_file[q["gr_file"]] = by_file.get(q["gr_file"], 0) + 1

    print(f"  questions : {len(questions)}")
    print(f"  tiers     : " + ", ".join(f"{t}={n}" for t, n in sorted(by_tier.items())))
    print(f"  gr_files  : " + ", ".join(f"{f}={n}" for f, n in sorted(by_file.items())))

    unknown = set(by_tier) - KNOWN_TIERS
    if unknown:
        print(f"  note      : unrecognised tier(s) {sorted(unknown)} — recorded as-is")

    dupes = [i for i, c in
             {q["id"]: sum(1 for x in questions if x["id"] == q["id"]) for q in questions}.items()
             if c > 1]
    if dupes:
        print(f"  WARNING   : duplicate id(s) {sorted(dupes)} — results will be ambiguous")


def login(client: httpx.Client, base_url: str, username: str, password: str) -> str:
    """Authenticates and returns a bearer token. Exits with guidance on failure."""
    try:
        res = client.post(f"{base_url}/api/auth/login",
                          json={"username": username, "password": password})
    except httpx.RequestError as e:
        sys.exit(f"error: cannot reach the backend at {base_url} — {type(e).__name__}: {e}\n"
                 f"       is it running?  uvicorn main:app")

    if res.status_code != 200:
        sys.exit(f"error: login failed for {username!r} "
                 f"(HTTP {res.status_code}: {res.text[:200]})\n"
                 f"       is MongoDB running and seeded? (db/users.py seed_default_users)")

    return res.json()["access_token"]


def ask(client: httpx.Client, base_url: str, headers: dict, question: dict,
        top_k, language, timeout: float) -> dict:
    """
    Sends ONE question to /api/query/chat and returns what came back.

    History is always empty — each question is judged standalone, so a
    previous answer cannot influence the next one.

    Never raises: transport and HTTP errors are recorded in the result so a
    single failure cannot abort a long run.
    """
    payload = {"query": question["question"], "history": []}
    if top_k is not None:
        payload["top_k"] = top_k
    if language is not None:
        payload["language"] = language

    started = time.time()
    try:
        res = client.post(f"{base_url}/api/query/chat", json=payload,
                          headers=headers, timeout=timeout)
    except httpx.RequestError as e:
        return {
            "http_status": None,
            "error":       f"{type(e).__name__}: {e}",
            "answer":      "",
            "citations":   [],
            "language":    "",
            "success":     False,
            "server_elapsed_sec": None,
            "client_elapsed_sec": round(time.time() - started, 2),
        }

    client_elapsed = round(time.time() - started, 2)

    if res.status_code != 200:
        return {
            "http_status": res.status_code,
            "error":       res.text[:500],
            "answer":      "",
            "citations":   [],
            "language":    "",
            "success":     False,
            "server_elapsed_sec": None,
            "client_elapsed_sec": client_elapsed,
        }

    body = res.json()
    return {
        "http_status": 200,
        "error":       "",
        # Recorded verbatim — never trimmed, normalised or compared.
        "answer":      body.get("answer", ""),
        "citations":   body.get("citations", []),
        "language":    body.get("language", ""),
        "success":     body.get("success", False),
        "server_elapsed_sec": body.get("elapsed_sec"),
        "client_elapsed_sec": client_elapsed,
    }


def write_output(path: Path, meta: dict, records: list) -> None:
    """
    Writes the results file. Called after EVERY question, not just at the
    end, so a long run that dies partway still leaves usable data on disk.
    """
    payload = dict(meta)
    payload["completed"] = len(records)
    payload["results"]   = records
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(
        description="Capture live RAG answers for a graded question set. Does not grade.",
    )
    ap.add_argument("--questions", type=Path, default=TESTS_DIR / "harness_questions.json",
                    help="question set (default: tests/harness_questions.json)")
    ap.add_argument("--out-dir", type=Path, default=TESTS_DIR,
                    help="where to write harness_results_<timestamp>.json (default: tests/)")
    ap.add_argument("--base-url", default="http://localhost:8000",
                    help="running backend (default: http://localhost:8000)")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--top-k", type=int, default=None,
                    help="override retrieval depth; omit to use the server default")
    ap.add_argument("--language", default=None,
                    help="force 'english' or 'marathi'; omit for per-question auto-detect")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-question timeout in seconds (default: 600)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N questions — useful for a smoke test")
    ap.add_argument("--preview-chars", type=int, default=100,
                    help="answer preview length on the progress line (default: 100)")
    ap.add_argument("--validate", action="store_true",
                    help="check the question file and exit without asking anything")
    args = ap.parse_args()

    print(f"question set: {args.questions}")
    questions = load_questions(args.questions)
    describe_set(questions)

    if args.validate:
        print("\nvalidate only — no questions sent.")
        return 0

    if args.limit:
        questions = questions[:args.limit]
        print(f"  limited to first {len(questions)} question(s)")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"harness_results_{timestamp}.json"

    meta = {
        "run_at":        str(datetime.now(timezone.utc)),
        "base_url":      args.base_url,
        "question_file": str(args.questions),
        "total":         len(questions),
        "top_k":         args.top_k,
        "language":      args.language,
        "graded":        False,
        "note": ("Raw capture only. 'answer' and 'citations' are what the live "
                 "system returned; 'gold_answer' is the untouched expected answer. "
                 "No comparison or scoring has been applied — grade separately."),
    }

    print(f"\nbackend: {args.base_url}")
    with httpx.Client() as client:
        token = login(client, args.base_url, args.username, args.password)
        headers = {"Authorization": f"Bearer {token}"}
        print(f"logged in as {args.username!r}\n")

        records = []
        started_all = time.time()

        for n, q in enumerate(questions, 1):
            print(f"[{n}/{len(questions)}] {q['id']} ({q['tier']}) ... ", end="", flush=True)

            with _Heartbeat(f"{q['id']}") as hb:
                outcome = ask(client, args.base_url, headers, q,
                              args.top_k, args.language, args.timeout)

            # If the heartbeat printed, the cursor is at the end of its last
            # line — start a fresh, indented line so the result stays readable.
            if hb.ticks:
                print("\n       -> ", end="", flush=True)

            records.append({
                # ── the question, carried through unchanged ──
                "id":          q["id"],
                "question":    q["question"],
                "gr_file":     q["gr_file"],
                "tier":        q["tier"],
                "gold_answer": q["gold_answer"],
                # ── what the live system actually said ──
                "actual_answer":       outcome["answer"],
                "actual_citations":    outcome["citations"],
                "detected_language":   outcome["language"],
                "api_success":         outcome["success"],
                "http_status":         outcome["http_status"],
                "error":               outcome["error"],
                "server_elapsed_sec":  outcome["server_elapsed_sec"],
                "client_elapsed_sec":  outcome["client_elapsed_sec"],
                # deliberately absent: any score, verdict or pass/fail field
            })

            # Written every iteration — a run that dies at question 17 keeps 16.
            write_output(out_path, meta, records)

            if outcome["error"]:
                print(f"ERROR ({outcome['http_status']}): {outcome['error'][:120]}")
            else:
                preview = " ".join(outcome["answer"].split())[:args.preview_chars]
                print(f"{outcome['client_elapsed_sec']}s "
                      f"[{len(outcome['citations'])} cite] {preview!r}")

        total_elapsed = round(time.time() - started_all, 2)

    meta["elapsed_sec"] = total_elapsed
    write_output(out_path, meta, records)

    failed = [r for r in records if r["error"]]
    print(f"\ncaptured {len(records)} answer(s) in {total_elapsed}s")
    if failed:
        print(f"  {len(failed)} request(s) failed: {[r['id'] for r in failed]}")
    print(f"  -> {out_path}")
    print("  not graded — 'gold_answer' is stored alongside for a separate grading pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
