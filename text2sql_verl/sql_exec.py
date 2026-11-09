"""Sandboxed SQLite execution shared by the tool and the reward.

Databases are opened read-only and immutable, so many rollout workers can
query the same file at once without locking, and a query cannot modify it.
Long-running queries are aborted via SQLite's progress handler.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any

# Matches SkyRL: observations longer than this are truncated to 50 rows.
MAX_OBS_CHARS = 9000
TRUNCATE_ROWS = 50
# Hard cap on rows pulled from SQLite, so `SELECT * FROM big_table` can't blow up memory.
MAX_FETCH_ROWS = 10_000


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    timed_out: bool = False


def execute(db_path: str, sql: str, timeout_s: float, max_rows: int | None = MAX_FETCH_ROWS) -> ExecResult:
    """Run `sql` against `db_path` read-only with a wall-clock timeout."""
    deadline = time.monotonic() + timeout_s
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    except sqlite3.Error as e:
        return ExecResult(ok=False, error=f"Cannot open database: {e}")
    # Non-zero return aborts the running statement with "interrupted".
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return ExecResult(ok=True, rows=rows, columns=cols)
    except sqlite3.OperationalError as e:
        if time.monotonic() > deadline:
            return ExecResult(ok=False, error="timeout", timed_out=True)
        return ExecResult(ok=False, error=str(e))
    except Exception as e:  # sqlite3.Warning (multiple statements), ProgrammingError, ...
        return ExecResult(ok=False, error=str(e))
    finally:
        conn.close()


def format_observation(res: ExecResult, sql: str) -> str:
    """Render an execution result for the model."""
    if res.timed_out:
        return f"SQL Timeout:\n{sql}"
    if not res.ok:
        return f"Error executing SQL: {res.error}"
    rows = res.rows or []
    if not rows:
        return "Query executed successfully. 0 rows returned." + (
            f"\nColumns: {', '.join(res.columns)}" if res.columns else ""
        )
    text = _table(res.columns, rows)
    if len(text) > MAX_OBS_CHARS:
        text = f"Truncated to {TRUNCATE_ROWS} rows since returned response too long:\n" + _table(
            res.columns, rows[:TRUNCATE_ROWS]
        )
    return text


def _table(columns: list[str] | None, rows: list[tuple]) -> str:
    def cell(v: Any) -> str:
        s = "NULL" if v is None else str(v)
        return s if len(s) <= 200 else s[:200] + "..."

    header = " | ".join(columns or [])
    body = "\n".join(" | ".join(cell(v) for v in r) for r in rows)
    return f"{header}\n{body}" if header else body
