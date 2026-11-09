"""Execution-match reward, matching SkyRL-SQL's scheme.

  -1  no final answer submitted (turn or token budget exhausted)
   0  submitted SQL fails, or its result set differs from the gold SQL's
   1  result sets match (compared as frozensets, like SkyRL: order/duplicates ignored)

Optional test-suite matching (reward kwarg `test_suite_manifest`, built by scripts/build_test_suites.py):
the result sets must also match on every pre-built random copy of the question's database. Questions
without a manifest entry or copies (e.g. validation) fall back to plain matching, so Spider dev scores
stay comparable across runs. `test_suite_scope="empty_gold"` applies the suite only where the gold
result on the real DB is empty (the loophole), leaving real-answer questions on plain matching. `plain_match` is always logged, so the gap shows what the suite rejects.
"""

from __future__ import annotations

import json
from functools import lru_cache

from text2sql_verl import sql_exec
from text2sql_verl.test_suite import suite_key
from text2sql_verl.tools import SUBMITTED_SQL_KEY

REWARD_TIMEOUT_S = 30.0  # SkyRL uses 30s for reward-time execution


@lru_cache(maxsize=65536)
def _gold_result(db_path: str, gold_sql: str) -> frozenset | None:
    res = sql_exec.execute(db_path, gold_sql, REWARD_TIMEOUT_S, max_rows=None)
    return frozenset(res.rows) if res.ok else None


@lru_cache(maxsize=4)
def _manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _pred_result(db_path: str, pred_sql: str) -> frozenset | None:
    res = sql_exec.execute(db_path, pred_sql, REWARD_TIMEOUT_S, max_rows=None)
    return frozenset(res.rows) if res.ok else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, test_suite_manifest=None,
                  test_suite_scope="all", **kwargs) -> dict:
    extra_info = extra_info or {}
    pred_sql = extra_info.get(SUBMITTED_SQL_KEY)
    out = {"score": -1.0, "submitted": 0.0, "pred_exec_ok": 0.0, "gold_exec_ok": 1.0,
           "plain_match": 0.0, "has_test_suite": 0.0}
    if not pred_sql:
        return out
    out["submitted"] = 1.0

    db_path = extra_info["db_path"]
    gold = _gold_result(db_path, ground_truth)
    if gold is None:
        out["gold_exec_ok"] = 0.0
    pred = sql_exec.execute(db_path, pred_sql, REWARD_TIMEOUT_S, max_rows=None)
    out["pred_exec_ok"] = float(pred.ok)
    match = pred.ok and gold is not None and frozenset(pred.rows) == gold
    out["plain_match"] = float(match)

    copies = _manifest(test_suite_manifest).get(suite_key(db_path, ground_truth), []) if test_suite_manifest else []
    if test_suite_scope == "empty_gold" and gold:  # suite only where the real DB's gold result is empty
        copies = []
    out["has_test_suite"] = float(bool(copies))
    for copy in copies:  # stop at the first copy that disagrees
        if not match:
            break
        g = _gold_result(copy, ground_truth)
        match = g is not None and _pred_result(copy, pred_sql) == g
    out["score"] = 1.0 if match else 0.0
    return out
