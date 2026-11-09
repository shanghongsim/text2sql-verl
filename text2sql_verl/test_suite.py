"""Test-suite execution matching (prototype), after Zhong et al. 2020 ("distilled test suites").

Execution match on ONE database can't tell two queries apart if they return the same rows *on that data*
(e.g. both return nothing on an empty DB). Here we also compare them on K synthetic copies of the
database: same schema, filled with random rows that respect primary/foreign keys, with the gold
query's literals ('Active', 10, ...) mixed into the value pools so its filters can actually match.

  copies = make_copies(db_path, gold_sql, k=5)          # cached per (db, gold) in practice
  test_suite_match(pred_sql, gold_sql, db_path, copies) -> (match: bool, per_db: list)
"""

from __future__ import annotations

import os
import random
import re
import shutil
import sqlite3
import tempfile

from text2sql_verl import sql_exec

WORDS = ["alpha", "beta", "gamma", "delta", "omega", "red", "blue", "north", "south", "pending"]


def _schema(con):
    tables = [t[0] for t in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    cols = {t: [dict(name=c[1], type=(c[2] or "").upper(), pk=c[5]) for c in con.execute(f'PRAGMA table_info("{t}")')]
            for t in tables}
    fks = {t: {f[3]: (f[2], f[4]) for f in con.execute(f'PRAGMA foreign_key_list("{t}")')} for t in tables}
    return tables, cols, fks


def _order(tables, fks):
    """Parents before children, so FK values can be drawn from rows that already exist."""
    done, out = set(), []
    def visit(t, stack=()):
        if t in done or t in stack or t not in fks:
            return
        for parent, _ in fks[t].values():
            visit(parent, stack + (t,))
        done.add(t)
        out.append(t)
    for t in tables:
        visit(t)
    return out


def _literals(sql):
    sql = re.sub(r"--[^\n]*", "", sql)  # drop comments
    # single-quoted strings, plus double-quoted ones (SQLite treats "x" as a string when no column x exists)
    strs = re.findall(r"'((?:[^']|'')*)'", sql) + re.findall(r'"([^"]*\s[^"]*)"', sql)
    nums = [float(x) if "." in x else int(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", sql)]
    return [s.replace("''", "'") for s in strs], nums


def _value(col, rng, strs, nums, used=None):
    t, n = col["type"], col["name"].lower()
    if "date" in n or "time" in n or "DATE" in t:
        return f"20{rng.randint(15, 25)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if any(k in t for k in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")):
        pool = list(range(0, 6)) + [x for x in nums if isinstance(x, (int, float))] + [x + 1 for x in nums]
        v = rng.choice(pool) if rng.random() < 0.5 else rng.randint(0, 100)
        return float(v) + rng.random() if ("REAL" in t or "FLOA" in t or "DOUB" in t) and rng.random() < 0.5 else v
    # Mostly-unique text: random duplicates (e.g. two experiments both named "gamma") create differences
    # that never occur on realistic data (GROUP BY name vs GROUP BY id). Gold literals are reused so its
    # WHERE filters can match.
    if strs and rng.random() < 0.3:
        v = rng.choice(strs)
        if used is None or v not in used:  # unique column: each gold literal at most once
            if used is not None:
                used.add(v)
            return v
    return f"{rng.choice(WORDS)}_{rng.randint(1, 10**6)}"


def _fill(db_path, dst, rng, strs, nums):
    shutil.copy(db_path, dst)
    con = sqlite3.connect(dst)
    tables, cols, fks = _schema(con)
    # Text columns whose values are all distinct in the real DB (names, titles) stay distinct in the copy:
    # otherwise e.g. two programs both named 'New York TNR' break a correct `= (SELECT id ... WHERE name = ...)`.
    unique = set()
    for t in tables:
        for c in cols[t]:
            n, d = con.execute(f'SELECT COUNT("{c["name"]}"), COUNT(DISTINCT "{c["name"]}") FROM "{t}"').fetchone()
            if n > 1 and n == d:
                unique.add((t, c["name"]))
    for t in tables:
        con.execute(f'DELETE FROM "{t}"')
    keys = {}  # (table, col) -> values present
    for t in _order(tables, fks):
        n_rows = rng.choice([8, 30, 80])
        # Skewed FKs: sometimes point most child rows at 1-3 parents, so GROUP BY ... HAVING COUNT(*) > 10 can fire
        hot = {c: rng.sample(keys.get(fks[t][c], []) or [None], k=min(rng.randint(1, 3), len(keys.get(fks[t][c], [])) or 1))
               for c in fks[t]} if rng.random() < 0.5 else {}
        rows = []
        used = {c["name"]: set() for c in cols[t] if (t, c["name"]) in unique}
        for r in range(n_rows):
            row = {}
            for c in cols[t]:
                name = c["name"]
                if name in fks[t]:
                    parent = keys.get(fks[t][name], [])
                    pool = hot.get(name) if name in hot and rng.random() < 0.8 else parent
                    row[name] = rng.choice(pool) if pool else None
                elif c["pk"] and "INT" in c["type"]:
                    row[name] = r + 1
                else:
                    row[name] = _value(c, rng, strs, nums, used.get(name))
            rows.append(row)
        names = [c["name"] for c in cols[t]]
        con.executemany(f'INSERT OR IGNORE INTO "{t}" ({",".join(chr(34) + n + chr(34) for n in names)}) '
                        f'VALUES ({",".join("?" * len(names))})', [[row[n] for n in names] for row in rows])
        for n in names:
            keys[(t, n)] = [row[n] for row in rows]
    con.commit()
    con.execute("VACUUM")  # the copy starts as the full DB file; shrink it to the random rows
    con.close()


def make_copies(db_path, gold_sql, k=5, max_tries=40, seed=0, out_dir=None):
    """Distilled test suite: up to K random, key-consistent copies of db_path on which the gold query
    returns at least one row. A copy where gold is empty can't distinguish queries, so it's discarded.
    Fewer than K copies (possibly 0) means the generator couldn't make gold fire within max_tries."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="testsuite_")
    strs, nums = _literals(gold_sql)
    copies = []
    for i in range(max_tries):
        dst = os.path.join(out_dir, f"copy{len(copies)}.sqlite")
        _fill(db_path, dst, random.Random(seed + i), strs, nums)
        g = _rows(dst, gold_sql)
        if g:  # informative: gold returns rows here
            copies.append(dst)
            if len(copies) == k:
                break
        else:
            os.remove(dst)
    return copies


def _rows(db, sql):
    res = sql_exec.execute(db, sql, 30.0, max_rows=None)
    return frozenset(res.rows) if res.ok else None


def test_suite_match(pred_sql, gold_sql, db_path, copies):
    """Match iff pred == gold (as row sets) on the real DB AND on every synthetic copy."""
    per_db = []
    for db in [db_path] + list(copies):
        g, p = _rows(db, gold_sql), _rows(db, pred_sql)
        per_db.append({"db": os.path.basename(db), "gold_rows": None if g is None else len(g),
                       "pred_rows": None if p is None else len(p), "match": g is not None and p == g})
    return all(d["match"] for d in per_db), per_db


def suite_key(db_path, gold_sql):
    """Stable id for a (database, gold query) pair; names its test-suite directory."""
    import hashlib
    return hashlib.sha1(f"{db_path}\n{gold_sql}".encode()).hexdigest()[:16]
