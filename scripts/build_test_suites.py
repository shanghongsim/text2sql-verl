"""Pre-build distilled test suites (random DB copies) for every training question, once, offline.

The copies depend only on (db_path, gold SQL), never on the model, so they're built before training and
reused by every rollout. Output: data/test_suite/<key>/copy*.sqlite plus data/test_suite/manifest.json
({key: [copy paths]}; an empty list means no informative copy, so the reward falls back to plain match).

  python scripts/build_test_suites.py data/verl/train.parquet [--k 5] [--max-tries 40]
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from text2sql_verl.test_suite import make_copies, suite_key  # noqa: E402

OUT = os.path.join(ROOT, "data", "test_suite")


def build(args):
    db, gold, k, tries, out = args
    key = suite_key(db, gold)
    d = os.path.join(out, key)
    os.makedirs(d, exist_ok=True)
    t = time.time()
    try:
        copies = make_copies(db, gold, k=k, max_tries=tries, seed=0, out_dir=d)
    except Exception as e:  # unusual schema: no suite, plain match
        print(f"{key}: {type(e).__name__}: {e}", file=sys.stderr)
        copies = []
    return key, copies, time.time() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tries", type=int, default=40)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out-dir", default=OUT, help="suite directory (manifest.json lives here)")
    a = ap.parse_args()
    rows = {}
    for p in a.parquet:
        for _, r in pd.read_parquet(p).iterrows():
            rows[suite_key(r["extra_info"]["db_path"], r["reward_model"]["ground_truth"])] = (
                r["extra_info"]["db_path"], r["reward_model"]["ground_truth"], a.k, a.max_tries, a.out_dir)
    mpath = os.path.join(a.out_dir, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    todo = [v for key, v in rows.items() if key not in manifest]
    print(f"{len(rows)} questions, {len(todo)} to build")
    with Pool(a.procs) as pool:
        for i, (key, copies, dt) in enumerate(pool.imap_unordered(build, todo), 1):
            manifest[key] = copies
            if i % 100 == 0:
                print(f"{i}/{len(todo)}")
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(manifest, open(mpath + ".new", "w"), indent=0)
    os.replace(mpath + ".new", mpath)
    n = [len(manifest[key]) for key in rows]
    print(f"coverage: >=1 copy {sum(x > 0 for x in n)}/{len(n)}, all {a.k}: {sum(x == a.k for x in n)}/{len(n)}")
