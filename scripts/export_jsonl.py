"""Export SkyRL-SQL-653 (newfmt) to the canonical text2sql JSONL that oumi consumes.

One line per example:
  {"id", "question", "schema", "external_knowledge", "db_id", "db_path", "gold_sql", "source"}

`db_path` is absolute, so downstream steps don't need to know the OmniSQL layout.

Usage:
  python scripts/export_jsonl.py --parquet raw/skyrl/newfmt_train.parquet \
      --db-root data/db --out data/jsonl/train.jsonl
"""

import argparse
import json
import os

import pandas as pd

# Same layout SkyRL's SQLEnv resolves, relative to the unzipped OmniSQL `data/` dir.
DB_SUBDIRS = {
    "synsql": "SynSQL-2.5M/databases",
    "spider": "spider/database",
    "bird": "bird/train/train_databases",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--db-root", required=True, help="directory containing SynSQL-2.5M/, spider/, ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    missing = 0
    with open(args.out, "w") as f:
        for i, r in df.iterrows():
            src = r["data"]
            db_path = os.path.abspath(os.path.join(args.db_root, DB_SUBDIRS[src], r["db_id"], f"{r['db_id']}.sqlite"))
            missing += not os.path.exists(db_path)
            ek = r.get("external_knowledge")
            rec = {
                "id": f"{src}-{i}",
                "question": r["question"],
                "schema": r["schema"],
                "external_knowledge": None if ek in (None, "None", "") else ek,
                "db_id": r["db_id"],
                "db_path": db_path,
                "gold_sql": r["sql"],
                "source": src,
            }
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(df)} rows to {args.out} ({missing} missing db files)")


if __name__ == "__main__":
    main()
