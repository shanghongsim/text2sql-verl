"""Canonical text2sql JSONL -> verl agent-loop parquet.

This is the conversion oumi will own; it has no verl imports. Each row gets:
  - prompt: system (task + tool usage rules) + user (schema, knowledge, question)
  - agent_name: text2sql_agent (our ToolAgentLoop subclass)
  - reward_model.ground_truth: gold SQL
  - extra_info.db_path + tools_kwargs.{execute_sql,submit_sql}.create_kwargs.db_path

Usage:
  python scripts/build_verl_dataset.py --jsonl data/jsonl/train.jsonl --out data/verl/train.parquet --max-turns 6
"""

import argparse
import json
import os

import pandas as pd

# Adapted from SkyRL-SQL's system prompt: same task framing and instructions,
# with the <sql>/<solution> tag protocol replaced by function calls.
SYSTEM_PROMPT = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question within limited turns. You should break down the problem, draft your reasoning process, and generate the solution.

Database Engine:
SQLite

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, think through the steps of how to write the query: analyze the question, summarize relevant findings, brainstorm new ideas, verify the accuracy of the current steps, refine any errors, and revisit previous steps.

Tools:
- Call `execute_sql` to explore the database or verify a draft query. Its output is the result table (truncated to 50 rows if too long).
- When no further exploration is needed, call `submit_sql` exactly once with your final SQL query. This ends the task.
- You have at most {max_turns} turns. If you are running out of turns, you MUST call `submit_sql` immediately.
- Think step by step before every tool call."""

# v2: v1 + a worked example in the exact tool-call format (SkyRL's prompt has a worked example; v1 dropped it,
# and Qwen2.5-Coder-1.5B then made 0 tool calls in 224 rollouts, writing ```sql blocks instead).
TOOL_EXAMPLE = """

Never write the SQL query in a markdown code block. The only way to run a query or give your answer is a tool call in exactly this format:
<tool_call>
{"name": "<function-name>", "arguments": {"sql": "<your SQL>"}}
</tool_call>

------------------------ START OF EXAMPLE ------------------------
Question: how many pigs are in the farm?
Database Schema:
CREATE TABLE animals (id INTEGER PRIMARY KEY, species TEXT, age INTEGER, name TEXT);

assistant: I need to count the rows of `animals` whose species is 'pig'. Let me check the data first.
<tool_call>
{"name": "execute_sql", "arguments": {"sql": "SELECT COUNT(*) FROM animals WHERE species = 'pig';"}}
</tool_call>

tool: COUNT(*)
12

You have 5 turns left to complete the task.

assistant: The query returns 12, which answers how many pigs there are. I will submit it.
<tool_call>
{"name": "submit_sql", "arguments": {"sql": "SELECT COUNT(*) FROM animals WHERE species = 'pig';"}}
</tool_call>
------------------------ END OF EXAMPLE ------------------------"""

# v2xml: v2 with the example's calls in the XML form that Qwen3.5 / Qwen3-Coder emit natively
# (verl multi_turn.format=qwen3_coder). Same words; only the call syntax differs.
def _xml_call(name, sql):
    return f"<tool_call>\n<function={name}>\n<parameter=sql>\n{sql}\n</parameter>\n</function>\n</tool_call>"


TOOL_EXAMPLE_XML = (
    TOOL_EXAMPLE.replace('<tool_call>\n{"name": "<function-name>", "arguments": {"sql": "<your SQL>"}}\n</tool_call>', _xml_call("<function-name>", "<your SQL>"))
    .replace('<tool_call>\n{"name": "execute_sql", "arguments": {"sql": "SELECT COUNT(*) FROM animals WHERE species = \'pig\';"}}\n</tool_call>', _xml_call("execute_sql", "SELECT COUNT(*) FROM animals WHERE species = 'pig';"))
    .replace('<tool_call>\n{"name": "submit_sql", "arguments": {"sql": "SELECT COUNT(*) FROM animals WHERE species = \'pig\';"}}\n</tool_call>', _xml_call("submit_sql", "SELECT COUNT(*) FROM animals WHERE species = 'pig';"))
)
assert '{"name"' not in TOOL_EXAMPLE_XML

PROMPT_SUFFIX = {"v1": "", "v2": TOOL_EXAMPLE, "v2xml": TOOL_EXAMPLE_XML}  # appended after .format() (the example contains JSON braces)

USER_TEMPLATE = """Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

External Knowledge:
{external_knowledge}

Question:
{question}"""


def to_row(rec: dict, idx: int, split: str, max_turns: int, prompt_version: str = "v2") -> dict:
    db_kwargs = {"create_kwargs": {"db_path": rec["db_path"]}}
    return {
        "data_source": f"text2sql/{rec.get('source', 'unknown')}",
        "agent_name": "text2sql_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT.format(max_turns=max_turns) + PROMPT_SUFFIX[prompt_version]},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    schema=rec["schema"].strip(),
                    external_knowledge=rec.get("external_knowledge") or "None",
                    question=rec["question"].strip(),
                ),
            },
        ],
        "ability": "text2sql",
        "reward_model": {"style": "rule", "ground_truth": rec["gold_sql"]},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": rec.get("id", str(idx)),
            "db_id": rec["db_id"],
            "db_path": rec["db_path"],
            "need_tools_kwargs": True,
            "tools_kwargs": {"execute_sql": db_kwargs, "submit_sql": db_kwargs},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-version", choices=["v1", "v2", "v2xml"], default="v2")
    args = ap.parse_args()

    split = args.split or os.path.splitext(os.path.basename(args.jsonl))[0]
    with open(args.jsonl) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        recs = recs[: args.limit]
    rows = [to_row(r, i, split, args.max_turns, args.prompt_version) for i, r in enumerate(recs)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {len(rows)} rows to {args.out} (prompt {args.prompt_version})")


if __name__ == "__main__":
    main()
