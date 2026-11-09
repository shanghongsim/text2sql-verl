"""CPU-only checks for the verl plugins: tool execution, turn reminder, submit, reward.

  python scripts/test_plugins.py
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf  # noqa: E402
from verl.tools.tool_registry import initialize_tools_from_config  # noqa: E402

from text2sql_verl.reward import compute_score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_db(path):
    c = sqlite3.connect(path)
    c.executescript(
        """CREATE TABLE animals(id INTEGER PRIMARY KEY, species TEXT, age INT);
        INSERT INTO animals(species, age) VALUES ('pig',1),('pig',2),('cow',3);
        CREATE TABLE big AS WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000)
          SELECT x, printf('row-%05d-padding-padding', x) AS s FROM n;"""
    )
    c.commit()
    c.close()


async def main():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "farm.sqlite")
    make_db(db)

    tools = {t.name: t for t in initialize_tools_from_config(f"{ROOT}/configs/tool_config.yaml")}
    assert set(tools) == {"execute_sql", "submit_sql"}, tools.keys()
    ex, sub = tools["execute_sql"], tools["submit_sql"]
    agent = SimpleNamespace(assistant_turns=2, extra_fields={})
    ck = {"db_path": db}

    async def run(tool, sql):
        iid, _ = await tool.create(create_kwargs=ck)
        try:
            r, _, m = await tool.execute(iid, {"sql": sql}, agent_data=agent)
        finally:
            await tool.release(iid)
        return r.text

    out = await run(ex, "SELECT COUNT(*) FROM animals WHERE species='pig'")
    print("--- ok query\n" + out)
    assert out.startswith("COUNT(*)\n2") and "4 turns left" in out

    out = await run(ex, "SELECT nope FROM animals")
    print("--- error\n" + out)
    assert "Error executing SQL" in out and "no such column" in out

    out = await run(ex, "DELETE FROM animals")
    print("--- write blocked\n" + out)
    assert "Error" in out

    out = await run(ex, "SELECT * FROM big")
    assert out.startswith("Truncated to 50 rows") and out.count("\n") < 60, out[:200]
    print("--- truncation ok")

    out = await run(ex, "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT count(*) FROM n")
    print("--- timeout\n" + out.splitlines()[0])
    assert out.startswith("SQL Timeout")

    agent.assistant_turns = 5
    out = await run(ex, "SELECT 1")
    assert "1 turns left" in out and "MUST call `submit_sql`" in out

    gold = "SELECT COUNT(*) FROM animals WHERE species = 'pig'"
    base = {"db_path": db}
    assert compute_score("t", "", gold, dict(base))["score"] == -1.0  # nothing submitted
    await run(sub, "SELECT count(id) FROM animals WHERE species='pig'")
    assert agent.extra_fields["submitted_sql"].startswith("SELECT count(id)")
    await run(sub, "SELECT 999")  # second submit ignored
    s = compute_score("t", "", gold, {**base, **agent.extra_fields})
    print("--- reward correct", s)
    assert s["score"] == 1.0
    assert compute_score("t", "", gold, {**base, "submitted_sql": "SELECT 3"})["score"] == 0.0
    assert compute_score("t", "", gold, {**base, "submitted_sql": "SELEC oops"})["score"] == 0.0
    # frozenset semantics (SkyRL): row order and duplicates ignored
    g2 = "SELECT species FROM animals ORDER BY id"
    assert compute_score("t", "", g2, {**base, "submitted_sql": "SELECT DISTINCT species FROM animals ORDER BY 1 DESC"})["score"] == 1.0

    # agent loop subclass + config resolves
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    from text2sql_verl.agent_loop import Text2SQLAgentLoop

    cfg = OmegaConf.load(f"{ROOT}/configs/agent_loop.yaml")
    assert cfg[0]["_target_"] == "text2sql_verl.agent_loop.Text2SQLAgentLoop" and issubclass(Text2SQLAgentLoop, ToolAgentLoop)
    print("\nALL PLUGIN CHECKS PASSED")


asyncio.run(main())
