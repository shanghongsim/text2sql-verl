"""verl tools for text-to-SQL: `execute_sql` (explore) and `submit_sql` (final answer).

Both receive the example's database via per-row `create_kwargs: {db_path: ...}`.
`submit_sql` records the answer in `agent_data.extra_fields`, which verl forwards
to the reward function's `extra_info`; `Text2SQLAgentLoop` ends the episode on it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from text2sql_verl import sql_exec

SUBMITTED_SQL_KEY = "submitted_sql"


class _DBTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._db_paths: dict[str, str] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        create_kwargs = kwargs.get("create_kwargs") or {}
        if "db_path" not in create_kwargs:
            raise ValueError(f"{self.name}: create_kwargs must contain 'db_path'")
        self._db_paths[instance_id] = create_kwargs["db_path"]
        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        self._db_paths.pop(instance_id, None)


class ExecuteSQLTool(_DBTool):
    """Runs a read-only query and returns the result table plus a turn-budget reminder."""

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        sql = parameters.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            obs = "Error: `sql` must be a non-empty string."
            res = None
        else:
            res = await asyncio.to_thread(
                sql_exec.execute, self._db_paths[instance_id], sql, float(self.config.get("timeout_s", 5))
            )
            obs = sql_exec.format_observation(res, sql)

        agent_data = kwargs.get("agent_data")
        max_turns = self.config.get("max_turns")
        if agent_data is not None and max_turns:
            turns_left = max(int(max_turns) - agent_data.assistant_turns, 0)
            obs += f"\n\nYou have {turns_left} turns left to complete the task."
            if turns_left <= 1:
                obs += " You MUST call `submit_sql` with your final query now."
        return ToolResponse(text=obs), 0.0, {"sql_ok": float(bool(res and res.ok))}


class SubmitSQLTool(_DBTool):
    """Records the final SQL answer; the agent loop terminates after this call."""

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        sql = parameters.get("sql")
        agent_data = kwargs.get("agent_data")
        if not isinstance(sql, str) or not sql.strip():
            return ToolResponse(text="Error: `sql` must be a non-empty string."), 0.0, {}
        if agent_data is not None and SUBMITTED_SQL_KEY not in agent_data.extra_fields:
            agent_data.extra_fields[SUBMITTED_SQL_KEY] = sql
        return ToolResponse(text="Final SQL submitted."), 0.0, {}
