"""verl's ToolAgentLoop, plus two episode-end rules for text2sql.

1. Stop the episode once `submit_sql` has been called. Stock ToolAgentLoop only stops when the model
   emits no tool call, or hits the turn/length budget.
2. Enforce the turn budget *after* the last turn's tool calls run. Stock ToolAgentLoop checks
   `assistant_turns >= max_assistant_turns` right after generation, before parsing tool calls, so a
   `submit_sql` on the final turn is silently dropped (reward -1), even though the turn reminder tells
   the model to submit on that turn. SkyRL scores the final turn's answer; so do we.
"""

from __future__ import annotations

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

from text2sql_verl.tools import SUBMITTED_SQL_KEY


class Text2SQLAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Take over the assistant-turn limit (rule 2); verl's own check would drop the last turn's calls.
        self.turn_budget = self.max_assistant_turns
        self.max_assistant_turns = None

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if SUBMITTED_SQL_KEY in agent_data.extra_fields:
            return AgentState.TERMINATED
        if self.turn_budget and agent_data.assistant_turns >= self.turn_budget:
            return AgentState.TERMINATED
        return state
