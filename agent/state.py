"""
State schema for the research assistant agent.

This TypedDict is the single object that flows through every node in the
LangGraph StateGraph. Each node reads from it and returns a partial update,
which LangGraph merges back in. Keeping the schema explicit here (rather than
relying on a framework-managed scratchpad) is what makes the agent's
reasoning trace inspectable after the fact.
"""

from typing import TypedDict, Literal


class ToolCall(TypedDict):
    """A single record of a tool invocation, kept for the visible trace."""

    tool_name: str
    tool_input: str
    tool_output: str


class AgentState(TypedDict):
    # The original task/question the agent was given.
    task: str

    # Every tool call made so far, in order. This list IS the demo artifact:
    # it's what gets printed/rendered to show "the agent decided to call X".
    tool_calls: list[ToolCall]

    # The agent's current reasoning/plan for what to do next. Set by the
    # agent node before a tool is selected.
    next_action: str  # e.g. "web_search", "calculator", "sql_query", "finish"
    next_action_input: str

    # The final answer, populated only once next_action == "finish".
    final_answer: str

    # Guard against infinite loops. Incremented by the agent node each turn.
    step_count: int
    max_steps: int


def initial_state(task: str, max_steps: int = 6) -> AgentState:
    """Build a fresh AgentState for a new task."""
    return AgentState(
        task=task,
        tool_calls=[],
        next_action="",
        next_action_input="",
        final_answer="",
        step_count=0,
        max_steps=max_steps,
    )
