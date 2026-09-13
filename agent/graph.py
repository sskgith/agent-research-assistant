"""
StateGraph wiring for the research assistant agent.

For this smoke test, the agent node uses a hardcoded decision sequence
instead of a real LLM call. This lets us confirm the graph's loop,
state-passing, and termination logic all work before any API key or
model choice enters the picture.
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from tools.calculator_tool import calculator
from tools.sql_query_tool import sql_query
from tools.web_search_tool import web_search


def agent_node(state: AgentState) -> dict:
    """
    Decide the next action. FAKE LOGIC for now — a real version will
    replace this with an LLM call that reasons over `state["task"]` and
    `state["tool_calls"]` so far.
    """
    step = state["step_count"]

    if step == 0:
        next_action, next_input = "calculator", "12 * (4 + 3)"
    elif step == 1:
        next_action, next_input = "sql_query", "SELECT * FROM products WHERE price > 10"
    else:
        next_action, next_input = "finish", ""

    return {
        "next_action": next_action,
        "next_action_input": next_input,
        "step_count": step + 1,
    }


def tool_executor_node(state: AgentState) -> dict:
    """Run whichever tool the agent node selected, and record the call."""
    action = state["next_action"]
    tool_input = state["next_action_input"]

    if action == "calculator":
        output = calculator(tool_input)
    elif action == "sql_query":
        output = sql_query(tool_input)
    elif action == "web_search":
        output = web_search(tool_input)
    else:
        output = f"[ERROR] Unknown tool: {action}"

    new_tool_calls = state["tool_calls"] + [
        {"tool_name": action, "tool_input": tool_input, "tool_output": output}
    ]

    return {"tool_calls": new_tool_calls}


def route_after_agent(state: AgentState) -> str:
    """Decide whether to run a tool or finish, based on the agent's decision."""
    if state["next_action"] == "finish" or state["step_count"] > state["max_steps"]:
        return "finish"
    return "use_tool"


def finish_node(state: AgentState) -> dict:
    """Compose the final answer from the tool call trace."""
    summary = "; ".join(
        f"{call['tool_name']}({call['tool_input']}) -> {call['tool_output']}"
        for call in state["tool_calls"]
    )
    return {"final_answer": f"Done. Trace: {summary}"}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("finish", finish_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"use_tool": "tool_executor", "finish": "finish"},
    )
    graph.add_edge("tool_executor", "agent")
    graph.add_edge("finish", END)

    return graph.compile()