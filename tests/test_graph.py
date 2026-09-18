"""
Tests for the graph's routing and tool-selection logic.

These don't call the real LLM (that would be slow, flaky, and rate-limited).
Instead they test the graph mechanics directly: given a specific state,
does the routing function make the correct decision, and does the tool
executor dispatch to the correct tool.

next_action_input values are JSON strings (e.g. '{"expression": "2 + 2"}')
because agent_node now uses native tool-calling, which returns structured
args that get serialized with json.dumps before being stored in state.
"""

from agent.graph import route_after_agent, tool_executor_node, finish_node
from agent.state import initial_state


def test_route_after_agent_selects_tool_when_action_is_not_finish():
    state = initial_state("dummy task")
    state["next_action"] = "calculator"
    assert route_after_agent(state) == "use_tool"


def test_route_after_agent_finishes_when_action_is_finish():
    state = initial_state("dummy task")
    state["next_action"] = "finish"
    assert route_after_agent(state) == "finish"


def test_tool_executor_dispatches_to_calculator():
    state = initial_state("dummy task")
    state["next_action"] = "calculator"
    state["next_action_input"] = '{"expression": "2 + 2"}'
    result = tool_executor_node(state)
    assert result["tool_calls"][0]["tool_name"] == "calculator"
    assert result["tool_calls"][0]["tool_output"] == "4"


def test_tool_executor_dispatches_to_sql_query():
    state = initial_state("dummy task")
    state["next_action"] = "sql_query"
    state["next_action_input"] = '{"query": "SELECT name FROM products WHERE price > 10"}'
    result = tool_executor_node(state)
    assert result["tool_calls"][0]["tool_name"] == "sql_query"
    assert "Widget B" in result["tool_calls"][0]["tool_output"]


def test_tool_executor_dispatches_to_web_search():
    state = initial_state("dummy task")
    state["next_action"] = "web_search"
    state["next_action_input"] = '{"query": "test query"}'
    result = tool_executor_node(state)
    assert result["tool_calls"][0]["tool_name"] == "web_search"
    assert "test query" in result["tool_calls"][0]["tool_output"]


def test_tool_executor_handles_unknown_tool():
    state = initial_state("dummy task")
    state["next_action"] = "not_a_real_tool"
    state["next_action_input"] = "irrelevant"
    result = tool_executor_node(state)
    assert result["tool_calls"][0]["tool_output"].startswith("[ERROR]")


def test_tool_executor_records_history_across_calls():
    state = initial_state("dummy task")
    state["tool_calls"] = [{"tool_name": "calculator", "tool_input": '{"expression": "1+1"}', "tool_output": "2"}]
    state["next_action"] = "calculator"
    state["next_action_input"] = '{"expression": "2+2"}'
    result = tool_executor_node(state)
    assert len(result["tool_calls"]) == 2


def test_finish_node_returns_final_answer_from_input():
    state = initial_state("dummy task")
    state["next_action_input"] = "The answer is 106"
    result = finish_node(state)
    assert result["final_answer"] == "The answer is 106"