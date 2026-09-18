"""
StateGraph wiring for the research assistant agent.

agent_node calls Groq's native tool-calling API — the model returns a
structured tool_call object rather than free-text JSON we have to parse,
removing the failure mode of the model wrapping its answer in markdown
fences or adding stray text around the JSON.

Tool call arguments are passed between nodes as a JSON string (json.dumps/
json.loads) rather than eval() — same reasoning as the calculator tool's
use of ast instead of raw eval(): never execute arbitrary text as code,
even text you generated yourself.
"""

import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from tools.calculator_tool import calculator as _calculator
from tools.sql_query_tool import sql_query as _sql_query
from tools.web_search_tool import web_search as _web_search

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '12 * (4 + 3)'. Never do arithmetic yourself — always call this."""
    return _calculator(expression)


@tool
def sql_query(query: str) -> str:
    """Run a SELECT query against a `products` table with columns (id, name, price)."""
    return _sql_query(query)


@tool
def web_search(query: str) -> str:
    """Search the web for a text query."""
    return _web_search(query)


TOOLS = [calculator, sql_query, web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
).bind_tools(TOOLS)

SYSTEM_PROMPT = """You are a research assistant agent. Use the available tools
to answer the task step by step. Never perform arithmetic yourself — always
call the calculator tool for any calculation, even a simple one. Once you
have enough information, respond with your final answer in plain text with
no further tool calls."""


def agent_node(state: AgentState) -> dict:
    step = state["step_count"]

    if step >= state["max_steps"]:
        return {"next_action": "finish", "next_action_input": "Max steps reached.", "step_count": step + 1}

    history = "\n".join(
        f"- {c['tool_name']}({c['tool_input']}) -> {c['tool_output']}"
        for c in state["tool_calls"]
    ) or "(none yet)"

    user_prompt = f"Task: {state['task']}\n\nTool calls so far:\n{history}"

    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    if response.tool_calls:
        call = response.tool_calls[0]  # one tool call per step, by design
        return {
            "next_action": call["name"],
            "next_action_input": json.dumps(call["args"]),
            "step_count": step + 1,
        }

    return {
        "next_action": "finish",
        "next_action_input": response.content,
        "step_count": step + 1,
    }


def tool_executor_node(state: AgentState) -> dict:
    action = state["next_action"]
    raw_input = state["next_action_input"]

    if action in TOOLS_BY_NAME:
        args = json.loads(raw_input)
        output = TOOLS_BY_NAME[action].invoke(args)
    else:
        output = f"[ERROR] Unknown tool: {action}"

    new_tool_calls = state["tool_calls"] + [
        {"tool_name": action, "tool_input": raw_input, "tool_output": output}
    ]

    return {"tool_calls": new_tool_calls}


def route_after_agent(state: AgentState) -> str:
    if state["next_action"] == "finish":
        return "finish"
    return "use_tool"


def finish_node(state: AgentState) -> dict:
    return {"final_answer": state["next_action_input"]}


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