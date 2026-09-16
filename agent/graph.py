"""
StateGraph wiring for the research assistant agent.

agent_node now calls a LLM (via Groq) to decide which tool to use
and with what input, based on the task and the tool-call history so far.
"""

import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from tools.calculator_tool import calculator
from tools.sql_query_tool import sql_query
from tools.web_search_tool import web_search

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

SYSTEM_PROMPT = """You are a research assistant agent with access to three tools:
- calculator: evaluate arithmetic expressions, e.g. "12 * (4 + 3)"
- sql_query: run a SELECT query against a `products` table with columns (id, name, price)
- web_search: search the web for a text query

IMPORTANT: Never perform arithmetic yourself. Any time the task requires a
calculation — even a simple one — you MUST call the calculator tool with the
exact expression, rather than computing the result mentally.

Given the task and the tool calls made so far, decide the SINGLE next action.
Respond with ONLY a JSON object, no other text, in this exact format:
{"next_action": "<calculator|sql_query|web_search|finish>", "next_action_input": "<input string, or final answer text if finishing>"}

Choose "finish" once you have enough information to answer the task directly.
When you finish, put your actual answer to the task in next_action_input.
"""


def agent_node(state: AgentState) -> dict:
    step = state["step_count"]

    if step >= state["max_steps"]:
        return {"next_action": "finish", "next_action_input": "Max steps reached.", "step_count": step + 1}

    history = "\n".join(
        f"- {c['tool_name']}({c['tool_input']}) -> {c['tool_output']}"
        for c in state["tool_calls"]
    ) or "(none yet)"

    user_prompt = f"Task: {state['task']}\n\nTool calls so far:\n{history}\n\nWhat's the next action?"

    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    try:
        decision = json.loads(response.content)
        next_action = decision["next_action"]
        next_input = decision["next_action_input"]
    except (json.JSONDecodeError, KeyError):
        next_action, next_input = "finish", f"[ERROR] Could not parse model response: {response.content}"

    return {
        "next_action": next_action,
        "next_action_input": next_input,
        "step_count": step + 1,
    }


def tool_executor_node(state: AgentState) -> dict:
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