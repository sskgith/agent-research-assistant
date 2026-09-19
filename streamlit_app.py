"""
Streamlit chat interface for the research assistant agent.

Shows the full tool-call trace alongside the final answer, since the
trace — not just the answer — is the actual point of this demo: it's
what shows the agent reasoning through tool selection step by step,
rather than following a fixed pipeline.
"""

import streamlit as st

from agent.graph import build_graph
from agent.state import initial_state

st.set_page_config(page_title="Research Assistant Agent", page_icon="🔎")
st.title("🔎 Research Assistant Agent")
st.caption(
    "A LangGraph agent that reasons over a task and decides for itself "
    "which tool to call — calculator, SQL query, or web search — rather "
    "than following a fixed sequence."
)

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

task = st.text_input(
    "Ask it something that needs a calculation and/or a product lookup:",
    placeholder="e.g. What is 12 times 7, plus the price of Widget C?",
)

if st.button("Run", type="primary") and task:
    with st.spinner("Agent is reasoning..."):
        result = st.session_state.graph.invoke(initial_state(task))

    st.subheader("Tool call trace")
    if result["tool_calls"]:
        for i, call in enumerate(result["tool_calls"], start=1):
            st.markdown(f"**Step {i}: `{call['tool_name']}`**")
            st.code(call["tool_input"], language="json")
            st.text(f"→ {call['tool_output']}")
    else:
        st.text("(no tools were called — the agent answered directly)")

    st.subheader("Final answer")
    st.success(result["final_answer"])