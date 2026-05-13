from config import llm
from logger import get_logger
from state import AgentState, WorkerInput
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from agents.prompt import ORDER_AGENT_PROMPT
from tools import order_tools
from langgraph.graph import END
from agents.common import build_context
from typing import Literal
from langgraph.types import Command, Send
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt
from tools.order_tools import get_order_status

logger = get_logger(__name__)
order_tools = [get_order_status]
order_tools_by_name = {t.name: t for t in order_tools}

support_llm   = llm.bind_tools(order_tools)

# ── Support subgraph ─────────────────────────────────────────

def support_model(state: AgentState) -> dict:
    """Call the support LLM. If it asks for info without calling tools,
    use interrupt() to pause the graph and collect user input."""
    response = support_llm.invoke(state["messages"])
    logger.info("[support:model] tool_calls=%s", bool(response.tool_calls))

    # If no tool calls and no tools have been called yet,
    # the agent is asking for missing info — interrupt for HITL
    if not response.tool_calls:
        any_tools_called = any(isinstance(m, ToolMessage) for m in state["messages"])
        if not any_tools_called:
            logger.info("[support:model] HITL: interrupting to collect user info")
            user_reply = interrupt(response.content)
            logger.info("[support:model] HITL: user replied %r", user_reply)
            return {"messages": [response, HumanMessage(content=str(user_reply))]}

    return {"messages": [response]}


def support_tools(state: AgentState) -> dict:
    """Execute tool calls from the support LLM."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[support:tools] %s(%s)", name, args)
        out = order_tools_by_name[name].invoke(args) if name in order_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}


def support_should_continue(state: AgentState) -> str:
    """Route after support model node. If the last message is a
    HumanMessage (user answered via HITL interrupt), loop back to model."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "model"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


sb = StateGraph(AgentState)
sb.add_node("model", support_model)
sb.add_node("tools", support_tools)
sb.add_edge(START, "model")
sb.add_conditional_edges("model", support_should_continue)
sb.add_edge("tools", "model")
support_subgraph = sb.compile()

# ═══════════════════════════════════════════════════════════
#  NODE 3 — Support Agent
# ═══════════════════════════════════════════════════════════

def support_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the sales-support agent via its model ⇄ tools subgraph.

    HITL is handled through conversation persistence: if the agent
    needs info (e.g. order ID), it responds with a question. The
    user's answer arrives on the next turn via the message history.
    """
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Support Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = support_subgraph.invoke({"messages": [
        SystemMessage(content=ORDER_AGENT_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "sales_support", "response": answer}]},
        goto="synthesizer",
    )
