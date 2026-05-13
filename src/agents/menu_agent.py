from tools.menu_tools import search_product_catalog
from state import AgentState, WorkerInput
from config import llm
from agents.prompt import MENU_AGENT_PROMPT
from logger import get_logger
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.graph import StateGraph, START
from langchain.messages import SystemMessage, HumanMessage
from agents.common import should_continue, build_context
from typing import Literal

menu_tools = [search_product_catalog]
menu_tools_by_name = {t.name: t for t in menu_tools}

logger = get_logger(__name__)
menu_llm = llm.bind_tools(menu_tools)

def menu_model(state: AgentState) -> dict:
    """Call the menu LLM (with tools bound)."""
    response = menu_llm.invoke(state["messages"])
    logger.info("[menu:model] tool_calls=%s", bool(response.tool_calls))
    return {"messages": [response]}


def menu_tools(state: AgentState) -> dict:
    """Execute tool calls from the menu LLM."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[menu:tools] %s(%s)", name, args)
        out = menu_tools_by_name[name].invoke(args) if name in menu_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}


pb = StateGraph(AgentState)
pb.add_node("model", menu_model)
pb.add_node("tools", menu_tools)
pb.add_edge(START, "model")
pb.add_conditional_edges("model", should_continue)
pb.add_edge("tools", "model")
product_subgraph = pb.compile()

# ═══════════════════════════════════════════════════════════
#  NODE 2 — Product Agent
# ═══════════════════════════════════════════════════════════

def product_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the product-discovery agent via its model ⇄ tools subgraph."""
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Product Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = product_subgraph.invoke({"messages": [
        SystemMessage(content=MENU_AGENT_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "product_discovery", "response": answer}]},
        goto="synthesizer",
    )
