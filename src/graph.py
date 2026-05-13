"""
Build and compile the LangGraph StateGraph.

Graph topology:

  START → orchestrator ─┬─ product_agent ──→ synthesizer → END
                        └─ support_agent ──↗

Each agent is internally a subgraph with a model ⇄ tools loop.
The MemorySaver checkpointer persists conversation history across
turns, enabling multi-turn HITL (agent asks a question on one turn,
user answers on the next).
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from logger import get_logger
from agents.orchestrator import orchestrator_node
from agents.menu_agent import product_agent
from agents.order_agent import support_agent
from agents.synthesizer import synthesizer_node
from state import SnackStackState

logger = get_logger("graph")


def build_graph() -> StateGraph:
    """Create, wire, and compile the AxiomCart multi-agent graph."""

    builder = StateGraph(SnackStackState)

    # ── Add nodes ────────────────────────────────────────
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("product_agent", product_agent)
    builder.add_node("support_agent", support_agent)
    builder.add_node("synthesizer", synthesizer_node)

    # ── Add edges ────────────────────────────────────────
    builder.add_edge(START, "orchestrator")
    builder.add_edge("synthesizer", END)

    # ── Compile with checkpointer ────────────────────────
    # MemorySaver persists graph state so that interrupt()-based
    # HITL can pause and resume, and conversation history carries
    # forward between queries.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    logger.info("Graph Structure: ")
    print(graph.get_graph().draw_ascii())     # Visualise the graph structure in the logs
    logger.info("Graph compiled  (with MemorySaver for conversation persistence)")
    return graph


# Module-level singleton
snackStack_graph = build_graph()
