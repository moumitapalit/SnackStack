from logger import get_logger

from agents.prompt import get_orchestrator_prompt
from config import llm
from state import SnackStackState, ClassificationResult


from typing import Literal
from langgraph.types import Command, Send


logger = get_logger(__name__)
# ═══════════════════════════════════════════════════════════
#  NODE 1 — Orchestrator
# ═══════════════════════════════════════════════════════════

def orchestrator_node(state: SnackStackState) -> Command[Literal["product_agent", "support_agent", "synthesizer"]]:
    """Classify the user query and dispatch to the right agent(s)."""
    user_query = state.get("user_query", "")
    if not user_query and state.get("messages"):
        user_query = state["messages"][-1].content

    logger.info("Orchestrator  query=%r", user_query)

    classifier = llm.with_structured_output(ClassificationResult)
    try:
        classification = classifier.invoke(get_orchestrator_prompt(user_query))
    except Exception:
        logger.exception("Classification failed — defaulting to support_agent")
        classification = ClassificationResult(
            tasks=[], requires_synthesis=False,
            reasoning="Fallback: classification error",
        )

    logger.info("  routing=%s  synthesis=%s",
                [t.agent for t in classification.tasks],
                classification.requires_synthesis)

    targets: list[Send] = []
    for task in classification.tasks:
        targets.append(Send(task.agent, {
            "messages": state.get("messages", []),
            "user_query": user_query,
            "task_description": task.task_description,
        }))

    if not targets:
        targets = [Send("synthesizer", {})]

    return Command(
        update={
            "tasks": classification.tasks,
            "requires_synthesis": classification.requires_synthesis,
            "user_query": user_query,
            "agent_results": [],  # reset stale results from prior turns
        },
        goto=targets,
    )
