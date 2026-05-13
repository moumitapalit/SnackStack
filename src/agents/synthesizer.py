# ═══════════════════════════════════════════════════════════
#  NODE 4 — Synthesizer
# ═══════════════════════════════════════════════════════════
from config import llm
from logger import get_logger
from state import SnackStackState

logger = get_logger(__name__)

def synthesizer_node(state: SnackStackState) -> dict:
    """Merge results from one or more agents into a single user-facing reply."""
    results = state.get("agent_results", [])
    user_query = state.get("user_query", "")

    if not results:
        logger.warning("Synthesizer received no agent results")
        return {"final_answer": "Sorry, I couldn't process that request. Please try again."}

    if len(results) == 1:
        logger.info("Synthesizer  single-agent pass-through")
        return {"final_answer": results[0]["response"]}

    logger.info("Synthesizer  merging %d agent responses", len(results))

    parts = "\n\n".join(
        f"[{r['source'].upper()}]:\n{r['response']}" for r in results
    )
    prompt = (
        f"You are combining responses from multiple specialist agents.\n\n"
        f"CUSTOMER QUERY: {user_query}\n\n"
        f"AGENT RESPONSES:\n{parts}\n\n"
        "Write a single, coherent reply that addresses every part of the "
        "customer's query. Be concise. Speak as 'AxiomCart Assistant'."
    )

    merged = llm.invoke(prompt)
    return {"final_answer": merged.content}