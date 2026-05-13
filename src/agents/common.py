from state import AgentState
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import END


def should_continue(state: AgentState) -> str:
    """Route after model node: tool_calls → tools, otherwise → END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

def build_context(messages: list[AnyMessage]) -> str:
    """Format prior conversation turns as text for agent context."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    if not parts:
        return ""
    return "CONVERSATION SO FAR:\n" + "\n".join(parts) + "\n\n"