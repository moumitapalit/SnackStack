"""
Agent Tools — the concrete actions agents can take.

order Support (2 tools):
  • get_order_status        – order lookup by ID or email
  • escalate_to_human       – HITL ticket + optional email via Resend
"""

from __future__ import annotations


from langchain_core.tools import tool

from data.orders import FOOD_ORDER_DATABASE
from logger import get_logger

logger = get_logger(__name__)


# ── helpers ──────────────────────────────────────────────────

def normalise_order_id(raw: str) -> str:
    """Accept 'ORD101', 'ORD-101', 'ord-101', or just '101' → 'ORD101'."""
    upper = raw.upper().strip()
    clean = upper.replace("ORD-", "").replace("ORD", "").strip()
    return f"ORD{clean}"


def lookup_order_by_email(email: str) -> dict | None:
    """Find the first order matching a customer email."""
    email_lower = email.lower().strip()
    for oid, order in FOOD_ORDER_DATABASE.items():
        if order["customer_email"].lower() == email_lower:
            return {"order_id": oid, **order}
    return None



# ═══════════════════════════════════════════════════════════
#  SALES SUPPORT TOOLS
# ═══════════════════════════════════════════════════════════

@tool
def get_order_status(identifier: str) -> str:
    """Look up the current status of a customer order.

    Args:
        identifier: an order ID (e.g. "ORD101") OR a customer email address
    """
    logger.info("get_order_status  identifier=%r", identifier)

    # Try as email first
    if "@" in identifier:
        match = lookup_order_by_email(identifier)
        if match:
            oid = match["order_id"]
            order = {k: v for k, v in match.items() if k != "order_id"}
        else:
            return f"No order found for email: {identifier}"
    else:
        oid = normalise_order_id(identifier)
        logger.info(f"Checking order {oid} for ID")
        order = FOOD_ORDER_DATABASE.get(oid)
        if not order:
            return f"Order {oid} not found. Please verify the order ID."

    info = (
        f"Order {oid}:\n"
        f"  Customer : {order['customer_name']} ({order['customer_email']})\n"
        f"  Dish     : {order['item_name']}\n"
        f"  Price    : ₹{order['price']:,}\n"
        f"  Status   : {order['status']}\n"
        f"  Ordered  : {order['order_date']}\n"
        f"  ETA      : {order['estimated_delivery']}"
    )
    if order.get("delay_reason"):
        info += f"\n  Delay    : {order['delay_reason']}"
    return info
