# ═══════════════════════════════════════════════════════════
#  PRODUCT DISCOVERY TOOL
# ═══════════════════════════════════════════════════════════
from __future__ import annotations
from logger import get_logger
from langchain.tools import tool
from tools.rag import retrieve_similar_products


logger = get_logger(__name__)

@tool
def search_product_catalog(query: str) -> str:
    """Search the SnackStack product catalog using semantic search (RAG).

    Args:
        query: natural-language search, e.g. "pasta dishes under 5000"
    """
    logger.info("search_product_catalog  query=%r", query)
    try:
        docs = retrieve_similar_products(query, k=3)
        if not docs:
            return "No products found matching your query."
        results = "Found the following products:\n\n"
        for i, doc in enumerate(docs, 1):
            results += f"Product {i}:\n{doc.page_content}\n\n"
        return results
    except Exception as exc:
        logger.exception("Catalog search failed")
        return f"Error searching catalog: {exc}"
