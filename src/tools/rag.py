"""
RAG: Build a ChromaDB vector store from the product catalog.

The vector store is created once at import time and re-used by the
search_product_catalog tool.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import embeddings
from logger import get_logger
from data.menu import MENU_ITEMS

logger = get_logger("rag")


def _build_documents() -> list[Document]:
    """Convert every catalog entry into a LangChain Document."""
    docs: list[Document] = []
    for p in MENU_ITEMS:
        content = (
            f"Dish: {p['name']}\n"
            f"Category: {p['category']}\n"
            f"Cuisine: {p['cuisine']}\n"
            f"Price: ₹{p['price']}\n"
            f"Rating: {p['rating']}/5\n"
            f"Dietary Tags: {p['dietary_tags']}\n"
            f"Description: {p['description']}\n"
            f"Availability: {p['availability']}\n"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "id": p["id"],
                    "name": p["name"],
                    "Dietary Tags": p["dietary_tags"],
                    "category": p["category"],
                    "price": p["price"],
                    "rating": p["rating"],
                },
            )
        )
    return docs


def build_vectorstore() -> Chroma:
    """Create an in-memory ChromaDB collection from the product catalog."""
    docs = _build_documents()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="menu_items_collection",
    )
    logger.info("Vector store ready  (%d products indexed)", len(docs))
    return store


# Module-level singleton so every importer shares the same store
product_vectorstore = build_vectorstore()

def retrieve_similar_products(query: str, k: int = 3) -> list[Document]:
    """Helper to query the vector store directly (not used by the agent)."""
    logger.info("Retrieving similar products for query: %r", query)
    return product_vectorstore.similarity_search(query, k=k)
