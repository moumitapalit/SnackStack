"""
Configuration: Logger, API clients, and settings.

This module is imported by everything else, so it must have zero
dependencies on other project modules.
"""

import os
import sys
from logger import get_logger
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# ── Load .env ────────────────────────────────────────────────
load_dotenv()

logger = get_logger("config")

# ── Validate API key ────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
    logger.error("OPENAI_API_KEY is missing. Copy .env.example → .env and add your key.")
    sys.exit(1)

# ── Clients ──────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

logger.info("OpenAI clients initialised  (model: gpt-4o, embeddings: text-embedding-3-small)")
