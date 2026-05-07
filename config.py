# ============================================
# config.py — All configuration in one place
# ============================================
# CONCEPT: Never scatter config across files
# Change chunk_size? Change it HERE, not in 5 places
# Change model? Change it HERE
# Java equivalent: application.properties
# ============================================

import os
from dotenv import load_dotenv
load_dotenv()

# ── API Keys ──────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ── Embedding Model ───────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking Settings ─────────────────────
# Learned on Day 4: 500 chars gives better retrieval
# for structured documents like tax guides
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# ── Retrieval Settings ────────────────────
# k=3 returns top 3 most relevant chunks
RETRIEVAL_K = 3

# ── LLM Settings ─────────────────────────
LLM_MODEL       = "claude-sonnet-4-5"
LLM_TEMPERATURE = 0  # 0 = deterministic, factual answers

# ── Paths ─────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
DATA_PATH      = "./data"