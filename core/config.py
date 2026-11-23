"""
Central Configuration for SmartResearch Chatbot (Agentic-RAG).
==============================================================
Orchestrates:
1. Vector Memory & RAG settings (with 100-batch rate pacing)
2. Live Web Crawling with 5-Pillar Academic Targeting
3. Dual-LLM Gateway: OpenRouter (Primary) + Google Gemini (Infinite Fallback)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- API Keys ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

# --- LLM Gateway Models ---
# Embeddings (Rock-solid Google AI Studio Free Tier)
EMBED_MODEL = "gemini-embedding-001"

# Primary Chat (OpenRouter Free Tier)
OR_PRIMARY_MODEL = "nvidia/nemotron-3.5-lightning:free"
OR_FALLBACK_MODEL = "liquid/lfm-3-1.7b:free"

# Native Gemini Backup (If OpenRouter 50/day limit is exhausted)
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
VISION_MODEL = "gemini-2.5-flash"

# Maximum Tokens for deep, authoritative research synthesis (~650-750 words)
MAX_RESEARCH_TOKENS = 1500

# --- RAG & Vector Engine Parameters ---
CHUNK_SIZE = 1500          # Characters per semantic chunk
CHUNK_OVERLAP = 200        # Overlap to prevent cut sentences
TOP_K = 5                  # Context passages retrieved per query
SIMILARITY_THRESHOLD = 0.35 # Below this -> Trigger Autonomous Web Research!

# --- Web Research Parameters (5-Pillar Architecture) ---
QUERIES_PER_TOPIC = 5      # 1-to-1 mapping for all 5 core academic pillars
RESULTS_PER_QUERY = 2      # Targeted top results per pillar
PAGE_LIMIT = 5             # 1 authoritative source per pillar
CHAR_LIMIT_PER_PAGE = 4000
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 10
USER_AGENT = "SmartResearchAgent/1.0 (Autonomous Educational Researcher; polite crawler)"

# --- Storage Directories ---
CACHE_DIR = BASE_DIR / ".cache"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# System Prompt ensuring strict grounding & zero hallucinations
SYSTEM_PROMPT = """\
You are SmartResearch AI, an elite autonomous research assistant combining Vector Document Knowledge with Real-Time Web Intelligence.

CRITICAL RULES:
1. Grounding: Answer strictly using the provided context snippets (either from Document or Live Web Search).
2. Citations:
   - For document-retrieved answers, explicitly cite page numbers: [Page X].
   - For live web research answers, explicitly cite source titles/links: [Source: Website Name (URL)].
3. Transparency: When answering from web research because the document lacked data, mention:
   "ℹ️ Note: This was gathered via real-time web research as it was not present in the indexed document."
4. Honesty: If neither document nor web provides sufficient information, state clearly that no verified data was found.
"""
