# 🧠 SmartResearch Chatbot (Agentic-RAG Architecture)

[![CI Pipeline](https://github.com/mr-sagarajput/SmartResearch-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/mr-sagarajput/SmartResearch-Agent/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-grade, enterprise autonomous research system combining **Persistent Vector Memory** with **Live Real-Time Web Intelligence** and **Automated Academic PDF Report Compilation**.

---

## 🌟 Key Architecture Innovations

1. **Dual-Intent Mode Engine:**
   - **Mode 1 (Smart Q&A & Document Chat):**
     - Queries local vector memory first.
     - If data confidence is below threshold (`score < 0.35`), it quietly fetches real-time web facts, responds in chat, and saves the new knowledge to vector memory.
     - **Zero clutter:** No unwanted PDF reports are generated during regular chat.
   - **Mode 2 (Deep Autonomous Topic Research):**
     - Triggered explicitly via `/research <topic>` in CLI or via Tab 2 in Streamlit.
     - Executes a structured **5-Pillar Academic Study** (Foundations, Taxonomy, Methodologies, Applications, Limitations).
     - Crawls multiple authoritative web sources, writes an executive synthesis, and **saves a formal PDF report in `output/`**.
     - **Instant In-Line Chat:** Automatically indexes that fresh report into active memory so you can immediately ask follow-up questions!

2. **Decoupled Domain Data Models:**
   - Strictly typed Python `@dataclass` domain entities in `core/models.py` (`ResearchPillar`, `SearchResult`, `VectorMatch`, `DocumentChunk`, `RouterDecision`, `DeepResearchResult`, `ChatMessage`, `ChatSessionState`).
   - Bidirectional serialization (`to_dict()`, `from_dict()`) with strict attribute access.

3. **Secure Vector Storage (.npz & .json):**
   - Eliminated Python `pickle` deserialization vulnerabilities entirely.
   - Dual-file atomic persistence (`.npz` for numpy embeddings via `allow_pickle=False`, `.chunks.json` for document metadata).
   - Atomic temporary-file swapping prevents cache corruption during interrupted sessions.

4. **100-Batch Embedding Pacing & Adaptive 429 Resilience:**
   - Slices large books and documents (1000+ pages) into strict 100-chunk batches.
   - Paces requests to respect Google Gemini Free Tier limits with dynamic pacing.
   - **Regex-based Adaptive Interceptor:** Intercepts `429 RESOURCE_EXHAUSTED`, auto-waits with exponential backoff, and resumes without crashing.

5. **Dual-LLM Gateway with Native HTTP Retry Resilience:**
   - **Primary:** OpenRouter Free Models (`nvidia/nemotron-3.5-lightning:free` & `liquid/lfm-3-1.7b:free`).
   - **Backup:** Seamlessly cascades to Google Gemini 2.5 Flash if OpenRouter daily limits are reached.
   - Built-in native SDK `HttpRetryOptions` with exponential backoff and randomized anti-scraping jitter.

---

## 🏗️ Repository Architecture

```text
SmartResearch Chatbot/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated Ruff linting & test suite execution on push/PR
├── core/
│   ├── __init__.py              # Core package initialization & exports
│   ├── models.py                # Typed domain dataclasses with serialization
│   ├── router.py                # Query intent classification & threshold routing
│   ├── agent_router.py          # Backward-compatibility facade
│   ├── vector_engine.py         # Caching, SHA-256 fingerprinting, cosine similarity
│   ├── llm_gateway.py           # Gemini & OpenRouter clients with retry backoff
│   ├── web_researcher.py        # 5-Pillar concurrent search dispatch & sanitization
│   └── report_builder.py        # Academic ReportLab PDF compiler with Unicode sanitization
├── tests/                       # Fully isolated, automated unit test suite (62 tests)
│   ├── __init__.py              # Sys.path test root resolution
│   ├── test_models.py           # Domain model serialization & attribute tests
│   ├── test_router.py           # Decision boundaries & fallback tests
│   ├── test_vector_engine.py    # Chunking, safe .npz persistence & similarity tests
│   ├── test_llm_gateway.py      # LLM response extraction, retry & client tests
│   ├── test_web_researcher.py   # Web scraper concurrency & validation tests
│   ├── test_report_builder.py   # PDF text sanitization & symbol tests
│   └── test_ui_contract.py      # Streamlit UI & CLI end-to-end integration contracts
├── app.py                       # Modern reactive Streamlit web interface
├── cli.py                       # Interactive terminal CLI with /research commands
├── config.py                    # Central configuration, env parsing, and thresholds
├── pyproject.toml               # PEP 621 packaging metadata & Ruff configuration
├── requirements.txt             # Pinned runtime dependencies
├── Run_SmartResearch.bat        # Windows desktop launcher batch script
├── .env.example                 # Example environment variables template
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone and Setup Environment
```bash
git clone https://github.com/mr-sagarajput/SmartResearch-Agent.git
cd "SmartResearch-Agent"

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Package in Editable Mode
```bash
# Install runtime dependencies and the package
pip install -e .

# Or with dev dependencies:
pip install -e ".[dev]"
```

### 4. Configure API Keys
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## 💻 Running the Application

### Modern Web UI (Streamlit)
```bash
# Direct Streamlit:
streamlit run app.py

# Or on Windows via the launcher shortcut:
Run_SmartResearch.bat
```
- **Tab 1:** Conversational Document & Smart Q&A with live web fallback.
- **Tab 2:** Autonomous Deep Research with 1-Click PDF Report Download and in-line chat.

### Interactive Terminal CLI
```bash
# Via installed entrypoint:
smartresearch

# Or directly with Python:
python cli.py

# Optional: Load a document directly:
python cli.py "path/to/document.pdf"
```

#### In-Terminal Commands:
- `/research <topic>` — Conducts 5-pillar deep web research, generates a PDF in `output/`, and indexes it for live chat.
- `/sources` — View citations & links from the previous answer.
- `/clear` — Clear current chat session memory.
- `/help` — View available commands.
- `/quit` — Exit the assistant.

---

## 🧪 Testing & Code Quality

### Running Unit Tests
All 62 test cases run in `< 0.8s` with zero external network dependencies:
```bash
python -m unittest discover -s tests -t .
```

### Running Linter (Ruff)
```bash
ruff check .
```

---

## 🛡️ License

Copyright (c) 2026 Sagar Rajput. All Rights Reserved. Proprietary software.
