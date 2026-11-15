# 🧠 SmartResearch Chatbot (Agentic-RAG Architecture)

A production-grade, enterprise autonomous research system combining **Persistent Vector Memory** with **Live Real-Time Web Intelligence**.

---

## 🌟 Key Architecture Innovations

1. **Dual-Intent Mode Engine:**
   - **Mode 1 (Smart Q&A & Document Chat):**
     - Queries local vector memory first.
     - If data is missing (`score < 0.35`), it quietly fetches real-time web facts, responds in chat, and saves the new knowledge to memory.
     - **Zero clutter:** No unwanted PDF reports are generated during regular chat.
   - **Mode 2 (Deep Autonomous Topic Research):**
     - Triggered explicitly via `/research <topic>` in CLI or via Tab 2 in Streamlit.
     - Crawls multiple live web pages, writes an executive synthesis, and **saves a formal PDF report in `output/`**.
     - **Instant Chat:** Automatically indexes that fresh report into active memory so you can immediately ask follow-up questions!

2. **100-Batch Embedding Pacing & Adaptive 429 Resilience:**
   - Slices large books/documents (1000+ pages) into strict 100-chunk batches.
   - Paces requests to respect Google Gemini Free Tier limits.
   - **Regex-based Adaptive Interceptor:** Intercepts `429 RESOURCE_EXHAUSTED`, auto-waits 60s, and resumes without ever crashing.

3. **Dual-LLM Gateway (Infinite Uptime):**
   - **Primary:** OpenRouter Free Models (`nvidia/nemotron-3.5-lightning:free` & `liquid/lfm-3-1.7b:free`).
   - **Backup:** Seamlessly cascades to Google Gemini 2.5 Flash if OpenRouter daily limits are reached.

---

## 🚀 How to Run

### 1. Modern Web UI (Streamlit)
Double-click the **`SmartResearch Chatbot`** shortcut on your Desktop, or run:
```powershell
cd "d:\Projects\SmartResearch Chatbot"
streamlit run app.py
```
* **Tab 1:** Conversational Document & Smart Q&A.
* **Tab 2:** Autonomous Deep Research with 1-Click PDF Report Download.

### 2. Interactive Terminal CLI
```powershell
cd "d:\Projects\SmartResearch Chatbot"

# Full Open Mode (Web & Q&A):
python cli.py

# Document-Grounded Mode (PDF + Auto Web Fallback):
python cli.py "path/to/document.pdf"
```

#### In-Terminal Commands:
- `/research <topic>` — Conducts deep web research, generates a PDF in `output/`, and lets you chat with it.
- `/sources` — View citations & links from the previous answer.
- `/clear` — Clear current chat session memory.
- `/help` — View available commands.
- `/quit` — Exit the assistant.
