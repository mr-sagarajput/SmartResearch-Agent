"""
Interactive Terminal CLI for SmartResearch Chatbot.
===================================================
Supports Dual-Intent Modes:
1. Regular Q&A:
   Ask any question or load a PDF. Missing info is fetched from web quietly (No PDF saved).
2. Deep Autonomous Topic Research:
   Type:  /research <Topic Name>
   - Crawls web, synthesizes facts, and generates a formal PDF report in output/
   - Immediately loads that report into conversational memory for instant Q&A!
"""

from pathlib import Path
import sys
import textwrap

from config import GEMINI_API_KEY, OPENROUTER_API_KEY
from core.llm_gateway import get_gemini_client
from core.router import execute_deep_research, process_query_with_router
from core.vector_engine import (
    build_document_chunks,
    compute_bytes_hash,
    embed_chunks,
    load_cached_embeddings,
    load_pdf_pages_from_bytes,
    save_cached_embeddings,
)

BANNER = r"""
  ____                      _   ____                               _     
 / ___| _ __ ___   __ _ _ _| |_|  _ \ ___  ___  ___  __ _ _ __ ___| |__  
 \___ \| '_ ` _ \ / _` | '__| __| |_) / _ \/ __|/ _ \/ _` | '__/ __| '_ \ 
  ___) | | | | | | (_| | |  | |_|  _ <  __/\__ \  __/ (_| | | | (__| | | |
 |____/|_| |_| |_|\__,_|_|   \__|_| \_\___||___/\___|\__,_|_|  \___|_| |_|
                                                                          
       Autonomous Agentic RAG: Vector Memory + Live Web Research
"""

HELP_TEXT = """
Available Commands:
  /research <topic> - Run full Autonomous Research & save PDF report to output/
  /help             - Show this help message
  /clear            - Reset current conversation history
  /sources          - View source links for the previous answer
  /quit             - Exit the application (or Ctrl+C)
"""


def print_wrapped(text: str, width: int = 90):
    for paragraph in text.split("\n"):
        if paragraph.strip():
            print(textwrap.fill(paragraph, width=width))
        else:
            print()


def main():
    print(BANNER)

    if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
        print("[ERROR] Please provide at least one API key in .env file.")
        sys.exit(1)

    gemini_client = get_gemini_client()
    chunks = []
    chunk_embeddings = None

    # Optional PDF Document argument check
    pdf_path = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            p = Path(arg)
            if p.exists() and p.is_file() and p.suffix.lower() == ".pdf":
                pdf_path = p
                break

    if pdf_path:
        print(f"📄 Loading Document: {pdf_path.name}")
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        file_hash = compute_bytes_hash(file_bytes)
        cached_data = load_cached_embeddings(file_hash)

        if cached_data is not None:
            chunks, chunk_embeddings = cached_data
            print(f"⚡ Loaded {len(chunks)} precomputed chunks from disk cache (0 API calls)!")
        else:
            print("📖 Reading and chunking document...")
            pages = load_pdf_pages_from_bytes(file_bytes, gemini_client)
            chunks = build_document_chunks(pages)
            print(f"✂️ Created {len(chunks)} semantic chunks. Embedding with Rate-Pacing...")
            chunk_embeddings = embed_chunks(gemini_client, chunks, progress_callback=lambda m: print(f"  ↳ {m}"))
            save_cached_embeddings(file_hash, chunks, chunk_embeddings)
            print("💾 Saved to persistent cache.")
    else:
        print("💡 Tip: Running in Full Open Mode! Ask anything, or type: /research <topic>")

    print("\n" + "─" * 65)
    print("✅ Ready! Type your question, or type: /research <topic>")
    print("─" * 65 + "\n")

    history = []
    last_sources = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("👋 Goodbye!")
            break

        if user_input.lower() == "/clear":
            history.clear()
            print("🗑️ Conversation history cleared.\n")
            continue

        if user_input.lower() == "/help":
            print(HELP_TEXT)
            continue

        if user_input.lower() == "/sources":
            if not last_sources:
                print("No recent sources to display.\n")
            else:
                print("\n📚 Sources retrieved for last answer:")
                for item, score in last_sources:
                    url_str = f" ({item.get('url')})" if item.get("url") else ""
                    print(f"  • {item['source']}{url_str}")
                    print(f"    Excerpt: \"{item['text'][:120]}...\"\n")
            continue

        def cli_status(msg):
            print(f"\n  ↳ {msg}", end="", flush=True)

        # -------------------------------------------------------------
        # Intent Check: Mode 2 (/research <topic>)
        # -------------------------------------------------------------
        if user_input.lower().startswith("/research"):
            research_topic = user_input[9:].strip()
            if not research_topic:
                print("\n⚠️ Please specify a topic: /research <topic name>\n")
                continue

            print(f"\n🔬 Starting Deep Autonomous Research on: '{research_topic}'...")
            res = execute_deep_research(
                topic=research_topic,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                max_pages=4,
                status_callback=cli_status,
            )

            if res.get("success"):
                if res.get("updated_embeddings") is not None:
                    chunk_embeddings = res["updated_embeddings"]

                print("\n" + "=" * 65)
                print(f"📄 Research Report Generated & Saved to:\n   {res['pdf_path']}")
                print("=" * 65 + "\n")
                print_wrapped(res["summary"])
                print("\n" + "─" * 65)
                print("💬 [Research Indexed into Memory! You can now chat directly about this report]")
                print("─" * 65 + "\n")

                history.append({"role": "user", "content": f"Research topic: {research_topic}"})
                history.append({"role": "assistant", "content": res["summary"]})
            else:
                print(f"\n❌ Research failed: {res.get('error')}\n")
            continue

        # -------------------------------------------------------------
        # Mode 1: Regular Q&A / Document Search (No PDF clutter)
        # -------------------------------------------------------------
        print("\n🤖 SmartResearch: ", end="", flush=True)
        try:
            result = process_query_with_router(
                user_query=user_input,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                history=history,
                status_callback=cli_status,
            )

            if "updated_embeddings" in result and result["updated_embeddings"] is not None:
                chunk_embeddings = result["updated_embeddings"]

            last_sources = result["sources"]

            print("\n")
            print_wrapped(result["answer"])

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result["answer"]})

            if result["is_web_research"]:
                print(f"\n🌐 [Web Fallback Active | Verified by {result['provider']}]")
            else:
                print(f"\n⚡ [Vector Hit | Confidence: {result['max_score']:.1%}]")
            print()

        except Exception as exc:
            print(f"\n[Error]: {exc}\n")


if __name__ == "__main__":
    main()
