"""
Autonomous Agent Router (The Brain).
====================================
Implements Dual-Mode Architecture with 5-Pillar Academic Synthesis:
1. Mode 1 (Document Q&A / Normal Chat):
   - Vector Search first.
   - If missing -> silent web fallback + cache into vector memory.
   - ZERO PDF generated (keeps storage clean).
2. Mode 2 (Deep Autonomous Topic Research):
   - Explicit 5-Pillar research (Definition, Branches, Principles, Applications, Limitations).
   - Generates & saves a formal branded PDF in output/ directory.
   - IMMEDIATELY indexes the report into Vector Memory for instant conversational Q&A!
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from core.config import MAX_RESEARCH_TOKENS, SIMILARITY_THRESHOLD, logger
from core.llm_gateway import execute_prompt, get_gemini_client
from core.models import DeepResearchResult, DocumentChunk, RouterDecision
from core.report_builder import generate_research_pdf_report
from core.vector_engine import (
    embed_chunks,
    retrieve,
)
from core.web_researcher import perform_5_pillar_research, perform_live_research


def process_query_with_router(
    user_query: str,
    chunks: List[DocumentChunk],
    chunk_embeddings: Optional[np.ndarray],
    history: Optional[List[Dict[str, str]]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> RouterDecision:
    """
    Evaluates incoming queries against active vector store confidence threshold.
    Routes to local vector retrieval or triggers live web research fallback.
    """
    history = history or []
    gemini_client = get_gemini_client()

    max_score = 0.0
    retrieved_chunks = []
    scores = []

    if chunks and chunk_embeddings is not None and len(chunks) > 0 and gemini_client is not None:
        if status_callback:
            status_callback("🔍 Checking local vector knowledge base...")
        retrieved_chunks, scores, max_score = retrieve(
            gemini_client=gemini_client,
            query=user_query,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
        )

    # High Confidence Vector Match
    if max_score >= SIMILARITY_THRESHOLD and len(retrieved_chunks) > 0:
        if status_callback:
            status_callback(f"⚡ Found verified answer in local knowledge (Confidence: {max_score:.1%})!")

        context_snippets = []
        for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores), start=1):
            context_snippets.append(f"[Source {i} | {chunk.source}]\n{chunk.text}")
        context_block = "\n\n---\n\n".join(context_snippets)

        prompt = (
            f"### Document Context\n\n{context_block}\n\n"
            f"### User Question\n{user_query}"
        )

        answer_text, provider = execute_prompt(prompt, history=history)

        sources_data = [
            ({"source": c.source, "page": c.page, "text": c.text, "url": c.url}, s)
            for c, s in zip(retrieved_chunks, scores)
        ]

        return RouterDecision(
            answer=answer_text,
            decision="VECTOR_HIT",
            provider=provider,
            max_score=max_score,
            sources=sources_data,
            is_web_research=False,
            updated_chunks=chunks,
        )

    # Information missing -> Silent web fallback
    if status_callback:
        status_callback("🌐 Question not in loaded document. Searching live web...")

    web_results = perform_live_research(
        topic=user_query,
        max_pages=3,
        status_callback=status_callback,
    )

    if not web_results:
        return RouterDecision(
            answer="I could not find relevant information in the local document, and live web research returned no accessible results for this query.",
            decision="NOT_FOUND",
            provider="None",
            max_score=max_score,
            sources=[],
            is_web_research=True,
            updated_chunks=chunks,
        )

    web_snippets = []
    new_chunks = []
    chunk_start_id = len(chunks) + 1

    for idx, item in enumerate(web_results, 1):
        title = item.title if hasattr(item, "title") else item.get("title", "")
        url = item.url if hasattr(item, "url") else item.get("url", "")
        notes = item.notes if hasattr(item, "notes") else item.get("notes", "")
        full_snippet = item.full_snippet if hasattr(item, "full_snippet") else item.get("full_snippet", "")
        web_snippets.append(f"[Web Source {idx}: {title}] ({url})\n{notes}\n{full_snippet}")
        new_chunks.append(
            DocumentChunk(
                chunk_id=chunk_start_id + idx,
                source=f"Web: {title[:40]}",
                page=0,
                text=f"{title}\n{notes}\n{full_snippet}",
                url=url,
            )
        )

    web_context_block = "\n\n---\n\n".join(web_snippets)

    synthesis_prompt = (
        f"### Live Web Research Findings\n\n{web_context_block}\n\n"
        f"### User Question\n{user_query}\n\n"
        "Provide a comprehensive, factual response. Synthesize the findings and explicitly cite the source URLs."
    )

    answer_text, provider = execute_prompt(synthesis_prompt, history=history)
    full_answer = (
        "*(Note: Information was not in the loaded document; gathered live from the web)*\n\n"
        + answer_text
    )

    # Append to vector memory
    if gemini_client and new_chunks:
        try:
            if status_callback:
                status_callback("💾 Learning and saving new web research into vector memory...")
            new_embeddings = embed_chunks(gemini_client, new_chunks)
            if chunk_embeddings is not None and len(chunk_embeddings) > 0:
                updated_embeddings = np.vstack([chunk_embeddings, new_embeddings])
            else:
                updated_embeddings = new_embeddings
            updated_chunks = list(chunks) + new_chunks
        except Exception as e:
            logger.warning(f"Vector Learning failed to append new chunks: {e}")
            updated_embeddings = chunk_embeddings
            updated_chunks = list(chunks)
    else:
        updated_embeddings = chunk_embeddings
        updated_chunks = list(chunks)

    sources_data = [
        ({
            "source": item.title if hasattr(item, "title") else item.get("title", ""),
            "page": 0,
            "text": item.notes if hasattr(item, "notes") else item.get("notes", ""),
            "url": item.url if hasattr(item, "url") else item.get("url", ""),
        }, 1.0)
        for item in web_results
    ]

    return RouterDecision(
        answer=full_answer,
        decision="WEB_RESEARCH_TRIGGERED",
        provider=provider,
        max_score=1.0,
        sources=sources_data,
        updated_embeddings=updated_embeddings,
        updated_chunks=updated_chunks,
        is_web_research=True,
    )


def execute_deep_research(
    topic: str,
    chunks: List[DocumentChunk],
    chunk_embeddings: Optional[np.ndarray],
    max_pages: int = 5,
    status_callback: Optional[Callable[[str], None]] = None,
) -> DeepResearchResult:
    """
    Mode 2: Explicit 5-Pillar Autonomous Research.
    Generates PDF & Indexes immediately for live chat.
    """
    gemini_client = get_gemini_client()

    if status_callback:
        status_callback(f"🚀 Initiating 5-Pillar Academic Research on: '{topic}'...")

    pillar_data = perform_5_pillar_research(topic=topic, status_callback=status_callback)

    if not pillar_data:
        return DeepResearchResult(
            success=False,
            topic=topic,
            error=f"Could not gather sufficient verified research data for '{topic}'.",
        )

    evidence_blocks = []
    for item in pillar_data:
        pillar_name = item.pillar if hasattr(item, "pillar") else item.get("pillar", "")
        title_str = item.title if hasattr(item, "title") else item.get("title", "")
        url_str = item.url if hasattr(item, "url") else item.get("url", "")
        notes_str = item.notes if hasattr(item, "notes") else item.get("notes", "")
        snippet_str = item.full_snippet if hasattr(item, "full_snippet") else item.get("full_snippet", "")
        evidence_blocks.append(
            f"### Pillar: {pillar_name}\n"
            f"Source: {title_str} ({url_str})\n"
            f"Key Extracted Points:\n{notes_str}\n"
            f"Context Excerpt:\n{snippet_str}\n"
        )
    combined_evidence = "\n---\n".join(evidence_blocks)

    if status_callback:
        status_callback("✍️ Synthesizing comprehensive 650-word report across all 5 Pillars...")

    academic_synthesis_prompt = f"""\
You are an authoritative senior academic researcher.
Topic: {topic}

Write the final research report text directly. Do not include your internal reasoning, scratchpad, or thoughts.
Format the report EXACTLY under these 5 numbered headings:

1. Introduction & Foundational Concepts
Explain core definition, what this field/topic studies, and its formal scope.

2. Core Sub-Fields & Taxonomy (Branches)
Detail the primary branches, subfields, classifications, and components.

3. Fundamental Principles & Methodologies
Detail the underlying theories, core mechanisms, frameworks, and scientific principles.

4. Real-World Applications & Modern Significance
Detail practical industry use cases, societal/technological impact, and modern relevance.

5. Critical Limitations & Future Outlook
Detail current challenges, unresolved questions, ethical concerns, and future developments.

TARGET TOTAL LENGTH: ~650 to 750 words.
Base all claims strictly on this verified evidence:
{combined_evidence}
"""

    report_text, provider = execute_prompt(
        prompt=academic_synthesis_prompt,
        max_tokens=MAX_RESEARCH_TOKENS,
    )

    if not report_text or len(report_text.strip()) < 50:
        report_text = f"Research synthesis for {topic} compiled across {len(pillar_data)} verified academic sources."

    # Build and save PDF Report
    if status_callback:
        status_callback("📄 Compiling branded PDF report in output/ directory...")

    pdf_file_path = generate_research_pdf_report(
        topic=topic,
        summary_text=report_text,
        sources_data=pillar_data,
    )

    # Index into Active Vector Session
    if status_callback:
        status_callback("🧠 Indexing report into Vector Memory for instant conversational Q&A...")

    report_full_text = f"Title: Comprehensive Research Report on {topic}\n\n{report_text}\n\n"
    for idx, s in enumerate(pillar_data, 1):
        p_name = s.pillar if hasattr(s, "pillar") else s.get("pillar", "")
        s_title = s.title if hasattr(s, "title") else s.get("title", "")
        s_url = s.url if hasattr(s, "url") else s.get("url", "")
        s_notes = s.notes if hasattr(s, "notes") else s.get("notes", "")
        report_full_text += f"\nPillar Reference {idx}: {p_name} - {s_title} ({s_url})\n{s_notes}\n"

    report_chunks_text = [p.strip() for p in report_full_text.split("\n\n") if len(p.strip()) > 30]

    new_chunks = []
    chunk_start_id = len(chunks) + 1
    for idx, text_block in enumerate(report_chunks_text):
        new_chunks.append(
            DocumentChunk(
                chunk_id=chunk_start_id + idx,
                source=f"Research Report: {topic[:30]}",
                page=1,
                text=text_block,
                url=str(pdf_file_path),
            )
        )

    if gemini_client and new_chunks:
        new_embeddings = embed_chunks(gemini_client, new_chunks)
        if chunk_embeddings is not None and len(chunk_embeddings) > 0:
            updated_embeddings = np.vstack([chunk_embeddings, new_embeddings])
        else:
            updated_embeddings = new_embeddings
        updated_chunks = list(chunks) + new_chunks
    else:
        updated_embeddings = chunk_embeddings
        updated_chunks = list(chunks)

    return DeepResearchResult(
        success=True,
        topic=topic,
        summary=report_text,
        pdf_path=str(pdf_file_path),
        sources=pillar_data,
        updated_embeddings=updated_embeddings,
        updated_chunks=updated_chunks,
        provider=provider,
    )
