"""
SmartResearch Chatbot Core Package.
===================================
Provides modular, decoupled components for vector RAG, LLM communication,
5-pillar web research, report generation, and autonomous query routing.
"""

from core.models import (
    ChatMessage,
    ChatSessionState,
    DeepResearchResult,
    DocumentChunk,
    ResearchPillar,
    RouterDecision,
    SearchResult,
    VectorMatch,
)

__all__ = [
    "ChatMessage",
    "ChatSessionState",
    "DeepResearchResult",
    "DocumentChunk",
    "ResearchPillar",
    "RouterDecision",
    "SearchResult",
    "VectorMatch",
]
