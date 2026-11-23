"""
Unit tests for core/models.py domain models.
Validates serialization, deserialization, attribute typing, and dictionary backward-compatibility for:
- ResearchPillar
- SearchResult
- DocumentChunk
- VectorMatch
- RouterDecision
- DeepResearchResult
- ChatMessage
- ChatSessionState
"""

from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


class TestModelsSerialization(unittest.TestCase):
    """Test suite for core domain dataclasses."""

    def test_research_pillar_to_and_from_dict(self):
        pillar = ResearchPillar(
            pillar="1. Introduction & Foundational Concepts",
            query="quantum computing overview",
            title="Quantum Basics",
            url="https://example.com/quantum",
            notes="Qubits are foundational.",
            full_snippet="Full text excerpt about qubits.",
        )
        d = pillar.to_dict()
        self.assertEqual(d["pillar"], "1. Introduction & Foundational Concepts")
        self.assertEqual(d["query"], "quantum computing overview")
        self.assertEqual(d["notes"], "Qubits are foundational.")

        # Reconstruct from dict
        reconstructed = ResearchPillar.from_dict(d)
        self.assertEqual(reconstructed.pillar, pillar.pillar)
        self.assertEqual(reconstructed.url, pillar.url)

        # Dict subscript access & in operator
        self.assertEqual(pillar["pillar"], "1. Introduction & Foundational Concepts")
        self.assertTrue("pillar" in pillar)
        self.assertTrue("query" in pillar)
        self.assertEqual(pillar.get("nonexistent", "default"), "default")
        pillar["notes"] = "Updated note"
        self.assertEqual(pillar.notes, "Updated note")

    def test_search_result_to_and_from_dict(self):
        res = SearchResult(
            title="MIT Tech Review",
            url="https://technologyreview.com/article",
            notes="Emerging technology trends.",
            full_snippet="Full snippet content.",
        )
        d = res.to_dict()
        self.assertEqual(d["title"], "MIT Tech Review")
        self.assertEqual(d["url"], "https://technologyreview.com/article")

        reconstructed = SearchResult.from_dict(d)
        self.assertEqual(reconstructed.title, res.title)
        self.assertEqual(reconstructed.notes, res.notes)

        # Dict subscript access
        self.assertEqual(res["title"], "MIT Tech Review")
        self.assertTrue("url" in res)
        self.assertEqual(res.get("notes"), "Emerging technology trends.")
        res["notes"] = "Changed"
        self.assertEqual(res.notes, "Changed")

    def test_document_chunk_to_and_from_dict(self):
        chunk = DocumentChunk(
            chunk_id=42,
            source="Page 7",
            page=7,
            text="Distributed consensus requires Paxos or Raft.",
            url="https://example.com/consensus.pdf",
        )
        d = chunk.to_dict()
        self.assertEqual(d["chunk_id"], 42)
        self.assertEqual(d["page"], 7)
        self.assertEqual(d["source"], "Page 7")

        reconstructed = DocumentChunk.from_dict(d)
        self.assertEqual(reconstructed.chunk_id, 42)
        self.assertEqual(reconstructed.text, chunk.text)

        # Dict subscript access
        self.assertEqual(chunk["chunk_id"], 42)
        self.assertTrue("text" in chunk)
        chunk["page"] = 8
        self.assertEqual(chunk.page, 8)

    def test_vector_match_to_and_from_dict(self):
        chunk = DocumentChunk(
            chunk_id=1,
            source="Page 1",
            page=1,
            text="Vector search uses cosine similarity.",
        )
        match = VectorMatch(chunk=chunk, score=0.92)
        d = match.to_dict()
        self.assertEqual(d["score"], 0.92)
        self.assertEqual(d["chunk"]["chunk_id"], 1)

        # Reconstruct from dict
        reconstructed = VectorMatch.from_dict(d)
        self.assertEqual(reconstructed.score, 0.92)
        self.assertEqual(reconstructed.chunk.chunk_id, 1)

        # Dict subscript access
        self.assertEqual(match["score"], 0.92)
        self.assertTrue("chunk" in match)
        self.assertEqual(match.get("score"), 0.92)

    def test_router_decision_to_and_from_dict(self):
        decision = RouterDecision(
            decision="VECTOR_HIT",
            answer="Here is the verified answer.",
            sources=[{"source": "Page 1", "page": 1, "text": "...", "url": ""}],
            max_score=0.88,
            is_web_research=False,
            provider="Google Gemini",
        )
        d = decision.to_dict()
        self.assertEqual(d["decision"], "VECTOR_HIT")
        self.assertEqual(d["max_score"], 0.88)
        self.assertFalse(d["is_web_research"])

        reconstructed = RouterDecision.from_dict(d)
        self.assertEqual(reconstructed.decision, "VECTOR_HIT")
        self.assertEqual(reconstructed.answer, decision.answer)
        self.assertEqual(reconstructed.max_score, 0.88)

        # Dict subscript access & methods
        self.assertEqual(decision["decision"], "VECTOR_HIT")
        self.assertTrue("answer" in decision)
        self.assertEqual(decision.get("provider"), "Google Gemini")
        decision["is_web_research"] = True
        self.assertTrue(decision.is_web_research)

    def test_deep_research_result_to_and_from_dict(self):
        result = DeepResearchResult(
            success=True,
            topic="Artificial Intelligence",
            summary="Comprehensive report summary across all 5 pillars.",
            pdf_path="/path/to/report.pdf",
            sources=[{"pillar": "1. Intro", "title": "AI Basics"}],
            provider="OpenRouter",
        )
        d = result.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["topic"], "Artificial Intelligence")
        self.assertEqual(d["pdf_path"], "/path/to/report.pdf")

        reconstructed = DeepResearchResult.from_dict(d)
        self.assertTrue(reconstructed.success)
        self.assertEqual(reconstructed.summary, result.summary)

        # Dict subscript access & methods
        self.assertEqual(result["topic"], "Artificial Intelligence")
        self.assertTrue("pdf_path" in result)
        self.assertEqual(result.get("pdf_path"), "/path/to/report.pdf")
        result["success"] = False
        self.assertFalse(result.success)

    def test_chat_message_to_and_from_dict(self):
        msg = ChatMessage(
            role="user",
            content="Explain transformers.",
            provider="Gemini",
        )
        d = msg.to_dict()
        self.assertEqual(d["role"], "user")
        self.assertEqual(d["content"], "Explain transformers.")
        self.assertEqual(d["provider"], "Gemini")

        reconstructed = ChatMessage.from_dict(d)
        self.assertEqual(reconstructed.role, "user")
        self.assertEqual(reconstructed.content, "Explain transformers.")

        # Dict subscript access
        self.assertEqual(msg["role"], "user")
        self.assertTrue("content" in msg)
        self.assertEqual(msg.get("role"), "user")

    def test_chat_session_state_to_and_from_dict(self):
        state = ChatSessionState(
            messages=[{"role": "user", "content": "Hello"}],
            research_messages=[],
            uploaded_pdf_name="sample.pdf",
            file_hash="abcdef123456",
            chunks_count=10,
            active_topic="Robotics",
            last_pdf_path="/output/report.pdf",
        )
        d = state.to_dict()
        self.assertEqual(d["uploaded_pdf_name"], "sample.pdf")
        self.assertEqual(d["chunks_count"], 10)
        self.assertEqual(d["active_topic"], "Robotics")

        reconstructed = ChatSessionState.from_dict(d)
        self.assertEqual(reconstructed.uploaded_pdf_name, "sample.pdf")
        self.assertEqual(reconstructed.chunks_count, 10)
        self.assertEqual(len(reconstructed.messages), 1)

        # Dict subscript access
        self.assertEqual(state["uploaded_pdf_name"], "sample.pdf")
        self.assertTrue("file_hash" in state)
        self.assertEqual(state.get("active_topic"), "Robotics")
        state["chunks_count"] = 25
        self.assertEqual(state.chunks_count, 25)


if __name__ == "__main__":
    unittest.main()
