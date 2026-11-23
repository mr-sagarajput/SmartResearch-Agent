"""
Unit tests for core/agent_router.py.
Validates router decision boundaries:
- High confidence vector match (score >= SIMILARITY_THRESHOLD -> VECTOR_HIT)
- Exact boundary condition (score == SIMILARITY_THRESHOLD -> VECTOR_HIT)
- Missing information triggering live web research (score < SIMILARITY_THRESHOLD -> WEB_RESEARCH_TRIGGERED)
- Web research returning empty results (-> NOT_FOUND)
- Deep 5-Pillar autonomous research execution
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SIMILARITY_THRESHOLD
from core.router import execute_deep_research, process_query_with_router
from core.vector_engine import DocumentChunk


class TestAgentRouterDecisionBoundaries(unittest.TestCase):
    """Test suite for agent router decision logic and fallback mechanisms."""

    def setUp(self):
        self.sample_chunk = DocumentChunk(
            chunk_id=1,
            source="Page 1",
            page=1,
            text="Quantum computing uses qubits which can exist in superposition.",
        )
        self.sample_embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    @patch("core.router.get_gemini_client")
    @patch("core.router.retrieve")
    @patch("core.router.execute_prompt")
    @patch("core.router.perform_live_research")
    def test_high_confidence_vector_match(
        self, mock_web_research, mock_execute_prompt, mock_retrieve, mock_gemini
    ):
        """When vector similarity is high (>= threshold), router must return VECTOR_HIT without web search."""
        mock_gemini.return_value = MagicMock()
        high_score = 0.85
        self.assertGreater(high_score, SIMILARITY_THRESHOLD)

        mock_retrieve.return_value = ([self.sample_chunk], [high_score], high_score)
        mock_execute_prompt.return_value = ("Superposition allows massive parallelism.", "Google Gemini")

        result = process_query_with_router(
            user_query="How do qubits work?",
            chunks=[self.sample_chunk],
            chunk_embeddings=self.sample_embeddings,
        )

        self.assertEqual(result["decision"], "VECTOR_HIT")
        self.assertFalse(result["is_web_research"])
        self.assertEqual(result["max_score"], high_score)
        self.assertEqual(result["answer"], "Superposition allows massive parallelism.")
        self.assertEqual(len(result["sources"]), 1)
        mock_web_research.assert_not_called()

    @patch("core.router.get_gemini_client")
    @patch("core.router.retrieve")
    @patch("core.router.execute_prompt")
    @patch("core.router.perform_live_research")
    def test_exact_threshold_vector_match(
        self, mock_web_research, mock_execute_prompt, mock_retrieve, mock_gemini
    ):
        """When vector similarity is exactly SIMILARITY_THRESHOLD, router should return VECTOR_HIT."""
        mock_gemini.return_value = MagicMock()
        exact_score = SIMILARITY_THRESHOLD

        mock_retrieve.return_value = ([self.sample_chunk], [exact_score], exact_score)
        mock_execute_prompt.return_value = ("Boundary match answer.", "OpenRouter")

        result = process_query_with_router(
            user_query="Boundary query",
            chunks=[self.sample_chunk],
            chunk_embeddings=self.sample_embeddings,
        )

        self.assertEqual(result["decision"], "VECTOR_HIT")
        self.assertFalse(result["is_web_research"])
        self.assertEqual(result["max_score"], exact_score)
        mock_web_research.assert_not_called()

    @patch("core.router.get_gemini_client")
    @patch("core.router.retrieve")
    @patch("core.router.execute_prompt")
    @patch("core.router.perform_live_research")
    @patch("core.router.embed_chunks")
    def test_below_threshold_triggers_web_research(
        self, mock_embed_chunks, mock_web_research, mock_execute_prompt, mock_retrieve, mock_gemini
    ):
        """When vector similarity is below threshold, router must trigger web research fallback."""
        mock_gemini.return_value = MagicMock()
        low_score = SIMILARITY_THRESHOLD - 0.05
        self.assertLess(low_score, SIMILARITY_THRESHOLD)

        mock_retrieve.return_value = ([self.sample_chunk], [low_score], low_score)
        mock_web_research.return_value = [
            {
                "title": "Quantum Supremacy Overview",
                "url": "https://example.com/quantum",
                "notes": "Detailed notes on quantum computing milestones.",
                "full_snippet": "Quantum supremacy was demonstrated using superconducting qubits.",
            }
        ]
        mock_execute_prompt.return_value = ("Synthesized answer from live web.", "OpenRouter")
        mock_embed_chunks.return_value = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)

        result = process_query_with_router(
            user_query="What was the latest quantum benchmark?",
            chunks=[self.sample_chunk],
            chunk_embeddings=self.sample_embeddings,
        )

        self.assertEqual(result["decision"], "WEB_RESEARCH_TRIGGERED")
        self.assertTrue(result["is_web_research"])
        self.assertIn("Synthesized answer from live web.", result["answer"])
        mock_web_research.assert_called_once()

    @patch("core.router.get_gemini_client")
    @patch("core.router.retrieve")
    @patch("core.router.perform_live_research")
    def test_missing_info_web_research_empty_returns_not_found(
        self, mock_web_research, mock_retrieve, mock_gemini
    ):
        """When vector match is below threshold and web research returns no results -> NOT_FOUND."""
        mock_gemini.return_value = MagicMock()
        mock_retrieve.return_value = ([], [], 0.10)
        mock_web_research.return_value = []

        result = process_query_with_router(
            user_query="Obscure unpublished query",
            chunks=[self.sample_chunk],
            chunk_embeddings=self.sample_embeddings,
        )

        self.assertEqual(result["decision"], "NOT_FOUND")
        self.assertTrue(result["is_web_research"])
        self.assertIn("could not find relevant information", result["answer"])
        self.assertEqual(result["sources"], [])

    @patch("core.router.get_gemini_client")
    @patch("core.router.execute_prompt")
    @patch("core.router.perform_live_research")
    def test_empty_chunks_triggers_web_research(
        self, mock_web_research, mock_execute_prompt, mock_gemini
    ):
        """When no document chunks exist, router directly triggers web research."""
        mock_gemini.return_value = MagicMock()
        mock_web_research.return_value = [
            {
                "title": "General Topic Info",
                "url": "https://example.com/topic",
                "notes": "General notes.",
                "full_snippet": "General snippet.",
            }
        ]
        mock_execute_prompt.return_value = ("Answer without local document.", "Gemini")

        result = process_query_with_router(
            user_query="Tell me about CRISPR",
            chunks=[],
            chunk_embeddings=None,
        )

        self.assertEqual(result["decision"], "WEB_RESEARCH_TRIGGERED")
        self.assertTrue(result["is_web_research"])

    @patch("core.router.get_gemini_client")
    @patch("core.router.retrieve")
    @patch("core.router.execute_prompt")
    def test_status_callback_invoked(self, mock_execute_prompt, mock_retrieve, mock_gemini):
        """Status callback should be called during vector search and routing."""
        mock_gemini.return_value = MagicMock()
        mock_retrieve.return_value = ([self.sample_chunk], [0.9], 0.9)
        mock_execute_prompt.return_value = ("Answer", "TestProvider")

        status_messages = []
        def callback(msg):
            status_messages.append(msg)

        process_query_with_router(
            user_query="Test query",
            chunks=[self.sample_chunk],
            chunk_embeddings=self.sample_embeddings,
            status_callback=callback,
        )

        self.assertTrue(any("local vector knowledge base" in msg for msg in status_messages))
        self.assertTrue(any("Found verified answer" in msg for msg in status_messages))

    @patch("core.router.get_gemini_client")
    @patch("core.router.perform_5_pillar_research")
    @patch("core.router.execute_prompt")
    @patch("core.router.generate_research_pdf_report")
    @patch("core.router.embed_chunks")
    def test_execute_deep_research_success(
        self, mock_embed, mock_pdf, mock_execute_prompt, mock_5_pillar, mock_gemini
    ):
        """execute_deep_research should synthesize 5 pillars, generate PDF, and index report."""
        mock_gemini.return_value = MagicMock()
        mock_5_pillar.return_value = [
            {
                "pillar": "1. Introduction & Foundational Concepts",
                "title": "Intro Source",
                "url": "https://example.com/intro",
                "notes": "Key definitions",
                "full_snippet": "Full text snippet",
            }
        ]
        synthesis_text = "Synthesized 5-Pillar Academic Report with comprehensive factual analysis across all pillars."
        mock_execute_prompt.return_value = (synthesis_text, "Google Gemini")
        mock_pdf.return_value = "/path/to/report.pdf"
        mock_embed.return_value = np.array([[0.1, 0.2]], dtype=np.float32)

        res = execute_deep_research(
            topic="Quantum Computing",
            chunks=[],
            chunk_embeddings=None,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["topic"], "Quantum Computing")
        self.assertEqual(res["summary"], synthesis_text)
        self.assertEqual(res["pdf_path"], "/path/to/report.pdf")

    @patch("core.router.get_gemini_client")
    @patch("core.router.perform_5_pillar_research")
    def test_execute_deep_research_failure(self, mock_5_pillar, mock_gemini):
        """execute_deep_research should return failure if 5-pillar data cannot be retrieved."""
        mock_gemini.return_value = MagicMock()
        mock_5_pillar.return_value = []

        res = execute_deep_research(
            topic="Fictional Impossible Topic",
            chunks=[],
            chunk_embeddings=None,
        )

        self.assertFalse(res["success"])
        self.assertIn("Could not gather sufficient verified research data", res["error"])


class TestAgentRouterBackwardCompatibility(unittest.TestCase):
    """Test suite ensuring core.agent_router preserves all functions and symbols."""

    def test_agent_router_reexports_router_functions(self):
        import core.agent_router as agent_router
        self.assertTrue(hasattr(agent_router, "process_query_with_router"))
        self.assertTrue(hasattr(agent_router, "execute_deep_research"))
        self.assertIs(agent_router.process_query_with_router, process_query_with_router)
        self.assertIs(agent_router.execute_deep_research, execute_deep_research)


if __name__ == "__main__":
    unittest.main()
