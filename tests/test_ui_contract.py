"""
Integration Contract Tests for UI and CLI interfaces.
Validates the data contract between core.router (process_query_with_router,
execute_deep_research) and the UI/CLI consumer states in app.py and cli.py.
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

from core.models import (
    DeepResearchResult,
    DocumentChunk,
    ResearchPillar,
    RouterDecision,
    SearchResult,
)
from core.router import execute_deep_research, process_query_with_router


class TestUIAndCLIContract(unittest.TestCase):
    """Test suite validating contracts expected by Streamlit (app.py) and CLI (cli.py)."""

    def setUp(self):
        self.chunk = DocumentChunk(
            chunk_id=1,
            source="Manual.pdf",
            page=2,
            text="Autonomous agents navigate complex state spaces using planning trees.",
        )
        self.embeddings = np.array([[0.05, 0.95, 0.15]], dtype=np.float32)

    @patch("core.router.get_gemini_client")
    @patch("core.router.retrieve")
    @patch("core.router.execute_prompt")
    def test_process_query_vector_hit_contract(self, mock_prompt, mock_retrieve, mock_client):
        """Validates that a vector hit returns a RouterDecision with all fields required by app.py & cli.py."""
        mock_client.return_value = MagicMock()
        mock_retrieve.return_value = ([self.chunk], [0.88], 0.88)
        mock_prompt.return_value = ("Autonomous agents use planning trees.", "Google Gemini")

        statuses = []
        result = process_query_with_router(
            user_query="How do autonomous agents navigate?",
            chunks=[self.chunk],
            chunk_embeddings=self.embeddings,
            history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            status_callback=lambda msg: statuses.append(msg),
        )

        # 1. Type verification
        self.assertIsInstance(result, RouterDecision)

        # 2. app.py expected attributes
        self.assertIsInstance(result.answer, str)
        self.assertFalse(result.is_web_research)
        self.assertEqual(result.provider, "Google Gemini")
        self.assertAlmostEqual(result.max_score, 0.88)
        self.assertIsInstance(result.sources, list)
        self.assertGreater(len(result.sources), 0)

        # 3. Source item contract: (item_dict_or_chunk, score)
        first_item, first_score = result.sources[0]
        self.assertIn("source", first_item)
        self.assertIn("text", first_item)
        self.assertIsInstance(first_score, float)

        # 4. Status callbacks triggered
        self.assertGreater(len(statuses), 0)

        # 5. CLI rendering simulation (assert no exceptions on format)
        cli_badge = f"⚡ Answered from Vector Memory (Confidence: {result.max_score:.1%}) · {result.provider}"
        self.assertIn("88.0%", cli_badge)
        self.assertIn("Google Gemini", cli_badge)

    @patch("core.router.get_gemini_client")
    @patch("core.router.retrieve")
    @patch("core.router.perform_live_research")
    @patch("core.router.execute_prompt")
    @patch("core.router.embed_chunks")
    def test_process_query_web_research_contract(
        self, mock_embed, mock_prompt, mock_live_research, mock_retrieve, mock_client
    ):
        """Validates that web research returns RouterDecision with updated chunks and embeddings."""
        mock_client.return_value = MagicMock()
        # Low confidence forces web research
        mock_retrieve.return_value = ([self.chunk], [0.15], 0.15)

        mock_search_item = SearchResult(
            title="TechNews Article",
            url="https://example.com/news",
            notes="Recent discoveries show new advancements.",
            full_snippet="Full details on the breakthrough.",
        )
        mock_live_research.return_value = [mock_search_item]
        mock_prompt.return_value = ("Recent discoveries show new advancements.", "OpenRouter Free")
        mock_embed.return_value = np.array([[0.8, 0.2, 0.1]], dtype=np.float32)

        result = process_query_with_router(
            user_query="What are the recent discoveries?",
            chunks=[self.chunk],
            chunk_embeddings=self.embeddings,
        )

        self.assertIsInstance(result, RouterDecision)
        self.assertTrue(result.is_web_research)
        self.assertEqual(result.provider, "OpenRouter Free")
        self.assertIsNotNone(result.updated_chunks)
        self.assertEqual(len(result.updated_chunks), 2)
        self.assertIsNotNone(result.updated_embeddings)
        self.assertEqual(result.updated_embeddings.shape[0], 2)

    @patch("core.router.get_gemini_client")
    @patch("core.router.perform_5_pillar_research")
    @patch("core.router.execute_prompt")
    @patch("core.router.generate_research_pdf_report")
    @patch("core.router.embed_chunks")
    def test_execute_deep_research_ui_contract(
        self, mock_embed, mock_generate_pdf, mock_prompt, mock_pillar_research, mock_client
    ):
        """Validates that execute_deep_research returns a DeepResearchResult satisfying app.py requirements."""
        mock_client.return_value = MagicMock()
        mock_source = ResearchPillar(
            pillar="Foundational Theory",
            query="quantum algorithms overview",
            title="Theory Paper",
            url="https://arxiv.org/1",
            notes="Speedup analysis.",
            full_snippet="Detailed theoretical analysis of quantum computing speedups.",
        )
        mock_pillar_research.return_value = [mock_source]
        mock_prompt.return_value = (
            "1. Introduction & Foundational Concepts\nQuantum computing overview.\n\n"
            "2. Core Sub-Fields\nHardware and software.\n\n"
            "3. Principles\nSuperposition and entanglement.\n\n"
            "4. Applications\nCryptography and simulation.\n\n"
            "5. Limitations\nDecoherence and error rates.",
            "Google Gemini",
        )
        fake_pdf_path = Path("reports/Deep_Research_Quantum_20261124.pdf")
        mock_generate_pdf.return_value = fake_pdf_path
        mock_embed.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        statuses = []
        res = execute_deep_research(
            topic="Quantum Computing",
            chunks=[],
            chunk_embeddings=None,
            max_pages=2,
            status_callback=lambda m: statuses.append(m),
        )

        # 1. Type & Success verification
        self.assertIsInstance(res, DeepResearchResult)
        self.assertTrue(res.success)
        self.assertEqual(res.topic, "Quantum Computing")

        # 2. app.py contract assertions:
        # - res.pdf_path can be safely parsed by Path(str(res.pdf_path)).name
        self.assertIsNotNone(res.pdf_path)
        pdf_name = Path(str(res.pdf_path)).name
        self.assertEqual(pdf_name, "Deep_Research_Quantum_20261124.pdf")

        # - res.summary is non-empty string
        self.assertIsInstance(res.summary, str)
        self.assertIn("Foundational Concepts", res.summary)

        # - res.sources is list of SearchResult objects with title, notes, url
        self.assertIsInstance(res.sources, list)
        self.assertGreater(len(res.sources), 0)
        source = res.sources[0]
        self.assertEqual(source.title, "Theory Paper")
        self.assertEqual(source.url, "https://arxiv.org/1")
        self.assertEqual(source.notes, "Speedup analysis.")

        # - res.updated_chunks and res.updated_embeddings populated
        self.assertIsNotNone(res.updated_chunks)
        self.assertGreater(len(res.updated_chunks), 0)
        self.assertIsNotNone(res.updated_embeddings)

        # 3. Streamlit message payload generation verification (as done in app.py:320-335)
        formatted_sources = []
        for s in res.sources:
            s_title = getattr(s, "title", None) or (s.get("title", "") if isinstance(s, dict) else "")
            s_notes = getattr(s, "notes", None) or (s.get("notes", "") if isinstance(s, dict) else "")
            s_url = getattr(s, "url", None) or (s.get("url", "") if isinstance(s, dict) else "")
            formatted_sources.append(({"source": s_title, "page": 0, "text": s_notes, "url": s_url}, 1.0))

        self.assertEqual(len(formatted_sources), 1)
        self.assertEqual(formatted_sources[0][0]["source"], "Theory Paper")


if __name__ == "__main__":
    unittest.main()
