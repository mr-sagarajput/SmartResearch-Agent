"""
Unit tests for core/web_researcher.py concurrency and polite resilience.
Validates:
- ThreadPoolExecutor parallel execution across the 5 academic pillars
- Error isolation across thread workers (failure in one pillar does not crash others)
- Canonical 1-to-5 ordering preserved after parallel completion
- Live research concurrent article fetching
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.web_researcher import (
    get_5_pillar_search_queries,
    is_supported_article_url,
    perform_5_pillar_research,
    perform_live_research,
)


class TestWebResearcherConcurrency(unittest.TestCase):
    """Test suite for concurrent web research and error isolation."""

    def test_get_5_pillar_search_queries_count(self):
        """Should return exactly 5 targeted pillar search queries."""
        queries = get_5_pillar_search_queries("Machine Learning")
        self.assertEqual(len(queries), 5)
        self.assertTrue(all(hasattr(q, "pillar") and hasattr(q, "query") and q.pillar and q.query for q in queries))

    def test_is_supported_article_url(self):
        """Should filter out YouTube links and direct PDF binaries."""
        valid, _ = is_supported_article_url("https://en.wikipedia.org/wiki/Python")
        self.assertTrue(valid)

        invalid_yt, reason_yt = is_supported_article_url("https://www.youtube.com/watch?v=123")
        self.assertFalse(invalid_yt)
        self.assertIn("YouTube", reason_yt)

        invalid_pdf, reason_pdf = is_supported_article_url("https://arxiv.org/pdf/123.pdf")
        self.assertFalse(invalid_pdf)
        self.assertIn("PDF", reason_pdf)

    @patch("core.web_researcher.time.sleep", return_value=None)
    @patch("core.web_researcher.search_web")
    @patch("core.web_researcher.fetch_article_text")
    @patch("core.web_researcher.execute_prompt")
    def test_perform_5_pillar_research_concurrent_success(
        self, mock_prompt, mock_fetch, mock_search, mock_sleep
    ):
        """All 5 pillars should execute concurrently and preserve 1-5 canonical ordering."""
        mock_search.side_effect = lambda q, max_results=3: [
            {"title": f"Title for {q[:15]}", "url": f"https://example.com/{abs(hash(q))}"}
        ]
        mock_fetch.return_value = "Detailed academic article body text with more than 150 characters to pass the threshold easily." * 3
        mock_prompt.return_value = ("• Point 1: Core definition\n• Point 2: Key facts", "Google Gemini")

        results = perform_5_pillar_research("Neuroscience")

        self.assertEqual(len(results), 5)
        # Verify canonical 1..5 ordering
        for idx, item in enumerate(results, start=1):
            self.assertTrue(item.pillar.startswith(f"{idx}."))
            self.assertTrue(bool(item.notes))
            self.assertTrue(bool(item.full_snippet))

    @patch("core.web_researcher.time.sleep", return_value=None)
    @patch("core.web_researcher.search_web")
    @patch("core.web_researcher.fetch_article_text")
    @patch("core.web_researcher.execute_prompt")
    def test_perform_5_pillar_research_error_isolation(
        self, mock_prompt, mock_fetch, mock_search, mock_sleep
    ):
        """If one pillar worker raises an exception, other pillars must succeed without crashing."""
        def search_side_effect(q, max_results=3):
            if "taxonomy" in q.lower():
                raise RuntimeError("Simulated network timeout for Pillar 2")
            return [{"title": f"Title for {q[:10]}", "url": f"https://example.com/{abs(hash(q))}"}]

        mock_search.side_effect = search_side_effect
        mock_fetch.return_value = "Sufficiently long article text over 150 characters for unit testing purposes." * 3
        mock_prompt.return_value = ("Extracted summary notes", "OpenRouter")

        results = perform_5_pillar_research("Genomics")

        # Pillar 2 failed gracefully, remaining 4 pillars succeeded
        self.assertEqual(len(results), 4)
        pillars_collected = [r.pillar for r in results]
        self.assertFalse(any("2. Core Sub-Fields" in p for p in pillars_collected))
        self.assertTrue(any("1. Introduction" in p for p in pillars_collected))

    @patch("core.web_researcher.time.sleep", return_value=None)
    @patch("core.web_researcher.search_web")
    @patch("core.web_researcher.fetch_article_text")
    @patch("core.web_researcher.execute_prompt")
    def test_perform_live_research_concurrency(
        self, mock_prompt, mock_fetch, mock_search, mock_sleep
    ):
        """perform_live_research should concurrently fetch and synthesize candidate articles."""
        mock_search.return_value = [
            {"title": "Result 1", "url": "https://example.com/1"},
            {"title": "Result 2", "url": "https://example.com/2"},
        ]
        mock_fetch.return_value = "Article content longer than one hundred characters to qualify as a valid article snippet." * 2
        mock_prompt.return_value = ("Key summary bullet points", "OpenRouter")

        results = perform_live_research("Robotics", max_pages=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Result 1")
        self.assertEqual(results[1].title, "Result 2")


if __name__ == "__main__":
    unittest.main()
