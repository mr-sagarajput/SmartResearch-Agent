"""
Unit tests for core/llm_gateway.py.
Focuses on extract_content_safely resilience and boundary conditions.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_gateway import extract_content_safely, get_gemini_client


class TestLLMGatewayContentExtraction(unittest.TestCase):
    """Test suite for extract_content_safely."""

    def test_message_is_none(self):
        """extract_content_safely should return empty string when message is None."""
        self.assertEqual(extract_content_safely(None), "")

    def test_message_is_empty_object(self):
        """extract_content_safely should return empty string for an empty object without attrs."""
        msg = SimpleNamespace()
        self.assertEqual(extract_content_safely(msg), "")

    def test_content_is_present(self):
        """When content is non-empty, it should be extracted and stripped."""
        msg = SimpleNamespace(content="Quantum mechanics is a fundamental theory.", reasoning=None)
        result = extract_content_safely(msg)
        self.assertEqual(result, "Quantum mechanics is a fundamental theory.")

    def test_content_with_whitespace(self):
        """Leading and trailing whitespace in content should be stripped."""
        msg = SimpleNamespace(content="   \nLeading and trailing space.   \t\n", reasoning=None)
        result = extract_content_safely(msg)
        self.assertEqual(result, "Leading and trailing space.")

    def test_content_none_with_single_paragraph_reasoning(self):
        """When content is None, fallback to reasoning text."""
        msg = SimpleNamespace(
            content=None,
            reasoning="The final synthesized answer is that Python is widely used for ML.",
        )
        result = extract_content_safely(msg)
        self.assertEqual(result, "The final synthesized answer is that Python is widely used for ML.")

    def test_content_empty_string_with_reasoning(self):
        """When content is empty string, fallback to reasoning."""
        msg = SimpleNamespace(
            content="",
            reasoning="Single paragraph reasoning content.",
        )
        result = extract_content_safely(msg)
        self.assertEqual(result, "Single paragraph reasoning content.")

    def test_content_whitespace_only_with_reasoning(self):
        """When content consists only of whitespace, fallback to reasoning."""
        msg = SimpleNamespace(
            content="   \n\t  ",
            reasoning="Fallback reasoning after blank content.",
        )
        result = extract_content_safely(msg)
        self.assertEqual(result, "Fallback reasoning after blank content.")

    def test_content_none_with_multi_paragraph_reasoning(self):
        """When reasoning has multiple paragraphs, extract the final paragraph as conclusion."""
        reasoning_text = (
            "Step 1: Analyze user question regarding photosynthesis.\n\n"
            "Step 2: Review light and dark reactions.\n\n"
            "In conclusion, photosynthesis converts light energy into chemical energy."
        )
        msg = SimpleNamespace(content=None, reasoning=reasoning_text)
        result = extract_content_safely(msg)
        self.assertEqual(result, "In conclusion, photosynthesis converts light energy into chemical energy.")

    def test_content_takes_precedence_over_reasoning(self):
        """If content is populated, reasoning should not override it."""
        msg = SimpleNamespace(
            content="Direct answer to user.",
            reasoning="Internal chain of thought that should not be shown.",
        )
        result = extract_content_safely(msg)
        self.assertEqual(result, "Direct answer to user.")

    def test_both_content_and_reasoning_empty(self):
        """When both content and reasoning are empty or None, return empty string."""
        msg1 = SimpleNamespace(content=None, reasoning=None)
        self.assertEqual(extract_content_safely(msg1), "")

        msg2 = SimpleNamespace(content="", reasoning="")
        self.assertEqual(extract_content_safely(msg2), "")

        msg3 = SimpleNamespace(content="   ", reasoning="   \n\n  ")
        self.assertEqual(extract_content_safely(msg3), "")

    def test_mock_chat_completion_message(self):
        """Test with MagicMock simulating OpenAI ChatCompletionMessage."""
        mock_msg = MagicMock()
        mock_msg.content = "OpenAI response text"
        del mock_msg.reasoning  # simulate missing attribute
        self.assertEqual(extract_content_safely(mock_msg), "OpenAI response text")


class TestGeminiClientInitialization(unittest.TestCase):
    """Test suite for get_gemini_client."""

    def test_get_gemini_client_missing_key(self):
        """When GEMINI_API_KEY is missing or empty, client returns None."""
        with patch("core.llm_gateway.GEMINI_API_KEY", ""):
            client = get_gemini_client()
            self.assertIsNone(client)

    def test_get_gemini_client_with_key(self):
        """When GEMINI_API_KEY is provided, initializes genai.Client with HttpOptions."""
        with patch("core.llm_gateway.GEMINI_API_KEY", "test-mock-key"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_instance = MagicMock()
            mock_client_cls.return_value = mock_instance

            client = get_gemini_client()
            self.assertIsNotNone(client)
            self.assertEqual(client, mock_instance)
            mock_client_cls.assert_called_once()
            _, kwargs = mock_client_cls.call_args
            self.assertEqual(kwargs.get("api_key"), "test-mock-key")
            self.assertIn("http_options", kwargs)

    def test_get_gemini_client_handles_exception(self):
        """When genai.Client raises an exception, return None without crashing."""
        with patch("core.llm_gateway.GEMINI_API_KEY", "test-mock-key"), \
             patch("google.genai.Client", side_effect=Exception("Initialization failure")):
            client = get_gemini_client()
            self.assertIsNone(client)


if __name__ == "__main__":
    unittest.main()
