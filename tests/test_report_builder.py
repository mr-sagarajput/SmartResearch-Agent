"""
Unit tests for core/report_builder.py.
Validates text sanitization for ReportLab PDF generation:
- Unicode hyphens and dashes
- Smart curly quotes and apostrophes
- Mathematical symbols
- Safe HTML escaping and tag preservation
"""

from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.report_builder import get_unique_filename, sanitize_text_for_pdf


class TestReportBuilderSanitization(unittest.TestCase):
    """Test suite for ReportLab PDF text sanitization."""

    def test_empty_and_none_text(self):
        """Sanitizer should return empty string on None or empty input."""
        self.assertEqual(sanitize_text_for_pdf(""), "")
        self.assertEqual(sanitize_text_for_pdf(None), "")

    def test_unicode_hyphens_and_dashes(self):
        """All Unicode hyphens, dashes, and bars must convert to standard ASCII hyphen."""
        test_cases = [
            ("hyphen\u2010test", "hyphen-test"),          # \u2010 Hyphen
            ("non\u2011breaking", "non-breaking"),        # \u2011 Non-breaking hyphen
            ("figure\u2012dash", "figure-dash"),          # \u2012 Figure dash
            ("en\u2013dash", "en-dash"),                  # \u2013 En dash
            ("em\u2014dash", "em-dash"),                  # \u2014 Em dash
            ("horizontal\u2015bar", "horizontal-bar"),    # \u2015 Horizontal bar
            ("soft\u00adhyphen", "soft-hyphen"),          # \u00ad Soft hyphen
            ("minus\u2212sign", "minus-sign"),            # \u2212 Minus sign
        ]
        for input_text, expected_output in test_cases:
            with self.subTest(input_text=input_text):
                self.assertEqual(sanitize_text_for_pdf(input_text), expected_output)

    def test_curly_quotes_and_apostrophes(self):
        """Smart single and double quotes must convert to plain ASCII equivalents."""
        single_quote_sample = "\u2018single\u2019 \u201alow\u201b"
        expected_single = "'single' 'low'"
        self.assertEqual(sanitize_text_for_pdf(single_quote_sample), expected_single)

        double_quote_sample = "\u201cquoted\u201d and \u201elow\u201e"
        expected_double = '"quoted" and "low"'
        self.assertEqual(sanitize_text_for_pdf(double_quote_sample), expected_double)

        contraction_sample = "It\u2019s a researcher\u2019s notebook"
        self.assertEqual(sanitize_text_for_pdf(contraction_sample), "It's a researcher's notebook")

    def test_math_symbols(self):
        """Mathematical operators and symbols should be converted to clean readable ASCII/HTML."""
        math_samples = [
            ("5 \u2212 3 = 2", "5 - 3 = 2"),             # Minus sign (\u2212)
            ("10 \u00b1 2", "10 +/- 2"),                  # Plus-minus (\u00b1)
            ("4 \u00d7 5 = 20", "4 * 5 = 20"),            # Multiplication (\u00d7)
            ("20 \u00f7 4 = 5", "20 / 4 = 5"),            # Division (\u00f7)
            ("x \u2260 y", "x != y"),                     # Not equal (\u2260)
            ("x \u2264 100", "x &lt;= 100"),             # Less than or equal (\u2264)
            ("y \u2265 50", "y &gt;= 50"),               # Greater than or equal (\u2265)
            ("limit \u221e", "limit inf"),                # Infinity (\u221e)
            ("pi \u2248 3.14", "pi ~ 3.14"),              # Almost equal (\u2248)
        ]
        for input_text, expected_output in math_samples:
            with self.subTest(input_text=input_text):
                self.assertEqual(sanitize_text_for_pdf(input_text), expected_output)

    def test_unicode_whitespace_and_ellipsis(self):
        """Non-standard spaces and ellipsis should be normalized."""
        spaces_text = "word1\u00a0word2\u2002word3\u2003word4\u2009word5"
        self.assertEqual(sanitize_text_for_pdf(spaces_text), "word1 word2 word3 word4 word5")

        ellipsis_text = "Continuing\u2026"
        self.assertEqual(sanitize_text_for_pdf(ellipsis_text), "Continuing...")

    def test_bullet_character_preserved(self):
        """Standard bullet symbol • (\u2022) is allowed for list items."""
        bullet_text = "\u2022 Item 1\n\u2022 Item 2"
        self.assertEqual(sanitize_text_for_pdf(bullet_text), "• Item 1\n• Item 2")

    def test_html_escaping_and_bold_tags(self):
        """Arbitrary HTML tags should be escaped, but <b> and </b> tags should be preserved."""
        text = "This is <b>bold</b> and this is <script>alert('xss')</script> & more"
        expected = "This is <b>bold</b> and this is &lt;script&gt;alert('xss')&lt;/script&gt; &amp; more"
        self.assertEqual(sanitize_text_for_pdf(text), expected)

    def test_unique_filename_generation(self):
        """get_unique_filename should return a valid Path inside output directory."""
        path = get_unique_filename("Test Research Topic: Quantum Computing", ext=".pdf")
        self.assertIsInstance(path, Path)
        self.assertTrue(path.name.startswith("research_Test_Research_Topic__Quantum_Computing_"))
        self.assertTrue(path.name.endswith(".pdf"))


if __name__ == "__main__":
    unittest.main()
