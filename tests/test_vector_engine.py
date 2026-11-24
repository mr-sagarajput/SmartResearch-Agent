"""
Unit tests for core/vector_engine.py.
Validates:
- DocumentChunk dataclass creation and attributes
- Text splitting with smart sentence-aware chunking and overlap preservation
- Multi-page document chunk construction
- Cosine similarity computation and bytes hashing
"""

from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.vector_engine import (
    DocumentChunk,
    build_document_chunks,
    compute_bytes_hash,
    cosine_similarity,
    embed_content_with_retry,
    get_cache_paths,
    load_cached_embeddings,
    save_cached_embeddings,
    split_text_smart,
)


class TestVectorEngine(unittest.TestCase):
    """Test suite for vector engine data structures and chunking algorithms."""

    def test_document_chunk_creation_default_url(self):
        """DocumentChunk should properly initialize with required fields and empty url."""
        chunk = DocumentChunk(
            chunk_id=1,
            source="Page 1",
            page=1,
            text="This is sample text from page 1.",
        )
        self.assertEqual(chunk.chunk_id, 1)
        self.assertEqual(chunk.source, "Page 1")
        self.assertEqual(chunk.page, 1)
        self.assertEqual(chunk.text, "This is sample text from page 1.")
        self.assertEqual(chunk.url, "")

    def test_document_chunk_with_url(self):
        """DocumentChunk originating from web search should hold url."""
        chunk = DocumentChunk(
            chunk_id=2,
            source="Web: Scientific American",
            page=0,
            text="Web article content here.",
            url="https://example.com/article",
        )
        self.assertEqual(chunk.chunk_id, 2)
        self.assertEqual(chunk.source, "Web: Scientific American")
        self.assertEqual(chunk.page, 0)
        self.assertEqual(chunk.url, "https://example.com/article")
        chunk_dict = asdict(chunk)
        self.assertEqual(chunk_dict["url"], "https://example.com/article")
        self.assertEqual(chunk_dict["chunk_id"], 2)

    def test_split_text_smart_short_text(self):
        """Text shorter than max_chars should return a single chunk."""
        text = "Short paragraph of research text."
        chunks = split_text_smart(text, max_chars=100, overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_split_text_smart_multiple_short_paragraphs_combined(self):
        """Multiple short paragraphs should be merged into a single chunk if within max_chars."""
        p1 = "First short paragraph."
        p2 = "Second short paragraph."
        text = f"{p1}\n\n{p2}"
        chunks = split_text_smart(text, max_chars=100, overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertIn(p1, chunks[0])
        self.assertIn(p2, chunks[0])

    def test_split_text_smart_long_paragraph_splits_on_sentences(self):
        """Paragraph exceeding max_chars should split at sentence boundaries."""
        sentences = [
            "Artificial intelligence is transforming computational biology.",
            "Deep learning models predict protein structures with high fidelity.",
            "Researchers can now simulate molecular dynamics faster than ever.",
            "These advances accelerate novel drug discovery pipelines significantly.",
        ]
        text = " ".join(sentences)
        # Using a small max_chars (e.g. 140) to force sentence splitting
        chunks = split_text_smart(text, max_chars=140, overlap=30)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 160)  # Account for overlap boundary
            self.assertTrue(any(s[:20] in chunk for s in sentences))

    def test_split_text_smart_overlap_preservation(self):
        """When text exceeds max_chars, subsequent chunks should contain overlap from previous chunk."""
        p1 = "Alpha " * 20  # ~120 chars
        p2 = "Beta " * 20   # ~100 chars
        text = f"{p1}\n\n{p2}"
        chunks = split_text_smart(text, max_chars=130, overlap=30)
        self.assertGreaterEqual(len(chunks), 2)
        # Check that overlap characters or words appear in chunk 2
        first_chunk_end = chunks[0][-20:].strip()
        self.assertTrue(
            any(word in chunks[1] for word in first_chunk_end.split() if len(word) > 2),
            "Expected overlap text from chunk 0 in chunk 1",
        )

    def test_split_text_smart_single_huge_sentence(self):
        """A single continuous sentence longer than max_chars should still be sliced."""
        huge_word = "A" * 300
        chunks = split_text_smart(huge_word, max_chars=100, overlap=20)
        self.assertGreaterEqual(len(chunks), 3)

    def test_build_document_chunks_multi_page(self):
        """build_document_chunks should create sequential chunks across multiple pages."""
        pages = [
            (1, "Page 1 Introduction to quantum physics.\n\nQuantum mechanics governs small scales."),
            (2, "Page 2 Wave-particle duality.\n\nElectrons exhibit both wave and particle traits."),
        ]
        chunks = build_document_chunks(pages)
        self.assertGreaterEqual(len(chunks), 2)

        # Validate sequential chunk_ids starting from 1
        for i, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk.chunk_id, i)

        # Validate sources and pages
        page1_chunks = [c for c in chunks if c.page == 1]
        page2_chunks = [c for c in chunks if c.page == 2]
        self.assertTrue(len(page1_chunks) >= 1)
        self.assertTrue(len(page2_chunks) >= 1)
        self.assertEqual(page1_chunks[0].source, "Page 1")
        self.assertEqual(page2_chunks[0].source, "Page 2")

    def test_compute_bytes_hash(self):
        """compute_bytes_hash should compute deterministic SHA-256 hex string."""
        data = b"SmartResearch test byte string"
        hash_val = compute_bytes_hash(data)
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)
        # Determinism check
        self.assertEqual(hash_val, compute_bytes_hash(data))

    def test_cosine_similarity(self):
        """cosine_similarity should return expected normalized dot products."""
        # Query vector: [1, 0, 0]
        q_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Doc vectors: [1, 0, 0] (identical), [0, 1, 0] (orthogonal), [-1, 0, 0] (opposite)
        doc_vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=np.float32)

        sims = cosine_similarity(q_vec, doc_vecs)
        self.assertEqual(len(sims), 3)
        self.assertAlmostEqual(sims[0], 1.0, places=4)
        self.assertAlmostEqual(sims[1], 0.0, places=4)
        self.assertAlmostEqual(sims[2], -1.0, places=4)

    @patch("time.sleep")
    def test_embed_content_with_retry_exceeds_max_retries(self, mock_sleep):
        """When 429 persists, embed_content_with_retry should raise RuntimeError after max_retries."""
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")

        with self.assertRaises(RuntimeError) as ctx:
            embed_content_with_retry(
                gemini_client=mock_client,
                model="gemini-embedding-001",
                contents="test content",
                config=MagicMock(),
                max_retries=2,
            )

        self.assertIn("embedding quota exhausted after 2 attempts", str(ctx.exception))
        # Initial call + 2 retries = 3 calls
        self.assertEqual(mock_client.models.embed_content.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("time.sleep")
    def test_embed_content_with_retry_succeeds_after_retry(self, mock_sleep):
        """embed_content_with_retry should recover and return response when a retry succeeds."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.models.embed_content.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED"),
            mock_response,
        ]

        res = embed_content_with_retry(
            gemini_client=mock_client,
            model="gemini-embedding-001",
            contents="test content",
            config=MagicMock(),
            max_retries=3,
        )

        self.assertEqual(res, mock_response)
        self.assertEqual(mock_client.models.embed_content.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    def test_safe_cache_roundtrip(self):
        """save_cached_embeddings and load_cached_embeddings should write npz+json and restore data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("core.vector_engine.CACHE_DIR", tmp_path):
                test_hash = "test_sha256_hash_12345"
                chunks = [
                    DocumentChunk(chunk_id=1, source="Doc A", page=1, text="Chunk 1 text", url="https://example.com/1"),
                    DocumentChunk(chunk_id=2, source="Doc A", page=2, text="Chunk 2 text", url="https://example.com/2"),
                ]
                embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

                save_cached_embeddings(test_hash, chunks, embeddings)

                npz_file, json_file = get_cache_paths(test_hash)
                self.assertTrue(npz_file.exists())
                self.assertTrue(json_file.exists())

                loaded = load_cached_embeddings(test_hash)
                self.assertIsNotNone(loaded)
                loaded_chunks, loaded_embeddings = loaded

                self.assertEqual(len(loaded_chunks), 2)
                self.assertEqual(loaded_chunks[0].text, "Chunk 1 text")
                self.assertEqual(loaded_chunks[1].source, "Doc A")
                np.testing.assert_array_almost_equal(loaded_embeddings, embeddings)

    def test_load_cached_embeddings_miss_and_legacy_cleanup(self):
        """Cache miss returns None, and unvalidated legacy .pkl files are removed safely."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("core.vector_engine.CACHE_DIR", tmp_path):
                test_hash = "nonexistent_hash"
                legacy_file = tmp_path / f"{test_hash}.pkl"
                legacy_file.write_bytes(b"dummy legacy content")
                self.assertTrue(legacy_file.exists())

                loaded = load_cached_embeddings(test_hash)
                self.assertIsNone(loaded)
                self.assertFalse(legacy_file.exists())

    def test_load_cached_embeddings_dimension_mismatch(self):
        """If chunk count does not match embedding matrix rows, cache load must reject and return None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("core.vector_engine.CACHE_DIR", tmp_path):
                test_hash = "dimension_mismatch_hash"
                chunks = [DocumentChunk(chunk_id=1, source="Doc", page=1, text="Only one chunk")]
                embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)  # 2 vectors for 1 chunk

                save_cached_embeddings(test_hash, chunks, embeddings)
                loaded = load_cached_embeddings(test_hash)
                self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
