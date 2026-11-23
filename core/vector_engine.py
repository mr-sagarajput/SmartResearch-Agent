"""
Vector & Document Processing Engine with Production-Grade Resilience.
=====================================================================
Features:
- Hybrid PDF Text Extraction + pypdfium2 Vision OCR for scanned pages
- Smart Recursive Chunking preserving sentence integrity
- Persistent Disk Caching (.cache/) with SHA-256 validation (0 API calls on reload)
- 100-Chunk Batched Embeddings with Free Tier Pacing (Prevents 429 Resource Exhaustion)
- Regex-based Adaptive 429 Retry Wrapper (Never crashes on rate limits)
- Dynamic In-Memory Knowledge Augmentation (adds web search chunks on the fly)
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import pickle
import re
import time
from typing import List, Optional, Tuple

from google import genai
from google.genai import errors as genai_errors, types as genai_types
import numpy as np
from pypdf import PdfReader
import pypdfium2 as pdfium

from config import (
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    TOP_K,
    VISION_MODEL,
)
from core.models import DocumentChunk


def compute_bytes_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def get_cache_path(file_hash: str) -> Path:
    return CACHE_DIR / f"{file_hash}.pkl"


def load_cached_embeddings(file_hash: str) -> Optional[Tuple[List[DocumentChunk], np.ndarray]]:
    cache_file = get_cache_path(file_hash)
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        if data.get("embed_model") != EMBED_MODEL:
            return None
        chunks = [
            DocumentChunk(
                chunk_id=c["chunk_id"],
                source=c.get("source", f"Page {c.get('page', 1)}"),
                page=c.get("page", 1),
                text=c["text"],
                url=c.get("url", ""),
            )
            for c in data["chunks"]
        ]
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        return chunks, embeddings
    except Exception:
        return None


def save_cached_embeddings(
    file_hash: str,
    chunks: List[DocumentChunk],
    embeddings: np.ndarray,
) -> None:
    cache_file = get_cache_path(file_hash)
    data = {
        "embed_model": EMBED_MODEL,
        "chunks": [asdict(c) for c in chunks],
        "embeddings": embeddings,
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[Warning] Failed to write cache: {e}")


def ocr_page_image(gemini_client: genai.Client, image_bytes: bytes) -> str:
    try:
        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        prompt = (
            "Transcribe all text from this page image verbatim. "
            "Maintain original layout. Return only the extracted text."
        )
        response = gemini_client.models.generate_content(
            model=VISION_MODEL,
            contents=[prompt, image_part],
        )
        return response.text.strip()
    except Exception as e:
        print(f"Vision OCR error: {e}")
        return ""


def load_pdf_pages_from_bytes(
    file_bytes: bytes,
    gemini_client: genai.Client,
    status_callback=None,
) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: List[Tuple[int, str]] = []
    pdfium_doc = None

    for i, page in enumerate(reader.pages, start=1):
        raw_text = (page.extract_text() or "").strip()
        if len(raw_text) >= 30:
            pages.append((i, raw_text))
            continue

        if status_callback:
            status_callback(f"🔍 Page {i} is scanned. Running Gemini Vision OCR...")

        try:
            if pdfium_doc is None:
                pdfium_doc = pdfium.PdfDocument(file_bytes)
            pdf_page = pdfium_doc[i - 1]
            pil_image = pdf_page.render(scale=2.0).to_pil()

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            ocr_text = ocr_page_image(gemini_client, buf.getvalue())

            if ocr_text:
                pages.append((i, ocr_text))
            if status_callback:
                status_callback(f"✅ Page {i}: OCR extracted {len(ocr_text)} characters.")
        except Exception as e:
            if status_callback:
                status_callback(f"⚠️ Page {i} OCR skipped: {e}")

    if not pages:
        raise ValueError("No readable text or images found in the uploaded document.")
    return pages


def split_text_smart(
    text: str,
    max_chars: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(current_chunk) + len(sent) + 1 <= max_chars:
                    current_chunk = f"{current_chunk} {sent}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        if len(overlap_text) + len(sent) + 1 <= max_chars:
                            current_chunk = f"{overlap_text} {sent}".strip()
                        else:
                            while len(sent) > max_chars:
                                chunks.append(sent[:max_chars])
                                sent = sent[max_chars - overlap:]
                            current_chunk = sent
                    else:
                        while len(sent) > max_chars:
                            chunks.append(sent[:max_chars])
                            sent = sent[max_chars - overlap:]
                        current_chunk = sent
        else:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = f"{overlap_text}\n\n{para}".strip()
                else:
                    chunks.append(para)
                    current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_document_chunks(pages: List[Tuple[int, str]]) -> List[DocumentChunk]:
    all_chunks: List[DocumentChunk] = []
    chunk_counter = 1

    for page_num, page_text in pages:
        page_chunks = split_text_smart(page_text, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for text_chunk in page_chunks:
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_counter,
                    source=f"Page {page_num}",
                    page=page_num,
                    text=text_chunk,
                )
            )
            chunk_counter += 1

    return all_chunks


# ---------------------------------------------------------------------------
# Rate-Paced Batch Embedding with Regex Adaptive 429 Interceptor
# ---------------------------------------------------------------------------

def embed_content_with_retry(
    gemini_client: genai.Client,
    model: str,
    contents: str | list[str],
    config: genai_types.EmbedContentConfig,
    status_callback=None,
) -> genai_types.EmbedContentResponse:
    """Wrapper around embed_content with dynamic 429 retry and rate limit backoff."""
    while True:
        try:
            return gemini_client.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )
        except (genai_errors.ClientError, genai_errors.APIError, Exception) as exc:
            err_msg = str(exc)
            if getattr(exc, "code", None) == 429 or "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                wait_sec = 60.0
                match = re.search(
                    r"(?:retryDelay|retry in|retry after)[\"':\s]*([0-9]+(?:\.[0-9]+)?)\s*s?",
                    err_msg,
                    re.IGNORECASE,
                )
                if match:
                    try:
                        wait_sec = float(match.group(1))
                    except ValueError:
                        wait_sec = 60.0

                total_wait = wait_sec + 2.0
                log_msg = f"⏳ [Rate Limit] Free tier quota reached (429). Auto-waiting {total_wait:.0f}s before resuming..."
                print(f"\n{log_msg}", flush=True)
                if status_callback:
                    status_callback(log_msg)
                time.sleep(total_wait)
            else:
                raise


def embed_chunks(
    gemini_client: genai.Client,
    chunks: List[DocumentChunk],
    progress_callback=None,
) -> np.ndarray:
    """
    Embeds chunks in strict batches of 100.
    For multi-batch large documents (e.g., 1000 pages), applies pacing pauses
    to prevent Google Gemini Free Tier 429 exhaustion.
    """
    texts = [c.text for c in chunks]
    batch_size = 100
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_idx = (i // batch_size) + 1
        batch = texts[i : i + batch_size]

        if progress_callback:
            progress_callback(f"🔢 Embedding Batch {batch_idx}/{total_batches} ({len(batch)} chunks)...")

        response = embed_content_with_retry(
            gemini_client=gemini_client,
            model=EMBED_MODEL,
            contents=batch,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            status_callback=progress_callback,
        )

        for emb in response.embeddings:
            all_embeddings.append(emb.values)

        # Apply safe pacing between batches for large multi-batch files
        if batch_idx < total_batches:
            if progress_callback:
                progress_callback("⏳ Pacing between batches to protect Free Tier quota (Waiting 15s)...")
            time.sleep(15.0)

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(gemini_client: genai.Client, query: str) -> np.ndarray:
    response = embed_content_with_retry(
        gemini_client=gemini_client,
        model=EMBED_MODEL,
        contents=query,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return np.array(response.embeddings[0].values, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return d @ q


def retrieve(
    gemini_client: genai.Client,
    query: str,
    chunks: List[DocumentChunk],
    chunk_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> Tuple[List[DocumentChunk], List[float], float]:
    if len(chunks) == 0 or chunk_embeddings is None or len(chunk_embeddings) == 0:
        return [], [], 0.0

    q_vec = embed_query(gemini_client, query)
    scores = cosine_similarity(q_vec, chunk_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = [chunks[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]
    max_score = top_scores[0] if top_scores else 0.0

    return retrieved, top_scores, max_score
