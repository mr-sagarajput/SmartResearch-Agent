"""
Live Web Research Module with 5-Pillar Academic Targeting & Robust Null-Safety.
=============================================================================
Executes 1-to-1 targeted search queries corresponding to the 5 Core Research Pillars:
1. Introduction & Foundational Concepts
2. Core Sub-Fields & Taxonomy (Branches)
3. Fundamental Principles & Methodologies
4. Real-World Applications & Significance
5. Critical Limitations & Future Outlook
Includes defensive error-handling to prevent NoneType attribute errors.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse, urlsplit, urlunsplit
import urllib.robotparser

from bs4 import BeautifulSoup
import requests

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from core.config import (
    CHAR_LIMIT_PER_PAGE,
    PAGE_LIMIT,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    RESULTS_PER_QUERY,
    USER_AGENT,
)
from core.llm_gateway import execute_prompt
from core.models import ResearchPillar, SearchResult


def encode_url(url: str) -> str:
    """Safely percent-encodes Hindi and non-ASCII URLs."""
    parts = urlsplit(url)
    encoded_path = quote(parts.path, safe="/:@&=+$,-_.!~*'()")
    encoded_query = quote(parts.query, safe="/:@&=+$,-_.!~*'()?")
    encoded_fragment = quote(parts.fragment, safe="/:@&=+$,-_.!~*'()")
    return urlunsplit((parts.scheme, parts.netloc, encoded_path, encoded_query, encoded_fragment))


def is_supported_article_url(url: str) -> Tuple[bool, str]:
    lower_url = url.lower()
    if "youtube.com" in lower_url or "youtu.be" in lower_url:
        return False, "YouTube link (cannot extract HTML text)"
    if lower_url.endswith(".pdf") or ".pdf?" in lower_url:
        return False, "PDF binary (skipped in web crawl)"
    return True, ""


def is_allowed_by_robots(url: str, user_agent: str = USER_AGENT) -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        headers = {"User-Agent": user_agent}
        response = requests.get(robots_url, headers=headers, timeout=4)
        if response.status_code == 404:
            return True
        if response.status_code == 200:
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(response.text.splitlines())
            return rp.can_fetch(user_agent, url)
        return True
    except Exception:
        return True


def search_web(query: str, max_results: int = RESULTS_PER_QUERY) -> List[Dict[str, str]]:
    results = []
    clean_query = query.strip()
    if not clean_query:
        return results

    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        with DDGS() as ddgs:
            search_gen = ddgs.text(clean_query, max_results=max_results + 3)
            for item in search_gen:
                if len(results) >= max_results:
                    break
                title = item.get("title", "No Title")
                url = item.get("href") or item.get("url", "")
                if not url or not (url.startswith("http://") or url.startswith("https://")):
                    continue
                is_valid, _ = is_supported_article_url(url)
                if not is_valid:
                    continue
                results.append({"title": title, "url": url})
    except Exception as e:
        print(f"[Search Notice]: {e}")

    return results


def fetch_article_text(url: str) -> Optional[str]:
    is_valid, _ = is_supported_article_url(url)
    if not is_valid:
        return None

    safe_url = encode_url(url)
    if not is_allowed_by_robots(safe_url):
        return None

    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = requests.get(safe_url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text_blocks = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
        cleaned_text = "\n\n".join(text_blocks)

        if not cleaned_text:
            cleaned_text = soup.get_text(separator="\n", strip=True)

        return cleaned_text[:CHAR_LIMIT_PER_PAGE].strip() if cleaned_text else None
    except Exception:
        return None


def get_5_pillar_search_queries(topic: str) -> List[ResearchPillar]:
    clean_topic = topic.strip()
    return [
        ResearchPillar(
            pillar="1. Introduction & Foundational Concepts",
            query=f"{clean_topic} definition introduction foundational concepts overview",
        ),
        ResearchPillar(
            pillar="2. Core Sub-Fields & Taxonomy (Branches)",
            query=f"{clean_topic} major branches subfields taxonomy types classification",
        ),
        ResearchPillar(
            pillar="3. Fundamental Principles & Methodologies",
            query=f"{clean_topic} fundamental principles core theories methodologies framework",
        ),
        ResearchPillar(
            pillar="4. Real-World Applications & Significance",
            query=f"{clean_topic} real world applications practical use cases modern impact",
        ),
        ResearchPillar(
            pillar="5. Critical Limitations & Future Outlook",
            query=f"{clean_topic} critical limitations challenges current debates future trends",
        ),
    ]


def _research_single_pillar(
    item: Dict[str, str],
    idx: int,
    topic: str,
    discovered_urls: set,
    url_lock: threading.Lock,
    status_callback=None,
) -> Optional[Dict[str, Any]]:
    """Researches a single academic pillar in an isolated thread worker."""
    pillar_name = item["pillar"]
    search_phrase = item["query"]

    # Polite stagger to prevent burst rate limits on search providers
    if idx > 1:
        time.sleep((idx - 1) * 0.25)

    if status_callback:
        try:
            status_callback(f"🔍 Pillar {idx}/5: Searching for {pillar_name.split('&')[0].strip()}...")
        except Exception:
            pass

    hits = search_web(search_phrase, max_results=3)
    best_article = None

    for hit in hits:
        url = hit.get("url", "")
        with url_lock:
            if not url or url in discovered_urls:
                continue
            discovered_urls.add(url)

        if status_callback:
            try:
                status_callback(f"📖 Reading Source for Pillar {idx}: {hit.get('title', '')[:45]}...")
            except Exception:
                pass

        raw_text = fetch_article_text(url)
        if raw_text and len(raw_text) > 150:
            best_article = {
                "pillar": pillar_name,
                "title": hit.get("title", "No Title"),
                "url": url,
                "text": raw_text,
            }
            break

    if best_article:
        prompt = (
            f"Pillar: {best_article['pillar']}\n"
            f"Topic: {topic}\n"
            f"Source: {best_article['title']} ({best_article['url']})\n\n"
            f"Article Text:\n{best_article['text']}\n\n"
            f"Extract 3 to 4 concise, high-value factual points that directly explain this specific pillar:"
        )
        try:
            raw_notes, _ = execute_prompt(prompt, max_tokens=300)
        except Exception as e:
            print(f"[LLM Extraction Note]: {e}")
            raw_notes = ""

        # Defensive Null-Safety: (raw_notes or "").strip()
        safe_notes = (raw_notes or "").strip()
        if not safe_notes:
            safe_notes = best_article["text"][:300].strip()

        return ResearchPillar(
            pillar=best_article["pillar"],
            query=search_phrase,
            title=best_article["title"],
            url=best_article["url"],
            notes=safe_notes,
            full_snippet=best_article["text"][:1200],
        )

    return None


def perform_5_pillar_research(topic: str, status_callback=None) -> List[ResearchPillar]:
    """
    Executes 5-Pillar Academic Research concurrently using ThreadPoolExecutor.
    Maintains polite pacing, error isolation, and canonical pillar ordering.
    """
    pillar_queries = get_5_pillar_search_queries(topic)
    discovered_urls = set()
    url_lock = threading.Lock()

    max_workers = min(5, len(pillar_queries))
    collected: List[Tuple[int, Dict[str, Any]]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _research_single_pillar,
                item,
                idx,
                topic,
                discovered_urls,
                url_lock,
                status_callback,
            ): idx
            for idx, item in enumerate(pillar_queries, 1)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result:
                    collected.append((idx, result))
            except Exception as e:
                print(f"[Pillar {idx} Research Error]: {e}")

    # Maintain canonical 1..5 pillar ordering
    collected.sort(key=lambda x: x[0])
    return [item for _, item in collected]


def perform_live_research(topic: str, max_pages: int = PAGE_LIMIT, status_callback=None) -> List[SearchResult]:
    """Legacy wrapper for Mode 1 quick Q&A web searches with concurrent page processing."""
    queries = [
        f"{topic} overview definition",
        f"{topic} key facts explanation",
    ]
    discovered_urls = set()
    sources_to_read = []

    for q in queries:
        hits = search_web(q, max_results=2)
        for hit in hits:
            if hit["url"] not in discovered_urls:
                discovered_urls.add(hit["url"])
                sources_to_read.append(hit)
                if len(sources_to_read) >= max_pages:
                    break
        if len(sources_to_read) >= max_pages:
            break

    if not sources_to_read:
        return []

    def _fetch_and_summarize(s: Dict[str, str], idx: int) -> Optional[Tuple[int, SearchResult]]:
        try:
            raw_text = fetch_article_text(s["url"])
            if raw_text and len(raw_text) > 100:
                prompt = (
                    f"Source: {s['title']} ({s['url']})\n\n"
                    f"Text:\n{raw_text}\n\n"
                    f"Extract 3 to 5 concise factual bullet points:"
                )
                try:
                    raw_notes, _ = execute_prompt(prompt, max_tokens=250)
                except Exception:
                    raw_notes = ""

                safe_notes = (raw_notes or "").strip()
                if not safe_notes:
                    safe_notes = raw_text[:250].strip()

                return idx, SearchResult(
                    title=s["title"],
                    url=s["url"],
                    notes=safe_notes,
                    full_snippet=raw_text[:1200],
                )
        except Exception as e:
            print(f"[Live Research Fetch Error]: {e}")
        return None

    results_indexed: List[Tuple[int, SearchResult]] = []
    with ThreadPoolExecutor(max_workers=min(max_pages, len(sources_to_read))) as executor:
        futures = [executor.submit(_fetch_and_summarize, s, idx) for idx, s in enumerate(sources_to_read)]
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    results_indexed.append(res)
            except Exception as e:
                print(f"[Live Fetch Thread Error]: {e}")

    results_indexed.sort(key=lambda x: x[0])
    return [item for _, item in results_indexed]
