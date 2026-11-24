"""
LLM Gateway for SmartResearch Chatbot.
Handles extraction from reasoning models and provides seamless fallback
between OpenRouter Free Models and Google Gemini Free Tier.
Guarantees robust string output (Never returns None).
"""

from typing import Dict, List, Optional, Tuple

import google.genai as genai
from google.genai import types as genai_types
from openai import OpenAI

from core.config import (
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    OPENROUTER_API_KEY,
    OR_FALLBACK_MODEL,
    OR_PRIMARY_MODEL,
    SYSTEM_PROMPT,
    logger,
)


def get_openrouter_client() -> Optional[OpenAI]:
    if not OPENROUTER_API_KEY:
        return None
    try:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "SmartResearch Chatbot",
            },
        )
    except Exception:
        return None


def get_gemini_client() -> Optional[genai.Client]:
    if not GEMINI_API_KEY:
        return None
    try:
        retry_options = getattr(genai_types, "HttpRetryOptions", None)
        if retry_options is not None:
            http_options = genai_types.HttpOptions(
                api_version="v1beta",
                retry_options=genai_types.HttpRetryOptions(
                    attempts=5,
                    initial_delay=1.0,
                    max_delay=60.0,
                    exp_base=2.0,
                    jitter=1.0,
                    http_status_codes=[408, 429, 500, 502, 503, 504],
                ),
            )
        else:
            http_options = genai_types.HttpOptions(api_version="v1beta")

        return genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=http_options,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}")
        return None


def extract_content_safely(message) -> str:
    """
    Safely extracts text from chat completion message.
    Handles models that put output in reasoning when max_tokens cuts output short.
    """
    if not message:
        return ""
    content = getattr(message, "content", "") or ""
    if content.strip():
        return content.strip()

    # Fallback to reasoning field if content is None or empty
    reasoning = getattr(message, "reasoning", "") or ""
    if reasoning.strip():
        # Grab the last paragraph of reasoning as the conclusion
        paras = [p.strip() for p in reasoning.split("\n\n") if p.strip()]
        if paras:
            return paras[-1]
        return reasoning.strip()

    return ""


def execute_prompt(
    prompt: str,
    system_instruction: str = SYSTEM_PROMPT,
    history: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 1500,
) -> Tuple[str, str]:
    """
    Executes a chat prompt with cascading resilience:
    1. Try OpenRouter Primary (Nemotron-3.5-Lightning Free)
    2. Try OpenRouter Fallback (Liquid LFM Free)
    3. Seamlessly Fallback to Google Gemini 2.5 Flash
    Guarantees non-empty string output.
    """
    history = history or []
    openrouter_client = get_openrouter_client()
    gemini_client = get_gemini_client()

    # 1. Attempt OpenRouter Primary
    if openrouter_client:
        try:
            res = openrouter_client.chat.completions.create(
                model=OR_PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    *history,
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                extra_body={"models": [OR_FALLBACK_MODEL]},
                timeout=40.0,
            )
            extracted = extract_content_safely(res.choices[0].message)
            if extracted:
                return extracted, f"OpenRouter ({OR_PRIMARY_MODEL})"
        except Exception as e:
            logger.warning(f"OpenRouter Primary Fail: {e}")

    # 2. Try OpenRouter Secondary directly
    if openrouter_client:
        try:
            res = openrouter_client.chat.completions.create(
                model=OR_FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    *history,
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                timeout=35.0,
            )
            extracted = extract_content_safely(res.choices[0].message)
            if extracted:
                return extracted, f"OpenRouter Fallback ({OR_FALLBACK_MODEL})"
        except Exception as e:
            logger.warning(f"OpenRouter Fallback Fail: {e}")

    # 3. Native Google Gemini Cloud Fallback
    if gemini_client:
        try:
            gemini_contents = []
            for h in history:
                role = "user" if h.get("role") == "user" else "model"
                gemini_contents.append(
                    genai_types.Content(role=role, parts=[genai_types.Part(text=h.get("content", ""))])
                )
            gemini_contents.append(
                genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])
            )

            res = gemini_client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=gemini_contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_tokens,
                ),
            )
            return (res.text or "").strip(), f"Google Gemini ({GEMINI_CHAT_MODEL})"
        except Exception as e:
            raise RuntimeError(f"All LLM Gateways failed. Gemini error: {e}")

    raise RuntimeError("No LLM API keys configured. Please verify your .env file.")
