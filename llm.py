#!/usr/bin/env python3
"""Unified LLM API wrapper."""

import os
import json
import threading
from pathlib import Path

# Global rate limiter - limits concurrent API calls across all threads
_api_semaphore = None
_max_concurrent = 500  # Default, safe for OpenAI Tier 5


def init_rate_limiter(max_concurrent: int = 500):
    """Initialize global rate limiter. Call from main() before evaluation."""
    global _api_semaphore, _max_concurrent
    _max_concurrent = max_concurrent
    _api_semaphore = threading.Semaphore(max_concurrent)


def _get_semaphore():
    """Get or create the rate limiter semaphore."""
    global _api_semaphore
    if _api_semaphore is None:
        init_rate_limiter()
    return _api_semaphore


def _load_api_key():
    """Load API key from file if not in environment."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return
    # Try to load from ../.openaiapi
    key_file = Path(__file__).parent.parent / ".openaiapi"
    if key_file.exists():
        key = key_file.read_text().strip()
        os.environ["OPENAI_API_KEY"] = key


_load_api_key()

# Model configuration by role
MODEL_CONFIG = {
    "planner": "gpt-5-nano",   # For knowledge/script generation in knot
    "worker": "gpt-5-nano",    # For script execution in knot
    "default": "gpt-5-nano",   # For cot and other methods
}
DEFAULT_PROVIDER = "openai"

TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
# gpt-5-nano uses reasoning tokens, needs higher limit
MAX_TOKENS_REASONING = int(os.environ.get("LLM_MAX_TOKENS_REASONING", "4096"))


def get_model(role: str = "default") -> str:
    """Get model for a specific role."""
    return MODEL_CONFIG.get(role, MODEL_CONFIG["default"])


def call_llm(prompt: str, system: str = "", provider: str = None, model: str = None, role: str = "default") -> str:
    """Call LLM API. Uses role-based model selection if model not specified."""
    if provider is None:
        provider = DEFAULT_PROVIDER
    if model is None:
        model = get_model(role)

    # Rate limit: acquire semaphore before API call
    sem = _get_semaphore()
    with sem:
        return _call_llm_impl(prompt, system, provider, model)


def _call_llm_impl(prompt: str, system: str, provider: str, model: str) -> str:
    """Internal implementation of LLM call (called within semaphore)."""
    if provider == "openai":
        import openai
        client = openai.OpenAI()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # gpt-5-nano and o1/o3 models have different API requirements
        is_new_model = "gpt-5" in model or "o1" in model or "o3" in model
        if is_new_model:
            # These models use max_completion_tokens, only temperature=1, and need higher limit for reasoning tokens
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_completion_tokens=MAX_TOKENS_REASONING
            )
        else:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS
            )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        if "claude" not in model.lower():
            model = "claude-sonnet-4-20250514"
        resp = client.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            system=system or "", messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    elif provider == "local":
        import urllib.request
        base_url = os.environ.get("LLM_BASE_URL", "")
        url = base_url.rstrip("/") + "/v1/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages,
                   "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                      headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown provider: {provider}")


# Global async clients (reused to avoid cleanup issues)
_async_openai_client = None
_async_anthropic_client = None


def _get_async_openai_client():
    """Get or create async OpenAI client (reused across calls)."""
    global _async_openai_client
    if _async_openai_client is None:
        import openai
        _async_openai_client = openai.AsyncOpenAI()
    return _async_openai_client


def _get_async_anthropic_client():
    """Get or create async Anthropic client (reused across calls)."""
    global _async_anthropic_client
    if _async_anthropic_client is None:
        import anthropic
        _async_anthropic_client = anthropic.AsyncAnthropic()
    return _async_anthropic_client


async def call_llm_async(prompt: str, system: str = "", provider: str = None, model: str = None, role: str = "default") -> str:
    """Async version of call_llm for parallel execution."""
    if provider is None:
        provider = DEFAULT_PROVIDER
    if model is None:
        model = get_model(role)

    if provider == "openai":
        client = _get_async_openai_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        is_new_model = "gpt-5" in model or "o1" in model or "o3" in model
        if is_new_model:
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                max_completion_tokens=MAX_TOKENS_REASONING
            )
        else:
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS
            )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        client = _get_async_anthropic_client()
        if "claude" not in model.lower():
            model = "claude-sonnet-4-20250514"
        resp = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            system=system or "", messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    raise ValueError(f"Unknown provider for async: {provider}")
