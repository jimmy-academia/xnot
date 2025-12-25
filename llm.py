#!/usr/bin/env python3
"""Unified LLM API wrapper."""

import os
import json

PROVIDER = os.environ.get("LLM_PROVIDER", "").lower()
MODEL = os.environ.get("LLM_MODEL", "")
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))


def _detect_provider():
    """Auto-detect provider from API keys."""
    if PROVIDER:
        return PROVIDER, MODEL or ("gpt-4o-mini" if PROVIDER == "openai" else "claude-sonnet-4-20250514")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", MODEL or "claude-sonnet-4-20250514"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", MODEL or "gpt-4o"
    raise EnvironmentError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY")


def call_llm(prompt: str, system: str = "", provider: str = None, model: str = None) -> str:
    """Call LLM API. Auto-detects provider if not specified."""
    if provider is None or model is None:
        auto_provider, auto_model = _detect_provider()
        provider = provider or auto_provider
        model = model or auto_model

    if provider == "openai":
        import openai
        client = openai.OpenAI()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
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
