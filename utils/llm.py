#!/usr/bin/env python3
"""
utils/llm.py
Unified LLM API wrapper.

Refactored to use LLMService class for better state management.
"""

import os
import json
import time
import random
import logging
import threading
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import httpx
# External dependencies (optional)
try:
    import openai
except ImportError:
    openai = None
try:
    import anthropic
except ImportError:
    anthropic = None

from utils.usage import get_usage_tracker

# Suppress verbose HTTP client logs
for _logger_name in ["httpx", "httpcore", "urllib3", "openai._base_client"]:
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


# -----------------------------
# Configuration & Constants
# -----------------------------

MODEL_CONFIG = {
    "planner": "gpt-5-nano",
    "worker": "gpt-5-nano",
    "default": "gpt-5-nano",
}

MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
}

DEFAULT_CONTEXT_LIMIT = 128000

MODEL_INPUT_LIMITS = {
    "gpt-5-nano": 270000,
}


def _load_api_key():
    """Load API key from file if not in environment."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return
    key_file = Path(__file__).parent.parent.parent / ".openaiapi"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key

_load_api_key()


class LLMService:
    """Singleton service to manage LLM configuration, rate limiting, and clients."""

    def __init__(self):
        self._config = {
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_tokens_reasoning": 4096,
            "provider": "openai",
            "model": None,
            "base_url": "",
            "request_timeout": 90.0,
            "max_retries": 6,
        }
        
        # Rate limiting
        self._max_concurrent = 200
        self._api_semaphore: Optional[threading.Semaphore] = None
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        
        # Client caching
        self._clients = {}
        self._client_locks = defaultdict(threading.Lock)

    def init_rate_limiter(self, max_concurrent: int = 200):
        """Initialize or update rate limiter."""
        self._max_concurrent = int(max_concurrent)
        self._api_semaphore = threading.Semaphore(self._max_concurrent)
        self._async_semaphore = None  # Will be recreated lazily

    def _get_semaphore(self) -> threading.Semaphore:
        if self._api_semaphore is None:
            self.init_rate_limiter(self._max_concurrent)
        return self._api_semaphore

    def _get_async_semaphore(self) -> asyncio.Semaphore:
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._async_semaphore

    def configure(self, **kwargs):
        """Update configuration."""
        for k, v in kwargs.items():
            if v is not None:
                if k in ["max_tokens", "max_tokens_reasoning", "max_retries"]:
                    self._config[k] = int(v)
                elif k in ["temperature", "request_timeout"]:
                    self._config[k] = float(v)
                else:
                    self._config[k] = str(v) if k != "base_url" else str(v).strip()

    def get_model(self, role: str = "default") -> str:
        if self._config["model"]:
            return self._config["model"]
        return MODEL_CONFIG.get(role, MODEL_CONFIG["default"])

    def get_token_budget(self, model: str = None, output_reserve: int = 2000, safety_pct: float = 0.05) -> int:
        if model is None:
            model = self.get_model()
        
        if model in MODEL_INPUT_LIMITS:
            return MODEL_INPUT_LIMITS[model]
        
        limit = MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)
        safety_margin = int(limit * safety_pct)
        return limit - output_reserve - safety_margin

    # -------------------------------------------------------------------------
    # Client Management
    # -------------------------------------------------------------------------

    def _get_client(self, provider: str, is_async: bool):
        key = (provider, is_async)
        if key in self._clients:
            return self._clients[key]

        with self._client_locks[key]:
            if key in self._clients:
                return self._clients[key]

            timeout = self._config["request_timeout"]
            limits = httpx.Limits(
                max_connections=self._max_concurrent, 
                max_keepalive_connections=min(50, self._max_concurrent)
            )
            base_url = self._config.get("base_url")

            if provider == "openai":
                if not openai:
                    raise ImportError("openai package not installed")
                
                if is_async:
                    http_client = httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=True)
                    client = openai.AsyncOpenAI(http_client=http_client, base_url=base_url if base_url else None)
                else:
                    http_client = httpx.Client(timeout=timeout, limits=limits, trust_env=True)
                    client = openai.OpenAI(http_client=http_client, base_url=base_url if base_url else None)

            elif provider == "anthropic":
                if not anthropic:
                    raise ImportError("anthropic package not installed")
                if is_async:
                    client = anthropic.AsyncAnthropic()
                else:
                    client = anthropic.Anthropic()
            
            else:
                raise ValueError(f"Unknown provider: {provider}")

            self._clients[key] = client
            return client

    # -------------------------------------------------------------------------
    # Execution Logic
    # -------------------------------------------------------------------------

    def _should_retry(self, e: Exception) -> bool:
        name = e.__class__.__name__
        if name in {
            "APIConnectionError", "APITimeoutError", "RateLimitError",
            "InternalServerError", "ServiceUnavailableError"
        }:
            return True
        status = getattr(e, "status_code", None)
        if status in {429, 500, 502, 503, 504}:
            return True
        return False

    def _retry_delay(self, attempt: int) -> float:
        base = 1.0 * (2 ** attempt)
        return min(30.0, base + random.random())

    # --- Sync Calls ---

    def call_sync(self, prompt: str, system: str = "", provider: str = None, 
                  model: str = None, role: str = "default", context: dict = None) -> str:
        provider = provider or self._config["provider"]
        model = model or self.get_model(role)
        
        sem = self._get_semaphore()
        with sem:
            if provider == "local":
                return self._call_local(prompt, system, model)
            elif provider == "openai":
                return self._call_openai_sync(prompt, system, model, context)
            elif provider == "anthropic":
                return self._call_anthropic_sync(prompt, system, model, context)
            else:
                raise ValueError(f"Unknown provider: {provider}")

    def _call_openai_sync(self, prompt: str, system: str, model: str, context: dict) -> str:
        client = self._get_client("openai", False)
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        is_new_model = any(x in model for x in ["gpt-5", "o1", "o3"])

        last_err = None
        for attempt in range(max(1, self._config["max_retries"])):
            try:
                start_time = time.time()
                if is_new_model:
                    resp = client.chat.completions.create(
                        model=model, messages=messages,
                        max_completion_tokens=self._config["max_tokens_reasoning"]
                    )
                else:
                    resp = client.chat.completions.create(
                        model=model, messages=messages,
                        temperature=self._config["temperature"],
                        max_tokens=self._config["max_tokens"]
                    )
                
                latency_ms = (time.time() - start_time) * 1000
                text = resp.choices[0].message.content or ""
                
                if resp.usage:
                    self._record_usage(
                        model, "openai", resp.usage.prompt_tokens, 
                        resp.usage.completion_tokens, latency_ms, context, prompt, text
                    )
                return text

            except Exception as e:
                last_err = e
                if self._should_retry(e) and attempt < self._config["max_retries"] - 1:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise
        raise last_err

    def _call_anthropic_sync(self, prompt: str, system: str, model: str, context: dict) -> str:
        client = self._get_client("anthropic", False)
        if "claude" not in model.lower():
            model = "claude-sonnet-4-20250514"
            
        start_time = time.time()
        resp = client.messages.create(
            model=model, max_tokens=self._config["max_tokens"],
            system=system or "", messages=[{"role": "user", "content": prompt}]
        )
        latency_ms = (time.time() - start_time) * 1000
        text = resp.content[0].text
        
        if resp.usage:
            self._record_usage(
                model, "anthropic", resp.usage.input_tokens, 
                resp.usage.output_tokens, latency_ms, context, prompt, text
            )
        return text

    def _call_local(self, prompt: str, system: str, model: str) -> str:
        import urllib.request
        base_url = self._config["base_url"].strip()
        if not base_url:
            raise ValueError("provider='local' requires base_url")
            
        url = base_url.rstrip("/") + "/v1/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
            
        payload = {
            "model": model, "messages": messages,
            "temperature": self._config["temperature"],
            "max_tokens": self._config["max_tokens"]
        }
        
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=int(self._config["request_timeout"])) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]

    # --- Async Calls ---

    async def call_async(self, prompt: str, system: str = "", provider: str = None, 
                         model: str = None, role: str = "default", context: dict = None,
                         return_usage: bool = False):
        provider = provider or self._config["provider"]
        model = model or self.get_model(role)
        
        sem = self._get_async_semaphore()
        async with sem:
            if provider == "openai":
                result = await self._call_openai_async(prompt, system, model, context)
            elif provider == "anthropic":
                result = await self._call_anthropic_async(prompt, system, model, context)
            else:
                # Fallback to sync for local/unknown
                text = self.call_sync(prompt, system, provider, model, role, context)
                result = (text, 0, 0)
                
            text, pt, ct = result
            if return_usage:
                return {"text": text, "prompt_tokens": pt, "completion_tokens": ct}
            return text

    async def _call_openai_async(self, prompt: str, system: str, model: str, context: dict) -> Tuple[str, int, int]:
        client = self._get_client("openai", True)
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        is_new_model = any(x in model for x in ["gpt-5", "o1", "o3"])
        last_err = None
        
        for attempt in range(max(1, self._config["max_retries"])):
            try:
                start_time = time.time()
                if is_new_model:
                    resp = await client.chat.completions.create(
                        model=model, messages=messages,
                        max_completion_tokens=self._config["max_tokens_reasoning"]
                    )
                else:
                    resp = await client.chat.completions.create(
                        model=model, messages=messages,
                        temperature=self._config["temperature"],
                        max_tokens=self._config["max_tokens"]
                    )
                
                latency_ms = (time.time() - start_time) * 1000
                text = resp.choices[0].message.content or ""
                
                pt, ct = 0, 0
                if resp.usage:
                    pt, ct = resp.usage.prompt_tokens, resp.usage.completion_tokens
                    self._record_usage(model, "openai", pt, ct, latency_ms, context, prompt, text)
                    
                return text, pt, ct

            except Exception as e:
                last_err = e
                if self._should_retry(e) and attempt < self._config["max_retries"] - 1:
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise
        raise last_err

    async def _call_anthropic_async(self, prompt: str, system: str, model: str, context: dict) -> Tuple[str, int, int]:
        client = self._get_client("anthropic", True)
        if "claude" not in model.lower():
            model = "claude-sonnet-4-20250514"
            
        start_time = time.time()
        resp = await client.messages.create(
            model=model, max_tokens=self._config["max_tokens"],
            system=system or "", messages=[{"role": "user", "content": prompt}]
        )
        latency_ms = (time.time() - start_time) * 1000
        text = resp.content[0].text
        
        pt, ct = 0, 0
        if resp.usage:
            pt, ct = resp.usage.input_tokens, resp.usage.output_tokens
            self._record_usage(model, "anthropic", pt, ct, latency_ms, context, prompt, text)
            
        return text, pt, ct

    def _record_usage(self, model, provider, prompt_tokens, completion_tokens, latency_ms, context, prompt, response):
        tracker = get_usage_tracker()
        tracker.record(
            model=model, provider=provider,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            latency_ms=latency_ms, context=context,
            prompt_preview=prompt[:200] if prompt else None,
            response_preview=response[:200] if response else None,
        )

from collections import defaultdict

# -----------------------------
# Public API Facade (Backward Compatibility)
# -----------------------------

_service = LLMService()

def init_rate_limiter(max_concurrent: int = 200):
    _service.init_rate_limiter(max_concurrent)

def configure(**kwargs):
    _service.configure(**kwargs)

def get_token_budget(model: str = None, output_reserve: int = 2000, safety_pct: float = 0.05) -> int:
    return _service.get_token_budget(model, output_reserve, safety_pct)

def get_model(role: str = "default") -> str:
    return _service.get_model(role)

def get_configured_model() -> str:
    return _service.get_model()

def call_llm(prompt: str, system: str = "", provider: str = None, 
             model: str = None, role: str = "default", context: dict = None) -> str:
    return _service.call_sync(prompt, system, provider, model, role, context)

async def call_llm_async(prompt: str, system: str = "", provider: str = None, 
                         model: str = None, role: str = "default", context: dict = None, 
                         return_usage: bool = False):
    return await _service.call_async(prompt, system, provider, model, role, context, return_usage)

def config_llm(args):
    """Configure LLM settings from args (if present)."""
    max_conc = getattr(args, "max_concurrent", None)
    if max_conc is not None:
        init_rate_limiter(max_conc)
    
    configure(
        temperature=getattr(args, "temperature", None),
        max_tokens=getattr(args, "max_tokens", None),
        max_tokens_reasoning=getattr(args, "max_tokens_reasoning", None),
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        request_timeout=getattr(args, "request_timeout", None),
        max_retries=getattr(args, "max_retries", None),
    )
