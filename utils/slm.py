#!/usr/bin/env python3
"""
utils/slm.py
Small Language Model (SLM) support via Ollama.

Provides local inference for open-source models with:
- True async parallel inference via Ollama's batching
- Automatic GPU compatibility handling
- Usage tracking compatible with existing framework

Prerequisites:
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh

    # Start server (if not running via systemd)
    ollama serve

    # Pull a model
    ollama pull qwen2.5:0.5b
"""

import os
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# -----------------------------
# Model Registry
# -----------------------------

# Map of short names to Ollama model names
# Use these short names with --model flag: e.g., --model qwen-0.5b
SLM_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Qwen 2.5 family
    "qwen-0.5b": {
        "ollama_name": "qwen2.5:0.5b",
        "context_limit": 32768,
    },
    "qwen-1.5b": {
        "ollama_name": "qwen2.5:1.5b",
        "context_limit": 32768,
    },
    "qwen-3b": {
        "ollama_name": "qwen2.5:3b",
        "context_limit": 32768,
    },
    "qwen-7b": {
        "ollama_name": "qwen2.5:7b",
        "context_limit": 32768,
    },
    # Microsoft Phi family
    "phi-3-mini": {
        "ollama_name": "phi3:mini",
        "context_limit": 4096,
    },
    "phi-3.5-mini": {
        "ollama_name": "phi3.5:latest",
        "context_limit": 128000,
    },
    # Meta Llama 3.2 family
    "llama-1b": {
        "ollama_name": "llama3.2:1b",
        "context_limit": 128000,
    },
    "llama-3b": {
        "ollama_name": "llama3.2:3b",
        "context_limit": 128000,
    },
    "llama-8b": {
        "ollama_name": "llama3.1:8b",
        "context_limit": 128000,
    },
    # Google Gemma
    "gemma-2b": {
        "ollama_name": "gemma2:2b",
        "context_limit": 8192,
    },
    # Mistral
    "mistral-7b": {
        "ollama_name": "mistral:7b",
        "context_limit": 32768,
    },
    # TinyLlama
    "tinyllama": {
        "ollama_name": "tinyllama:latest",
        "context_limit": 2048,
    },
    # SmolLM
    "smollm-1.7b": {
        "ollama_name": "smollm2:1.7b",
        "context_limit": 8192,
    },
}


def get_slm_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model info by short name or Ollama name."""
    if model_name in SLM_REGISTRY:
        return SLM_REGISTRY[model_name]
    # Check if it's already an Ollama name
    for short_name, info in SLM_REGISTRY.items():
        if model_name == info["ollama_name"]:
            return info
    # Partial match
    for short_name, info in SLM_REGISTRY.items():
        if model_name.lower() in short_name:
            return info
    return None


def is_slm_model(model_name: str) -> bool:
    """Check if model name refers to a registered SLM."""
    return get_slm_info(model_name) is not None


def list_slm_models() -> List[str]:
    """List all available SLM short names."""
    return list(SLM_REGISTRY.keys())


def get_ollama_name(model_name: str) -> str:
    """Get Ollama model name from short name."""
    info = get_slm_info(model_name)
    if info:
        return info["ollama_name"]
    # Assume it's already an Ollama name
    return model_name


# -----------------------------
# SLM Service (Ollama Backend)
# -----------------------------

class SLMService:
    """
    Service for managing Small Language Model inference via Ollama.

    Features:
    - True async parallel inference (Ollama handles batching)
    - Automatic GPU compatibility
    - Simple HTTP API
    """

    def __init__(self):
        self._config = {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "max_new_tokens": 2048,
            "temperature": 0.0,
            "max_concurrent": 16,  # Ollama's default OLLAMA_NUM_PARALLEL
            "timeout": 120.0,
        }
        self._http_client = None
        self._http_client_loop = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop = None

    def configure(self, **kwargs):
        """Update configuration."""
        # Map max_new_tokens from old interface
        if "max_new_tokens" in kwargs and kwargs["max_new_tokens"] is not None:
            self._config["max_new_tokens"] = kwargs["max_new_tokens"]
        for k, v in kwargs.items():
            if v is not None and k in self._config:
                self._config[k] = v
        # Reset client on config change
        self._http_client = None
        self._semaphore = None
        self._semaphore_loop = None  # Track which loop the semaphore belongs to
        self._http_client_loop = None  # Track which loop the client belongs to

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create async semaphore for rate limiting (per event loop)."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Create new semaphore if loop changed or doesn't exist
        if self._semaphore is None or self._semaphore_loop != current_loop:
            self._semaphore = asyncio.Semaphore(self._config["max_concurrent"])
            self._semaphore_loop = current_loop
        return self._semaphore

    async def _get_http_client(self):
        """Get or create httpx async client (per event loop)."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Create new client if loop changed or doesn't exist
        if self._http_client is None or self._http_client_loop != current_loop:
            # Close old client if exists
            if self._http_client is not None:
                try:
                    await self._http_client.aclose()
                except Exception:
                    pass
            import httpx
            self._http_client = httpx.AsyncClient(
                base_url=self._config["base_url"],
                timeout=self._config["timeout"],
            )
            self._http_client_loop = current_loop
        return self._http_client

    def call_sync(
        self,
        prompt: str,
        system: str = "",
        model: str = "qwen-0.5b",
        context: dict = None,
    ) -> str:
        """
        Synchronous inference call via Ollama.

        Args:
            prompt: User prompt text
            system: Optional system prompt
            model: Model short name or Ollama name
            context: Optional context dict for usage tracking

        Returns:
            Generated text response
        """
        import httpx

        ollama_model = get_ollama_name(model)

        # Get context limit for model
        model_info = get_slm_info(model)
        num_ctx = model_info.get("context_limit", 4096) if model_info else 4096

        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._config["temperature"],
                "num_predict": self._config["max_new_tokens"],
                "num_ctx": num_ctx,  # Set context window
            },
        }

        if system:
            payload["system"] = system

        start_time = time.time()

        try:
            with httpx.Client(base_url=self._config["base_url"], timeout=self._config["timeout"]) as client:
                resp = client.post("/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._config['base_url']}. "
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

        latency_ms = (time.time() - start_time) * 1000

        text = data.get("response", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        # Record usage
        self._record_usage(
            model=ollama_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            context=context,
            prompt=prompt,
            response=text,
        )

        return text

    async def call_async(
        self,
        prompt: str,
        system: str = "",
        model: str = "qwen-0.5b",
        context: dict = None,
        return_usage: bool = False,
    ):
        """
        Async inference call via Ollama (true parallel).

        Args:
            prompt: User prompt text
            system: Optional system prompt
            model: Model short name or Ollama name
            context: Optional context dict for usage tracking
            return_usage: If True, return dict with text and token counts

        Returns:
            Generated text, or dict if return_usage=True
        """
        ollama_model = get_ollama_name(model)

        sem = self._get_semaphore()
        async with sem:
            start_time = time.time()

            client = await self._get_http_client()

            # Get context limit for model
            model_info = get_slm_info(model)
            num_ctx = model_info.get("context_limit", 4096) if model_info else 4096

            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self._config["temperature"],
                    "num_predict": self._config["max_new_tokens"],
                    "num_ctx": num_ctx,  # Set context window
                },
            }

            if system:
                payload["system"] = system

            try:
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                # Check if it's a connection error
                if "connect" in str(e).lower():
                    raise RuntimeError(
                        f"Cannot connect to Ollama at {self._config['base_url']}. "
                        "Make sure Ollama is running: ollama serve"
                    )
                logger.error(f"Ollama API error: {e}")
                raise

            latency_ms = (time.time() - start_time) * 1000

            text = data.get("response", "")
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)

            # Record usage
            self._record_usage(
                model=ollama_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                context=context,
                prompt=prompt,
                response=text,
            )

            if return_usage:
                return {
                    "text": text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            return text

    def _record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        context: dict,
        prompt: str,
        response: str,
    ):
        """Record usage to tracker if available."""
        try:
            from utils.usage import get_usage_tracker
            tracker = get_usage_tracker()
            tracker.record(
                model=model,
                provider="slm",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                context=context,
                prompt_preview=prompt[:200] if prompt else None,
                response_preview=response[:200] if response else None,
            )
        except Exception:
            pass  # Usage tracking is optional

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._config['base_url']}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """List models available on the Ollama server."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._config['base_url']}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
        return []

    def get_token_budget(self, model_name: str) -> int:
        """Get context limit for a model."""
        info = get_slm_info(model_name)
        if info is None:
            return 4096
        return info.get("context_limit", 4096) - self._config["max_new_tokens"] - 500

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# -----------------------------
# Singleton Instance
# -----------------------------

_service: Optional[SLMService] = None


def get_slm_service() -> SLMService:
    """Get or create the SLM service singleton."""
    global _service
    if _service is None:
        _service = SLMService()
    return _service


# -----------------------------
# Public API (matches llm.py interface)
# -----------------------------

def call_slm(
    prompt: str,
    system: str = "",
    model: str = "qwen-0.5b",
    context: dict = None,
) -> str:
    """Synchronous SLM call."""
    return get_slm_service().call_sync(prompt, system, model, context)


async def call_slm_async(
    prompt: str,
    system: str = "",
    model: str = "qwen-0.5b",
    context: dict = None,
    return_usage: bool = False,
):
    """Async SLM call."""
    return await get_slm_service().call_async(prompt, system, model, context, return_usage)


def configure_slm(**kwargs):
    """Configure SLM service."""
    get_slm_service().configure(**kwargs)


def get_slm_token_budget(model: str = "qwen-0.5b") -> int:
    """Get token budget for SLM model."""
    return get_slm_service().get_token_budget(model)


# -----------------------------
# Ollama Startup Checks & Automation
# -----------------------------

# Default parallelism setting for auto-started server
DEFAULT_NUM_PARALLEL = 64


def check_ollama_installed() -> bool:
    """Check if ollama command is available."""
    import shutil
    return shutil.which("ollama") is not None


def check_ollama_server() -> bool:
    """Check if Ollama server is running (sync version for startup)."""
    import httpx
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def detect_gpu() -> dict:
    """Detect GPU environment for optimal Ollama configuration."""
    import shutil
    import subprocess

    info = {"has_nvidia": False, "gpu_name": None, "gpu_memory_gb": None}

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]
                parts = line.split(", ")
                info["has_nvidia"] = True
                info["gpu_name"] = parts[0] if len(parts) > 0 else None
                if len(parts) > 1:
                    try:
                        info["gpu_memory_gb"] = int(parts[1]) // 1024
                    except ValueError:
                        pass
        except Exception:
            pass

    return info


def get_available_models() -> List[str]:
    """Get list of models currently available on Ollama server."""
    import httpx
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def start_ollama_server(num_parallel: int = DEFAULT_NUM_PARALLEL, wait_timeout: float = 30.0) -> bool:
    """
    Start Ollama server as a background process with specified parallelism.

    Args:
        num_parallel: OLLAMA_NUM_PARALLEL value for concurrent request handling
        wait_timeout: Seconds to wait for server to become ready

    Returns:
        True if server started successfully, False otherwise
    """
    import subprocess
    import shutil

    if not shutil.which("ollama"):
        logger.error("Ollama is not installed")
        return False

    # Check if already running
    if check_ollama_server():
        logger.info("Ollama server is already running")
        return True

    print(f"[SLM] Starting Ollama server with OLLAMA_NUM_PARALLEL={num_parallel}...")

    # Set up environment
    env = os.environ.copy()
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)

    # Start server as background process
    try:
        # Use Popen to start in background without waiting
        subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent process
        )
    except Exception as e:
        logger.error(f"Failed to start Ollama server: {e}")
        return False

    # Wait for server to become ready
    start_time = time.time()
    while time.time() - start_time < wait_timeout:
        if check_ollama_server():
            print(f"[SLM] Ollama server started successfully (OLLAMA_NUM_PARALLEL={num_parallel})")
            return True
        time.sleep(0.5)

    logger.error(f"Ollama server did not become ready within {wait_timeout}s")
    return False


def pull_model(model_name: str, timeout: float = 600.0) -> bool:
    """
    Pull a model if not already available.

    Args:
        model_name: Short name (e.g., "qwen-3b") or Ollama name (e.g., "qwen2.5:3b")
        timeout: Maximum time to wait for pull (models can be large)

    Returns:
        True if model is available (already present or pulled), False on failure
    """
    import subprocess

    ollama_name = get_ollama_name(model_name)

    # Check if already available
    available = get_available_models()
    # Check for exact match or base name match (e.g., "qwen2.5:3b" matches "qwen2.5:3b")
    for avail in available:
        if avail == ollama_name or avail.split(":")[0] == ollama_name.split(":")[0]:
            if avail == ollama_name:
                logger.info(f"Model {ollama_name} is already available")
                return True

    # Need to pull
    print(f"[SLM] Pulling model {ollama_name}... (this may take a few minutes)")

    try:
        result = subprocess.run(
            ["ollama", "pull", ollama_name],
            capture_output=False,  # Show progress to user
            timeout=timeout,
        )
        if result.returncode == 0:
            print(f"[SLM] Model {ollama_name} pulled successfully")
            return True
        else:
            logger.error(f"Failed to pull model {ollama_name}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Model pull timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        return False


def ensure_ollama_ready(
    model: str = None,
    num_parallel: int = DEFAULT_NUM_PARALLEL,
    auto_start: bool = True,
    auto_pull: bool = True,
) -> bool:
    """
    Ensure Ollama is running and model is available.

    This is the main entry point for automation. Call this before using SLM:

        from utils.slm import ensure_ollama_ready
        ensure_ollama_ready(model="qwen-3b", num_parallel=128)

    Args:
        model: Model to ensure is available (short name or Ollama name)
        num_parallel: OLLAMA_NUM_PARALLEL for auto-started server
        auto_start: If True, start server if not running
        auto_pull: If True, pull model if not available

    Returns:
        True if ready, False on failure
    """
    import sys

    # Step 1: Check/install Ollama
    if not check_ollama_installed():
        print("\n" + "=" * 60)
        print("ERROR: Ollama is not installed")
        print("=" * 60)
        print("\nTo install Ollama, run:")
        print()
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        print()
        print("Or visit: https://ollama.com/download")
        print("=" * 60 + "\n")
        return False

    # Step 2: Check/start server
    if not check_ollama_server():
        if auto_start:
            if not start_ollama_server(num_parallel=num_parallel):
                print("\n" + "=" * 60)
                print("ERROR: Failed to start Ollama server")
                print("=" * 60)
                print("\nTry starting manually:")
                print()
                print(f"  OLLAMA_NUM_PARALLEL={num_parallel} ollama serve")
                print("=" * 60 + "\n")
                return False
        else:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            print("\n" + "=" * 60)
            print("ERROR: Ollama server is not running")
            print("=" * 60)
            print(f"\nCannot connect to Ollama at: {base_url}")
            print("\nTo start the server, run:")
            print()
            print(f"  OLLAMA_NUM_PARALLEL={num_parallel} ollama serve")
            print("=" * 60 + "\n")
            return False

    # Step 3: Check/pull model
    if model:
        ollama_name = get_ollama_name(model)
        available = get_available_models()

        # Check if model is available
        model_present = any(
            avail == ollama_name or avail.startswith(ollama_name.split(":")[0] + ":")
            for avail in available
        ) if available else False

        if not model_present:
            if auto_pull:
                if not pull_model(model):
                    print("\n" + "=" * 60)
                    print(f"ERROR: Failed to pull model {ollama_name}")
                    print("=" * 60)
                    print("\nTry pulling manually:")
                    print()
                    print(f"  ollama pull {ollama_name}")
                    print("=" * 60 + "\n")
                    return False
            else:
                print("\n" + "=" * 60)
                print(f"ERROR: Model {ollama_name} is not available")
                print("=" * 60)
                print("\nAvailable models:", available if available else "(none)")
                print("\nTo pull the model, run:")
                print()
                print(f"  ollama pull {ollama_name}")
                print("=" * 60 + "\n")
                return False

    return True


def check_ollama_or_exit(
    model: str = None,
    num_parallel: int = DEFAULT_NUM_PARALLEL,
    auto_start: bool = True,
    auto_pull: bool = True,
):
    """
    Ensure Ollama is ready, exit if not.

    This is a convenience wrapper that exits the program on failure:

        from utils.slm import check_ollama_or_exit
        check_ollama_or_exit(model="qwen-3b")  # Auto-starts server & pulls model

    Args:
        model: Model to ensure is available
        num_parallel: OLLAMA_NUM_PARALLEL for auto-started server
        auto_start: If True, start server if not running
        auto_pull: If True, pull model if not available
    """
    import sys

    if not ensure_ollama_ready(
        model=model,
        num_parallel=num_parallel,
        auto_start=auto_start,
        auto_pull=auto_pull,
    ):
        sys.exit(1)


# -----------------------------
# CLI for testing
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLM (Ollama) Provider Test")
    parser.add_argument("--model", "-m", default="qwen-0.5b", help="Model to use")
    parser.add_argument("--test", action="store_true", help="Run parallel test")
    parser.add_argument("--health", action="store_true", help="Check server health")
    parser.add_argument("--setup", action="store_true", help="Auto-start server and pull model")
    parser.add_argument("--parallel", "-p", type=int, default=64, help="OLLAMA_NUM_PARALLEL value")
    parser.add_argument("--requests", "-n", type=int, default=8, help="Number of parallel requests")
    args = parser.parse_args()

    if args.setup:
        # Automation demo: auto-start server and pull model
        print(f"Ensuring Ollama is ready with model={args.model}, num_parallel={args.parallel}...")
        check_ollama_or_exit(model=args.model, num_parallel=args.parallel)
        print("\nOllama is ready!")
        print(f"  Server: running with OLLAMA_NUM_PARALLEL={args.parallel}")
        print(f"  Model:  {get_ollama_name(args.model)} available")

    elif args.health:
        async def check():
            service = get_slm_service()
            healthy = await service.health_check()
            if healthy:
                print("Ollama server is running")
                models = await service.list_models()
                print(f"Available models: {models}")
            else:
                print("Ollama server is NOT reachable")
                print("Run: ollama serve")
        asyncio.run(check())

    elif args.test:
        async def test():
            service = get_slm_service()

            if not await service.health_check():
                print("ERROR: Ollama server not reachable")
                print("Run: ollama serve")
                return

            print(f"Testing parallel inference with {args.model}...")
            print(f"Requests: {args.requests}")

            prompts = [
                "What is 2+2? Answer with just the number.",
                "What is the capital of France? One word answer.",
                "Is water wet? Answer yes or no.",
                "What color is the sky? One word.",
            ] * (args.requests // 4 + 1)
            prompts = prompts[:args.requests]

            # Sequential test
            print(f"\nRunning {args.requests} requests SEQUENTIALLY...")
            seq_start = time.perf_counter()
            for i, p in enumerate(prompts):
                result = await service.call_async(p, model=args.model)
                print(f"  {i+1}/{args.requests}: {result[:40]}...")
            seq_time = time.perf_counter() - seq_start
            print(f"Sequential time: {seq_time:.2f}s")

            # Parallel test
            print(f"\nRunning {args.requests} requests IN PARALLEL...")
            par_start = time.perf_counter()
            tasks = [service.call_async(p, model=args.model) for p in prompts]
            results = await asyncio.gather(*tasks)
            par_time = time.perf_counter() - par_start

            for i, result in enumerate(results):
                print(f"  {i+1}/{args.requests}: {result[:40]}...")
            print(f"Parallel time: {par_time:.2f}s")

            # Results
            speedup = seq_time / par_time if par_time > 0 else 0
            print(f"\n{'='*50}")
            print(f"Sequential: {seq_time:.2f}s")
            print(f"Parallel:   {par_time:.2f}s")
            print(f"Speedup:    {speedup:.1f}x")

            if speedup > 1.5:
                print("\nPASSED: Parallelism is working!")
            else:
                print("\nCheck OLLAMA_NUM_PARALLEL setting for better parallelism")

        asyncio.run(test())

    else:
        # Quick test
        print("Quick test with qwen-0.5b...")
        try:
            result = call_slm("What is 2+2? Answer briefly.", model=args.model)
            print(f"Response: {result}")
        except Exception as e:
            print(f"Error: {e}")
            print("\nMake sure Ollama is running: ollama serve")
