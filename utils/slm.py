#!/usr/bin/env python3
"""
utils/slm.py
Small Language Model (SLM) support via Hugging Face Transformers.

Provides local inference for open-source models with:
- Lazy model loading (first call triggers download/load)
- Apple Silicon (MPS) acceleration when available
- Async support via thread pool
- Usage tracking compatible with existing framework
"""

import os
import time
import logging
import threading
import asyncio
from typing import Optional, Dict, Any, Tuple, List
from functools import partial
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Suppress transformers verbosity
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Suppress HuggingFace hub file lock debug messages
for _hf_logger in ["huggingface_hub", "filelock"]:
    logging.getLogger(_hf_logger).setLevel(logging.WARNING)

# -----------------------------
# Model Registry
# -----------------------------

# Map of short names to HuggingFace repo IDs
# Use these short names with --model flag: e.g., --model qwen-0.5b
SLM_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Qwen 2.5 family
    "qwen-0.5b": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "context_limit": 32768,
        "template": "qwen",
    },
    "qwen-1.5b": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "context_limit": 32768,
        "template": "qwen",
    },
    "qwen-3b": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "context_limit": 32768,
        "template": "qwen",
    },
    # Microsoft Phi family
    "phi-3-mini": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct",
        "context_limit": 4096,
        "template": "phi",
    },
    "phi-3.5-mini": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "context_limit": 128000,
        "template": "phi",
    },
    # Meta Llama 3.2 family
    "llama-1b": {
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "context_limit": 128000,
        "template": "llama",
        "requires_auth": True,
    },
    "llama-3b": {
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "context_limit": 128000,
        "template": "llama",
        "requires_auth": True,
    },
    # Google Gemma
    "gemma-2b": {
        "repo_id": "google/gemma-2-2b-it",
        "context_limit": 8192,
        "template": "gemma",
        "requires_auth": True,
    },
    # HuggingFace SmolLM
    "smollm-1.7b": {
        "repo_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "context_limit": 8192,
        "template": "default",
    },
    # TinyLlama
    "tinyllama": {
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "context_limit": 2048,
        "template": "tinyllama",
    },
}

# Reverse mapping: repo_id -> short_name
_REPO_TO_SHORT = {v["repo_id"]: k for k, v in SLM_REGISTRY.items()}


def get_slm_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model info by short name or repo ID."""
    if model_name in SLM_REGISTRY:
        return SLM_REGISTRY[model_name]
    # Try as repo ID
    if model_name in _REPO_TO_SHORT:
        return SLM_REGISTRY[_REPO_TO_SHORT[model_name]]
    # Try partial match
    for short_name, info in SLM_REGISTRY.items():
        if model_name.lower() in short_name or model_name.lower() in info["repo_id"].lower():
            return info
    return None


def is_slm_model(model_name: str) -> bool:
    """Check if model name refers to a registered SLM."""
    return get_slm_info(model_name) is not None


def list_slm_models() -> List[str]:
    """List all available SLM short names."""
    return list(SLM_REGISTRY.keys())


# -----------------------------
# Device Detection
# -----------------------------

_device_cache = None


def get_device() -> str:
    """Get best available device: mps (Apple Silicon) > cuda > cpu."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            _device_cache = "mps"
        elif torch.cuda.is_available():
            _device_cache = "cuda"
        else:
            _device_cache = "cpu"
    except ImportError:
        _device_cache = "cpu"

    logger.info(f"SLM device: {_device_cache}")
    return _device_cache


# -----------------------------
# SLM Service
# -----------------------------

class SLMService:
    """
    Service for managing Small Language Model inference.

    Features:
    - Lazy model loading (models loaded on first use)
    - Thread-safe model access
    - Async support via thread pool
    - Automatic device selection (MPS/CUDA/CPU)
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}  # {repo_id: (tokenizer, model)}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._config = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "do_sample": False,
            "max_concurrent": 16,  # Increased for better throughput (was 4)
        }

    def configure(self, **kwargs):
        """Update configuration."""
        for k, v in kwargs.items():
            if v is not None and k in self._config:
                self._config[k] = v

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool for async inference."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._config["max_concurrent"],
                thread_name_prefix="slm_"
            )
        return self._executor

    def _get_model_lock(self, repo_id: str) -> threading.Lock:
        """Get per-model lock for thread-safe loading."""
        with self._global_lock:
            if repo_id not in self._model_locks:
                self._model_locks[repo_id] = threading.Lock()
            return self._model_locks[repo_id]

    def _load_model(self, model_name: str) -> Tuple[Any, Any]:
        """
        Load model and tokenizer from HuggingFace.

        Returns: (tokenizer, model)
        Raises: ImportError if transformers not installed
                RuntimeError if model loading fails
        """
        info = get_slm_info(model_name)
        if info is None:
            raise ValueError(f"Unknown SLM model: {model_name}. Available: {list_slm_models()}")

        repo_id = info["repo_id"]

        # Check cache first
        if repo_id in self._models:
            return self._models[repo_id]

        lock = self._get_model_lock(repo_id)
        with lock:
            # Double-check after acquiring lock
            if repo_id in self._models:
                return self._models[repo_id]

            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
            except ImportError as e:
                raise ImportError(
                    "SLM requires: pip install torch transformers\n"
                    "For Apple Silicon: pip install torch transformers --extra-index-url https://download.pytorch.org/whl/cpu"
                ) from e

            device = get_device()
            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

            logger.info(f"Loading SLM: {repo_id} on {device} ({dtype})")
            start = time.time()

            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    repo_id,
                    trust_remote_code=True,
                    padding_side="left",
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    torch_dtype=dtype,
                    device_map=device if device != "cpu" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

                if device == "cpu":
                    model = model.to(device)

                model.eval()

                elapsed = time.time() - start
                logger.info(f"Loaded {repo_id} in {elapsed:.1f}s")

                self._models[repo_id] = (tokenizer, model)
                return tokenizer, model

            except Exception as e:
                logger.error(f"Failed to load {repo_id}: {e}")
                if info.get("requires_auth"):
                    logger.error(
                        f"Model {repo_id} requires authentication. "
                        "Run: huggingface-cli login"
                    )
                raise RuntimeError(f"Failed to load model {repo_id}: {e}") from e

    def _format_prompt(self, prompt: str, system: str, template: str) -> str:
        """Format prompt according to model's chat template."""
        if template == "qwen":
            parts = []
            if system:
                parts.append(f"<|im_start|>system\n{system}<|im_end|>")
            parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

        elif template == "phi":
            parts = []
            if system:
                parts.append(f"<|system|>\n{system}<|end|>")
            parts.append(f"<|user|>\n{prompt}<|end|>")
            parts.append("<|assistant|>")
            return "\n".join(parts)

        elif template == "llama":
            parts = ["<|begin_of_text|>"]
            if system:
                parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>")
            parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)

        elif template == "gemma":
            parts = []
            if system:
                parts.append(f"<start_of_turn>user\n{system}\n\n{prompt}<end_of_turn>")
            else:
                parts.append(f"<start_of_turn>user\n{prompt}<end_of_turn>")
            parts.append("<start_of_turn>model\n")
            return "\n".join(parts)

        elif template == "tinyllama":
            parts = []
            if system:
                parts.append(f"<|system|>\n{system}</s>")
            parts.append(f"<|user|>\n{prompt}</s>")
            parts.append("<|assistant|>")
            return "\n".join(parts)

        else:  # default
            if system:
                return f"System: {system}\n\nUser: {prompt}\n\nAssistant:"
            return f"User: {prompt}\n\nAssistant:"

    def call_sync(
        self,
        prompt: str,
        system: str = "",
        model: str = "qwen-0.5b",
        context: dict = None,
    ) -> str:
        """
        Synchronous inference call.

        Args:
            prompt: User prompt text
            system: Optional system prompt
            model: Model short name or repo ID
            context: Optional context dict for usage tracking

        Returns:
            Generated text response
        """
        info = get_slm_info(model)
        if info is None:
            raise ValueError(f"Unknown SLM model: {model}")

        repo_id = info["repo_id"]
        template = info.get("template", "default")

        try:
            import torch
        except ImportError:
            raise ImportError("SLM requires: pip install torch transformers")

        tokenizer, model_obj = self._load_model(model)
        device = get_device()

        # Format prompt
        formatted = self._format_prompt(prompt, system, template)

        # Tokenize
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=info.get("context_limit", 4096) - self._config["max_new_tokens"],
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        prompt_tokens = input_ids.shape[1]

        start_time = time.time()

        # Generate
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self._config["max_new_tokens"],
                "do_sample": self._config["do_sample"],
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            if self._config["temperature"] > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = self._config["temperature"]

            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask

            outputs = model_obj.generate(input_ids, **gen_kwargs)

        latency_ms = (time.time() - start_time) * 1000

        # Decode only new tokens
        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = len(new_tokens)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Record usage
        self._record_usage(
            model=repo_id,
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
        Async inference call (runs sync inference in thread pool).

        Args:
            prompt: User prompt text
            system: Optional system prompt
            model: Model short name or repo ID
            context: Optional context dict for usage tracking
            return_usage: If True, return dict with text and token counts

        Returns:
            Generated text, or dict if return_usage=True
        """
        loop = asyncio.get_event_loop()
        executor = self._get_executor()

        # Run sync call in thread pool
        func = partial(self.call_sync, prompt, system, model, context)
        text = await loop.run_in_executor(executor, func)

        if return_usage:
            # Note: actual token counts recorded in _record_usage
            # Return estimates here since we don't have them from thread
            return {"text": text, "prompt_tokens": 0, "completion_tokens": 0}
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

    def unload_model(self, model_name: str):
        """Unload a model to free memory."""
        info = get_slm_info(model_name)
        if info is None:
            return

        repo_id = info["repo_id"]
        lock = self._get_model_lock(repo_id)

        with lock:
            if repo_id in self._models:
                del self._models[repo_id]
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
                logger.info(f"Unloaded model: {repo_id}")

    def unload_all(self):
        """Unload all models."""
        with self._global_lock:
            for repo_id in list(self._models.keys()):
                self.unload_model(repo_id)

    def get_token_budget(self, model_name: str) -> int:
        """Get context limit for a model."""
        info = get_slm_info(model_name)
        if info is None:
            return 4096
        return info.get("context_limit", 4096) - self._config["max_new_tokens"] - 500


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
# Standalone Benchmark Script
# -----------------------------

def run_benchmark():
    """
    Standalone benchmark to test all SLM models.

    Run with: python -m utils.slm
    """
    import json
    from datetime import datetime

    print("=" * 60)
    print("SLM Benchmark")
    print("=" * 60)

    # Check dependencies
    missing = []
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        missing.append("torch")

    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")

    if missing:
        print(f"\nMissing packages: {missing}")
        print("Install with: pip install torch transformers")
        return

    print(f"Device: {get_device()}")
    print()

    # Test prompts (simplified for benchmark)
    TEST_PROMPTS = [
        {
            "prompt": "Given these restaurants: 1) Italian Bistro (rating 4.5), 2) Thai Express (rating 4.2), 3) Burger Joint (rating 3.8). Which is best for a romantic dinner? Output only the number.",
            "gold": "1",
        },
        {
            "prompt": "Restaurants: 1) Sushi Bar - expensive, 2) Pizza Place - cheap, 3) Steakhouse - moderate. Best for a budget meal? Output only the number.",
            "gold": "2",
        },
        {
            "prompt": "Options: 1) Cafe (quiet, coffee), 2) Sports Bar (loud, beer), 3) Library Cafe (quiet, tea). Best for studying? Output only the number.",
            "gold": "3",
        },
    ]

    # Models to test (start with smaller ones)
    test_models = ["qwen-0.5b", "tinyllama", "smollm-1.7b"]

    results = []
    service = get_slm_service()
    service.configure(max_new_tokens=32)  # Short for benchmark

    for model_name in test_models:
        info = get_slm_info(model_name)
        if info is None:
            continue

        print(f"\nTesting: {model_name} ({info['repo_id']})")
        print("-" * 40)

        model_results = {
            "model": model_name,
            "repo_id": info["repo_id"],
            "tests": [],
            "correct": 0,
            "total": 0,
            "avg_latency_ms": 0,
            "status": "success",
        }

        try:
            latencies = []
            for i, test in enumerate(TEST_PROMPTS):
                start = time.time()
                try:
                    response = service.call_sync(
                        prompt=test["prompt"],
                        system="You are a helpful assistant. Output only the requested number, nothing else.",
                        model=model_name,
                    )
                    latency = (time.time() - start) * 1000
                    latencies.append(latency)

                    # Check if correct
                    answer = "".join(c for c in response if c.isdigit())[:1]
                    correct = answer == test["gold"]
                    if correct:
                        model_results["correct"] += 1
                    model_results["total"] += 1

                    model_results["tests"].append({
                        "prompt_id": i,
                        "response": response[:100],
                        "extracted": answer,
                        "gold": test["gold"],
                        "correct": correct,
                        "latency_ms": latency,
                    })

                    status = "✓" if correct else "✗"
                    print(f"  Test {i+1}: {status} (got '{answer}', expected '{test['gold']}') - {latency:.0f}ms")

                except Exception as e:
                    print(f"  Test {i+1}: ERROR - {e}")
                    model_results["tests"].append({
                        "prompt_id": i,
                        "error": str(e),
                    })

            if latencies:
                model_results["avg_latency_ms"] = sum(latencies) / len(latencies)

            accuracy = model_results["correct"] / model_results["total"] if model_results["total"] > 0 else 0
            print(f"  Accuracy: {accuracy:.1%} ({model_results['correct']}/{model_results['total']})")
            print(f"  Avg latency: {model_results['avg_latency_ms']:.0f}ms")

        except Exception as e:
            model_results["status"] = f"failed: {e}"
            print(f"  FAILED: {e}")

        results.append(model_results)

        # Unload to free memory for next model
        service.unload_model(model_name)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Accuracy':<10} {'Latency':<12} {'Status'}")
    print("-" * 60)
    for r in results:
        acc = f"{r['correct']}/{r['total']}" if r['total'] > 0 else "N/A"
        lat = f"{r['avg_latency_ms']:.0f}ms" if r['avg_latency_ms'] > 0 else "N/A"
        print(f"{r['model']:<15} {acc:<10} {lat:<12} {r['status']}")

    # Save results
    output_file = f"slm_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    run_benchmark()
