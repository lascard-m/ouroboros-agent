"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (OpenRouter).
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import ollama

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """Multi-provider LLM API wrapper. Supports OpenRouter, OpenAI, Ollama, Mistral, Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        # Initialize clients for all available providers
        self._clients = {}
        
        # OpenRouter
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if or_key:
            self._clients["openrouter"] = {
                "client": None,
                "api_key": or_key,
                "base_url": "https://openrouter.ai/api/v1"
            }
        
        # OpenAI
        oai_key = os.environ.get("OPENAI_API_KEY", "")
        if oai_key:
            self._clients["openai"] = {
                "client": None,
                "api_key": oai_key,
                "base_url": "https://api.openai.com/v1"
            }
        
        # Ollama
        self._clients["ollama"] = {
            "client": None,
            "api_key": "ollama",
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        }
        
        # vLLM
        self._clients["vllm"] = {
            "client": None,
            "api_key": "",
            "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        }
        
        # Mistral
        mistral_key = os.environ.get("MISTRAL_API_KEY", "")
        if mistral_key:
            self._clients["mistral"] = {
                "client": None,
                "api_key": mistral_key,
                "base_url": "https://api.mistral.ai/v1"
            }
        
        # Gemini
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            self._clients["gemini"] = {
                "client": None,
                "api_key": gemini_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
            }
        
        # Fallback to openrouter if no clients
        if not self._clients:
            log.warning("No LLM providers configured. Set API keys for OPENROUTER_API_KEY, OPENAI_API_KEY, etc.")
            self._clients["openrouter"] = {
                "client": None,
                "api_key": "",
                "base_url": "https://openrouter.ai/api/v1"
            }

    def _get_client_for_provider(self, provider: str):
        if provider not in self._clients:
            raise ValueError(f"Provider {provider} not configured")
        
        if self._clients[provider]["client"] is None:
            from openai import OpenAI
            self._clients[provider]["client"] = OpenAI(
                base_url=self._clients[provider]["base_url"],
                api_key=self._clients[provider]["api_key"],
            )
        return self._clients[provider]["client"]

    def _parse_provider_from_model(self, model: str) -> Tuple[str, str]:
        """Ex: 'openrouter/anthropic/claude-3' -> ('openrouter', 'anthropic/claude-3')"""
        if "/" in model:
            parts = model.split("/", 1)
            provider = parts[0].lower()
            model_id = parts[1]
            if provider == "openrouter" and model_id == "free":
                # Special case: 'openrouter/free' is a valid model ID on OpenRouter
                return "openrouter", "openrouter/free"
            if provider in ["openrouter", "openai", "mistral", "gemini", "anthropic", "ollama", "vllm"]:
                return provider, model_id
        return "openrouter", model

    def _fetch_generation_cost(self, generation_id: str, provider: str = "openrouter") -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        if provider != "openrouter" or "openrouter" not in self._clients:
            return None
        base_url = self._clients["openrouter"]["base_url"]
        api_key = self._clients["openrouter"]["api_key"]
        try:
            import requests
            url = f"{base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call with fallback support. Returns: (response_message_dict, usage_dict with cost)."""
        
        # Map generic 'free' to OpenRouter's free router
        if model.lower() == "free":
            model = "openrouter/free"

        # Define fallback chain for free models (IDs verified 2026-03-02)
        fallbacks = [
            model, # Try requested first
            "openrouter/free", # Global free router on OpenRouter (requires full ID)
            "gemini/gemini-2.0-flash-lite-preview-02-05", # Gemini API direct
            "mistral/mistral-large-latest", # Mistral API direct
        ]
        
        # Remove duplicates while preserving order
        unique_fallbacks = []
        for f in fallbacks:
            if f and f not in unique_fallbacks:
                unique_fallbacks.append(f)
        
        last_error = None
        for current_model in unique_fallbacks:
            provider = "unknown"
            try:
                provider, model_name = self._parse_provider_from_model(current_model)
                client = self._get_client_for_provider(provider)
                
                # Prepare arguments
                env_max = int(os.environ.get("OUROBOROS_MAX_TOKENS", "0"))
                actual_max = max_tokens
                if env_max > 0:
                    actual_max = min(max_tokens, env_max)

                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": actual_max,
                }

                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice

                # Provider-specific logic
                if provider == "openrouter":
                    effort = normalize_reasoning_effort(reasoning_effort)
                    extra_body: Dict[str, Any] = {
                        "reasoning": {"effort": effort, "exclude": True},
                    }
                    if model_name.startswith("anthropic/"):
                        extra_body["provider"] = {
                            "order": ["Anthropic"],
                            "allow_fallbacks": False,
                            "require_parameters": True,
                        }
                    kwargs["extra_body"] = extra_body

                completion = client.chat.completions.create(**kwargs)
                
                if not completion or not completion.choices:
                    raise ValueError(f"Empty response from {provider}")

                response_message = completion.choices[0].message
                usage = getattr(completion, "usage", None)

                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                    "cost": 0.0,
                }

                if provider == "openrouter":
                    if usage and hasattr(usage, "cost") and usage.cost is not None:
                        usage_dict["cost"] = usage.cost
                    else:
                        resp_dict = completion.model_dump()
                        gen_id = resp_dict.get("id") or ""
                        if gen_id:
                            cost = self._fetch_generation_cost(gen_id, provider)
                            if cost is not None:
                                usage_dict["cost"] = cost

                return response_message.model_dump(), usage_dict

            except Exception as e:
                last_error = e
                log.warning(f"LLM call failed for {current_model} ({provider}): {e}. Trying fallback if available.")
                continue
        
        # If all fail
        try:
            print(f"All LLM fallbacks failed. Last error: {last_error}")
        except UnicodeEncodeError:
            print(f"All LLM fallbacks failed. Last error: {repr(last_error)}")
        raise last_error

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
