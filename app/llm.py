import copy
import json
import logging
import os
import random
import re
import threading
import time
from typing import Any

import litellm
from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)

# Configure LiteLLM defaults
litellm.drop_params = True
litellm.set_verbose = False
litellm.num_retries = 5

_lock = threading.Lock()
_usage = {"prompt": 0, "completion": 0}


class ContentFilterError(RuntimeError):
    """Content filter blocked prompt/response."""


def _get_chat_model() -> tuple[str, dict]:
    """Resolves model name and kwargs for LiteLLM completion."""
    kwargs: dict[str, Any] = {}
    
    if settings.llm_model:
        model = settings.llm_model
    elif settings.azure_openai_endpoint:
        model = f"azure/{settings.azure_chat_deployment}"
        kwargs["api_base"] = settings.azure_openai_endpoint
        kwargs["api_key"] = settings.azure_openai_api_key
        kwargs["api_version"] = settings.azure_openai_api_version
    else:
        model = "gpt-4o"

    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    if settings.litellm_api_base:
        kwargs["api_base"] = settings.litellm_api_base
    if settings.litellm_api_key:
        kwargs["api_key"] = settings.litellm_api_key

    return model, kwargs


def _get_embed_model() -> tuple[str, dict]:
    """Resolves embedding model name and kwargs for LiteLLM embedding."""
    kwargs: dict[str, Any] = {}
    
    if settings.embed_model:
        model = settings.embed_model
    elif settings.azure_openai_endpoint:
        model = f"azure/{settings.azure_embed_deployment}"
        kwargs["api_base"] = settings.azure_openai_endpoint
        kwargs["api_key"] = settings.azure_openai_api_key
        kwargs["api_version"] = settings.azure_openai_api_version
    else:
        model = "text-embedding-3-small"

    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    if settings.litellm_api_base:
        kwargs["api_base"] = settings.litellm_api_base
    if settings.litellm_api_key:
        kwargs["api_key"] = settings.litellm_api_key

    return model, kwargs


def _record(usage) -> None:
    if usage is None:
        return
    with _lock:
        _usage["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
        _usage["completion"] += getattr(usage, "completion_tokens", 0) or 0


def usage_snapshot() -> dict:
    with _lock:
        return dict(_usage)


def reset_usage() -> None:
    with _lock:
        _usage["prompt"] = 0
        _usage["completion"] = 0


def _parse_retry_after(error_str: str, default: float = 10.0) -> float:
    """Extracts seconds from 'retry after X seconds' message if present."""
    match = re.search(r"retry after (\d+)", error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + random.uniform(1.0, 3.0)
        except Exception:
            pass
    return default


def _backoff_on_rate_limit(exc: Exception, attempt: int) -> None:
    """Handles rate limits (429) gracefully with exponential backoff."""
    exc_str = str(exc)
    delay = _parse_retry_after(exc_str, default=min(60.0, 4.0 * (2 ** attempt)))
    logger.warning(f"Rate limit / 429 encountered (attempt {attempt+1}). Backing off for {delay:.1f}s: {exc}")
    time.sleep(delay)


def make_strict_schema(schema: dict) -> dict:
    """Patches Pydantic JSON schema to be strict-mode compliant."""
    schema = copy.deepcopy(schema)

    def patch(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                props = node.get("properties")
                if isinstance(props, dict):
                    node["required"] = list(props.keys())
            if "$ref" in node:
                for k in list(node.keys()):
                    if k != "$ref":
                        del node[k]
            for v in list(node.values()):
                patch(v)
        elif isinstance(node, list):
            for item in node:
                patch(item)

    patch(schema)
    return schema


# ------------------------------------------------------------
# Universal Chat with LiteLLM
# ------------------------------------------------------------
def chat(
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    retries: int = 8,
) -> str:
    model, kwargs = _get_chat_model()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_exc = None
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            _record(resp.usage)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "429" in exc_str or "ratelimit" in exc_str:
                _backoff_on_rate_limit(exc, attempt)
                continue
            if "content_filter" in exc_str:
                if attempt == retries - 1:
                    raise ContentFilterError(f"Content filter triggered: {exc}") from exc
                time.sleep(2.0)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(min(30.0, 2.0 * (2 ** attempt)))

    if last_exc:
        raise last_exc
    return ""


# ------------------------------------------------------------
# Universal Structured Chat with LiteLLM
# ------------------------------------------------------------
def chat_structured(
    system: str,
    user: str,
    model_cls,
    *,
    temperature: float = 0.1,
    max_tokens: int = 6000,
    retries: int = 8,
):
    model, kwargs = _get_chat_model()
    schema = make_strict_schema(model_cls.model_json_schema())
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_exc = None
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": model_cls.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
                **kwargs,
            )
            _record(resp.usage)
            raw = resp.choices[0].message.content
            if not raw:
                raise ValueError("Empty structured response from LLM")
            return model_cls.model_validate_json(raw)
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "429" in exc_str or "ratelimit" in exc_str:
                _backoff_on_rate_limit(exc, attempt)
                continue
            if "content_filter" in exc_str:
                if attempt == retries - 1:
                    raise ContentFilterError(f"Content filter triggered: {exc}") from exc
                time.sleep(2.0)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(min(30.0, 2.0 * (2 ** attempt)))

    if last_exc:
        raise last_exc
    raise RuntimeError("Structured LLM call failed")


# ------------------------------------------------------------
# Universal Embeddings with LiteLLM & Rate-Limit Resilience
# ------------------------------------------------------------
def embed(texts: list[str], retries: int = 8) -> list[list[float]]:
    """
    Batch-embed with LiteLLM.
    Uses safe sub-batching (32 items) and automatic 429 exponential backoff
    to safely embed thousands of pages without failing.
    """
    if not texts:
        return []

    model, kwargs = _get_embed_model()
    out: list[list[float]] = []
    
    # Safe sub-batch size to prevent rate limits on large books
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        chunk = [t.replace("\n", " ")[:8000] for t in texts[i:i + batch_size]]
        for attempt in range(retries):
            try:
                resp = litellm.embedding(
                    model=model,
                    input=chunk,
                    **kwargs,
                )
                out.extend([d["embedding"] for d in resp.data])
                # Tiny breather between batches for high-volume documents
                if len(texts) > 100:
                    time.sleep(0.1)
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "ratelimit" in exc_str:
                    _backoff_on_rate_limit(exc, attempt)
                    continue
                if attempt == retries - 1:
                    raise
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))

    return out
