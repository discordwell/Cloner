"""Shared OpenAI client helper for autopitch scripts."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def complete(prompt: str, model: str = "gpt-5.4",
              fallback_model: str = "gpt-4o",
              max_tokens: int = 800,
              temperature: float = 0.8) -> str:
    """Run a single-turn chat completion with fallback on model-not-found."""
    from openai import OpenAI
    from openai import NotFoundError

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
    except NotFoundError:
        logger.info("model %s unavailable; falling back to %s", model, fallback_model)
        resp = client.chat.completions.create(
            model=fallback_model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )

    return (resp.choices[0].message.content or "").strip()
