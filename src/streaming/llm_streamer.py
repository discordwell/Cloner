"""
Streaming LLM client for the pipeline.

Streams tokens from Anthropic or OpenAI, feeds them through a SentenceSplitter,
and pushes complete SentenceChunk objects into a queue.
"""

import os
import queue
import logging
import threading
from typing import Optional, Callable

from src.streaming.types import SentenceChunk
from src.streaming.sentence_splitter import SentenceSplitter

logger = logging.getLogger(__name__)


class LLMStreamer:
    """
    Streams LLM responses token-by-token, splitting into sentences.

    Runs in a worker thread. Feeds tokens into SentenceSplitter,
    which emits SentenceChunk objects into the output queue.

    Usage:
        streamer = LLMStreamer(provider="openai")
        streamer.stream_response(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="You are helpful.",
            output_queue=my_queue,
            stop_event=threading.Event(),
        )
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.8,
    ):
        """
        Args:
            provider: "anthropic" or "openai"
            model: Model name. Defaults to provider's best fast model.
            api_key: API key. Falls back to env var.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.
        """
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature

        if provider == "anthropic":
            import anthropic
            self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self._model = model or "claude-sonnet-4-20250514"
            self._client = anthropic.Anthropic(api_key=self._api_key)
        elif provider == "openai":
            import openai
            self._api_key = api_key or os.getenv("OPENAI_API_KEY")
            self._model = model or "gpt-4o-mini"
            self._client = openai.OpenAI(api_key=self._api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")

    def stream_response(
        self,
        messages: list,
        system_prompt: str,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        on_sentence: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Stream LLM response, splitting into sentences pushed to output_queue.

        Blocks until the full response is streamed. Caller should run this
        in a worker thread.

        Args:
            messages: Chat messages (user/assistant history).
            system_prompt: System prompt.
            output_queue: Queue to push SentenceChunk objects into.
            stop_event: Set this to cancel mid-stream.
            on_sentence: Optional callback when each sentence is emitted.

        Returns:
            The full response text.
        """
        seq_counter = [0]
        full_text_parts = []

        def emit_sentence(text: str):
            chunk = SentenceChunk(seq=seq_counter[0], text=text)
            seq_counter[0] += 1
            output_queue.put(chunk)
            if on_sentence:
                on_sentence(text)
            logger.debug(f"[LLM] Sentence {chunk.seq}: {text[:60]}...")

        splitter = SentenceSplitter(on_sentence=emit_sentence)

        try:
            if self.provider == "anthropic":
                full_text = self._stream_anthropic(
                    messages, system_prompt, splitter, full_text_parts, stop_event
                )
            else:
                full_text = self._stream_openai(
                    messages, system_prompt, splitter, full_text_parts, stop_event
                )

            # Flush any remaining text in the splitter
            splitter.flush()

            # Signal completion
            final = SentenceChunk(
                seq=seq_counter[0], text="", is_final=True
            )
            output_queue.put(final)

            return full_text

        except Exception as e:
            if stop_event.is_set():
                logger.info("[LLM] Stream cancelled")
            else:
                logger.error(f"[LLM] Stream error: {e}")
            # Still signal completion so downstream doesn't hang
            output_queue.put(SentenceChunk(seq=seq_counter[0], text="", is_final=True))
            raise

    def _stream_anthropic(self, messages, system_prompt, splitter, parts, stop_event):
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
            temperature=self.temperature,
        ) as stream:
            for text in stream.text_stream:
                if stop_event.is_set():
                    break
                parts.append(text)
                splitter.feed(text)

        return "".join(parts)

    def _stream_openai(self, messages, system_prompt, splitter, parts, stop_event):
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            if stop_event.is_set():
                break
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
                splitter.feed(delta.content)

        return "".join(parts)
