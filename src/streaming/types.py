"""Data types for the streaming pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SentenceChunk:
    """One sentence extracted from LLM stream."""
    seq: int
    text: str
    is_final: bool = False


@dataclass
class AudioChunk:
    """Audio bytes for one sentence."""
    seq: int
    audio_bytes: bytes
    sentence_text: str
    is_final: bool = False
