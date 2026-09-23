from __future__ import annotations

import hashlib
import math
from typing import Protocol

from .text import tokenize


class EmbeddingBackend(Protocol):
    model_id: str
    provider: str
    revision: str
    dimension: int

    def embed_documents(self, texts: list[str], *, private: bool = False) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LocalHashingEmbedding:
    """Deterministic dependency-free baseline; not a production semantic model."""

    model_id = "local-hashing-v1"
    provider = "local"
    revision = "1"

    def __init__(self, dimension: int = 512) -> None:
        if dimension < 64:
            raise ValueError("向量维度至少为 64")
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_documents(self, texts: list[str], *, private: bool = False) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
