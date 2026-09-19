from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    path: str
    text: str
    content_hash: str
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    path: str
    text: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(**value)


@dataclass(slots=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int
    method: str = "vector"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "method": self.method,
        }


@dataclass(slots=True)
class Answer:
    question: str
    text: str
    status: str
    citations: list[str] = field(default_factory=list)
    hits: list[RetrievalHit] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "status": self.status,
            "citations": self.citations,
            "hits": [hit.to_dict() for hit in self.hits],
            "error": self.error,
        }

