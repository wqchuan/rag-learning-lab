from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Protocol

from .embeddings import EmbeddingBackend, LocalHashingEmbedding
from .indexing import LoadedIndex
from .models import Chunk, RetrievalHit
from .text import informative_tokens, normalize_text


class Retriever(Protocol):
    method: str

    def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievalHit]: ...


class VectorRetriever:
    method = "dense"

    def __init__(
        self,
        index: LoadedIndex,
        embedder: EmbeddingBackend,
        *,
        require_lexical_support: bool = False,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.require_lexical_support = require_lexical_support
        if index.manifest["embedding_model"] != embedder.model_id:
            raise ValueError("检索模型与索引模型不匹配")
        if index.manifest["embedding_dimension"] != embedder.dimension:
            raise ValueError("检索向量维度与索引不匹配")

    def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievalHit]:
        _validate_query(question, top_k)
        query = self.embedder.embed_query(question)
        query_terms = informative_tokens(question)
        scored: list[tuple[int, float]] = []
        for position, (chunk, vector) in enumerate(zip(self.index.chunks, self.index.vectors)):
            lexical = bool(query_terms & informative_tokens(chunk.text))
            score = sum(a * b for a, b in zip(query, vector))
            if self.require_lexical_support and not lexical:
                score = 0.0
            scored.append((position, score))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            RetrievalHit(self.index.chunks[pos], score, rank, self.method)
            for rank, (pos, score) in enumerate(ranked, start=1)
        ]


class BM25Retriever:
    method = "bm25"

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 要求 k1 > 0 且 0 <= b <= 1")
        if tokenizer is None:
            try:
                import jieba
            except ImportError as exc:
                raise RuntimeError("bm25 需要可选依赖：python -m pip install -e .[retrieval]") from exc
            tokenizer = lambda text: [token for token in jieba.lcut(normalize_text(text)) if token.strip()]
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer
        self._documents = [Counter(tokenizer(chunk.text)) for chunk in chunks]
        self._lengths = [sum(document.values()) for document in self._documents]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        frequencies = Counter(token for document in self._documents for token in document)
        total = len(self._documents)
        self._idf = {
            token: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for token, count in frequencies.items()
        }

    def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievalHit]:
        _validate_query(question, top_k)
        terms = self._tokenizer(question)
        scored: list[tuple[int, float]] = []
        for position, document in enumerate(self._documents):
            length = self._lengths[position]
            score = 0.0
            for term in terms:
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                normalizer = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self._average_length or 1)
                )
                score += self._idf.get(term, 0.0) * frequency * (self.k1 + 1) / normalizer
            scored.append((position, score))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            RetrievalHit(self.chunks[pos], score, rank, self.method)
            for rank, (pos, score) in enumerate(ranked, start=1)
        ]


class RRFRetriever:
    method = "rrf"

    def __init__(self, retrievers: list[Retriever], *, rrf_k: int = 60, candidates: int = 20) -> None:
        if len(retrievers) < 2 or rrf_k <= 0 or candidates <= 0:
            raise ValueError("RRF 至少需要两个检索器，且参数必须为正数")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.candidates = candidates

    def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievalHit]:
        _validate_query(question, top_k)
        fused: dict[str, dict] = {}
        for retriever in self.retrievers:
            for hit in retriever.retrieve(question, top_k=self.candidates):
                entry = fused.setdefault(hit.chunk.chunk_id, {"chunk": hit.chunk, "score": 0.0, "ranks": {}})
                entry["score"] += 1.0 / (self.rrf_k + hit.rank)
                entry["ranks"][retriever.method] = hit.rank
        ranked = sorted(fused.values(), key=lambda item: (-item["score"], item["chunk"].chunk_id))[:top_k]
        return [
            RetrievalHit(item["chunk"], item["score"], rank, self.method, {"component_ranks": item["ranks"]})
            for rank, item in enumerate(ranked, start=1)
        ]


def _validate_query(question: str, top_k: int) -> None:
    if not question.strip():
        raise ValueError("问题不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")


def retrieve(
    question: str,
    index: LoadedIndex,
    embedder: LocalHashingEmbedding,
    *,
    top_k: int = 4,
) -> list[RetrievalHit]:
    """Compatibility wrapper for the phase-one hashing retriever."""
    return VectorRetriever(index, embedder, require_lexical_support=True).retrieve(
        question, top_k=top_k
    )
