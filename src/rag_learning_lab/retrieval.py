from __future__ import annotations

from .embeddings import LocalHashingEmbedding
from .indexing import LoadedIndex
from .models import RetrievalHit
from .text import informative_tokens


def retrieve(
    question: str,
    index: LoadedIndex,
    embedder: LocalHashingEmbedding,
    *,
    top_k: int = 4,
) -> list[RetrievalHit]:
    if not question.strip():
        raise ValueError("问题不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if index.manifest["embedding_model"] != embedder.model_id:
        raise ValueError("检索模型与索引模型不匹配")
    if index.manifest["embedding_dimension"] != embedder.dimension:
        raise ValueError("检索向量维度与索引不匹配")

    query = embedder.embed([question])[0]
    query_terms = informative_tokens(question)
    scored = []
    for chunk, vector in zip(index.chunks, index.vectors):
        has_lexical_support = bool(query_terms & informative_tokens(chunk.text))
        cosine = sum(a * b for a, b in zip(query, vector))
        scored.append(cosine if has_lexical_support else 0.0)
    ranked = sorted(enumerate(scored), key=lambda item: (-item[1], item[0]))[:top_k]
    return [
        RetrievalHit(index.chunks[position], score, rank)
        for rank, (position, score) in enumerate(ranked, start=1)
    ]
