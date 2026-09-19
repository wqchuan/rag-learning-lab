from __future__ import annotations

from pathlib import Path

from .embeddings import LocalHashingEmbedding
from .generation import ExtractiveGenerator, Generator, OpenAICompatibleGenerator
from .indexing import load_index
from .models import Answer
from .retrieval import retrieve


def answer_question(
    question: str,
    index_path: Path,
    *,
    top_k: int = 4,
    min_score: float = 0.08,
    generator_name: str = "extractive",
) -> Answer:
    if not question.strip():
        return Answer(question, "", "error", error="问题不能为空")
    try:
        index = load_index(index_path)
        embedder = LocalHashingEmbedding(index.manifest["embedding_dimension"])
        if index.manifest["embedding_model"] != embedder.model_id:
            raise ValueError("第一版只支持 local-hashing-v1 索引")
        hits = retrieve(question, index, embedder, top_k=top_k)
        generator: Generator
        if generator_name == "extractive":
            generator = ExtractiveGenerator(embedder)
        elif generator_name == "openai":
            generator = OpenAICompatibleGenerator()
        else:
            raise ValueError(f"未知生成器: {generator_name}")
        return generator.generate(question, hits, min_score)
    except (OSError, ValueError) as exc:
        return Answer(question, "", "error", error=str(exc))

