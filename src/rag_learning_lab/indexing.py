from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import chunk_documents
from .embeddings import LocalHashingEmbedding
from .models import Chunk, Document

SCHEMA_VERSION = 1


def _corpus_hash(documents: list[Document]) -> str:
    joined = "\n".join(
        f"{doc.path}:{doc.content_hash}:{doc.status}" for doc in sorted(documents, key=lambda d: d.path)
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class LoadedIndex:
    manifest: dict[str, Any]
    chunks: list[Chunk]
    vectors: list[list[float]]


def build_index(
    documents: list[Document],
    output_path: Path,
    *,
    chunk_size: int = 300,
    overlap: int = 50,
    embedder: LocalHashingEmbedding | None = None,
) -> dict[str, Any]:
    embedder = embedder or LocalHashingEmbedding()
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("没有可索引的文本片段")
    vectors = embedder.embed([chunk.text for chunk in chunks])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_hash": _corpus_hash(documents),
        "document_count": sum(doc.status == "ok" for doc in documents),
        "failed_document_count": sum(doc.status != "ok" for doc in documents),
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "embedding_model": embedder.model_id,
        "embedding_dimension": embedder.dimension,
    }
    payload = {
        "manifest": manifest,
        "chunks": [chunk.to_dict() for chunk in chunks],
        "vectors": vectors,
        "document_errors": [doc.to_dict() for doc in documents if doc.status != "ok"],
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output_path.parent, delete=False, suffix=".tmp"
        ) as temporary:
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
            temp_name = temporary.name
        os.replace(temp_name, output_path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
    return manifest


def load_index(
    path: Path, *, expected_model: str | None = None, expected_dimension: int | None = None
) -> LoadedIndex:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"索引不存在，请先执行 build: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"索引无法读取: {exc}") from exc

    manifest = payload.get("manifest", {})
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("索引版本不受支持或索引未完整构建")
    if expected_model and manifest.get("embedding_model") != expected_model:
        raise ValueError("索引的向量模型与当前配置不匹配，请重新 build")
    if expected_dimension and manifest.get("embedding_dimension") != expected_dimension:
        raise ValueError("索引的向量维度与当前配置不匹配，请重新 build")

    chunks = [Chunk.from_dict(value) for value in payload.get("chunks", [])]
    vectors = payload.get("vectors", [])
    dimension = manifest.get("embedding_dimension")
    if len(chunks) != len(vectors) or len(chunks) != manifest.get("chunk_count"):
        raise ValueError("索引中的片段与向量数量不一致")
    if any(not isinstance(vector, list) or len(vector) != dimension for vector in vectors):
        raise ValueError("索引中存在维度不正确的向量")
    return LoadedIndex(manifest, chunks, vectors)
