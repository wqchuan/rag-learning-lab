from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embeddings import LocalHashingEmbedding
from .local_embeddings import SentenceTransformerEmbedding
from .remote_embeddings import SiliconFlowEmbedding
from .evaluation import evaluate_answers, validate_dataset
from .generation import ExtractiveGenerator
from .indexing import build_index, load_index
from .ingestion import load_documents
from .models import Answer
from .retrieval import BM25Retriever, RRFRetriever, Retriever, VectorRetriever


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    required = {"name", "retriever"}
    if not required.issubset(config):
        raise ValueError(f"实验配置缺少字段: {sorted(required - set(config))}")
    return config


def select_best(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen selection order: Recall@5, then MRR, then latency."""
    if not reports:
        raise ValueError("没有可供选择的实验报告")
    return min(
        reports,
        key=lambda report: (
            -report["summary"]["recall_at_5"],
            -report["summary"]["mrr"],
            report["runtime"]["mean_query_ms"],
        ),
    )


def _backend(config: dict[str, Any], *, allow_remote_api: bool):
    kind = config["retriever"]
    if kind == "hash":
        return LocalHashingEmbedding(int(config.get("dimension", 512)))
    if kind in {"dense-local", "hybrid-local"}:
        return SentenceTransformerEmbedding(
            config.get("model", "BAAI/bge-small-zh-v1.5"),
            revision=config.get("model_revision", "main"),
            query_instruction=bool(config.get("query_instruction", False)),
        )
    if kind == "dense-api-siliconflow":
        return SiliconFlowEmbedding(
            allow_remote_api=allow_remote_api,
            model_id=config.get("model", "BAAI/bge-m3"),
            timeout=float(config.get("timeout", 30)),
            retries=int(config.get("retries", 2)),
            batch_size=int(config.get("batch_size", 32)),
            cache_path=Path(config.get("cache_path", ".cache/siliconflow_embeddings.json")),
        )
    if kind == "bm25":
        return LocalHashingEmbedding(64)
    raise ValueError(f"未知检索器: {kind}")


def run_experiment(
    config_path: Path,
    *,
    allow_remote_api: bool = False,
    split: str = "dev",
    output_path: Path | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    corpus = Path(config.get("corpus", "data/corpora/python-3.14.7"))
    dataset_path = Path(config.get("dataset", "data/eval/python_docs_v2.json"))
    dataset = validate_dataset(dataset_path, corpus_root=corpus)
    if split not in {"dev", "test", "all"}:
        raise ValueError("split 必须是 dev、test 或 all")
    if split == "test" and not dataset.get("sealed", False):
        raise ValueError("封存测试集尚未完成人工审核；请先审核并封存")
    cases = dataset["cases"] if split == "all" else [
        case for case in dataset["cases"] if case["split"] == split
    ]
    documents = load_documents(corpus)
    backend = _backend(config, allow_remote_api=allow_remote_api)
    index_path = Path(config.get("index", f"data/index-{config['name']}.json"))
    started = time.perf_counter()
    build_index(
        documents, index_path, chunk_size=int(config.get("chunk_size", 300)),
        overlap=int(config.get("overlap", 50)), embedder=backend,
    )
    index = load_index(index_path)
    kind = config["retriever"]
    retriever: Retriever
    if kind == "bm25":
        retriever = BM25Retriever(
            index.chunks, k1=float(config.get("k1", 1.5)), b=float(config.get("b", 0.75))
        )
    else:
        dense = VectorRetriever(
            index, backend, require_lexical_support=kind == "hash"
        )
        if kind == "hybrid-local":
            retriever = RRFRetriever(
                [BM25Retriever(index.chunks, k1=float(config.get("k1", 1.5)),
                               b=float(config.get("b", 0.75))), dense],
                rrf_k=int(config.get("rrf_k", 60)),
                candidates=int(config.get("candidates", 20)),
            )
        else:
            retriever = dense
    generator = ExtractiveGenerator(LocalHashingEmbedding(512))
    top_k = int(config.get("context_count", 5))
    min_score = float(config.get("min_score", 0.0))
    latencies: list[float] = []
    def answerer(case: dict[str, Any]) -> Answer:
        before = time.perf_counter()
        hits = retriever.retrieve(case["question"], top_k=top_k)
        latencies.append((time.perf_counter() - before) * 1000)
        if not hits or hits[0].score <= min_score:
            return Answer(case["question"], "当前资料不足以回答这个问题。", "insufficient", hits=hits)
        return generator.generate(case["question"], hits, -1.0)
    evaluated = evaluate_answers(cases, answerer)
    total_seconds = time.perf_counter() - started
    peak_vram_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    except ImportError:
        pass
    versions = {}
    for package in ("jieba-py", "sentence-transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": _git_revision(),
        "corpus": {"path": str(corpus), "hash": index.manifest["corpus_hash"]},
        "dataset": {"path": str(dataset_path), "hash": _hash_file(dataset_path),
                    "id": dataset.get("dataset_id"), "review_status": dataset.get("review_status"),
                    "split": split},
        "configuration": config,
        "model": {"provider": index.manifest.get("embedding_provider"),
                  "id": index.manifest.get("embedding_model"),
                  "revision": index.manifest.get("embedding_revision"),
                  "dimension": index.manifest.get("embedding_dimension"),
                  "device": getattr(backend, "device", "cpu")},
        "runtime": {"total_seconds": round(total_seconds, 4),
                    "mean_query_ms": round(sum(latencies) / len(latencies), 4) if latencies else None,
                    "peak_vram_mb": peak_vram_mb, "packages": versions},
        **evaluated,
    }
    destination = output_path or Path("experiments") / f"{config['name']}-{split}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
