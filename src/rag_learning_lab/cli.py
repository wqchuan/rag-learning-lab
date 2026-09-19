from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import LocalHashingEmbedding
from .evaluation import run_evaluation
from .indexing import build_index
from .ingestion import load_documents
from .pipeline import answer_question


def _build(args: argparse.Namespace) -> int:
    documents = load_documents(args.data)
    manifest = build_index(
        documents, args.index, chunk_size=args.chunk_size, overlap=args.overlap,
        embedder=LocalHashingEmbedding(args.dimension),
    )
    print(json.dumps({"manifest": manifest, "index": str(args.index)}, ensure_ascii=False, indent=2))
    for document in documents:
        if document.status != "ok":
            print(f"读取失败: {document.path}: {document.error}")
    return 0


def _ask(args: argparse.Namespace) -> int:
    answer = answer_question(
        args.question, args.index, top_k=args.top_k, min_score=args.min_score,
        generator_name=args.generator,
    )
    print(f"状态: {answer.status}")
    if answer.error:
        print(f"错误: {answer.error}")
    for hit in answer.hits:
        print(
            f"[C{hit.rank}] score={hit.score:.4f} "
            f"{hit.chunk.path}:{hit.chunk.start_line}-{hit.chunk.end_line} "
            f"chunk={hit.chunk.chunk_id}"
        )
    if answer.text:
        print(f"回答: {answer.text}")
    if answer.citations:
        print("引用片段: " + ", ".join(answer.citations))
    return 0 if answer.status in {"ok", "insufficient"} else 1


def _evaluate(args: argparse.Namespace) -> int:
    report = run_evaluation(
        args.cases, args.index, args.output, top_k=args.top_k,
        min_score=args.min_score, generator_name=args.generator,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"完整报告: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可检查的 RAG 学习实验室")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="读取资料并建立索引")
    build.add_argument("--data", type=Path, default=Path("data/raw"))
    build.add_argument("--index", type=Path, default=Path("data/index.json"))
    build.add_argument("--chunk-size", type=int, default=300)
    build.add_argument("--overlap", type=int, default=50)
    build.add_argument("--dimension", type=int, default=512)
    build.set_defaults(handler=_build)

    ask = subparsers.add_parser("ask", help="检索资料并回答问题")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=Path("data/index.json"))
    ask.add_argument("--top-k", type=int, default=4)
    ask.add_argument("--min-score", type=float, default=0.08)
    ask.add_argument("--generator", choices=["extractive", "openai"], default="extractive")
    ask.set_defaults(handler=_ask)

    evaluate = subparsers.add_parser("evaluate", help="运行固定题集并保存报告")
    evaluate.add_argument("--cases", type=Path, default=Path("data/eval/cases.json"))
    evaluate.add_argument("--index", type=Path, default=Path("data/index.json"))
    evaluate.add_argument("--output", type=Path, default=Path("experiments/baseline.json"))
    evaluate.add_argument("--top-k", type=int, default=4)
    evaluate.add_argument("--min-score", type=float, default=0.08)
    evaluate.add_argument("--generator", choices=["extractive", "openai"], default="extractive")
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
