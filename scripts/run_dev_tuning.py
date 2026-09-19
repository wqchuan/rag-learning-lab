"""Run only the approved development-set grid and select by Recall@5, MRR, latency."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_learning_lab.experiments import run_experiment, select_best


def toml(name: str, retriever: str, extra: dict) -> str:
    values = {
        "name": name, "retriever": retriever,
        "model": "BAAI/bge-small-zh-v1.5",
        "model_revision": "main",
        "corpus": "data/corpora/python-3.14.7",
        "dataset": "data/eval/python_docs_v2.json",
        "index": f"data/index-{name}.json",
        "chunk_size": 300, "overlap": 50, "context_count": 5,
        "candidates": 20, "min_score": 0.0, **extra,
    }
    lines = []
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, str):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def main() -> None:
    reports = []
    specs = []
    for k1 in (1.2, 1.5, 2.0):
        for b in (0.5, 0.75):
            specs.append((f"bm25-k{k1}-b{b}", "bm25", {"k1": k1, "b": b}))
    for rrf_k in (10, 30, 60):
        for candidates in (10, 20):
            specs.append((f"hybrid-rrf{rrf_k}-c{candidates}", "hybrid-local", {
                "k1": 1.5, "b": 0.75, "rrf_k": rrf_k,
                "candidates": candidates, "query_instruction": False,
            }))
    for enabled in (False, True):
        specs.append((f"dense-instruction-{str(enabled).lower()}", "dense-local", {
            "query_instruction": enabled,
        }))
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        for name, retriever, extra in specs:
            config = temp / f"{name}.toml"
            config.write_text(toml(name, retriever, extra), encoding="utf-8")
            reports.append(run_experiment(config, split="dev"))
    best = select_best(reports)
    summary = {
        "selection_order": ["recall_at_5", "mrr", "mean_query_ms"],
        "selected": best["configuration"]["name"],
        "experiments": [{
            "name": report["configuration"]["name"],
            "recall_at_5": report["summary"]["recall_at_5"],
            "mrr": report["summary"]["mrr"],
            "exact_term_recall_at_5": report["summary_by_type"].get("exact_term", {}).get("recall_at_5"),
            "mean_query_ms": report["runtime"]["mean_query_ms"],
        } for report in reports],
    }
    output = ROOT / "experiments/tuning-dev.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
