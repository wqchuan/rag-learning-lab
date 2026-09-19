from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .models import Answer, RetrievalHit
from .pipeline import answer_question


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dataset(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"评估题集不存在: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"评估题集不是有效 JSON: {exc}") from exc
    if isinstance(value, list):
        return {"schema_version": 1, "dataset_id": "regression-v1", "cases": value}
    if not isinstance(value, dict):
        raise ValueError("评估题集必须是数组或对象")
    return value


def load_cases(path: Path) -> list[dict[str, Any]]:
    return validate_dataset(path)["cases"]


def validate_dataset(path: Path, *, corpus_root: Path | None = None) -> dict[str, Any]:
    dataset = load_dataset(path)
    schema_version = dataset.get("schema_version", 1)
    if schema_version not in {1, 2}:
        raise ValueError(f"不支持的题集 schema: {schema_version}")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("评估题集必须包含非空 cases 数组")
    ids: set[str] = set()
    required_v1 = {"id", "question", "type", "answerable", "expected_keywords", "evidence_paths"}
    required_v2 = {"id", "question", "family_id", "type", "split", "answerable", "expected_keywords", "evidence"}
    errors: list[str] = []
    counts = {"answerable": 0, "unanswerable": 0, "dev": 0, "test": 0}
    family_splits: dict[str, set[str]] = {}
    for position, case in enumerate(cases):
        required = required_v2 if schema_version == 2 else required_v1
        if not isinstance(case, dict) or not required.issubset(case):
            errors.append(f"case[{position}] 缺少必要字段")
            continue
        if case["id"] in ids:
            errors.append(f"重复题号: {case['id']}")
        ids.add(case["id"])
        counts["answerable" if case["answerable"] else "unanswerable"] += 1
        if schema_version == 2:
            split = case["split"]
            if split not in {"dev", "test"}:
                errors.append(f"{case['id']} 的 split 无效")
            else:
                counts[split] += 1
            family_splits.setdefault(case["family_id"], set()).add(split)
            evidence = case["evidence"]
            if case["answerable"] and not evidence:
                errors.append(f"{case['id']} 可回答但没有证据")
            if not case["answerable"] and evidence:
                errors.append(f"{case['id']} 无答案但提供了证据")
            for ref in evidence:
                needed = {"doc_id", "source_version", "start_char", "end_char", "evidence_hash"}
                if not needed.issubset(ref):
                    errors.append(f"{case['id']} 的证据字段不完整")
                    continue
                if ref["start_char"] < 0 or ref["end_char"] <= ref["start_char"]:
                    errors.append(f"{case['id']} 的证据区间无效")
                if corpus_root and ref.get("path"):
                    try:
                        source = (corpus_root / ref["path"]).read_text(encoding="utf-8")
                        excerpt = source[ref["start_char"] : ref["end_char"]]
                        if _sha256(excerpt) != ref["evidence_hash"]:
                            errors.append(f"{case['id']} 的证据哈希不匹配")
                    except OSError as exc:
                        errors.append(f"{case['id']} 的证据文件无法读取: {exc}")
    for family, splits in family_splits.items():
        if len(splits) > 1:
            errors.append(f"问题家族 {family} 跨越 dev/test")
    if schema_version == 2 and dataset.get("dataset_id") == "python-docs-3.14.7-v2":
        expected = {"answerable": 64, "unanswerable": 16, "dev": 48, "test": 32}
        if len(cases) != 80 or counts != expected:
            errors.append(f"标准题集计数错误: {counts}, total={len(cases)}")
    if errors:
        raise ValueError("；".join(errors))
    dataset["validation"] = {"case_count": len(cases), **counts}
    return dataset


def _overlaps(hit: RetrievalHit, evidence: dict[str, Any]) -> bool:
    return (
        hit.chunk.doc_id == evidence.get("doc_id")
        and hit.chunk.start_char < evidence["end_char"]
        and hit.chunk.end_char > evidence["start_char"]
        and hit.chunk.metadata.get("source_version", evidence.get("source_version"))
        == evidence.get("source_version")
    )


def evaluate_answers(
    cases: list[dict[str, Any]], answerer: Callable[[dict[str, Any]], Answer]
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        answer = answerer(case)
        evidence = case.get("evidence", [])
        if "evidence_paths" in case:
            evidence = [
                {"doc_id": hit.chunk.doc_id, "source_version": hit.chunk.metadata.get("source_version"),
                 "start_char": hit.chunk.start_char, "end_char": hit.chunk.end_char}
                for hit in answer.hits if hit.chunk.path in case["evidence_paths"]
            ]
        ranks = [hit.rank for hit in answer.hits if any(_overlaps(hit, ref) for ref in evidence)]
        first_rank = min(ranks) if ranks else None
        required = [ref for ref in evidence if ref.get("required", True)]
        covered = [ref for ref in required if any(_overlaps(hit, ref) for hit in answer.hits[:5])]
        cited_ids = set(answer.citations)
        cited_hits = [hit for hit in answer.hits if hit.chunk.chunk_id in cited_ids]
        citations_valid = (
            bool(cited_ids) and cited_ids.issubset({hit.chunk.chunk_id for hit in answer.hits})
            if answer.status == "ok" else answer.status == "insufficient"
        )
        evidence_supported = (
            bool(cited_hits) and all(any(_overlaps(hit, ref) for ref in evidence) for hit in cited_hits)
            if case["answerable"] and answer.status == "ok" else answer.status == "insufficient"
        )
        keywords = case["expected_keywords"]
        keyword_coverage = (
            sum(word in answer.text for word in keywords) / len(keywords) if keywords else None
        )
        abstention_correct = answer.status == "insufficient" if not case["answerable"] else None
        false_refusal = bool(case["answerable"] and answer.status == "insufficient")
        if answer.status == "error":
            failure = "runtime_error"
        elif case["answerable"] and first_rank is None:
            failure = "retrieval_miss"
        elif false_refusal:
            failure = "generation_abstained"
        elif not citations_valid:
            failure = "citation_error"
        elif case["answerable"] and not evidence_supported:
            failure = "unsupported_citation"
        elif not case["answerable"] and not abstention_correct:
            failure = "failed_to_abstain"
        elif keyword_coverage is not None and keyword_coverage < 1:
            failure = "answer_incomplete"
        else:
            failure = None
        results.append({
            "case": case,
            "answer": answer.to_dict(),
            "first_evidence_rank": first_rank,
            "recall_at_1": first_rank is not None and first_rank <= 1 if case["answerable"] else None,
            "recall_at_5": first_rank is not None and first_rank <= 5 if case["answerable"] else None,
            "recall_at_10": first_rank is not None and first_rank <= 10 if case["answerable"] else None,
            "reciprocal_rank": 1 / first_rank if first_rank else 0.0 if case["answerable"] else None,
            "required_evidence_coverage": len(covered) / len(required) if required else None,
            "citations_valid": citations_valid,
            "evidence_supported": evidence_supported,
            "keyword_coverage": keyword_coverage,
            "abstention_correct": abstention_correct,
            "false_refusal": false_refusal,
            "failure_type": failure,
        })
    answerable = [r for r in results if r["case"]["answerable"]]
    unanswerable = [r for r in results if not r["case"]["answerable"]]
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None
    summary = {
            "case_count": len(results),
            "recall_at_1": mean([float(r["recall_at_1"]) for r in answerable]),
            "recall_at_5": mean([float(r["recall_at_5"]) for r in answerable]),
            "recall_at_10": mean([float(r["recall_at_10"]) for r in answerable]),
            "mrr": mean([r["reciprocal_rank"] for r in answerable]),
            "required_evidence_coverage": mean([r["required_evidence_coverage"] for r in answerable if r["required_evidence_coverage"] is not None]),
            "citation_validity": mean([float(r["citations_valid"]) for r in results]),
            "evidence_support_rate": mean([float(r["evidence_supported"]) for r in answerable]),
            "abstention_accuracy": mean([float(r["abstention_correct"]) for r in unanswerable]),
            "false_refusal_rate": mean([float(r["false_refusal"]) for r in answerable]),
            "fully_passed_count": sum(r["failure_type"] is None for r in results),
    }
    # Phase-one compatibility: this was the original name for answerable Recall@k.
    summary["evidence_hit_rate"] = summary["recall_at_5"]
    summary["correct_abstention_rate"] = summary["abstention_accuracy"]
    by_type: dict[str, dict[str, float | int | None]] = {}
    for case_type in sorted({r["case"].get("type", "unknown") for r in results}):
        subset = [r for r in answerable if r["case"].get("type", "unknown") == case_type]
        by_type[case_type] = {
            "case_count": len([r for r in results if r["case"].get("type", "unknown") == case_type]),
            "recall_at_5": mean([float(r["recall_at_5"]) for r in subset]),
            "mrr": mean([r["reciprocal_rank"] for r in subset]),
        }
    return {"summary": summary, "summary_by_type": by_type, "results": results}


def run_evaluation(
    cases_path: Path,
    index_path: Path,
    output_path: Path,
    *,
    top_k: int = 4,
    min_score: float = 0.08,
    generator_name: str = "extractive",
) -> dict[str, Any]:
    dataset = validate_dataset(cases_path)
    evaluated = evaluate_answers(
        dataset["cases"],
        lambda case: answer_question(
            case["question"], index_path, top_k=top_k,
            min_score=min_score, generator_name=generator_name,
        ),
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {key: value for key, value in dataset.items() if key not in {"cases", "validation"}},
        "configuration": {"index_path": str(index_path), "top_k": top_k, "min_score": min_score, "generator": generator_name},
        **evaluated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
