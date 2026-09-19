from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pipeline import answer_question


def load_cases(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"评估题集不存在: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"评估题集不是有效 JSON: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("评估题集必须是非空数组")
    required = {"id", "question", "type", "answerable", "expected_keywords", "evidence_paths"}
    for case in value:
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError("评估题缺少必要字段")
    return value


def run_evaluation(
    cases_path: Path,
    index_path: Path,
    output_path: Path,
    *,
    top_k: int = 4,
    min_score: float = 0.08,
    generator_name: str = "extractive",
) -> dict[str, Any]:
    results = []
    for case in load_cases(cases_path):
        answer = answer_question(
            case["question"], index_path, top_k=top_k, min_score=min_score,
            generator_name=generator_name,
        )
        retrieved_paths = {hit.chunk.path for hit in answer.hits}
        evidence_hit = (
            any(path in retrieved_paths for path in case["evidence_paths"])
            if case["answerable"] else None
        )
        keyword_matches = [word for word in case["expected_keywords"] if word in answer.text]
        keyword_coverage = (
            len(keyword_matches) / len(case["expected_keywords"])
            if case["expected_keywords"] else None
        )
        context_ids = {hit.chunk.chunk_id for hit in answer.hits}
        citations_valid = bool(answer.citations) and all(
            value in context_ids for value in answer.citations
        ) if answer.status == "ok" else answer.status == "insufficient"
        abstention_correct = answer.status == "insufficient" if not case["answerable"] else None

        if answer.status == "error":
            failure_type = "runtime_error"
        elif case["answerable"] and not evidence_hit:
            failure_type = "retrieval_miss"
        elif case["answerable"] and answer.status == "insufficient":
            failure_type = "generation_abstained"
        elif not citations_valid or answer.status == "citation_error":
            failure_type = "citation_error"
        elif not case["answerable"] and not abstention_correct:
            failure_type = "failed_to_abstain"
        elif keyword_coverage is not None and keyword_coverage < 1:
            failure_type = "answer_incomplete"
        else:
            failure_type = None

        results.append({
            "case": case,
            "answer": answer.to_dict(),
            "evidence_hit": evidence_hit,
            "keyword_coverage": keyword_coverage,
            "citations_valid": citations_valid,
            "abstention_correct": abstention_correct,
            "failure_type": failure_type,
        })

    answerable = [result for result in results if result["case"]["answerable"]]
    unanswerable = [result for result in results if not result["case"]["answerable"]]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "index_path": str(index_path), "top_k": top_k,
            "min_score": min_score, "generator": generator_name,
        },
        "summary": {
            "case_count": len(results),
            "evidence_hit_rate": (
                sum(result["evidence_hit"] is True for result in answerable) / len(answerable)
                if answerable else None
            ),
            "correct_abstention_rate": (
                sum(result["abstention_correct"] is True for result in unanswerable) / len(unanswerable)
                if unanswerable else None
            ),
            "fully_passed_count": sum(result["failure_type"] is None for result in results),
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

