"""Seal schema-v2 test data only after every review row is approved."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def seal_dataset(dataset_path: Path, review_path: Path) -> dict:
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("审核表为空")
    incomplete = [
        row.get("case_id", "<unknown>") for row in rows
        if row.get("review_status") != "approved" or not row.get("reviewer", "").strip()
    ]
    if incomplete:
        raise ValueError(f"仍有 {len(incomplete)} 题未审核通过: {', '.join(incomplete[:5])}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_id = {row["case_id"]: row for row in rows}
    if set(by_id) != {case["id"] for case in dataset["cases"]}:
        raise ValueError("审核表题号与数据集不一致")
    for case in dataset["cases"]:
        case["review_status"] = "approved"
        case["reviewer"] = by_id[case["id"]]["reviewer"].strip()
        note = by_id[case["id"]].get("reviewer_notes", "").strip()
        if note:
            case["reviewer_notes"] = note
    dataset["review_status"] = "approved"
    dataset["sealed"] = True
    dataset["sealed_at"] = datetime.now(timezone.utc).isoformat()
    dataset["review_hash"] = hashlib.sha256(review_path.read_bytes()).hexdigest()
    dataset["dataset_hash"] = hashlib.sha256(
        json.dumps(dataset["cases"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return dataset


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = seal_dataset(
        root / "data/eval/python_docs_v2.json",
        root / "data/eval/python_docs_v2_review.csv",
    )
    print(json.dumps({
        "review_status": result["review_status"], "sealed": result["sealed"],
        "dataset_hash": result["dataset_hash"], "review_hash": result["review_hash"],
    }, ensure_ascii=False, indent=2))
