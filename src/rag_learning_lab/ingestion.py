from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import Document

SUPPORTED_SUFFIXES = {".txt", ".md"}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_documents(root: Path) -> list[Document]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"资料目录不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"资料路径不是目录: {root}")

    documents: list[Document] = []
    metadata_by_path: dict[str, dict] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata_by_path = {
                item["path"]: item for item in manifest.get("documents", []) if "path" in item
            }
        except (OSError, json.JSONDecodeError, TypeError):
            metadata_by_path = {}
    candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        doc_id = _digest(relative)[:16]
        try:
            text = path.read_text(encoding="utf-8")
            metadata = dict(metadata_by_path.get(relative, {}))
            metadata.setdefault("privacy", "public")
            documents.append(Document(doc_id, relative, text, _digest(text), metadata=metadata))
        except (OSError, UnicodeError) as exc:
            documents.append(Document(doc_id, relative, "", "", "error", str(exc)))
    return documents
