from __future__ import annotations

import hashlib

from .models import Chunk, Document


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def chunk_documents(
    documents: list[Document], *, chunk_size: int = 300, overlap: int = 50
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for document in documents:
        if document.status != "ok" or not document.text.strip():
            continue
        start = 0
        while start < len(document.text):
            end = min(start + chunk_size, len(document.text))
            piece = document.text[start:end]
            if piece.strip():
                identity = (
                    f"{document.doc_id}:{document.content_hash}:{start}:{end}:"
                    f"{chunk_size}:{overlap}"
                )
                chunks.append(
                    Chunk(
                        hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                        document.doc_id,
                        document.path,
                        piece,
                        start,
                        end,
                        _line_number(document.text, start),
                        _line_number(document.text, max(start, end - 1)),
                    )
                )
            if end == len(document.text):
                break
            start += step
    return chunks
