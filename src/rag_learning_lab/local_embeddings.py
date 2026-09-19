from __future__ import annotations

from pathlib import Path


class SentenceTransformerEmbedding:
    """Local BGE inference backend. The model is downloaded once and then cached."""

    provider = "huggingface-local"

    def __init__(
        self,
        model_id: str = "BAAI/bge-small-zh-v1.5",
        *,
        revision: str = "main",
        device: str | None = None,
        query_instruction: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "dense-local 需要可选依赖：python -m pip install -e .[semantic]"
            ) from exc
        self.model_id = model_id
        self.query_instruction = query_instruction
        if device is None:
            try:
                import torch
                requested = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                requested = "cpu"
        else:
            requested = device
        try:
            self._model = SentenceTransformer(
                model_id, revision=revision, device=requested, local_files_only=True
            )
        except OSError:
            self._model = SentenceTransformer(model_id, revision=revision, device=requested)
        self.device = requested
        try:
            from huggingface_hub import snapshot_download
            snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
            self.revision = snapshot.name
        except (ImportError, OSError):
            self.revision = revision
        dimension_getter = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        self.dimension = int(dimension_getter())

    def _encode(self, texts: list[str]) -> list[list[float]]:
        values = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [row.tolist() if hasattr(row, "tolist") else list(row) for row in values]

    def embed_documents(self, texts: list[str], *, private: bool = False) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.query_instruction:
            text = "为这个句子生成表示以用于检索相关文章：" + text
        return self._encode([text])[0]
