from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class SiliconFlowEmbedding:
    """Opt-in OpenAI-compatible embeddings client with an on-disk vector cache."""

    provider = "siliconflow"
    revision = "api"

    def __init__(
        self,
        *,
        allow_remote_api: bool = False,
        model_id: str = "BAAI/bge-m3",
        base_url: str = "https://api.siliconflow.cn/v1",
        dimension: int = 1024,
        timeout: float = 30,
        retries: int = 2,
        batch_size: int = 32,
        cache_path: Path = Path(".cache/siliconflow_embeddings.json"),
    ) -> None:
        if not allow_remote_api:
            raise ValueError("远程 embeddings 默认关闭；必须显式添加 --allow-remote-api")
        api_key = os.environ.get("SILICONFLOW_API_KEY", "")
        if not api_key:
            raise ValueError("缺少环境变量 SILICONFLOW_API_KEY")
        if batch_size < 1 or batch_size > 64:
            raise ValueError("API 批量大小必须在 1 到 64 之间")
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        self.timeout = timeout
        self.retries = retries
        self.batch_size = batch_size
        self.cache_path = cache_path
        self._api_key = api_key
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, list[float]]:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_id}\0{text}".encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )

    def _request(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model_id, "input": texts}).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                ordered = sorted(payload["data"], key=lambda item: item["index"])
                return [list(map(float, item["embedding"])) for item in ordered]
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.25 * (2 ** attempt))
        raise RuntimeError(f"SiliconFlow embeddings 请求失败: {last_error}")

    def embed_documents(self, texts: list[str], *, private: bool = False) -> list[list[float]]:
        if private:
            raise ValueError("私人资料禁止发送到远程 embedding API")
        missing = [text for text in dict.fromkeys(texts) if self._cache_key(text) not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._request(batch)
            if len(vectors) != len(batch):
                raise RuntimeError("SiliconFlow 返回的向量数量不匹配")
            for text, vector in zip(batch, vectors):
                self._cache[self._cache_key(text)] = vector
        if missing:
            self._save_cache()
        return [self._cache[self._cache_key(text)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
