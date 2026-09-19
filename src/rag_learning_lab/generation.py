from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

from .embeddings import LocalHashingEmbedding
from .models import Answer, RetrievalHit
from .text import split_sentences

_CITATION_PATTERN = re.compile(r"\[C(\d+)\]")


class Generator(Protocol):
    def generate(self, question: str, hits: list[RetrievalHit], min_score: float) -> Answer: ...


def _insufficient(question: str, hits: list[RetrievalHit]) -> Answer:
    return Answer(question, "当前资料不足以回答这个问题。", "insufficient", hits=hits)


class ExtractiveGenerator:
    def __init__(self, embedder: LocalHashingEmbedding, max_sentences: int = 2) -> None:
        self.embedder = embedder
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[RetrievalHit], min_score: float) -> Answer:
        if not hits or hits[0].score < min_score:
            return _insufficient(question, hits)

        query_vector = self.embedder.embed([question])[0]
        candidates: list[tuple[float, int, str]] = []
        for hit_index, hit in enumerate(hits, start=1):
            for sentence in split_sentences(hit.chunk.text):
                if sentence.lstrip().startswith("#"):
                    continue
                sentence_vector = self.embedder.embed([sentence])[0]
                score = sum(a * b for a, b in zip(query_vector, sentence_vector))
                candidates.append((score, hit_index, sentence))
        candidates.sort(key=lambda item: item[0], reverse=True)

        selected: list[tuple[int, str]] = []
        seen: set[str] = set()
        for score, hit_index, sentence in candidates:
            if score <= 0 or sentence in seen:
                continue
            selected.append((hit_index, sentence))
            seen.add(sentence)
            if len(selected) == self.max_sentences:
                break
        if not selected:
            return _insufficient(question, hits)

        citations: list[str] = []
        parts: list[str] = []
        for hit_index, sentence in selected:
            chunk_id = hits[hit_index - 1].chunk.chunk_id
            if chunk_id not in citations:
                citations.append(chunk_id)
            parts.append(f"{sentence} [C{hit_index}]")
        return Answer(question, " ".join(parts), "ok", citations, hits)


class OpenAICompatibleGenerator:
    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("RAG_LLM_MODEL", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def generate(self, question: str, hits: list[RetrievalHit], min_score: float) -> Answer:
        if not hits or hits[0].score < min_score:
            return _insufficient(question, hits)
        if not self.api_key or not self.model:
            return Answer(
                question,
                "",
                "error",
                hits=hits,
                error="使用 openai 生成器需要 OPENAI_API_KEY 和 RAG_LLM_MODEL",
            )

        context = "\n\n".join(
            f"[C{index}] 来源={hit.chunk.path}:{hit.chunk.start_line}-{hit.chunk.end_line}\n{hit.chunk.text}"
            for index, hit in enumerate(hits, start=1)
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "只根据给出的资料回答。每个事实后使用 [C数字] 引用。"
                        "资料不足时只回答：当前资料不足以回答这个问题。不要编造引用。"
                    ),
                },
                {"role": "user", "content": f"问题：{question}\n\n资料：\n{context}"},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer_text = result["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return Answer(question, "", "error", hits=hits, error=str(exc))

        if answer_text == "当前资料不足以回答这个问题。":
            return _insufficient(question, hits)
        labels = [int(value) for value in _CITATION_PATTERN.findall(answer_text)]
        if not labels or any(label < 1 or label > len(hits) for label in labels):
            return Answer(
                question,
                answer_text,
                "citation_error",
                hits=hits,
                error="回答缺少有效引用或引用了未提供的片段",
            )
        citations = list(dict.fromkeys(hits[label - 1].chunk.chunk_id for label in labels))
        return Answer(question, answer_text, "ok", citations, hits)
