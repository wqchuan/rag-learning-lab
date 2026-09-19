from __future__ import annotations

import re
import unicodedata

_LATIN_OR_NUMBER = re.compile(r"[a-z0-9]+")
_HAN_BLOCK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")
_QUESTION_WORDS = {
    "什么", "哪些", "怎么", "如何", "是否", "为何", "为什么",
    "多少", "需要", "应该", "可以",
}


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).lower()


def tokenize(value: str) -> list[str]:
    """Tokenize English words plus Chinese unigrams and bigrams."""
    normalized = normalize_text(value)
    tokens = _LATIN_OR_NUMBER.findall(normalized)
    for block in _HAN_BLOCK.findall(normalized):
        tokens.extend(block)
        tokens.extend(block[index : index + 2] for index in range(len(block) - 1))
    return tokens


def informative_tokens(value: str) -> set[str]:
    """Return concrete Latin tokens and Chinese bigrams used by the local baseline."""
    return {
        token
        for token in tokenize(value)
        if (token.isascii() or len(token) > 1) and token not in _QUESTION_WORDS
    }


def split_sentences(value: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(value) if part.strip()]
