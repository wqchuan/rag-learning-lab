from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_learning_lab.chunking import chunk_documents
from rag_learning_lab.embeddings import LocalHashingEmbedding
from rag_learning_lab.evaluation import run_evaluation
from rag_learning_lab.indexing import build_index, load_index
from rag_learning_lab.ingestion import load_documents
from rag_learning_lab.pipeline import answer_question


class RagLabTests(unittest.TestCase):
    def test_ingestion_and_traceable_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "资料.md").write_text("第一行\n第二行包含 RAG。\n第三行", encoding="utf-8")
            documents = load_documents(root)
            chunks = chunk_documents(documents, chunk_size=12, overlap=3)
            self.assertEqual(documents[0].status, "ok")
            self.assertTrue(chunks)
            for chunk in chunks:
                source = documents[0].text[chunk.start_char : chunk.end_char]
                self.assertEqual(chunk.text, source)
                self.assertGreaterEqual(chunk.start_line, 1)
                self.assertGreaterEqual(chunk.end_line, chunk.start_line)

    def test_invalid_chunk_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_documents([], chunk_size=100, overlap=100)

    def test_index_roundtrip_and_dimension_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "a.txt").write_text("RAG 先检索资料，再生成回答。", encoding="utf-8")
            path = root / "index.json"
            build_index(load_documents(raw), path, chunk_size=50, overlap=5,
                        embedder=LocalHashingEmbedding(128))
            index = load_index(path, expected_model="local-hashing-v1", expected_dimension=128)
            self.assertEqual(len(index.chunks), len(index.vectors))
            with self.assertRaises(ValueError):
                load_index(path, expected_dimension=256)

    def test_retrieval_and_answer_citations_are_from_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "a.txt").write_text(
                "RAG 的中文名称是检索增强生成。它先检索资料，再组织答案。", encoding="utf-8"
            )
            path = root / "index.json"
            build_index(load_documents(raw), path, embedder=LocalHashingEmbedding(128))
            answer = answer_question("RAG 的中文名称是什么？", path, top_k=2, min_score=0.01)
            self.assertEqual(answer.status, "ok")
            self.assertIn("检索增强生成", answer.text)
            self.assertTrue(set(answer.citations).issubset(
                {hit.chunk.chunk_id for hit in answer.hits}
            ))

    def test_empty_question_and_unrelated_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "a.txt").write_text("RAG 用于知识库问答。", encoding="utf-8")
            path = root / "index.json"
            build_index(load_documents(raw), path)
            self.assertEqual(answer_question("", path).status, "error")
            answer = answer_question("量子纠错码有哪些？", path)
            self.assertEqual(answer.status, "insufficient")

    def test_evaluation_report_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "a.txt").write_text("API 密钥应通过环境变量提供。", encoding="utf-8")
            index_path = root / "index.json"
            build_index(load_documents(raw), index_path)
            cases_path = root / "cases.json"
            cases_path.write_text(json.dumps([{
                "id": "1", "question": "API 密钥通过什么提供？",
                "type": "direct_fact", "answerable": True,
                "expected_keywords": ["环境变量"], "evidence_paths": ["a.txt"],
            }], ensure_ascii=False), encoding="utf-8")
            report_path = root / "report.json"
            report = run_evaluation(cases_path, index_path, report_path, min_score=0.01)
            self.assertTrue(report_path.exists())
            self.assertEqual(report["summary"]["evidence_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
