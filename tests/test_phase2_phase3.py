from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_learning_lab.local_embeddings import SentenceTransformerEmbedding
from rag_learning_lab.remote_embeddings import SiliconFlowEmbedding
from rag_learning_lab.evaluation import evaluate_answers, validate_dataset
from rag_learning_lab.models import Answer, Chunk, RetrievalHit
from rag_learning_lab.retrieval import BM25Retriever, RRFRetriever
from rag_learning_lab.experiments import select_best
from rag_learning_lab.indexing import load_index


def chunk(identity: str, text: str, start: int = 0, *, version: str = "v1") -> Chunk:
    return Chunk(identity, identity.split("-")[0], f"{identity}.md", text, start,
                 start + len(text), 1, 1, {"source_version": version})


class PhaseTwoThreeTests(unittest.TestCase):
    def test_generated_schema_and_evidence_hashes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dataset = validate_dataset(
            root / "data/eval/python_docs_v2.json",
            corpus_root=root / "data/corpora/python-3.14.7",
        )
        self.assertEqual(dataset["validation"], {
            "case_count": 80, "answerable": 64, "unanswerable": 16,
            "dev": 48, "test": 32,
        })

    def test_multi_evidence_coverage_uses_character_ranges(self) -> None:
        first = chunk("doc-a", "alpha evidence", 0)
        second = chunk("doc-b", "beta evidence", 20)
        case = {
            "id": "multi", "question": "q", "answerable": True,
            "expected_keywords": ["alpha"], "evidence": [
                {"doc_id": "doc", "source_version": "v1", "start_char": 0, "end_char": 5, "required": True},
                {"doc_id": "doc", "source_version": "v1", "start_char": 20, "end_char": 24, "required": True},
            ],
        }
        report = evaluate_answers([case], lambda _: Answer(
            "q", "alpha", "ok", [first.chunk_id],
            [RetrievalHit(first, 1.0, 1), RetrievalHit(second, 0.9, 2)],
        ))
        self.assertEqual(report["results"][0]["required_evidence_coverage"], 1.0)
        self.assertEqual(report["summary"]["recall_at_1"], 1.0)

    def test_bm25_ranks_exact_term_first(self) -> None:
        chunks = [chunk("a-1", "普通文本"), chunk("b-1", "Path.read_text 读取文本")]
        retriever = BM25Retriever(chunks, tokenizer=lambda value: value.lower().split())
        hits = retriever.retrieve("path.read_text", top_k=2)
        self.assertEqual(hits[0].chunk.chunk_id, "b-1")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_schema_one_index_remains_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            item = chunk("legacy-1", "legacy")
            payload = {
                "manifest": {
                    "schema_version": 1, "status": "complete", "chunk_count": 1,
                    "embedding_model": "local-hashing-v1", "embedding_dimension": 2,
                },
                "chunks": [{key: value for key, value in item.to_dict().items()
                            if key != "metadata"}],
                "vectors": [[1.0, 0.0]],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_index(path, expected_model="local-hashing-v1", expected_dimension=2)
            self.assertEqual(loaded.chunks[0].metadata, {})

    def test_rrf_combines_component_ranks(self) -> None:
        first, second = chunk("a-1", "a"), chunk("b-1", "b")
        class Fixed:
            def __init__(self, method, order):
                self.method, self.order = method, order
            def retrieve(self, question, *, top_k=5):
                return [RetrievalHit(item, 1.0, rank, self.method)
                        for rank, item in enumerate(self.order[:top_k], 1)]
        hits = RRFRetriever([Fixed("one", [first, second]), Fixed("two", [second, first])],
                            rrf_k=10, candidates=2).retrieve("q", top_k=2)
        self.assertEqual(set(hits[0].details["component_ranks"]), {"one", "two"})

    def test_missing_sentence_transformers_has_clear_error(self) -> None:
        real_import = __import__
        def blocked(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=blocked):
            with self.assertRaisesRegex(RuntimeError, "semantic"):
                SentenceTransformerEmbedding()

    def test_remote_api_is_disabled_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "默认关闭"):
            SiliconFlowEmbedding()

    def test_remote_cache_and_private_data_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"SILICONFLOW_API_KEY": "test-only"}, clear=False
        ):
            cache = Path(directory) / "cache.json"
            backend = SiliconFlowEmbedding(allow_remote_api=True, cache_path=cache, dimension=2)
            key = backend._cache_key("cached")
            cache.write_text(json.dumps({key: [1.0, 0.0]}), encoding="utf-8")
            backend = SiliconFlowEmbedding(allow_remote_api=True, cache_path=cache, dimension=2)
            with mock.patch("urllib.request.urlopen") as request:
                self.assertEqual(backend.embed_documents(["cached"]), [[1.0, 0.0]])
                request.assert_not_called()
            with self.assertRaisesRegex(ValueError, "私人资料"):
                backend.embed_documents(["secret"], private=True)

    def test_unsealed_test_split_is_rejected(self) -> None:
        root = Path(__file__).resolve().parents[1]
        from rag_learning_lab.experiments import run_experiment
        with self.assertRaisesRegex(ValueError, "尚未完成人工审核"):
            run_experiment(root / "configs/hash.toml", split="test")

    def test_sealing_rejects_incomplete_review(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        from seal_eval import seal_dataset
        with self.assertRaisesRegex(ValueError, "未审核通过"):
            seal_dataset(
                root / "data/eval/python_docs_v2.json",
                root / "data/eval/python_docs_v2_review.csv",
            )

    def test_selection_order_is_recall_then_mrr_then_latency(self) -> None:
        def report(name, recall, mrr, latency):
            return {"configuration": {"name": name},
                    "summary": {"recall_at_5": recall, "mrr": mrr},
                    "runtime": {"mean_query_ms": latency}}
        selected = select_best([
            report("fast", 0.9, 1.0, 1),
            report("higher-recall", 1.0, 0.8, 9),
            report("higher-mrr", 1.0, 0.9, 20),
            report("winner", 1.0, 0.9, 10),
        ])
        self.assertEqual(selected["configuration"]["name"], "winner")


if __name__ == "__main__":
    unittest.main()
