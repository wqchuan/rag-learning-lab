# A→B 路线状态

## 已完成：P2 可信评估

- 保留 30 题 `regression-v1`。
- 新增 20 份 Python 3.14.7 可追溯语料、80 题 schema v2 数据集和逐题审核表。
- 实现字符区间证据、内容哈希、多证据覆盖、Recall@1/5/10、MRR、引用有效性、证据支持率、拒答准确率、误拒率和失败分类。
- 测试集在人工审核与封存前由代码拒绝运行。

## 已完成：P3 精简检索

- 保留哈希基线，新增统一 `EmbeddingBackend` 与 `Retriever` 契约。
- 新增 jieba BM25、BGE 本地语义检索和 RRF 排名融合。
- 本地模型记录实际 Hugging Face 快照 revision，并支持下载后离线加载。
- SiliconFlow embeddings 是显式开启的公开语料对照，私人资料强制拒绝远程发送。
- 仍使用 JSON 索引和暴力检索，不引入向量数据库、reranker、训练或微调。

## 当前审核门

`data/eval/python_docs_v2_review.csv` 的 80 行需要逐题人工确认。确认后才能把题集标记为 sealed、运行一次 32 题封存测试集、合并到 `main` 并创建 `v0.2.0-p3`。

## 下一阶段：B 文档接入

1. 接入文本型 PDF，保留页码、标题和字符区间。
2. 加入解析失败报告和文档内容哈希。
3. 再覆盖扫描 PDF 和 OCR。
4. 最后复用现有 pipeline 开发本地单用户界面。

私人 PDF 始终只允许本地 embedding。界面不会先于解析、引用定位和删除一致性验证。
