# RAG Learning Lab

一个便于学习和检查的中文 RAG 实验项目。当前版本完成可信评估、哈希基线、BM25、本地 BGE 语义检索、RRF 混合检索和可选的 SiliconFlow embeddings 适配器。模型只做推理，不训练、不微调。

PDF、OCR 和可视化界面属于下一阶段。

## 快速开始

基础哈希流程没有第三方依赖：

```powershell
python run.py build
python run.py ask "RAG 的基本流程是什么？"
python run.py evaluate
python -m unittest discover -s tests -v
```

原有 30 题保存在 `data/eval/cases.json`，作为 `regression-v1` 持续回归。

## 可信评估

项目包含 20 份基于 Python 3.14.7 官方中文文档整理的简明语料，以及 80 道 schema v2 题目：

- 64 道可回答，16 道无答案；
- 48 道开发集，32 道封存测试集；
- 按问题家族划分，家族不会跨开发集和测试集；
- 证据使用 `doc_id + source_version + 字符区间 + evidence_hash` 定位；
- 测试集在逐题审核完成前拒绝运行。

校验题集和证据：

```powershell
python run.py validate-eval
```

逐题审核表位于 `data/eval/python_docs_v2_review.csv`。当前状态为 `pending_human`。

审核者填写每行的 `review_status=approved` 和 `reviewer` 后，运行：

```powershell
python scripts/seal_eval.py
python run.py experiment --config configs/dense-local.toml --split test
```

封存脚本只有在 80 行全部通过时才会写入审核哈希和数据集哈希。

## 检索实验

安装 BM25：

```powershell
python -m pip install -e ".[retrieval]"
python run.py experiment --config configs/bm25.toml
```

安装本地语义模型：

```powershell
python -m pip install -e ".[semantic]"
python run.py experiment --config configs/dense-local.toml
python run.py experiment --config configs/hybrid-local.toml
```

默认模型是 `BAAI/bge-small-zh-v1.5`。首次运行下载权重，以后优先从本地 Hugging Face 缓存加载。代码自动选择 CUDA；CUDA 不可用时使用 CPU。

完整开发集调参：

```powershell
$env:HF_HUB_OFFLINE = "1"
python scripts/run_dev_tuning.py
```

调参只比较计划中锁定的 BM25、RRF、候选数和查询指令网格，选择顺序固定为 Recall@5、MRR、查询延迟。

## 可选远程 embeddings

SiliconFlow 适配器默认关闭，必须同时提供环境变量和命令开关：

```powershell
$env:SILICONFLOW_API_KEY = "你的密钥"
python run.py experiment --config configs/dense-api-siliconflow.toml --allow-remote-api
```

密钥只从 `SILICONFLOW_API_KEY` 读取，不会进入配置、缓存、日志或报告。客户端包含超时、有限重试、批量上限和本地向量缓存。带有 `privacy = "private"` 的资料会被代码强制拒绝发送到远程 API。远程 API 不参与默认验收。

## 主要目录

- `configs/`：五种可复现实验配置。
- `data/corpora/python-3.14.7/`：20 份语料和来源清单。
- `data/eval/`：旧回归题集、schema v2 题集和逐题审核表。
- `src/rag_learning_lab/`：摄取、分块、索引、检索、生成和评估实现。
- `scripts/`：确定性数据生成和开发集调参。
- `docs/EXPERIMENTS.md`：本机实验结果、采用判断和失败分析。

所有索引仍是可直接检查的 JSON 和暴力向量检索。本阶段不引入向量数据库、reranker、PDF 或 OCR。
