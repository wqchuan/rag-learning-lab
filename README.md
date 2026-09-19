# RAG 学习实验室（第一版）

这个项目用一条可检查的最小流程学习 RAG：读取资料、分块、建立索引、检索、回答并显示出处，最后用固定题集评估。

第一版只支持 UTF-8 的 `.txt` 和 `.md`。默认方案完全在本地运行，不需要 API 密钥，也不依赖第三方库。默认的“回答生成”是抽取式基线，用于看清流程；配置 API 后可切换到兼容 OpenAI Chat Completions 的模型。

## 1. 准备环境

在 PowerShell 中进入项目目录：

```powershell
cd D:\myproject\rag-learning-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

第一版本身没有第三方运行依赖，不需要联网安装。若希望使用 `rag-lab` 短命令，可选执行 `python -m pip install -e .`；下文使用无需安装的 `python run.py`。

## 2. 建立索引

```powershell
python run.py build
```

命令会读取 `data/raw`，将索引写入 `data/index.json`，并报告成功、失败、文档数和片段数。原始资料不会被修改。

## 3. 提问并查看出处

```powershell
python run.py ask "RAG 的基本流程是什么？"
```

输出包含回答状态、检索分数、文件位置、回答和引用的片段编号。`insufficient` 表示现有资料不足；`error` 表示运行失败，二者不会混淆。

## 4. 运行基线评估

```powershell
python run.py evaluate
```

报告保存到 `experiments/baseline.json`。题集包含直接事实、多片段、精确词和无答案问题。每道题保留检索结果、回答、引用有效性和失败类型。

## 5. 使用大模型生成回答（可选）

先设置环境变量，再明确选择 `openai`：

```powershell
$env:OPENAI_API_KEY = "你的密钥"
$env:RAG_LLM_MODEL = "你的模型名"
python run.py ask "RAG 的基本流程是什么？" --generator openai
```

可用 `OPENAI_BASE_URL` 指定兼容端点，默认是 `https://api.openai.com/v1`。密钥不要写入代码、配置文件或实验报告。

## 6. 运行测试

```powershell
python -m unittest discover -s tests -v
```

## 目录职责

- `src/rag_learning_lab/ingestion.py`：读取文件并保留来源。
- `chunking.py`：按字符窗口分块并保留偏移量和行号。
- `embeddings.py`：本地哈希向量，用于教学基线。
- `indexing.py`：原子地保存和校验索引。
- `retrieval.py`：相似度检索。
- `generation.py`：抽取式回答或可选大模型回答，并检查引用。
- `pipeline.py`：串联检索和回答，不重复底层实现。
- `evaluation.py`：运行固定题集并分类失败。
- `cli.py`：命令行入口。

## 第一版边界

- 哈希向量适合理解流程，不代表生产级语义检索效果。
- 抽取式生成器只从证据中摘取句子，不负责流畅改写。
- PDF、OCR、BM25、混合检索、重排序和 Graph RAG 留到后续实验。
- 修改向量维度、向量模型或分块参数后，应重新执行 `build`。

## 后续阶段规划

参见 [后续阶段规划与选项](docs/ROADMAP.md)：包含检索实验、知识库应用、图谱与复杂问答三条路线，以及 GitHub 参考项目、工作量和验收标准。
