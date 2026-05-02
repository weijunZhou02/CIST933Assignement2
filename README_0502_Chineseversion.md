# README 0502 — Shakespeare RAG 系统使用说明

## 项目概述

本项目为 CSCI433/933 Assignment 2，构建了一个**基于检索增强生成（RAG）的莎士比亚戏剧问答系统**。

系统流程：用户提问 → 从莎士比亚原文中检索相关段落 → 将段落作为上下文喂给本地小语言模型 → 生成对初学者友好的回答。

同时实现了一个 **Baseline（无检索纯模型）** 用于对比评估。

---

## 项目结构

```
assignment2_starter_code/
├── data/processed/              # 数据集（3个JSON文件）
│   ├── hamlet.json
│   ├── macbeth.json
│   └── romeo_and_juliet.json
├── prompts/
│   └── system_prompt.txt        # RAG系统的system prompt
├── results/
│   ├── instructor_questions.json # 评估用的问题
│   └── evaluation_results.csv   # 评估输出（运行evaluate后生成）
├── data/cache/                  # Embedding缓存（自动生成）
├── src/
│   ├── config.py                # 全局配置
│   ├── model.py                 # 共享LLM加载器（baseline+RAG共用）
│   ├── data_loader.py           # 数据加载器
│   ├── chunking.py              # 分块策略
│   ├── retrieval.py             # 嵌入与检索（带缓存）
│   ├── baseline.py              # Baseline系统（无检索）
│   ├── rag_chatbot.py           # RAG聊天机器人（核心）
│   ├── build_index.py           # 检索索引构建与测试
│   └── evaluate.py              # 评估脚本
├── requirements.txt             # Python依赖
└── README_0502.md               # 本文件
```

---

## 各文件功能说明

### `src/config.py` — 全局配置

集中管理所有可调参数：
- **数据路径**：`DATA_DIR`, `PLAY_FILES` 指向三部戏剧的JSON文件
- **嵌入模型**：`EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"`（轻量级，用于把文本变成向量）
- **生成模型**：`GENERATION_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"`（1.1B参数的本地小模型）
- **检索数量**：`DEFAULT_TOP_K = 3`（每次检索返回3个最相关的chunk）
- **分块粒度**：`CHUNK_LEVEL = "scene"`（按场景分块）
- **生成参数**：`GENERATION_MAX_NEW_TOKENS = 512`, `GENERATION_TEMPERATURE = 0.7`

> 如果需要切换模型或调参，只改这个文件即可。

---

### `src/data_loader.py` — 数据加载器

**功能**：读取三部莎士比亚戏剧的JSON文件，支持两种粒度提取：

| 粒度 | 说明 | 记录数量 |
|------|------|---------|
| `scene` | 一个场景 = 一条记录（含完整场景文本、摘要、关键词） | 73条 |
| `utterance` | 一句台词 = 一条记录（含说话人、所属场景信息） | 3613条 |

**关键函数**：
- `load_all_plays(level="scene")` — 加载全部三部戏剧，返回记录列表
- `load_play(path, level)` — 加载单部戏剧

---

### `src/chunking.py` — 分块策略

**功能**：将 data_loader 加载的记录转换为适合检索的 chunk。

**策略：Scene-level + Summary 增强**
- 每个 scene 成为一个 chunk
- 在原文前面拼接 `scene_summary`（摘要）+ `keywords`（关键词）+ `location`（地点），让 embedding 更好地捕捉语义
- 超过4000字符的长场景自动拆分，保留400字符重叠防止上下文断裂

**关键函数**：
- `create_chunks(records)` — 输入记录列表，输出 chunk 列表
- `format_chunk_for_display(chunk)` — 格式化chunk用于终端显示

---

### `src/retrieval.py` — 嵌入与检索

**功能**：使用 sentence-transformers 生成嵌入向量，用 cosine similarity 做检索。

**核心类：`EmbeddingRetriever`**
- `build_index(chunks)` — 对所有 chunk 的文本生成嵌入，构建检索索引
- `retrieve(query, top_k=3)` — 输入查询，返回 top-k 个最相关的 (chunk, score) 对

---

### `src/baseline.py` — Baseline 系统（无检索）

**功能**：**纯 prompt 生成**，不检索任何上下文，直接让模型凭自己的知识回答。

这是与 RAG 系统做对比的基线。两者使用同一个 TinyLlama 模型，唯一区别是 baseline 没有检索到的上下文。

**关键函数**：
- `baseline_answer(query)` — 输入问题，返回模型的纯生成回答

---

### `src/rag_chatbot.py` — RAG 聊天机器人（核心）

**功能**：完整的 RAG 流程：

```
用户提问 → embedding 检索 top-k 相关场景 → 构建带上下文的 prompt → TinyLlama 生成回答
```

**关键函数**：
- `build_pipeline()` — 加载数据 → 分块 → 构建检索索引，返回 `(retriever, chunks)`
- `rag_answer(query, retriever)` — 对一个问题执行完整 RAG 流程，返回 `{query, answer, retrieved, prompt}`
- `generate_answer(prompt)` — 调用 TinyLlama 生成回答
- `main()` — 交互式聊天循环

---

### `src/build_index.py` — 检索索引测试

**功能**：验证数据加载 → 分块 → 嵌入 → 检索整条链路是否正常工作。

跑3个测试查询，打印每个查询的 top-3 检索结果（含play、act、scene、score、summary）。

**用途**：快速 sanity check，不涉及生成模型，运行速度快。

---

### `src/evaluate.py` — 评估脚本

**功能**：对 `results/instructor_questions.json` 中的所有问题，分别运行 baseline 和 RAG 系统，将结果写入 `results/evaluation_results.csv`。

CSV 中每个问题有两行（baseline + rag），包含：
- 系统生成的回答
- RAG检索到的段落
- 5个评分列（留空，需手动打分）：correctness / grounding / retrieval_relevance / usefulness / style_quality

---

## 快速上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要的包：`numpy`, `pandas`, `scikit-learn`, `sentence-transformers`, `tqdm`, `torch`, `transformers`, `accelerate`

> 首次运行时，sentence-transformers 和 TinyLlama 模型会自动从 HuggingFace 下载（分别约 90MB 和 2GB）。

### 2. 测试检索（不需要生成模型）

```bash
cd assignment2_starter_code
python src/build_index.py
```

验证数据加载、分块、嵌入、检索是否正常。输出每个测试查询的 top-3 相关场景。

### 3. 启动 RAG 聊天机器人

```bash
python src/rag_chatbot.py
```

进入交互模式，输入问题得到 RAG 回答。输入 `quit` 退出。

### 4. 测试 Baseline

```bash
python src/baseline.py
```

运行一个简单的 baseline 测试。

### 5. 运行完整评估

```bash
python src/evaluate.py
```

自动对 14 个问题（6 instructor + 8 group-designed）跑 baseline + RAG，输出到 `results/evaluation_results.csv`。

然后手动在 CSV 中打分（1-5分），用于报告的评估部分。

---

## 4 种交互类型

系统支持 spec 要求的全部 4 种交互类型：

| 类型 | 说明 | 示例 |
|------|------|------|
| **concept_explanation** | 解释角色/概念 | "Who is Lady Macbeth?" |
| **contextual_qa** | 基于情节的问答 | "Why does Macbeth kill Duncan?" |
| **evidence_retrieval** | 检索并展示原文证据 | "What do the witches predict?" |
| **stylised_generation** | 莎翁风格创意生成（≤150词） | "Generate a Shakespearean-style speech from Hamlet" |

Stylised generation 会**自动检测**（通过关键词如 "shakespearean", "generate", "style of" 等），并在输出前加上 `[NOTE: creative output, not factual evidence]` 标记。

---

## 评估问题分布

共 15 个问题：

| 来源 | 数量 | 类型覆盖 |
|------|------|---------|
| Instructor | 6 | contextual_qa |
| Group (G1-G3) | 3 | concept_explanation |
| Group (G4-G5) | 2 | evidence_retrieval |
| Group (G6) | 1 | contextual_qa |
| Group (G7-G8) | 2 | stylised_generation |
| Group (G9) | 1 | robustness |

---

## Embedding 缓存

首次运行时 embedding 会自动缓存到 `data/cache/`，后续运行直接加载，跳过计算。如果修改了分块策略或嵌入模型，缓存会自动失效（基于内容 hash）。

---

## 可调整的配置

在 `src/config.py` 中修改：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `GENERATION_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 换其他模型如 `microsoft/phi-2` |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | 嵌入模型 |
| `DEFAULT_TOP_K` | `3` | 检索返回的chunk数量 |
| `CHUNK_LEVEL` | `"scene"` | 改为 `"utterance"` 可用台词级分块 |
| `GENERATION_MAX_NEW_TOKENS` | `512` | 生成回答的最大token数 |
| `GENERATION_TEMPERATURE` | `0.7` | 越低越确定性，越高越多样 |

---

## 系统流程图

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ 用户提问  │───▶│ Embedding    │───▶│  Cosine      │
│          │    │ (MiniLM)     │    │  Similarity   │
└──────────┘    └──────────────┘    └──────┬───────┘
                                           │ top-k chunks
                                           ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ 最终回答  │◀───│ TinyLlama    │◀───│ RAG Prompt   │
│          │    │ 生成          │    │ (context +   │
└──────────┘    └──────────────┘    │  question)   │
                                    └──────────────┘
```
