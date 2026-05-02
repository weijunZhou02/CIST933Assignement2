# Shakespeare RAG 系统 — 中文使用说明

## 项目概述

**课程**：CSCI433/933 Machine Learning Algorithms and Applications — Assignment 2

本项目实现了一个**基于检索增强生成（RAG）的莎士比亚戏剧问答系统**。系统面向**没有莎士比亚背景知识的初学者**，支持提问、检索原文证据、生成易懂的回答。

### 系统流程

```
用户提问
   │
   ▼
┌─────────────────┐   嵌入查询      ┌────────────────────┐
│  嵌入模型         │────────────────▶│  余弦相似度检索      │
│  (MiniLM-L6-v2)  │                │  在 73 个 chunk 中  │
└─────────────────┘                 └────────┬───────────┘
                                              │ top-k 个最相关 chunk
                                              ▼
                                    ┌────────────────────┐
                                    │  构建 RAG Prompt    │
                                    │ （系统提示 + 检索文本 │
                                    │  + 用户问题）        │
                                    └────────┬───────────┘
                                              │
                                              ▼
                                    ┌────────────────────┐
                                    │  TinyLlama 1.1B    │
                                    │  生成回答            │
                                    └────────┬───────────┘
                                              │
                                              ▼
                                    ┌────────────────────┐
                                    │  展示回答 + 检索证据  │
                                    └────────────────────┘
```

同时实现了一个 **Baseline 系统**（纯 prompt 生成，无检索）用于对比评估。

### 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 生成模型 | TinyLlama-1.1B-Chat-v1.0 | 最小的聊天模型，可在 CPU 运行，符合 SLM 约束 |
| 嵌入模型 | all-MiniLM-L6-v2 | 快速轻量（80MB），语义检索领域广泛使用的基线模型 |
| 分块策略 | 场景级 + 摘要增强 | 保留叙事完整性，摘要弥补古英语与现代查询的词汇差异 |
| 检索方法 | 密集嵌入 + 余弦相似度 | 简单透明，易于评估，无外部依赖 |
| 共享模型实例 | baseline 和 RAG 共用一个模型 | 节省一半内存，保证公平对比 |

---

## 项目结构

```
assignment2_starter_code/
├── data/
│   ├── processed/                   # 结构化莎士比亚数据集
│   │   ├── hamlet.json              #   20 个场景，1657 条台词
│   │   ├── macbeth.json             #   28 个场景，830 条台词
│   │   └── romeo_and_juliet.json    #   25 个场景，1126 条台词
│   └── cache/                       # Embedding 缓存（自动生成）
│       └── embeddings_<hash>.pkl    #   chunk/模型变更时自动失效
├── prompts/
│   └── system_prompt.txt            # RAG 生成用的系统提示
├── results/
│   ├── instructor_questions.json    # 15 个评估问题（6 instructor + 9 group）
│   └── evaluation_results.csv       # 完整评估输出（30 行）
├── src/
│   ├── config.py                    # 全局配置与环境设置
│   ├── model.py                     # 共享 LLM 加载器（单例模式）
│   ├── data_loader.py               # 数据加载器（scene/utterance 两级）
│   ├── chunking.py                  # 分块策略（摘要增强）
│   ├── retrieval.py                 # 嵌入检索器（带磁盘缓存）
│   ├── baseline.py                  # Baseline 系统（无检索）
│   ├── rag_chatbot.py               # RAG 聊天机器人（含莎翁风格生成）
│   ├── build_index.py               # 检索索引健全性测试
│   └── evaluate.py                  # 自动化评估脚本
├── requirements.txt                 # Python 依赖
└── README_0502_Chineseversion.md    # 本文件
```

---

## 各文件详细说明

### `src/config.py` — 全局配置

集中管理所有可调参数，并处理环境兼容性问题。

**环境修复**（第 7–18 行）：
- 设置 `TRANSFORMERS_NO_TF=1` 和 `USE_TF=0`，阻止 transformers 导入 TensorFlow
- 注入假的 `h5py` 模块，避免某些 Windows 环境下的 DLL 导入错误

**配置参数**：

| 参数 | 值 | 用途 |
|------|-----|------|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | 检索用的嵌入模型 |
| `GENERATION_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 本地生成模型（1.1B 参数） |
| `DEFAULT_TOP_K` | `3` | 每次检索返回的 chunk 数量 |
| `CHUNK_LEVEL` | `"scene"` | 分块粒度：`"scene"` 或 `"utterance"` |
| `GENERATION_MAX_NEW_TOKENS` | `512` | 最大输出长度 |
| `GENERATION_TEMPERATURE` | `0.7` | 采样温度（越低越确定性） |

---

### `src/model.py` — 共享语言模型加载器

**功能**：使用单例模式加载 TinyLlama 模型，确保 baseline 和 RAG 系统**共用同一个模型实例**。

**为什么需要共享**：
- TinyLlama 在内存中占用约 2GB（FP16）。加载两份会导致内存不足。
- 两个系统使用完全相同的生成参数，保证对比的公平性。

**关键函数**：
- `get_model()` — 返回 `(model, tokenizer)`，首次调用时加载，之后返回缓存的实例。
  - 有 CUDA 时自动使用 GPU（FP16），否则使用 CPU（FP32）。
- `generate(prompt, max_new_tokens, temperature)` — 应用 TinyLlama 聊天模板，分词、生成、解码。
  - 使用 `top_p=0.9` 核采样。
  - 输入截断到 2048 tokens 以适应模型上下文窗口。

---

### `src/data_loader.py` — 数据加载器

**功能**：读取三部莎士比亚戏剧的结构化 JSON 文件。

每个 JSON 文件结构：
```json
{
  "metadata": { "play": "Macbeth", "source": "Project Gutenberg", ... },
  "scenes": [
    {
      "scene_id": "macbeth_1_3",
      "play": "Macbeth",
      "act": 1,
      "scene": 3,
      "location": "A heath",
      "scene_summary": "女巫向 Macbeth 预言...",
      "keywords": ["prophecy", "ambition"],
      "utterances": [ { "speaker": "MACBETH", "text": "...", ... } ],
      "text": "...完整场景文本..."
    }
  ]
}
```

**两种提取粒度**：

| 粒度 | 记录数 | 适用场景 |
|------|--------|---------|
| `scene`（默认） | 73 | 每个场景一条记录，含 `text`、`scene_summary`、`keywords`。适合场景级分块。 |
| `utterance` | 3,613 | 每句台词一条记录，含 `speaker`、场景元数据。适合细粒度检索或自定义分块。 |

**关键函数**：
- `load_all_plays(level="scene")` — 加载三部戏剧，返回扁平记录列表。
- `load_play(path, level)` — 加载单部戏剧。

---

### `src/chunking.py` — 分块策略

**策略**：场景级分块 + 摘要/关键词增强。

**工作原理**：
1. 每个场景记录变成一个 chunk。
2. 在原始场景文本前拼接元数据：
   ```
   Summary: <场景摘要>
   Summary: <场景摘要>
   Keywords: <关键词1>, <关键词2>
   Location: <地点>
   <原始场景文本>
   ```
   这种增强帮助嵌入模型捕捉到早期近代英语中不易直接匹配的语义。
3. **长场景拆分**：超过 4,000 字符的场景自动拆成子 chunk，保留 400 字符重叠以维持上下文连续性。每个子 chunk 保留原始元数据。

**设计理由**：
- 场景级保留了叙事连贯性（角色互动、情节推进）。
- 摘要增强弥补了现代英语查询与莎士比亚古英语之间的词汇鸿沟。
- 4,000 字符阈值是在上下文完整性和嵌入模型容量之间的平衡（MiniLM 有效处理 256 tokens）。

**关键函数**：
- `create_chunks(records)` — 将记录转换为检索就绪的 chunk。
- `format_chunk_for_display(chunk)` — 格式化 chunk 用于终端显示。

---

### `src/retrieval.py` — 嵌入检索器（带磁盘缓存）

**功能**：将 chunk 编码为密集向量，并基于余弦相似度检索最相关的结果。

**核心类：`EmbeddingRetriever`**

| 方法 | 说明 |
|------|------|
| `build_index(chunks)` | 对所有 chunk 生成嵌入向量。优先从磁盘缓存加载；缓存未命中时计算并保存。 |
| `retrieve(query, top_k)` | 编码查询，计算与所有 chunk 的余弦相似度，返回 top-k 个 `(chunk, score)` 对。 |

**缓存机制**：
- 缓存路径：`data/cache/embeddings_<hash>.pkl`
- hash 基于嵌入模型名称 + 所有 chunk 文本的 MD5。
- 分块策略或嵌入模型变更时，hash 变化，自动重新计算。
- 首次运行：编码 73 个 chunk 约需 10 秒。后续运行：秒加载。

---

### `src/baseline.py` — Baseline 系统

**功能**：使用 TinyLlama **不带任何检索上下文**地生成回答，模型完全依赖自身预训练知识。

**Prompt 格式**：
```
You are a helpful Shakespeare assistant. Answer the following question about Shakespeare's plays.
Your answer should be beginner-friendly. If you are unsure, say so.

Question: <用户问题>

Answer:
```

这是一个受控对比：同一个模型、同样的生成参数，但没有检索。输出质量的差异可以归因于 RAG 检索组件。

**关键函数**：
- `baseline_answer(query)` — 返回模型的纯 prompt 生成回答。

---

### `src/rag_chatbot.py` — RAG 聊天机器人（核心）

**功能**：实现完整的 RAG 流程，支持全部 4 种交互类型。

**流程**：
```
用户查询 → is_stylised_query() 判断 → 检索 top-k chunk
         → build_rag_prompt() 或 build_stylised_prompt()
         → generate_answer() 通过共享模型生成
         → 返回回答 + 检索证据
```

**莎翁风格生成自动检测**：
- 查询包含 "shakespearean"、"style of"、"generate"、"poetic response" 等关键词时，触发专用的风格化 prompt。
- 风格化 prompt 指示模型用早期近代英语写 ≤150 词的回答。
- 输出前自动加上 `[NOTE: 以下是创意莎翁风格输出，非事实证据。]`

**关键函数**：

| 函数 | 说明 |
|------|------|
| `build_pipeline()` | 加载数据 → 分块 → 构建检索索引。返回 `(retriever, chunks)`。 |
| `rag_answer(query, retriever)` | 完整 RAG 流程，返回 `{query, answer, retrieved, prompt, stylised}`。 |
| `build_rag_prompt(query, retrieved)` | 构建 RAG 提示：系统 prompt + 检索上下文 + 查询。 |
| `build_stylised_prompt(query, retrieved)` | 构建创意生成提示，限制 150 词。 |
| `is_stylised_query(query)` | 判断查询是否请求莎翁风格生成。 |
| `main()` | 交互式聊天循环。 |

---

### `src/build_index.py` — 检索健全性测试

**功能**：验证 数据加载 → 分块 → 嵌入 → 检索 整条流水线，**不加载生成模型**。

运行 3 个测试查询，打印每个的 top-3 结果（含戏剧、幕、场、相似度分数、摘要）。用于快速验证。

---

### `src/evaluate.py` — 自动化评估脚本

**功能**：对全部 15 个评估问题分别运行 baseline 和 RAG，将结构化结果保存到 CSV。

**输出**：`results/evaluation_results.csv`，共 30 行（15 题 × 2 系统）。

**CSV 列说明**：

| 列名 | 说明 |
|------|------|
| `question_id` | Q1–Q6（instructor），G1–G9（group） |
| `question` | 评估问题 |
| `question_type` | `contextual_qa`、`concept_explanation`、`evidence_retrieval`、`stylised_generation`、`robustness` |
| `question_source` | `instructor` 或 `group` |
| `system` | `baseline` 或 `rag` |
| `is_stylised` | RAG 是否使用了莎翁风格生成 prompt |
| `retrieved_passages` | top-k 检索到的 chunk 及分数（仅 RAG） |
| `generated_response` | 系统生成的回答 |
| `correctness_score` | 手动评分 1–5（空白） |
| `grounding_score` | 手动评分 1–5（空白） |
| `retrieval_relevance_score` | 手动评分 1–5（空白） |
| `usefulness_score` | 手动评分 1–5（空白） |
| `style_quality_score` | 手动评分 1–5（空白） |
| `comments` | 自由文本备注（空白） |

---

### `prompts/system_prompt.txt` — RAG 系统提示

指示模型：
1. 使用检索到的上下文回答问题
2. 保持回答对初学者友好
3. 上下文不足时明确说明
4. 不编造无依据的细节
5. 莎翁风格回答限 150 词，并标明为创意输出

---

## 快速上手

### 前置要求

- Python 3.9+
- 约 4GB 磁盘空间（首次下载模型）
- 至少 4GB 内存（CPU 模式）；有 ≥4GB 显存的 GPU 更好

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要的包：`numpy`、`pandas`、`scikit-learn`、`sentence-transformers`、`tqdm`、`torch`、`transformers`、`accelerate`

> 首次运行时模型自动从 HuggingFace 下载：
> - `sentence-transformers/all-MiniLM-L6-v2`（约 90MB）
> - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`（约 2GB）

### 2. 测试检索流水线（快速，不需要 LLM）

```bash
cd assignment2_starter_code
python src/build_index.py
```

预期输出：加载 73 个 chunk，3 个测试查询，每个显示 top-3 检索结果及相似度分数。

### 3. 启动 RAG 交互式聊天

```bash
python src/rag_chatbot.py
```

输入问题后回车，系统显示：
- 检索到的证据（戏剧、幕、场、相似度分数、文本摘录）
- 生成的回答

示例查询：
- `Why does Macbeth kill Duncan?` — 情节问答
- `Who is Ophelia?` — 概念解释
- `Generate a short Shakespearean-style speech from Hamlet` — 莎翁风格生成

输入 `quit` 退出。

### 4. 测试 Baseline 系统

```bash
python src/baseline.py
```

用一个样例问题测试纯 prompt 生成。

### 5. 运行完整评估

```bash
python src/evaluate.py
```

对全部 15 个问题运行 baseline + RAG，输出：`results/evaluation_results.csv`（30 行）。

运行后，手动在 CSV 中填写 5 个评分列（1–5 分），用于报告的评估表格。

---

## 评估问题

### 分布（共 15 题）

| 来源 | 编号 | 类型 | 戏剧 |
|------|------|------|------|
| Instructor | Q1 | contextual_qa | Macbeth |
| Instructor | Q2 | contextual_qa | Macbeth |
| Instructor | Q3 | contextual_qa | Hamlet |
| Instructor | Q4 | contextual_qa | Hamlet |
| Instructor | Q5 | contextual_qa | Romeo and Juliet |
| Instructor | Q6 | contextual_qa | Romeo and Juliet |
| Group | G1 | concept_explanation | Macbeth |
| Group | G2 | concept_explanation | Hamlet |
| Group | G3 | concept_explanation | Romeo and Juliet |
| Group | G4 | evidence_retrieval | Macbeth |
| Group | G5 | evidence_retrieval | Hamlet |
| Group | G6 | contextual_qa | Romeo and Juliet |
| Group | G7 | stylised_generation | Macbeth |
| Group | G8 | stylised_generation | Hamlet |
| Group | G9 | robustness | N/A（域外问题） |

G9（"Who is Mingzhao Zhu?"）是一个故意设计的域外问题，测试系统如何处理与莎士比亚无关的查询。

### 评分标准

| 分数 | 含义 |
|------|------|
| 5 | 优秀 — 准确、有据、对初学者有帮助 |
| 4 | 良好 — 基本正确，有小瑕疵 |
| 3 | 一般 — 部分正确，有遗漏 |
| 2 | 较差 — 有明显错误或缺少关键信息 |
| 1 | 很差 — 错误、幻觉或无关 |

---

## 4 种交互类型

| 类型 | 说明 | 示例 | 检测方式 |
|------|------|------|---------|
| **概念解释** | 解释角色、关系或主题 | "Who is Lady Macbeth?" | 默认 |
| **情节问答** | 回答关于事件和动机的问题 | "Why does Macbeth kill Duncan?" | 默认 |
| **证据检索** | 检索并展示带元数据的原文段落 | "What do the witches predict?" | 默认 |
| **莎翁风格生成** | 创意莎士比亚风格回答（≤150词） | "Generate a Shakespearean speech from Hamlet" | 关键词自动检测 |

莎翁风格查询通过关键词匹配自动检测（如 "shakespearean"、"style of"、"generate"、"poetic"），输出明确标记为创意内容而非事实证据。

---

## 已知局限性

1. **TinyLlama 生成质量有限**：作为 1.1B 参数模型，TinyLlama 有时会幻觉（编造事实）或不能精确遵循指令。这是在普通硬件上运行的已知代价。

2. **莎翁风格输出长度不可控**：模型不一定能遵守 150 词限制。

3. **域外问题处理差**：面对与莎士比亚无关的问题（如 G9），模型可能编造答案或复读系统 prompt，而不是明确拒绝。

4. **Prompt 泄漏**：在某些边界情况（检索分数很低时），模型可能把系统 prompt 的部分内容原样输出。

5. **上下文窗口限制**：TinyLlama 有效上下文约 2048 tokens。过长的检索段落会被截断，可能丢失相关信息。

这些局限性在报告的评估部分作为 failure analysis 讨论。

---

## Embedding 缓存

首次运行时，chunk 嵌入向量计算后保存到 `data/cache/embeddings_<hash>.pkl`。hash 由嵌入模型名称 + chunk 内容的 MD5 生成。后续运行秒加载。

缓存自动失效条件：
- 分块策略变更（不同的 chunk → 不同的 hash）
- 嵌入模型变更
- 数据集修改

强制重新计算：删除 `data/cache/` 目录。

---

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| `h5py` DLL 导入错误 | 已在 `config.py` 中通过假模块注入解决 |
| TensorFlow 导入警告 | 已在 `config.py` 中通过环境变量屏蔽 |
| CPU 上模型太慢 | 正常：每个回答约 30–60 秒。有 GPU 的话会快很多。 |
| 内存不足 | 关闭其他应用；TinyLlama 需约 2GB（FP16）或 4GB（FP32）内存 |
| Windows 上的 `symlink` 警告 | 不影响功能，可忽略 |
| Flash attention 警告 | 不影响功能，TinyLlama 会回退到标准 attention |
