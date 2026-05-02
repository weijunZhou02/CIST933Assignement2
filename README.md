# Shakespeare RAG System — README

## Project Overview

**Course**: CSCI433/933 Machine Learning Algorithms and Applications — Assignment 2

This project implements a **Retrieval-Augmented Generation (RAG) system** for Shakespeare play question answering. The system is designed to help a user with **no prior Shakespeare knowledge** ask questions, retrieve relevant textual evidence, and receive beginner-friendly explanations.

### System Pipeline

```
User Question
      │
      ▼
┌─────────────────┐   embed query    ┌───────────────────┐
│  Embedding Model │─────────────────▶│  Cosine Similarity │
│  (MiniLM-L6-v2) │                  │  against 73 chunks │
└─────────────────┘                  └────────┬──────────┘
                                              │ top-k chunks
                                              ▼
                                     ┌───────────────────┐
                                     │  Build RAG Prompt  │
                                     │  (system prompt +  │
                                     │   retrieved text + │
                                     │   user question)   │
                                     └────────┬──────────┘
                                              │
                                              ▼
                                     ┌───────────────────┐
                                     │  TinyLlama 1.1B   │
                                     │  Generate Answer   │
                                     └────────┬──────────┘
                                              │
                                              ▼
                                     ┌───────────────────┐
                                     │  Display Answer +  │
                                     │  Retrieved Evidence │
                                     └───────────────────┘
```

A **Baseline system** (prompt-only, no retrieval) is also implemented for comparison evaluation.

### Key Design Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| Generation Model | TinyLlama-1.1B-Chat-v1.0 | Smallest chat-capable model; runs on CPU; aligns with SLM constraints |
| Embedding Model | all-MiniLM-L6-v2 | Fast, lightweight (80MB); widely used baseline for semantic search |
| Chunking Strategy | Scene-level + summary enrichment | Preserves narrative context; summaries improve retrieval accuracy |
| Retrieval Method | Cosine similarity on dense embeddings | Simple, transparent, easy to evaluate; no external dependencies |
| Shared Model Instance | Single model for both baseline and RAG | Halves memory usage; ensures fair comparison |

---

## Project Structure

```
assignment2_starter_code/
├── data/
│   ├── processed/                   # Structured Shakespeare dataset
│   │   ├── hamlet.json              #   20 scenes, 1657 utterances
│   │   ├── macbeth.json             #   28 scenes, 830 utterances
│   │   └── romeo_and_juliet.json    #   25 scenes, 1126 utterances
│   └── cache/                       # Embedding cache (auto-generated)
│       └── embeddings_<hash>.pkl    #   Invalidates on chunk/model change
├── prompts/
│   └── system_prompt.txt            # System prompt for RAG generation
├── results/
│   ├── instructor_questions.json    # 15 evaluation questions (6 instructor + 9 group)
│   └── evaluation_results.csv       # Full evaluation output (30 rows)
├── src/
│   ├── config.py                    # Global configuration and environment setup
│   ├── model.py                     # Shared LLM loader (singleton pattern)
│   ├── data_loader.py               # Dataset loader (scene/utterance level)
│   ├── chunking.py                  # Chunking strategy with summary enrichment
│   ├── retrieval.py                 # Embedding retriever with disk caching
│   ├── baseline.py                  # Baseline system (no retrieval)
│   ├── rag_chatbot.py               # RAG chatbot with stylised generation support
│   ├── build_index.py               # Retrieval index sanity check
│   └── evaluate.py                  # Automated evaluation script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Source File Descriptions

### `src/config.py` — Global Configuration

Centralizes all tunable parameters and handles environment setup.

**Environment fixes** (lines 7–18):
- Sets `TRANSFORMERS_NO_TF=1` and `USE_TF=0` to prevent transformers from importing TensorFlow
- Injects a fake `h5py` module to avoid DLL import errors on some Windows environments

**Configuration parameters**:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for retrieval |
| `GENERATION_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Local generation model (1.1B params) |
| `DEFAULT_TOP_K` | `3` | Number of retrieved chunks per query |
| `CHUNK_LEVEL` | `"scene"` | Granularity: `"scene"` or `"utterance"` |
| `GENERATION_MAX_NEW_TOKENS` | `512` | Max output length |
| `GENERATION_TEMPERATURE` | `0.7` | Sampling temperature (lower = more deterministic) |

---

### `src/model.py` — Shared Language Model Loader

**Purpose**: Loads the TinyLlama model **once** and shares it across baseline and RAG systems using a singleton pattern.

**Why this matters**:
- TinyLlama occupies ~2GB in memory (FP16). Loading two copies would cause OOM on machines with limited RAM/VRAM.
- Both systems use identical generation parameters, ensuring a fair comparison.

**Key functions**:
- `get_model()` — Returns `(model, tokenizer)`. Loads on first call, returns cached instance afterwards.
  - Automatically uses GPU (FP16) if CUDA is available; falls back to CPU (FP32).
- `generate(prompt, max_new_tokens, temperature)` — Applies the TinyLlama chat template, tokenizes, generates, and decodes the response.
  - Uses `top_p=0.9` nucleus sampling with the configured temperature.
  - Truncates input to 2048 tokens to fit within the model's context window.

---

### `src/data_loader.py` — Dataset Loader

**Purpose**: Reads the three structured Shakespeare play JSON files.

Each JSON file contains:
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
      "scene_summary": "The witches greet Macbeth with prophecies...",
      "keywords": ["prophecy", "ambition"],
      "utterances": [ { "speaker": "MACBETH", "text": "...", ... } ],
      "text": "...full scene text..."
    }
  ]
}
```

**Two extraction levels**:

| Level | Records | Use Case |
|-------|---------|----------|
| `scene` (default) | 73 | One record per scene. Includes `text`, `scene_summary`, `keywords`. Ideal for scene-level chunking. |
| `utterance` | 3,613 | One record per spoken line. Includes `speaker`, scene metadata. For fine-grained retrieval or custom chunking. |

**Key functions**:
- `load_all_plays(level="scene")` — Loads all three plays, returns a flat list of records.
- `load_play(path, level)` — Loads a single play from a JSON file.

---

### `src/chunking.py` — Chunking Strategy

**Strategy**: Scene-level chunking with summary and keyword enrichment.

**How it works**:
1. Each scene record becomes one chunk.
2. The chunk text is constructed by prepending metadata to the raw scene text:
   ```
   Summary: <scene_summary>
   Summary: <scene_summary>
   Keywords: <keyword1>, <keyword2>
   Location: <location>
   <original scene text>
   ```
   This enrichment helps the embedding model capture semantic meaning that may not be explicit in Early Modern English.
3. **Long scene splitting**: Scenes exceeding 4,000 characters are split into sub-chunks with 400-character overlap to preserve context continuity. Each sub-chunk retains the original metadata.

**Justification**:
- Scene-level preserves narrative coherence (character interactions, plot progression).
- Summary enrichment bridges the vocabulary gap between modern queries and Shakespearean text.
- The 4,000-character threshold was chosen to balance context completeness against embedding model capacity (MiniLM processes up to 256 tokens effectively).

**Key functions**:
- `create_chunks(records)` — Converts records to retrieval-ready chunks.
- `format_chunk_for_display(chunk)` — Formats a chunk for terminal output with metadata.

---

### `src/retrieval.py` — Embedding Retriever with Disk Caching

**Purpose**: Encodes chunks into dense vectors and retrieves the most relevant ones for a query.

**Core class: `EmbeddingRetriever`**

| Method | Description |
|--------|-------------|
| `build_index(chunks)` | Encodes all chunk texts into embeddings. Checks disk cache first; computes and saves to cache on miss. |
| `retrieve(query, top_k)` | Encodes the query, computes cosine similarity against all chunk embeddings, returns top-k `(chunk, score)` pairs. |

**Caching mechanism**:
- Cache stored in `data/cache/embeddings_<hash>.pkl`
- Hash is based on the embedding model name and an MD5 of all chunk texts.
- If chunking strategy or embedding model changes, the hash changes, and embeddings are recomputed automatically.
- First run: ~10 seconds to encode 73 chunks. Subsequent runs: instant.

---

### `src/baseline.py` — Baseline System

**Purpose**: Generates answers using TinyLlama **without any retrieved context**. The model relies entirely on its pretrained knowledge.

**Prompt format**:
```
You are a helpful Shakespeare assistant. Answer the following question about Shakespeare's plays.
Your answer should be beginner-friendly. If you are unsure, say so.

Question: <user query>

Answer:
```

This serves as a controlled comparison: same model, same generation parameters, but no retrieval. Differences in output quality can be attributed to the RAG retrieval component.

**Key function**:
- `baseline_answer(query)` — Returns the model's prompt-only answer.

---

### `src/rag_chatbot.py` — RAG Chatbot (Core)

**Purpose**: Implements the full RAG pipeline and supports all 4 interaction types.

**Pipeline**:
```
User query → is_stylised_query() check → retrieve top-k chunks
           → build_rag_prompt() or build_stylised_prompt()
           → generate_answer() via shared model
           → return answer + retrieved evidence
```

**Stylised generation auto-detection**:
- Queries containing keywords like "shakespearean", "style of", "generate", "poetic response" trigger a special stylised prompt.
- Stylised prompt instructs the model to write ≤150 words in Early Modern English.
- Output is prefixed with `[NOTE: The following is creative Shakespearean-style output, not factual evidence.]`

**Key functions**:

| Function | Description |
|----------|-------------|
| `build_pipeline()` | Load data → chunk → build retrieval index. Returns `(retriever, chunks)`. |
| `rag_answer(query, retriever)` | Full RAG pipeline. Returns `{query, answer, retrieved, prompt, stylised}`. |
| `build_rag_prompt(query, retrieved)` | Constructs the RAG prompt with system prompt + retrieved context + query. |
| `build_stylised_prompt(query, retrieved)` | Constructs the creative generation prompt with 150-word limit. |
| `is_stylised_query(query)` | Returns `True` if query contains stylised generation keywords. |
| `main()` | Interactive chat loop. |

---

### `src/build_index.py` — Retrieval Sanity Check

**Purpose**: Validates the data loading → chunking → embedding → retrieval pipeline without loading the generation model.

Runs 3 test queries and prints top-3 results for each, showing play, act, scene, similarity score, and summary. Useful for quick verification.

---

### `src/evaluate.py` — Automated Evaluation Script

**Purpose**: Runs both baseline and RAG on all 15 evaluation questions and saves structured results to CSV.

**Output**: `results/evaluation_results.csv` with 30 rows (15 questions × 2 systems).

**CSV columns**:

| Column | Description |
|--------|-------------|
| `question_id` | Q1–Q6 (instructor), G1–G9 (group) |
| `question` | The evaluation question |
| `question_type` | `contextual_qa`, `concept_explanation`, `evidence_retrieval`, `stylised_generation`, `robustness` |
| `question_source` | `instructor` or `group` |
| `system` | `baseline` or `rag` |
| `is_stylised` | Whether RAG used the stylised generation prompt |
| `retrieved_passages` | Top-k retrieved chunks with scores (RAG only) |
| `generated_response` | The system's answer |
| `correctness_score` | Manual score 1–5 (blank) |
| `grounding_score` | Manual score 1–5 (blank) |
| `retrieval_relevance_score` | Manual score 1–5 (blank) |
| `usefulness_score` | Manual score 1–5 (blank) |
| `style_quality_score` | Manual score 1–5 (blank) |
| `comments` | Free-text notes (blank) |

---

### `prompts/system_prompt.txt` — RAG System Prompt

Instructs the model to:
1. Use retrieved context to answer the question.
2. Keep answers beginner-friendly.
3. Say clearly when retrieved context is insufficient.
4. Not invent unsupported details.
5. Limit Shakespearean-style responses to 150 words and label them as creative output.

---

## Quick Start

### Prerequisites

- Python 3.9+
- ~4GB free disk space (for models on first download)
- ~4GB RAM minimum (CPU mode); GPU with ≥4GB VRAM recommended

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `scikit-learn`, `sentence-transformers`, `tqdm`, `torch`, `transformers`, `accelerate`

> On first run, models are downloaded automatically from HuggingFace:
> - `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)
> - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~2 GB)

### 2. Test Retrieval Pipeline (fast, no LLM needed)

```bash
cd assignment2_starter_code
python src/build_index.py
```

Expected output: 73 chunks loaded, 3 test queries, each showing top-3 retrieved scenes with similarity scores.

### 3. Launch Interactive RAG Chatbot

```bash
python src/rag_chatbot.py
```

Type a question and press Enter. The system displays:
- Retrieved evidence (play, act, scene, similarity score, text excerpt)
- Generated answer

Example queries:
- `Why does Macbeth kill Duncan?` — contextual QA
- `Who is Ophelia?` — concept explanation
- `Generate a short Shakespearean-style speech from Hamlet` — stylised generation

Type `quit` to exit.

### 4. Test Baseline System

```bash
python src/baseline.py
```

Runs a quick test with a sample question using prompt-only generation.

### 5. Run Full Evaluation

```bash
python src/evaluate.py
```

Runs all 15 questions through both systems. Output: `results/evaluation_results.csv` (30 rows).

After running, manually fill in the 5 scoring columns (1–5 scale) for each row. This scoring data is used in the report's evaluation tables.

---

## Evaluation Questions

### Distribution (15 total)

| Source | ID | Type | Play |
|--------|----|------|------|
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
| Group | G9 | robustness | N/A (out-of-domain) |

G9 ("Who is Mingzhao Zhu?") is an intentional out-of-domain question to test how the system handles queries unrelated to Shakespeare.

### Scoring Rubric

| Score | Meaning |
|-------|---------|
| 5 | Excellent — accurate, well-grounded, helpful |
| 4 | Good — mostly correct, minor issues |
| 3 | Adequate — partially correct, some gaps |
| 2 | Poor — significant errors or missing key information |
| 1 | Very poor — incorrect, hallucinated, or irrelevant |

---

## 4 Interaction Types

| Type | Description | Example | Detection |
|------|-------------|---------|-----------|
| **Concept Explanation** | Explain a character, relationship, or theme | "Who is Lady Macbeth?" | Default |
| **Contextual QA** | Answer questions about events and motivations | "Why does Macbeth kill Duncan?" | Default |
| **Evidence Retrieval** | Retrieve and display source passages with metadata | "What do the witches predict?" | Default |
| **Stylised Generation** | Creative Shakespearean-style response (≤150 words) | "Generate a Shakespearean speech from Hamlet" | Auto-detected via keywords |

Stylised queries are detected by keyword matching (e.g., "shakespearean", "style of", "generate", "poetic"). The output is clearly labelled as creative content, not factual evidence.

---

## Known Limitations

1. **TinyLlama generation quality**: As a 1.1B parameter model, TinyLlama sometimes hallucinates facts or fails to follow instructions precisely. This is an expected trade-off for running locally on modest hardware.

2. **Stylised output length**: The model does not always respect the 150-word limit for stylised generation.

3. **Out-of-domain handling**: When asked questions unrelated to Shakespeare (e.g., G9), the model may fabricate answers or repeat the system prompt instead of clearly declining.

4. **Prompt leakage**: In some edge cases (very low retrieval scores), the model may echo parts of the system prompt in its response.

5. **Context window**: TinyLlama's effective context is ~2048 tokens. Very long retrieved passages are truncated, which may lose relevant information.

These limitations are discussed in the evaluation section of the report as part of failure analysis.

---

## Embedding Cache

On first run, chunk embeddings are computed and saved to `data/cache/embeddings_<hash>.pkl`. The hash is derived from the embedding model name and chunk content (MD5). Subsequent runs load from cache instantly.

Cache invalidates automatically when:
- The chunking strategy changes (different chunks → different hash)
- The embedding model changes
- The dataset is modified

To force recomputation, delete the `data/cache/` directory.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `h5py` DLL import error | Already handled in `config.py` via fake module injection |
| TensorFlow import warnings | Already suppressed in `config.py` via environment variables |
| Model too slow on CPU | Expected: ~30–60s per answer on CPU. Use GPU if available. |
| Out of memory | Close other applications; TinyLlama needs ~2GB RAM (FP16) or ~4GB (FP32) |
| `symlink` warnings on Windows | Non-blocking; can be ignored |
| Flash attention warnings | Non-blocking; TinyLlama falls back to standard attention |
