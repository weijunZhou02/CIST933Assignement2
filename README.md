# README 0502 — Shakespeare RAG System

## Project Overview

This project is CSCI433/933 Assignment 2, implementing a **Retrieval-Augmented Generation (RAG) system for Shakespeare play Q&A**.

System flow: User query → Retrieve relevant passages from Shakespeare plays → Feed passages as context to a local small language model → Generate a beginner-friendly answer.

A **Baseline system (no retrieval)** is also implemented for comparison.

---

## Project Structure

```
assignment2_starter_code/
├── data/processed/              # Dataset (3 JSON files)
│   ├── hamlet.json
│   ├── macbeth.json
│   └── romeo_and_juliet.json
├── prompts/
│   └── system_prompt.txt        # System prompt for the RAG system
├── results/
│   ├── instructor_questions.json # Questions for evaluation
│   └── evaluation_results.csv   # Evaluation output (generated after running evaluate)
├── data/cache/                  # Embedding cache (auto-generated)
├── src/
│   ├── config.py                # Global configuration
│   ├── model.py                 # Shared LLM loader (used by both baseline and RAG)
│   ├── data_loader.py           # Data loader
│   ├── chunking.py              # Chunking strategies
│   ├── retrieval.py             # Embedding and retrieval (with cache)
│   ├── baseline.py              # Baseline system (no retrieval)
│   ├── rag_chatbot.py           # RAG chatbot (core)
│   ├── build_index.py           # Retrieval index construction and testing
│   └── evaluate.py              # Evaluation script
├── requirements.txt             # Python dependencies
└── README_0502.md               # This file
```

---

## File Descriptions

### `src/config.py` — Global Configuration

Centralizes all tunable parameters:
- **Data paths**: `DATA_DIR`, `PLAY_FILES` pointing to the three play JSON files
- **Embedding model**: `EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"` (lightweight, converts text to vectors)
- **Generation model**: `GENERATION_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"` (1.1B parameter local model)
- **Retrieval count**: `DEFAULT_TOP_K = 3` (returns 3 most relevant chunks per query)
- **Chunk granularity**: `CHUNK_LEVEL = "scene"` (chunk by scene)
- **Generation parameters**: `GENERATION_MAX_NEW_TOKENS = 512`, `GENERATION_TEMPERATURE = 0.7`

> To switch models or tune parameters, only this file needs to be modified.

---

### `src/data_loader.py` — Data Loader

**Function**: Reads the three Shakespeare play JSON files, supporting two levels of extraction:

| Level | Description | Record Count |
|-------|-------------|--------------|
| `scene` | One scene = one record (includes full scene text, summary, keywords) | 73 records |
| `utterance` | One line = one record (includes speaker, scene information) | 3,613 records |

**Key functions**:
- `load_all_plays(level="scene")` — Loads all three plays, returns a list of records
- `load_play(path, level)` — Loads a single play

---

### `src/chunking.py` — Chunking Strategy

**Function**: Converts records loaded by `data_loader` into chunks suitable for retrieval.

**Strategy: Scene-level + Summary Enhancement**
- Each scene becomes one chunk
- Prepends `scene_summary`, `keywords`, and `location` to the raw text, helping the embedding capture semantics more effectively
- Scenes longer than 4,000 characters are automatically split with a 400-character overlap to preserve context continuity

**Key functions**:
- `create_chunks(records)` — Takes a list of records, returns a list of chunks
- `format_chunk_for_display(chunk)` — Formats a chunk for terminal display

---

### `src/retrieval.py` — Embedding and Retrieval

**Function**: Uses sentence-transformers to generate embedding vectors, then retrieves using cosine similarity.

**Core class: `EmbeddingRetriever`**
- `build_index(chunks)` — Generates embeddings for all chunks and builds the retrieval index
- `retrieve(query, top_k=3)` — Takes a query, returns the top-k most relevant `(chunk, score)` pairs

---

### `src/baseline.py` — Baseline System (No Retrieval)

**Function**: **Pure prompt generation** — no context retrieval, the model answers directly from its own knowledge.

This is the baseline for comparison against the RAG system. Both use the same TinyLlama model; the only difference is that the baseline receives no retrieved context.

**Key function**:
- `baseline_answer(query)` — Takes a question, returns the model's generated answer

---

### `src/rag_chatbot.py` — RAG Chatbot (Core)

**Function**: Full RAG pipeline:

```
User query → Embedding retrieval of top-k relevant scenes → Build context-aware prompt → TinyLlama generates answer
```

**Key functions**:
- `build_pipeline()` — Load data → chunk → build retrieval index, returns `(retriever, chunks)`
- `rag_answer(query, retriever)` — Runs the full RAG pipeline for one question, returns `{query, answer, retrieved, prompt}`
- `generate_answer(prompt)` — Calls TinyLlama to generate an answer
- `main()` — Interactive chat loop

---

### `src/build_index.py` — Retrieval Index Testing

**Function**: Validates the full pipeline — data loading → chunking → embedding → retrieval.

Runs 3 test queries and prints the top-3 retrieved results for each (including play, act, scene, score, summary).

**Purpose**: Quick sanity check. Does not involve the generation model, so it runs fast.

---

### `src/evaluate.py` — Evaluation Script

**Function**: Runs both the baseline and RAG systems on all questions in `results/instructor_questions.json`, writing results to `results/evaluation_results.csv`.

Each question produces two rows in the CSV (baseline + rag), containing:
- The system's generated answer
- RAG-retrieved passages
- 5 scoring columns (left blank for manual scoring): correctness / grounding / retrieval_relevance / usefulness / style_quality

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `scikit-learn`, `sentence-transformers`, `tqdm`, `torch`, `transformers`, `accelerate`

> On first run, the sentence-transformers and TinyLlama models will be downloaded automatically from HuggingFace (~90MB and ~2GB respectively).

### 2. Test Retrieval (No Generation Model Required)

```bash
cd assignment2_starter_code
python src/build_index.py
```

Verifies that data loading, chunking, embedding, and retrieval work correctly. Outputs the top-3 relevant scenes for each test query.

### 3. Launch the RAG Chatbot

```bash
python src/rag_chatbot.py
```

Enters interactive mode. Type a question to get a RAG-generated answer. Type `quit` to exit.

### 4. Test the Baseline

```bash
python src/baseline.py
```

Runs a simple baseline test.

### 5. Run Full Evaluation

```bash
python src/evaluate.py
```

Automatically runs baseline + RAG on all 15 questions (6 instructor + 9 group-designed) and outputs results to `results/evaluation_results.csv`.

Manually score each response in the CSV (1–5 scale) for use in the evaluation section of the report.

---

## 4 Interaction Types

The system supports all 4 interaction types required by the spec:

| Type | Description | Example |
|------|-------------|---------|
| **concept_explanation** | Explain a character or concept | "Who is Lady Macbeth?" |
| **contextual_qa** | Plot-based Q&A | "Why does Macbeth kill Duncan?" |
| **evidence_retrieval** | Retrieve and present textual evidence | "What do the witches predict?" |
| **stylised_generation** | Shakespearean-style creative generation (≤150 words) | "Generate a Shakespearean-style speech from Hamlet" |

Stylised generation is **auto-detected** via keywords such as "shakespearean", "generate", "style of", and outputs are prefixed with `[NOTE: creative output, not factual evidence]`.

---

## Evaluation Question Distribution

15 questions in total:

| Source | Count | Types Covered |
|--------|-------|---------------|
| Instructor | 6 | contextual_qa |
| Group (G1–G3) | 3 | concept_explanation |
| Group (G4–G5) | 2 | evidence_retrieval |
| Group (G6) | 1 | contextual_qa |
| Group (G7–G8) | 2 | stylised_generation |
| Group (G9) | 1 | robustness |

---

## Embedding Cache

On first run, embeddings are automatically cached to `data/cache/`. Subsequent runs load from cache directly, skipping recomputation. If the chunking strategy or embedding model changes, the cache is automatically invalidated (based on content hash).

---

## Tunable Configuration

Modify in `src/config.py`:

| Parameter | Current Value | Description |
|-----------|---------------|-------------|
| `GENERATION_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Swap for another model e.g. `microsoft/phi-2` |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `DEFAULT_TOP_K` | `3` | Number of chunks returned by retrieval |
| `CHUNK_LEVEL` | `"scene"` | Change to `"utterance"` for utterance-level chunking |
| `GENERATION_MAX_NEW_TOKENS` | `512` | Maximum tokens in generated answer |
| `GENERATION_TEMPERATURE` | `0.7` | Lower = more deterministic; higher = more diverse |

---

## System Flow Diagram

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  User    │───▶│  Embedding   │───▶│   Cosine     │
│  Query   │    │  (MiniLM)    │    │  Similarity  │
└──────────┘    └──────────────┘    └──────┬───────┘
                                           │ top-k chunks
                                           ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  Final   │◀───│  TinyLlama   │◀───│  RAG Prompt  │
│  Answer  │    │  Generation  │    │  (context +  │
└──────────┘    └──────────────┘    │   question)  │
                                    └──────────────┘
```
