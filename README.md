# RAG Arena

A benchmarking framework for evaluating RAG retrieval hypotheses. 55+ retrieval strategies tested across 250+ experiment runs on standardised IR benchmarks.

## Motivation

Most RAG systems use the same retrieve-and-generate pattern, but the retrieval step has many possible strategies — geometric reranking, hubness correction, diversity sampling, spectral methods, and more. RAG Arena provides a controlled environment to test these hypotheses head-to-head on real benchmarks with published baselines.

## Key Findings

- **Composability breakdown**: Individually beneficial retrieval strategies (anti-hubness, DPP diversity, score calibration) lose their gains when composed together — the interactions cancel out.
- **Cross-encoder reranking dominates**: A strong cross-encoder reranker outperforms all embedding-space retrieval tricks. The reranker quality is the bottleneck, not candidate generation.
- **Sentence-level retrieval matters**: Indexing at sentence granularity surfaces documents that passage-level embeddings miss, improving nDCG@10 by +9-16% across benchmarks.
- **Hybrid always wins**: BM25 + dense retrieval with RRF fusion gives up to 580% recall improvement over either alone.

## Getting Started

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.ai)** running locally — used for embeddings and (optionally) LLM generation

### 1. Install

```bash
git clone https://github.com/hsamuelson/rag-arena.git
cd rag-arena
pip install -e ".[dev]"
```

### 2. Start Ollama and pull an embedding model

```bash
# In a separate terminal:
ollama serve

# Pull the default embedding model:
ollama pull nomic-embed-text
```

### 3. Run your first experiment

The fastest way to see results — run the default 10 hypotheses on the synthetic benchmark (no downloads needed, no LLM generation):

```bash
rag-arena run --skip-llm
```

This retrieves with cosine similarity, applies each hypothesis's reranking strategy, and prints a comparison table with recall@K, precision@K, nDCG@K, and MRR.

### 4. Try a real benchmark

HotpotQA is a good next step — multi-hop questions over Wikipedia passages (auto-downloads from HuggingFace):

```bash
rag-arena run -b hotpotqa -H flat dpp anti-hubness --skip-llm --max-samples 50
```

### 5. Run the full benchmark suite

Test all 10 default hypotheses across synthetic, HotpotQA, and BEIR:

```bash
rag-arena bench --max-samples 50
```

Results are saved as JSON to `results/` and can be compared later:

```bash
rag-arena compare results/*.json
```

## Configuration

Generate a default config file:

```bash
rag-arena init-config
```

This creates `arena.yaml`:

```yaml
ollama:
  base_url: http://localhost:11434
  embed_model: nomic-embed-text
  chat_model: qwen3.5:122b    # only needed without --skip-llm
  embed_dimensions: 768

data_dir: ./data
results_dir: ./results
top_k: 10
n_components: 3                # PCA components for SEMDA hypotheses
```

You can also use a different embedding model. For sentence-transformers models (e.g., Snowflake Arctic, GTE), add:

```yaml
st_embed_model: "Snowflake/snowflake-arctic-embed-l"
```

Use a custom config with any command:

```bash
rag-arena run --config my-config.yaml -b hotpotqa --skip-llm
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `rag-arena run` | Run an experiment (benchmark + backend + hypotheses) |
| `rag-arena bench` | Run full suite: 10 hypotheses x 3 benchmarks |
| `rag-arena compare <files>` | Compare saved JSON result files side-by-side |
| `rag-arena baselines` | Print published industry baselines for reference |
| `rag-arena init-config` | Generate a default `arena.yaml` |

### Common flags

```
-b, --benchmark    synthetic | locomo | nq | hotpotqa | beir
--backend          direct | bm25 | hybrid
-H, --hypotheses   Space-separated list (e.g., flat dpp anti-hubness)
-n, --max-samples  Limit number of queries (useful for quick tests)
--skip-llm         Skip LLM answer generation (retrieval metrics only)
-c, --config       Path to config YAML
-o, --output       Output JSON path
-q, --quiet        Suppress per-query output
```

## Hypotheses

### Default Scaling Set (10)

These run by default with `rag-arena run` and `rag-arena bench`:

| CLI Key | Strategy | Principle |
|---------|----------|-----------|
| `flat` | Flat Baseline | Control (standard cosine similarity) |
| `dpp` | DPP Selection | Parallelotope volume maximisation |
| `anti-hubness` | Anti-Hubness + CSLS | Hub detection + cross-domain local scaling |
| `mean-bias` | Mean Bias Correction | Anisotropy removal (centering) |
| `capacity-partition` | Capacity Partitioning | Embedding capacity ceiling sharding |
| `score-calibration` | Score Calibration | Z-score by local density |
| `spectral-gap` | Spectral Gap Clustering | Laplacian eigengap heuristic |
| `centroid-drift` | Centroid Drift | Centroid-query distance minimisation |
| `lid-weighted` | Local Intrinsic Dimension | TwoNN LID estimator |
| `pca-diversity` | PCA-MMR Diversity | SVD + MMR in PCA space |

### Full Catalog (55+)

Additional hypotheses span generative methods (HyDE, query decomposition), cross-encoder reranking, deep candidate pools, LLM-powered retrieval (IRCoT), multi-resolution memory (MRAM), and hybrid compositions. See `rag-arena run -H --help` for the complete list.

## Benchmarks

| CLI Key | Benchmark | Source | Published Baselines |
|---------|-----------|--------|---------------------|
| `synthetic` | Synthetic Multi-hop | Generated locally | N/A |
| `locomo` | LoCoMo Conversational Memory | HuggingFace | GPT-4 F1: 32.1 |
| `nq` | Natural Questions (21M passages) | HuggingFace | BM25: 59.1%, DPR: 78.4% top-20 |
| `hotpotqa` | HotpotQA Multi-hop | HuggingFace | BM25: 0.603 nDCG@10 |
| `beir` | BEIR Subset (SciFact/NFCorpus/FiQA) | HuggingFace | BM25: ~0.437 avg nDCG@10 |

Datasets auto-download from HuggingFace on first use (except `synthetic`, which is generated locally).

## Adding a New Hypothesis

1. Pick the right subpackage for your technique (e.g., `arena/hypotheses/geometric/`)

2. Create your hypothesis file implementing the `Hypothesis` ABC:

```python
# arena/hypotheses/geometric/my_idea.py
import numpy as np
from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

class MyIdeaHypothesis(Hypothesis):
    @property
    def name(self) -> str:
        return "my-idea"

    def apply(self, query, results, embeddings, query_embedding):
        # Your reranking logic here — reorder `results` based on
        # embeddings, query_embedding, or any other signal.
        reranked = results  # replace with your logic

        context = "\n\n".join(
            f"[{i+1}] {r.text}" for i, r in enumerate(reranked)
        )
        return HypothesisResult(results=reranked, context_prompt=context)
```

3. Register in `arena/hypotheses/__init__.py`:
```python
from .geometric.my_idea import MyIdeaHypothesis
```

4. Register the CLI key in `arena/cli.py` in the `HYPOTHESES` dict:
```python
"my-idea": MyIdeaHypothesis,
```

5. Test it:
```bash
rag-arena run -H flat my-idea --benchmark synthetic --skip-llm
```

## Project Structure

```
arena/
├── backends/              Direct embeddings, BM25, hybrid retrieval
├── benchmarks/            LoCoMo, NQ, HotpotQA, BEIR, synthetic
├── hypotheses/
│   ├── geometric/         Embedding-space reranking, diversity, spectral (47)
│   ├── cross_encoder/     Cross-encoder reranking variants (31)
│   ├── deep_pool/         Deep candidate pool + CE reranking (8)
│   ├── llm/               LLM-powered query decomposition, IRCoT (9)
│   ├── multi_resolution/  MRAM, advanced rerankers (9)
│   └── hybrid/            Hybrid composition experiments (3)
├── metrics/               EM, F1, recall@K, precision@K, nDCG@K, MRR
├── runners/               Experiment orchestration
├── cli.py                 CLI entry point
└── config.py              YAML configuration
experiments/
├── run_*.py               Experiment scripts
└── results/               Organized by topic (benchmarks/, rounds/, mram/, ...)
```

## License

MIT
