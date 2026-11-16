# How Far Are KV-Cache Eviction Policies from the Offline Optimum?

model: Qwen2.5-0.5B

## Overview

LLMs with long contexts are bottlenecked by KV-cache memory. Existing eviction policies
(H2O, TOVA, LRU) hard-code a fixed mix of signals, but no single signal dominates
across all text types. This project:

1. **Formulates KV-cache eviction as an online problem** and computes the **exact
   offline optimum** via a reduction to min-cost flow.
2. **Benchmarks 8+ eviction policies** (Random, Sliding Window, StreamingLLM,
   H2O, Hybrid, Imitation Learning, DAgger, Chameleon) against the optimum.
3. Introduces **Chameleon** — a training-free adaptive policy that uses
   Fixed-Share Hedge to re-weight five expert heuristics online.
4. Evaluates everything through the **Gauntlet** — a pre-registered test battery
   covering competitive ratio, distribution shift, adversarial robustness,
   real perplexity, needle retrieval, ablations, and overhead.

## Key Results

Run the notebook to generate results. The pre-registered decision rule in
`verdict()` determines the winner — see Section 10e.

## Project Structure

```
kvcache/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml              # Experiment hyperparameters
├── src/                          # Core source modules
│   ├── __init__.py
│   ├── core.py                   # Policies, simulator, min-cost flow optimum
│   ├── model.py                  # Attention extraction, pruned-cache decoding
│   ├── imitation.py              # Imitation learning + DAgger
│   ├── chameleon.py              # Chameleon online-learning policy
│   └── gauntlet.py               # Unified test battery
├── notebooks/
│   └── kv_cache_eviction.ipynb   # Main experiment notebook
├── configs/
│   └── default.yaml
├── data/                         # Input data
├── outputs/                      # Generated figures, tables, checkpoints
│   ├── figures/
│   ├── tables/
│   ├── checkpoints/
│   └── logs/
└── docs/
    ├── paper/
    │   └── kv_cache_eviction_paper.pdf
    └── notes/
        └── build_log.txt
```

## Setup & Installation

```bash
pip install -r requirements.txt
```

Requires a GPU (tested on Colab T4). For CPU-only testing of core algorithms,
the model-related cells can be skipped.

## Running Experiments

1. Open `notebooks/kv_cache_eviction.ipynb` in Jupyter or Colab
2. Set `PRESET = "quick"` for a fast sanity run, `"full"` for the complete battery
3. Run all cells sequentially

## Modules

| Module | What it does |
|--------|-------------|
| `src/core.py` | Policy base class, 6 online policies, Belady oracle, exact optimum (min-cost flow), simulator |
| `src/model.py` | Full-cache attention extraction, token-by-token decoding with physical cache pruning |
| `src/imitation.py` | Imitation learning scorer, DAgger training, feature extraction |
| `src/chameleon.py` | Chameleon (Fixed-Share Hedge ensemble), LastAttention baseline |
| `src/gauntlet.py` | Gauntlet test battery: scoreboard, adversary, stress traces, ablations, verdict |
