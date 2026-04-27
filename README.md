# BUGLEX: Semantic–Lexical Fusion for Performance Bug Classification

[![Smoke Test](https://github.com/siddharth-shringarpure/buglex/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/siddharth-shringarpure/buglex/actions/workflows/smoke-test.yml)

This repository provides the code and framework used to evaluate machine learning models for performance bug report classification, including feature engineering (TF-IDF and embeddings) and hybrid model training.

## Setup

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for fast dependency management.

### macOS / Linux (Recommended)
```bash
# Install uv (if not already installed):
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the local virtual environment:
uv sync

# Activate the virtual environment
source .venv/bin/activate  # bash/zsh
```

### Windows

Please see instructions at the [Installing uv](https://docs.astral.sh/uv/getting-started/installation/) page.

## Running Experiments

Commands can be run directly using `uv`:

```bash
# Run the baseline model on a single dataset
uv run python -m src.run_baseline --dataset caffe

# Run the full experiment suite across all datasets
uv run python -m src.run_experiments

# Run all experiments and generate comparison plots automatically
uv run python -m src.run_experiments --with-plots

# Run experiments with a specific preprocessing mode (eg: lemmatize)
uv run python -m src.run_experiments --preprocessing-mode lemmatize 

# Run all preprocessing ablations
uv run python -m src.run_experiments --all-preprocessing
```

## Results

Mean macro-F1 ± std across 30 stratified runs (70/30 split). Source: [`results/main_table_macro_f1.csv`](results/main_table_macro_f1.csv).

| Dataset | Baseline NB + TF-IDF | TF-IDF + LogReg | Embedding LogReg | **Hybrid LogReg** |
|---|---|---|---|---|
| Caffe | 0.623 ± 0.073 | 0.758 ± 0.061 | 0.744 ± 0.046 | **0.788 ± 0.063** |
| Incubator-MXNet | 0.503 ± 0.037 | 0.779 ± 0.031 | 0.832 ± 0.026 | **0.844 ± 0.030** |
| Keras | 0.530 ± 0.039 | 0.805 ± 0.034 | 0.845 ± 0.025 | **0.856 ± 0.023** |
| PyTorch | 0.529 ± 0.039 | 0.751 ± 0.037 | 0.796 ± 0.028 | **0.814 ± 0.027** |
| TensorFlow | 0.628 ± 0.027 | 0.820 ± 0.026 | 0.875 ± 0.016 | **0.882 ± 0.017** |

The hybrid model outperforms the baseline on all five datasets, with statistically significant improvements over the baseline.

## Generating Documentation & Reports

To compile the results into the final LaTeX PDF report:

```bash
uv run python -m src.tools.build_docs
```

This will automatically generate the figures and tables before compiling the PDF.

## Repository Layout

```text
.
├── datasets/              # Raw data used for models
├── docs/                  # Documentation and report source files
├── main.py                # Local runner
├── pyproject.toml         # Dependencies
├── results/               # Generated results
├── src/                   # Main source code
├── README.md              # This file
```
