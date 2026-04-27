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
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
.venv\Scripts\activate
```

## Running Experiments

Commands can be run directly using `uv`:

```bash
# Run the baseline model on a single dataset
uv run python src/run_baseline.py --dataset caffe

# Run the full experiment suite across all datasets
uv run python src/run_experiments.py

# Run all experiments and generate comparison plots automatically
uv run python src/run_experiments.py --with-plots

# Run experiments with a specific preprocessing mode (eg: lemmatize)
uv run python src/run_experiments.py --preprocessing-mode lemmatize

# Run all preprocessing ablations
uv run python src/run_experiments.py --all-preprocessing
```

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
