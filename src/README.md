# Source Code

This folder contains the experiment pipeline for bug report classification.

The code is split so the baseline, embedding models, and experiment runner stay
modular rather than being packed into one script.

## Structure

- `config.py`: shared experiment settings such as dataset names, split size,
  number of runs, seeds, embedding dimensions, and output directories
- `run_baseline.py`: entry point for running only the Naive Bayes + TF-IDF
  baseline
- `run_experiments.py`: entry point for the full comparison workflow
- `plot_results.py`: creates report-ready plots from saved summary CSV files
- `features/`: dataset loading, text preprocessing, TF-IDF helpers, and
  embedding caching
- `models/`: baseline, linear, centroid, and kNN model implementations
- `experiments/`: paired evaluation, statistics, ablation logic, and result
  writing

## Main Workflow

1. Load one dataset from `datasets/`
2. Preprocess the text field
3. Run the baseline with fixed paired splits
4. Build or reuse cached full embeddings for
   `nomic-ai/nomic-embed-text-v1.5`
5. Run the main comparison at `768` dimensions
6. Run a separate embedding ablation at `512`, `256`, `128`, and `64`
7. Save per-run and summary CSV outputs to `results/`

The first embedding run may download the model weights and trust-remote-code
implementation from Hugging Face. After that, embeddings are cached locally.

## Main Commands

Run the baseline only:

```bash
uv run python -m src.run_baseline --dataset caffe
```

Run all implemented models for one dataset:

```bash
uv run python -m src.run_experiments --dataset caffe
```

Run all implemented models for every dataset:

```bash
uv run python -m src.run_experiments
```

Run the default full experiment and regenerate the plots right after:

```bash
uv run python -m src.run_experiments --with-plots
```

Run every preprocessing ablation in one go:

```bash
uv run python -m src.run_experiments --all-preprocessing
```

Run every preprocessing ablation and then regenerate the default plots:

```bash
uv run python -m src.run_experiments --all-preprocessing --with-plots
```

Run one specific preprocessing variant across every dataset:

```bash
uv run python -m src.run_experiments --preprocessing-mode lemmatize
```

Generate a macro-F1 comparison figure from the main summary:

```bash
uv run python -m src.plot_results
```

To activate the virtual environment manually first:

```bash
source .venv/bin/activate
python -m src.run_experiments --dataset caffe
```

## Outputs

The `results/` directory stores all experimental outputs, including:

- **Experiment Summaries**: Detailed metrics and cross-dataset evaluation results (`.csv`).
- **Statistical Tests**: Wilcoxon signed-rank and Friedman test outputs for significance testing.
- **Visualizations**: Generated plots comparing macro-F1 scores, runtimes, and ablation studies (`results/figures/`).
- **Embeddings**: Cached model-aware embedding arrays (`results/embeddings/`).
The `results/embeddings/` folder stores model-aware cached embedding arrays and
row mapping files, so embeddings do not need to be recomputed every run and
different embedding models or dimensions do not overwrite each other.

The preprocessing ablations are:

- `none`
- `stopwords_all`
- `stopwords_keep_negation`
- `lemmatize`
- `stopwords_keep_negation+lemmatize`

Recommended order:

1. run the default `none` experiment first
2. add `--with-plots` to regenerate the report figures
3. run `--all-preprocessing` to include extra preprocessing ablation
   outputs too

Cached embeddings and results do not normally need to be deleted or reset first,
because both result files and embedding caches are now separated by
preprocessing mode.
