"""Run the reproducible Naive Bayes plus TF-IDF baseline."""

# pylint: disable=import-outside-toplevel

import argparse
import logging

from config import AVAILABLE_DATASETS, N_RUNS, RESULTS_DIR, SEEDS, TEST_SIZE
from logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run the baseline classifier with fixed paired splits."
    )
    parser.add_argument(
        "--dataset",
        default="pytorch",
        choices=AVAILABLE_DATASETS,
        help="Dataset name from the datasets folder.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the baseline experiment and save results."""
    configure_logging()
    args = parse_args()
    logging.info("Starting baseline with dataset: %s", args.dataset)

    from experiments.evaluate import evaluate_baseline_model
    from features.data_load import load_dataset
    from features.text_prep import preprocess_texts
    from models.baseline_nb_tfidf import BaselineNbTfidf

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_text, y = load_dataset(args.dataset)
    x_text = preprocess_texts(x_text)

    per_run_results, summary = evaluate_baseline_model(
        x_text,
        y,
        model_factory=BaselineNbTfidf,
        seeds=SEEDS,
    )

    per_run_path = RESULTS_DIR / f"{args.dataset}_baseline_runs.csv"
    summary_path = RESULTS_DIR / f"{args.dataset}_baseline_summary.csv"

    per_run_results.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)

    logging.info("Naive Bayes + TF-IDF baseline complete")
    logging.info("Dataset: %s", args.dataset)
    logging.info("Test size: %s", TEST_SIZE)
    logging.info("Runs: %s", N_RUNS)
    logging.info("Seeds: %s..%s", SEEDS[0], SEEDS[-1])
    logging.info("Per-run results: %s", per_run_path)
    logging.info("Summary results: %s", summary_path)
    logging.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
