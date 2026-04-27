"""Run all classification experiments."""

# pylint: disable=import-outside-toplevel

import argparse
import logging

from .config import AVAILABLE_DATASETS, PREPROCESSING_MODES
from .logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run all experiments across one or more datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=AVAILABLE_DATASETS,
        help="Run experiments for one dataset only.",
    )
    parser.add_argument(
        "--preprocessing-mode",
        default="none",
        choices=PREPROCESSING_MODES,
        help="Preprocessing mode applied to every model in the run.",
    )
    parser.add_argument(
        "--all-preprocessing",
        action="store_true",
        help="Run every configured preprocessing mode in sequence.",
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="Generate plots after a default full-dataset run completes.",
    )
    return parser.parse_args()


def main() -> None:
    """Run experiments and save outputs."""
    configure_logging()
    args = parse_args()
    from .experiments.evaluate import save_full_experiment_outputs
    from .experiments.evaluate import save_single_dataset_outputs
    from .plot_results import main as plot_results_main

    preprocessing_modes = (
        list(PREPROCESSING_MODES)
        if args.all_preprocessing
        else [args.preprocessing_mode]
    )

    if args.all_preprocessing and args.preprocessing_mode != "none":
        logging.info(
            "Ignoring --preprocessing-mode=%s because --all-preprocessing was set",
            args.preprocessing_mode,
        )

    if args.dataset:
        logging.info("Running single dataset: %s", args.dataset)
        for preprocessing_mode in preprocessing_modes:
            logging.info(
                "Starting experiments with preprocessing mode: %s",
                preprocessing_mode,
            )
            save_single_dataset_outputs(
                args.dataset,
                preprocessing_mode=preprocessing_mode,
            )
        if args.with_plots:
            logging.info("Skipping plot generation for single-dataset runs")
        return

    for preprocessing_mode in preprocessing_modes:
        logging.info(
            "Starting experiments with preprocessing mode: %s",
            preprocessing_mode,
        )
        logging.info("Running full dataset run")
        save_full_experiment_outputs(preprocessing_mode=preprocessing_mode)

    if not args.with_plots:
        return
    if args.all_preprocessing or args.preprocessing_mode == "none":
        logging.info("Generating plots from the default summary outputs")
        plot_results_main()
        return
    logging.info(
        "Skipping plot generation because plots currently target the default "
        "preprocessing outputs"
    )


if __name__ == "__main__":
    main()
