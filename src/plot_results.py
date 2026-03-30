"""Plot experiment summaries."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    ABLATION_DIMENSIONS_SLUG,
    MAIN_EMBEDDING_DIMENSION,
    RESULTS_DIR,
)
from logging_config import configure_logging
from models.registry import (
    BASELINE_NB_TFIDF,
    EMBEDDING_CENTROID,
    EMBEDDING_KNN,
    EMBEDDING_LOGREG,
    HYBRID_LOGREG,
    LATE_FUSION_LOGREG,
    MODEL_NAMES,
    TFIDF_LOGREG,
)


FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_LABELS = {
    BASELINE_NB_TFIDF: "NB + TF-IDF (baseline)",
    TFIDF_LOGREG: "TF-IDF + LogReg",
    EMBEDDING_LOGREG: "Embedding LogReg",
    LATE_FUSION_LOGREG: "Late Fusion LogReg",
    HYBRID_LOGREG: "Hybrid LogReg",
    EMBEDDING_CENTROID: "Embedding Centroid",
    EMBEDDING_KNN: "Embedding kNN",
}
MODEL_COLOURS = {
    # see https://matplotlib.org/stable/gallery/color/named_colors.html
    BASELINE_NB_TFIDF: "silver",
    TFIDF_LOGREG: "slateblue",
    EMBEDDING_LOGREG: "steelblue",
    LATE_FUSION_LOGREG: "cornflowerblue",
    HYBRID_LOGREG: "mediumseagreen",
    EMBEDDING_CENTROID: "orange",
    EMBEDDING_KNN: "indianred",
}


def _prepare_summary(summary_path: Path) -> pd.DataFrame:
    """Load and order the summary dataframe.

    Args:
        summary_path: Path to summary CSV

    Returns:
        Ordered summary dataframe
    """
    summary = pd.read_csv(summary_path)
    summary["model"] = pd.Categorical(
        summary["model"],
        categories=MODEL_NAMES,
        ordered=True,
    )
    return summary.sort_values(["dataset", "model"]).reset_index(drop=True)


def plot_macro_f1(summary_path: Path | None = None) -> Path:
    """Plot mean macro-F1 for every model and dataset.

    Args:
        summary_path: Optional summary CSV path

    Returns:
        Path to generated figure
    """
    summary_csv = summary_path or (
        RESULTS_DIR / f"summary_{MAIN_EMBEDDING_DIMENSION}.csv"
    )
    summary = _prepare_summary(summary_csv)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "macro_f1_comparison.png"

    datasets = list(summary["dataset"].unique())
    model_names = list(MODEL_NAMES)

    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(11, 2.6 * len(datasets)),
        sharex=True,
        constrained_layout=True,
    )

    if len(datasets) == 1:
        axes = [axes]

    for axis, dataset_name in zip(axes, datasets):
        dataset_rows = (
            summary[summary["dataset"] == dataset_name]
            .set_index("model")
            .reindex(model_names)
            .reset_index()
        )
        model_labels = [MODEL_LABELS[model] for model in dataset_rows["model"]]
        model_colours = [MODEL_COLOURS[model] for model in dataset_rows["model"]]

        axis.bar(
            model_labels,
            dataset_rows["f1_macro_mean"],
            yerr=dataset_rows["f1_macro_std"],
            color=model_colours,
            alpha=0.9,
            capsize=4,
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Macro-F1")
        axis.set_title(dataset_name)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.tick_params(axis="x", rotation=15)

    fig.suptitle("Model Comparison by Dataset", fontsize=14)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_hybrid_ablation(summary_path: Path | None = None) -> Path:
    """Plot hybrid LogReg macro-F1 across embedding dimensions.

    Args:
        summary_path: Optional ablation summary CSV path

    Returns:
        Path to generated figure
    """
    summary_csv = summary_path or (
        RESULTS_DIR / f"embedding_ablation_summary_{ABLATION_DIMENSIONS_SLUG}.csv"
    )
    summary = pd.read_csv(summary_csv)
    hybrid = summary[summary["model"] == HYBRID_LOGREG].copy()

    main_summary = pd.read_csv(RESULTS_DIR / f"summary_{MAIN_EMBEDDING_DIMENSION}.csv")
    main_hybrid = main_summary[main_summary["model"] == HYBRID_LOGREG].copy()
    main_hybrid = main_hybrid[
        ["dataset", "embedding_dimension", "f1_macro_mean", "f1_macro_std"]
    ]

    hybrid = pd.concat([hybrid, main_hybrid], ignore_index=True)
    hybrid = hybrid.sort_values(["dataset", "embedding_dimension"])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "hybrid_ablation.png"

    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)

    for dataset_name, dataset_rows in hybrid.groupby("dataset"):
        axis.plot(
            dataset_rows["embedding_dimension"],
            dataset_rows["f1_macro_mean"],
            marker="o",
            linewidth=2,
            label=dataset_name,
        )

    axis.set_xlabel("Embedding dimension")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Hybrid LogReg Ablation Across Embedding Dimensions")
    axis.set_xticks(sorted(hybrid["embedding_dimension"].unique()))
    axis.set_ylim(0.7, 0.92)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(ncols=2, frameon=False)

    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_runtime(summary_path: Path | None = None) -> Path:
    """Plot mean runtime for every model and dataset."""
    summary_csv = summary_path or (
        RESULTS_DIR / f"summary_{MAIN_EMBEDDING_DIMENSION}.csv"
    )
    summary = _prepare_summary(summary_csv)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "runtime_comparison.png"

    datasets = list(summary["dataset"].unique())
    model_names = list(MODEL_NAMES)

    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(11, 2.6 * len(datasets)),
        sharex=True,
        constrained_layout=True,
    )

    if len(datasets) == 1:
        axes = [axes]

    for axis, dataset_name in zip(axes, datasets):
        dataset_rows = (
            summary[summary["dataset"] == dataset_name]
            .set_index("model")
            .reindex(model_names)
            .reset_index()
        )
        model_labels = [MODEL_LABELS[model] for model in dataset_rows["model"]]
        model_colours = [MODEL_COLOURS[model] for model in dataset_rows["model"]]

        axis.bar(
            model_labels,
            dataset_rows["runtime_seconds_mean"],
            yerr=dataset_rows["runtime_seconds_std"],
            color=model_colours,
            alpha=0.9,
            capsize=4,
        )
        axis.set_ylabel("Seconds")
        axis.set_title(dataset_name)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.tick_params(axis="x", rotation=15)

    fig.suptitle("Runtime Comparison by Dataset", fontsize=14)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_memory(summary_path: Path | None = None) -> Path:
    """Plot peak Python memory for every model and dataset."""
    summary_csv = summary_path or (
        RESULTS_DIR / f"summary_{MAIN_EMBEDDING_DIMENSION}.csv"
    )
    summary = _prepare_summary(summary_csv)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "memory_comparison.png"

    datasets = list(summary["dataset"].unique())
    model_names = list(MODEL_NAMES)

    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(11, 2.6 * len(datasets)),
        sharex=True,
        constrained_layout=True,
    )

    if len(datasets) == 1:
        axes = [axes]

    for axis, dataset_name in zip(axes, datasets):
        dataset_rows = (
            summary[summary["dataset"] == dataset_name]
            .set_index("model")
            .reindex(model_names)
            .reset_index()
        )
        model_labels = [MODEL_LABELS[model] for model in dataset_rows["model"]]
        model_colours = [MODEL_COLOURS[model] for model in dataset_rows["model"]]

        axis.bar(
            model_labels,
            dataset_rows["peak_python_memory_mb_mean"],
            yerr=dataset_rows["peak_python_memory_mb_std"],
            color=model_colours,
            alpha=0.9,
            capsize=4,
        )
        axis.set_ylabel("MB")
        axis.set_title(dataset_name)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.tick_params(axis="x", rotation=15)

    fig.suptitle("Peak Python Memory by Dataset", fontsize=14)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_inference(summary_path: Path | None = None) -> Path:
    """Plot mean inference time for every model and dataset."""
    summary_csv = summary_path or (
        RESULTS_DIR / f"summary_{MAIN_EMBEDDING_DIMENSION}.csv"
    )
    summary = _prepare_summary(summary_csv)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "inference_comparison.png"

    datasets = list(summary["dataset"].unique())
    model_names = list(MODEL_NAMES)

    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(11, 2.6 * len(datasets)),
        sharex=True,
        constrained_layout=True,
    )

    if len(datasets) == 1:
        axes = [axes]

    for axis, dataset_name in zip(axes, datasets):
        dataset_rows = (
            summary[summary["dataset"] == dataset_name]
            .set_index("model")
            .reindex(model_names)
            .reset_index()
        )
        model_labels = [MODEL_LABELS[model] for model in dataset_rows["model"]]
        model_colours = [MODEL_COLOURS[model] for model in dataset_rows["model"]]

        axis.bar(
            model_labels,
            dataset_rows["predict_seconds_mean"],
            yerr=dataset_rows["predict_seconds_std"],
            color=model_colours,
            alpha=0.9,
            capsize=4,
        )
        axis.set_yscale("log")
        axis.set_ylabel("Seconds (log)")
        axis.set_title(dataset_name)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.tick_params(axis="x", rotation=15)

    fig.suptitle("Inference Time Comparison by Dataset", fontsize=14)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_runtime_stages(summary_path: Path | None = None) -> Path:
    """Plot average runtime split into prep, fit, and predict stages."""
    summary_csv = summary_path or (
        RESULTS_DIR / f"summary_{MAIN_EMBEDDING_DIMENSION}.csv"
    )
    summary = _prepare_summary(summary_csv)
    stage_summary = (
        summary.groupby("model")[
            [
                "feature_prep_seconds_mean",
                "fit_seconds_mean",
                "predict_seconds_mean",
            ]
        ]
        .mean()
        .reindex(MODEL_NAMES)
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "runtime_stage_breakdown.png"

    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    labels = [MODEL_LABELS[model] for model in stage_summary.index]
    axis.bar(labels, stage_summary["feature_prep_seconds_mean"], label="Feature prep")
    axis.bar(
        labels,
        stage_summary["fit_seconds_mean"],
        bottom=stage_summary["feature_prep_seconds_mean"],
        label="Fit",
    )
    axis.bar(
        labels,
        stage_summary["predict_seconds_mean"],
        bottom=(
            stage_summary["feature_prep_seconds_mean"]
            + stage_summary["fit_seconds_mean"]
        ),
        label="Inference",
    )
    axis.set_ylabel("Seconds")
    axis.set_title("Average Runtime Breakdown by Model")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(frameon=False)

    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_memory_stages(summary_path: Path | None = None) -> Path:
    """Plot average peak Python memory split by stage."""
    summary_csv = summary_path or (
        RESULTS_DIR / f"summary_{MAIN_EMBEDDING_DIMENSION}.csv"
    )
    summary = _prepare_summary(summary_csv)
    stage_summary = (
        summary.groupby("model")[
            [
                "feature_prep_memory_mb_mean",
                "fit_memory_mb_mean",
                "predict_memory_mb_mean",
            ]
        ]
        .mean()
        .reindex(MODEL_NAMES)
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "memory_stage_breakdown.png"

    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    labels = [MODEL_LABELS[model] for model in stage_summary.index]
    x_positions = range(len(labels))
    width = 0.25
    axis.bar(
        [position - width for position in x_positions],
        stage_summary["feature_prep_memory_mb_mean"],
        width=width,
        label="Feature prep",
    )
    axis.bar(
        list(x_positions),
        stage_summary["fit_memory_mb_mean"],
        width=width,
        label="Fit",
    )
    axis.bar(
        [position + width for position in x_positions],
        stage_summary["predict_memory_mb_mean"],
        width=width,
        label="Inference",
    )
    axis.set_ylabel("MB")
    axis.set_title("Average Peak Python Memory Breakdown by Model")
    axis.set_xticks(list(x_positions), labels, rotation=20)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(frameon=False)

    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main() -> None:
    """Generate summary plots from experiment outputs."""
    configure_logging()
    figure_paths = (
        plot_macro_f1(),
        plot_hybrid_ablation(),
        plot_runtime(),
        plot_memory(),
        plot_inference(),
        plot_runtime_stages(),
        plot_memory_stages(),
    )
    for figure_path in figure_paths:
        logging.info("Saved figure to %s", figure_path)


if __name__ == "__main__":
    main()
