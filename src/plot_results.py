"""Plot experiment summaries.

NB: Titles are omitted from some figures used in the report
as captions are provided instead to save on page space.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import FormatStrFormatter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import (
    ABLATION_DIMENSIONS_SLUG,
    MAIN_EMBEDDING_DIMENSION,
    REPO_ROOT,
    RESULTS_DIR,
)
from .logging_config import configure_logging
from .models.registry import (
    BASELINE_NB_TFIDF,
    EMBEDDING_CENTROID,
    EMBEDDING_KNN,
    EMBEDDING_LOGREG,
    HYBRID_LOGREG,
    LATE_FUSION_LOGREG,
    MODEL_NAMES,
    TFIDF_LOGREG,
)


_console = Console()

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
HEATMAP_MODEL_LABELS = {  # shorter labels to avoid overlap
    BASELINE_NB_TFIDF: "Baseline",
    TFIDF_LOGREG: "TF-IDF + LR",
    EMBEDDING_LOGREG: "Embed LR",
    LATE_FUSION_LOGREG: "Late Fusion LR",
    HYBRID_LOGREG: "Hybrid LR",
    EMBEDDING_CENTROID: "Centroid",
    EMBEDDING_KNN: "kNN",
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
STAGE_NAMES = ("Feature prep", "Fit", "Inference")
RUNTIME_PROFILE_MODELS = tuple(MODEL_NAMES)
RUNTIME_PROFILE_LABELS = {
    BASELINE_NB_TFIDF: "Baseline",
    TFIDF_LOGREG: "TF-IDF + LogReg",
    EMBEDDING_LOGREG: "Embedding LogReg",
    LATE_FUSION_LOGREG: "Late Fusion",
    HYBRID_LOGREG: "Hybrid LogReg",
    EMBEDDING_CENTROID: "Centroid",
    EMBEDDING_KNN: "kNN",
}


def _blend_with_white(colour: str, white_amount: float) -> tuple[float, float, float]:
    """Create a lighter shade from an existing plot colour."""
    r, g, b = to_rgb(colour)
    return (
        (1.0 - white_amount) * r + white_amount,
        (1.0 - white_amount) * g + white_amount,
        (1.0 - white_amount) * b + white_amount,
    )


STAGE_COLOURS = {
    stage: _blend_with_white(MODEL_COLOURS[EMBEDDING_LOGREG], white_amount)
    for stage, white_amount in zip(STAGE_NAMES, (0.0, 0.38, 0.7))
}


def _format_seconds(value: float) -> str:
    """Format short runtime values without hiding small non-zero stages."""
    if value == 0:
        return "0"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _runtime_axis_formatter(x_limit: float) -> FormatStrFormatter:
    """Choose readable tick precision for each runtime panel."""
    if x_limit >= 0.1:
        return FormatStrFormatter("%.1f")
    if x_limit >= 0.02:
        return FormatStrFormatter("%.2f")
    return FormatStrFormatter("%.3f")


def _prepare_summary(summary_path: Path | None = None) -> pd.DataFrame:
    """Load and order the summary dataframe.

    Args:
        summary_path: Optional path to summary CSV

    Returns:
        Ordered summary dataframe
    """
    path = summary_path or (RESULTS_DIR / f"summary_{MAIN_EMBEDDING_DIMENSION}.csv")
    summary = pd.read_csv(path)
    summary["model"] = pd.Categorical(
        summary["model"],
        categories=MODEL_NAMES,
        ordered=True,
    )
    return summary.sort_values(["dataset", "model"]).reset_index(drop=True)


def _get_figure_path(filename: str) -> Path:
    """Ensure figures directory exists and return path for filename."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename


def _plot_bar_comparison(
    summary_path: Path | None,
    figure_filename: str,
    column_prefix: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    yscale: str = "linear",
) -> Path:
    """Helper to generate bar charts comparing models across datasets."""
    summary = _prepare_summary(summary_path)
    figure_path = _get_figure_path(figure_filename)

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
            dataset_rows[f"{column_prefix}_mean"],
            yerr=dataset_rows[f"{column_prefix}_std"],
            color=model_colours,
            alpha=0.9,
            capsize=4,
        )
        if ylim:
            axis.set_ylim(*ylim)
        if yscale != "linear":
            axis.set_yscale(yscale)

        axis.set_ylabel(ylabel)
        axis.set_title(dataset_name)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.tick_params(axis="x", rotation=15)

    fig.suptitle(title, fontsize=14)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_macro_f1(summary_path: Path | None = None) -> Path:
    """Plot mean macro-F1 for every model and dataset.

    Args:
        summary_path: Optional summary CSV path

    Returns:
        Path to generated figure
    """
    return _plot_bar_comparison(
        summary_path=summary_path,
        figure_filename="macro_f1_comparison.png",
        column_prefix="f1_macro",
        ylabel="Macro-F1",
        title="Model Comparison by Dataset",
        ylim=(0.0, 1.0),
    )


def plot_macro_f1_heatmap(summary_path: Path | None = None) -> Path:
    """Plot a compact annotated heatmap of mean macro-F1 by dataset and model.

    Args:
        summary_path: Optional summary CSV path

    Returns:
        Path to generated figure
    """
    summary = _prepare_summary(summary_path)
    figure_path = _get_figure_path("macro_f1_heatmap.png")

    model_names = list(MODEL_NAMES)
    heatmap_data = summary.pivot(
        index="dataset", columns="model", values="f1_macro_mean"
    ).reindex(columns=model_names)
    model_labels = [HEATMAP_MODEL_LABELS[model] for model in model_names]
    cmap = LinearSegmentedColormap.from_list(
        "macro_f1_heatmap",
        [
            _blend_with_white(MODEL_COLOURS[EMBEDDING_LOGREG], 0.9),
            _blend_with_white(MODEL_COLOURS[HYBRID_LOGREG], 0.15),
        ],
    )

    fig, axis = plt.subplots(figsize=(9.8, 2.45), constrained_layout=True)
    image = axis.imshow(
        heatmap_data,
        cmap=cmap,
        aspect="auto",
        vmin=0.5,
        vmax=0.9,
    )

    for row_index, (_, row) in enumerate(heatmap_data.iterrows()):
        best_column = int(row.to_numpy().argmax())
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if column_index == best_column else "normal",
            )

    axis.set_xticks(range(len(model_labels)), model_labels)
    axis.set_yticks(range(len(heatmap_data.index)), heatmap_data.index)
    axis.tick_params(length=0)
    axis.set_xticks(
        [position - 0.5 for position in range(1, len(model_labels))],
        minor=True,
    )
    axis.set_yticks(
        [position - 0.5 for position in range(1, len(heatmap_data.index))],
        minor=True,
    )
    axis.grid(which="minor", color="white", linewidth=1.0)

    colourbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colourbar.set_label("Macro-F1")

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

    figure_path = _get_figure_path("hybrid_ablation.png")

    fig, axis = plt.subplots(figsize=(9.8, 3.1), constrained_layout=True)

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
    axis.set_xticks(sorted(hybrid["embedding_dimension"].unique()))
    axis.set_ylim(0.7, 0.92)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(ncols=5, frameon=False, loc="upper center")

    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_runtime(summary_path: Path | None = None) -> Path:
    """Plot mean runtime for every model and dataset.

    Args:
        summary_path: Optional summary CSV path

    Returns:
        Path to generated figure
    """
    return _plot_bar_comparison(
        summary_path=summary_path,
        figure_filename="runtime_comparison.png",
        column_prefix="runtime_seconds",
        ylabel="Seconds",
        title="Runtime Comparison by Dataset",
    )


def plot_memory(summary_path: Path | None = None) -> Path:
    """Plot peak Python memory for every model and dataset.

    Args:
        summary_path: Optional summary CSV path

    Returns:
        Path to generated figure
    """
    return _plot_bar_comparison(
        summary_path=summary_path,
        figure_filename="memory_comparison.png",
        column_prefix="peak_python_memory_mb",
        ylabel="MB",
        title="Peak Python Memory by Dataset",
    )


def plot_inference(summary_path: Path | None = None) -> Path:
    """Plot mean inference time for every model and dataset.

    Args:
        summary_path: Optional summary CSV path

    Returns:
        Path to generated figure
    """
    return _plot_bar_comparison(
        summary_path=summary_path,
        figure_filename="inference_comparison.png",
        column_prefix="predict_seconds",
        ylabel="Seconds (log)",
        title="Inference Time Comparison by Dataset",
        yscale="log",
    )


def plot_avg_runtime_profile(summary_path: Path | None = None) -> Path:
    """Plot stacked average runtime profile across all compared models.

    Args:
        summary_path: Optional summary CSV path

    Returns:
        Path to generated figure
    """
    summary = _prepare_summary(summary_path)
    stage_summary = (
        summary.groupby("model")[
            [
                "feature_prep_seconds_mean",
                "fit_seconds_mean",
                "predict_seconds_mean",
            ]
        ]
        .mean()
        .reindex(RUNTIME_PROFILE_MODELS)
    )

    figure_path = _get_figure_path("avg_runtime_profile.png")

    labels = [RUNTIME_PROFILE_LABELS[model] for model in stage_summary.index]
    x_positions = list(range(len(labels)))
    prep_values = stage_summary["feature_prep_seconds_mean"]
    fit_values = stage_summary["fit_seconds_mean"]
    infer_values = stage_summary["predict_seconds_mean"]

    prep_colour = STAGE_COLOURS["Feature prep"]
    fit_colour = STAGE_COLOURS["Fit"]
    infer_colour = STAGE_COLOURS["Inference"]

    fig, axis = plt.subplots(figsize=(9.5, 3.75), constrained_layout=True)
    axis.bar(
        x_positions,
        prep_values,
        color=prep_colour,
        label="Prep",
    )
    axis.bar(
        x_positions,
        fit_values,
        bottom=prep_values,
        color=fit_colour,
        label="Fit",
    )
    axis.bar(
        x_positions,
        infer_values,
        bottom=prep_values + fit_values,
        color=infer_colour,
        label="Predict",
    )

    axis.set_xticks(x_positions, labels)
    axis.set_ylabel("Runtime (s)")
    axis.tick_params(axis="y", labelsize=9)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        ncols=1,
        handlelength=1.2,
        columnspacing=0.9,
    )

    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_memory_stages(summary_path: Path | None = None) -> Path:
    """Plot average peak Python memory split by stage.

    Args:
        summary_path: Optional summary CSV path

    Returns:
        Path to generated figure
    """
    summary = _prepare_summary(summary_path)
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

    figure_path = _get_figure_path("memory_stage_breakdown.png")

    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    labels = [MODEL_LABELS[model] for model in stage_summary.index]
    x_positions = range(len(labels))
    width = 0.25
    axis.bar(
        [position - width for position in x_positions],
        stage_summary["feature_prep_memory_mb_mean"],
        width=width,
        color=STAGE_COLOURS["Feature prep"],
        label="Feature prep",
    )
    axis.bar(
        list(x_positions),
        stage_summary["fit_memory_mb_mean"],
        width=width,
        color=STAGE_COLOURS["Fit"],
        label="Fit",
    )
    axis.bar(
        [position + width for position in x_positions],
        stage_summary["predict_memory_mb_mean"],
        width=width,
        color=STAGE_COLOURS["Inference"],
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

    tasks = (
        ("Macro-F1", plot_macro_f1),
        ("Macro-F1 Heatmap", plot_macro_f1_heatmap),
        ("Hybrid Ablation", plot_hybrid_ablation),
        ("Runtime", plot_runtime),
        ("Memory", plot_memory),
        ("Inference", plot_inference),
        ("Average Runtime Profile", plot_avg_runtime_profile),
        ("Memory Stages", plot_memory_stages),
    )

    generated_paths = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("", total=len(tasks))
        for label, plot_func in tasks:
            progress.update(task_id, description=f"Plotting {label}")
            figure_path = plot_func()
            generated_paths.append(figure_path)
            progress.advance(task_id)

    for figure_path in generated_paths:
        _console.print(f"✓ {figure_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
