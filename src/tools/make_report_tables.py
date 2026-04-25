"""Generate report-facing LaTeX tables and summary notes from CSV outputs."""

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import DATASETS_DIR, REPO_ROOT, RESULTS_DIR

_console = Console()

REPORT_DIR = REPO_ROOT / "docs" / "report"
TABLES_DIR = REPORT_DIR / "tables"

MAIN_RESULTS_PATH = RESULTS_DIR / "main_table_macro_f1.csv"
WILCOXON_PATH = RESULTS_DIR / "wilcoxon_summary_768.csv"
SECONDARY_METRICS_PATH = RESULTS_DIR / "secondary_metrics_table.csv"
METRIC_DELTAS_PATH = RESULTS_DIR / "metric_deltas_hybrid_vs_baseline.csv"
PREPROC_ABLATION_PATH = RESULTS_DIR / "ablation_preproc_summary.csv"
EMBEDDING_ABLATION_PATH = RESULTS_DIR / "embedding_ablation_summary_512_256_128_64.csv"
LATE_FUSION_PATH = RESULTS_DIR / "late_fusion_vs_hybrid.csv"
SUMMARY_PATH = RESULTS_DIR / "summary_768.csv"
BENCHMARK_PATH = RESULTS_DIR / "benchmark_stage_summary_768.csv"
REPORT_NOTES_PATH = RESULTS_DIR / "report_notes.txt"


def main() -> None:
    """Build all report tables and extracted notes."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            "Main results table",
            TABLES_DIR / "main_results.tex",
            lambda: _build_main_results_table(pd.read_csv(MAIN_RESULTS_PATH)),
        ),
        (
            "Dataset overview table",
            TABLES_DIR / "dataset_overview.tex",
            _build_dataset_overview_table,
        ),
        (
            "Wilcoxon results table",
            TABLES_DIR / "wilcoxon_results.tex",
            lambda: _build_wilcoxon_table(pd.read_csv(WILCOXON_PATH)),
        ),
        (
            "Secondary metrics table",
            TABLES_DIR / "secondary_metrics.tex",
            lambda: _build_secondary_metrics_table(pd.read_csv(SECONDARY_METRICS_PATH)),
        ),
        (
            "Metric deltas table",
            TABLES_DIR / "metric_deltas.tex",
            lambda: _build_metric_deltas_table(pd.read_csv(METRIC_DELTAS_PATH)),
        ),
        (
            "Preprocessing ablation table",
            TABLES_DIR / "preproc_ablation.tex",
            lambda: _build_preproc_ablation_table(pd.read_csv(PREPROC_ABLATION_PATH)),
        ),
        (
            "Embedding ablation table",
            TABLES_DIR / "embedding_ablation.tex",
            lambda: _build_embedding_ablation_table(
                pd.read_csv(EMBEDDING_ABLATION_PATH)
            ),
        ),
        (
            "Efficiency summary table",
            TABLES_DIR / "efficiency_summary.tex",
            lambda: _build_efficiency_table(
                summary_df=pd.read_csv(SUMMARY_PATH),
                benchmark_df=pd.read_csv(BENCHMARK_PATH),
            ),
        ),
        (
            "Report notes",
            REPORT_NOTES_PATH,
            lambda: _build_report_notes(
                summary_df=pd.read_csv(SUMMARY_PATH),
                benchmark_df=_read_optional_csv(BENCHMARK_PATH),
                embedding_ablation_df=pd.read_csv(EMBEDDING_ABLATION_PATH),
                late_fusion_df=pd.read_csv(LATE_FUSION_PATH),
            ),
        ),
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("", total=len(tasks))
        for label, path, build_fn in tasks:
            progress.update(task_id, description=label)
            _write_text(path, build_fn())
            progress.advance(task_id)

    _console.print(f"✓ {len(tasks)} files written to [bold]{TABLES_DIR.parent}[/bold]")


def _build_main_results_table(df: pd.DataFrame) -> str:
    """Create the main macro-F1 table."""
    display = df.rename(
        columns={
            "dataset": "Dataset",
            "baseline_nb_tfidf": "Baseline NB",
            "tfidf_lr": "TF-IDF + LogReg",
            "embedding_lr": "Embedding LogReg",
            "hybrid_lr": "Hybrid LogReg",
        }
    )
    rows = [
        [
            _format_dataset_name(row["Dataset"]),
            row["Baseline NB"],
            row["TF-IDF + LogReg"],
            row["Embedding LogReg"],
            row["Hybrid LogReg"],
        ]
        for _, row in display.iterrows()
    ]
    return _tabular(
        "lcccc",
        [
            "Dataset",
            "Baseline NB",
            "TF-IDF + LogReg",
            "Embedding LogReg",
            "Hybrid LogReg",
        ],
        rows,
    )


def _build_dataset_overview_table() -> str:
    """Create the dataset overview table from raw dataset CSVs."""
    rows = []
    for dataset_name in [
        "caffe",
        "incubator-mxnet",
        "keras",
        "pytorch",
        "tensorflow",
    ]:
        df = pd.read_csv(DATASETS_DIR / f"{dataset_name}.csv")
        total_reports = len(df)
        positive_reports = int((df["class"] == 1).sum())
        positive_rate = 100.0 * positive_reports / total_reports
        rows.append(
            [
                _format_dataset_name(dataset_name),
                str(total_reports),
                str(positive_reports),
                _format_percent(positive_rate, places=2),
            ]
        )
    return _tabular(
        "lccc",
        ["Project", "Reports", "Positive reports", "Positive rate"],
        rows,
    )


def _build_wilcoxon_table(df: pd.DataFrame) -> str:
    """Create the Wilcoxon results table."""
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                _format_dataset_name(row["dataset"]),
                _format_decimal(row["f1_gain_mean"], places=4),
                _format_percent(row["f1_percent_improvement"]),
                _format_p_value(row["wilcoxon_p_value"]),
                str(int(row["n_positive_deltas"])),
            ]
        )
    return _tabular(
        "lcccc",
        ["Dataset", "Mean $\\Delta$F1", "Improvement", "$p$", "Positive deltas"],
        rows,
    )


def _build_secondary_metrics_table(df: pd.DataFrame) -> str:
    """Create a compact baseline-vs-hybrid secondary metrics table."""
    filtered = df[df["model"].isin(["baseline_nb_tfidf", "hybrid_lr"])].copy()
    pivoted = filtered.pivot(index="dataset", columns="model")
    rows = []
    for dataset in pivoted.index:
        rows.append(
            [
                _format_dataset_name(dataset),
                _format_decimal(
                    pivoted.loc[dataset, ("accuracy_mean", "baseline_nb_tfidf")],
                    places=4,
                ),
                _format_decimal(
                    pivoted.loc[dataset, ("accuracy_mean", "hybrid_lr")],
                    places=4,
                ),
                _format_decimal(
                    pivoted.loc[
                        dataset,
                        ("precision_macro_mean", "baseline_nb_tfidf"),
                    ],
                    places=4,
                ),
                _format_decimal(
                    pivoted.loc[dataset, ("precision_macro_mean", "hybrid_lr")],
                    places=4,
                ),
                _format_decimal(
                    pivoted.loc[
                        dataset,
                        ("recall_macro_mean", "baseline_nb_tfidf"),
                    ],
                    places=4,
                ),
                _format_decimal(
                    pivoted.loc[dataset, ("recall_macro_mean", "hybrid_lr")],
                    places=4,
                ),
                _format_decimal(
                    pivoted.loc[dataset, ("auc_mean", "baseline_nb_tfidf")],
                    places=4,
                ),
                _format_decimal(
                    pivoted.loc[dataset, ("auc_mean", "hybrid_lr")],
                    places=4,
                ),
            ]
        )
    return _tabular(
        "lcccccccc",
        [
            "Dataset",
            "Base Acc",
            "Hybrid Acc",
            "Base Prec",
            "Hybrid Prec",
            "Base Rec",
            "Hybrid Rec",
            "Base AUC",
            "Hybrid AUC",
        ],
        rows,
    )


def _build_metric_deltas_table(df: pd.DataFrame) -> str:
    """Create the hybrid-vs-baseline delta table."""
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                _format_dataset_name(row["dataset"]),
                _format_decimal(row["accuracy_delta_mean"], places=4),
                _format_decimal(row["precision_macro_delta_mean"], places=4),
                _format_decimal(row["recall_macro_delta_mean"], places=4),
                _format_decimal(row["f1_macro_delta_mean"], places=4),
                _format_decimal(row["auc_delta_mean"], places=4),
            ]
        )
    return _tabular(
        "lccccc",
        [
            "Dataset",
            "$\\Delta$Acc",
            "$\\Delta$Prec",
            "$\\Delta$Rec",
            "$\\Delta$F1",
            "$\\Delta$AUC",
        ],
        rows,
    )


def _build_preproc_ablation_table(df: pd.DataFrame) -> str:
    """Create a compact preprocessing ablation summary table."""
    baseline = df[df["preprocessing_mode"] == "none"].set_index("dataset")["hybrid_lr"]
    rows = []
    for mode in df["preprocessing_mode"].unique():
        mode_rows = df[df["preprocessing_mode"] == mode].set_index("dataset")
        hybrid_delta = mode_rows["hybrid_lr"] - baseline
        rows.append(
            [
                _format_mode_name(mode),
                _format_decimal(hybrid_delta.mean(), places=4),
                _format_decimal(hybrid_delta.min(), places=4),
                _format_dataset_name(hybrid_delta.idxmin()),
            ]
        )
    return _tabular(
        "lccc",
        ["Mode", "Mean $\\Delta$F1 vs default", "Worst $\\Delta$F1", "Worst dataset"],
        rows,
    )


def _build_embedding_ablation_table(df: pd.DataFrame) -> str:
    """Create the hybrid-dimension ablation table."""
    hybrid = df[df["model"] == "hybrid_lr"].copy()
    pivoted = hybrid.pivot(
        index="dataset",
        columns="embedding_dimension",
        values="f1_macro_mean",
    ).reindex(columns=[64, 128, 256, 512])
    summary = pd.read_csv(SUMMARY_PATH)
    hybrid_main = summary[summary["model"] == "hybrid_lr"].set_index("dataset")[
        "f1_macro_mean"
    ]
    rows = []
    for dataset in pivoted.index:
        rows.append(
            [
                _format_dataset_name(dataset),
                _format_decimal(pivoted.loc[dataset, 64], places=4),
                _format_decimal(pivoted.loc[dataset, 128], places=4),
                _format_decimal(pivoted.loc[dataset, 256], places=4),
                _format_decimal(pivoted.loc[dataset, 512], places=4),
                _format_decimal(hybrid_main.loc[dataset], places=4),
            ]
        )
    return _tabular(
        "lccccc",
        ["Dataset", "64", "128", "256", "512", "768"],
        rows,
    )


def _build_efficiency_table(
    summary_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> str:
    """Create the average efficiency summary table."""
    f1_mean = summary_df.groupby("model")["f1_macro_mean"].mean()
    benchmark_mean = benchmark_df.groupby("model")[
        ["runtime_seconds_mean", "peak_python_memory_mb_mean"]
    ].mean()
    combined = benchmark_mean.join(f1_mean, how="inner").reset_index()
    model_order = [
        "baseline_nb_tfidf",
        "tfidf_lr",
        "embedding_lr",
        "late_fusion_lr",
        "hybrid_lr",
        "embedding_centroid",
        "embedding_knn",
    ]
    combined["order"] = combined["model"].apply(model_order.index)
    combined = combined.sort_values("order")
    rows = []
    for _, row in combined.iterrows():
        rows.append(
            [
                _format_model_name(row["model"]),
                _format_decimal(row["f1_macro_mean"], places=4),
                _format_decimal(row["runtime_seconds_mean"], places=4),
                _format_decimal(row["peak_python_memory_mb_mean"], places=2),
            ]
        )
    return _tabular(
        "lccc",
        ["Model", "Mean macro-F1", "Runtime (s)", "Peak Memory (MB)"],
        rows,
    )


def _build_report_notes(
    summary_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    embedding_ablation_df: pd.DataFrame,
    late_fusion_df: pd.DataFrame,
) -> str:
    """Extract concise report-writing notes from existing outputs."""
    hybrid_gains = pd.read_csv(WILCOXON_PATH)[["dataset", "f1_gain_mean"]]
    largest_gain = hybrid_gains.loc[hybrid_gains["f1_gain_mean"].idxmax()]
    smallest_gain = hybrid_gains.loc[hybrid_gains["f1_gain_mean"].idxmin()]

    non_hybrid = summary_df[summary_df["model"] != "hybrid_lr"].copy()
    non_hybrid_mean = (
        non_hybrid.groupby("model")["f1_macro_mean"].mean().sort_values(ascending=False)
    )
    best_non_hybrid_model = non_hybrid_mean.index[0]

    hybrid_dim = embedding_ablation_df[
        embedding_ablation_df["model"] == "hybrid_lr"
    ].copy()
    hybrid_main = summary_df[summary_df["model"] == "hybrid_lr"].set_index("dataset")[
        "f1_macro_mean"
    ]

    smallest_close_dims = []
    for dataset, target_f1 in hybrid_main.items():
        dataset_dims = hybrid_dim[hybrid_dim["dataset"] == dataset].copy()
        dataset_dims["difference_to_768"] = (
            target_f1 - dataset_dims["f1_macro_mean"]
        ).abs()
        within_tolerance = dataset_dims[dataset_dims["difference_to_768"] <= 0.01]
        if within_tolerance.empty:
            smallest_close_dims.append(
                f"- {_format_dataset_name(_as_str(dataset))}: "
                "no lower dimension within 0.01"
            )
            continue
        smallest_dim = int(within_tolerance["embedding_dimension"].min())
        smallest_close_dims.append(
            f"- {_format_dataset_name(_as_str(dataset))}: {smallest_dim} dimensions"
        )

    largest_gain_dataset = _as_str(largest_gain["dataset"])
    smallest_gain_dataset = _as_str(smallest_gain["dataset"])
    late_fusion_worst_dataset = _as_str(
        late_fusion_df.loc[
            late_fusion_df["delta_late_minus_hybrid"].idxmin(),
            "dataset",
        ]
    )
    best_non_hybrid_name = _as_str(best_non_hybrid_model)

    lines = [
        "- Largest hybrid-vs-baseline macro-F1 gain: "
        f"{_format_dataset_name(largest_gain_dataset)} "
        f"({_to_float(largest_gain['f1_gain_mean']):.4f})",
        "- Smallest hybrid-vs-baseline macro-F1 gain: "
        f"{_format_dataset_name(smallest_gain_dataset)} "
        f"({_to_float(smallest_gain['f1_gain_mean']):.4f})",
        f"- Best non-hybrid model overall: {_format_model_name(best_non_hybrid_name)}",
        "- Late fusion vs hybrid: competitive, but weaker on every dataset; "
        f"largest gap on {_format_dataset_name(late_fusion_worst_dataset)}",
        "- Smallest embedding dimension within 0.01 of hybrid 768 per dataset:",
        *smallest_close_dims,
    ]

    if benchmark_df is not None:
        benchmark_mean = (
            benchmark_df.groupby("model")[
                ["predict_seconds_mean", "peak_python_memory_mb_mean"]
            ]
            .mean()
            .sort_index()
        )
        fastest_model = _as_str(benchmark_mean["predict_seconds_mean"].idxmin())
        lightest_model = _as_str(benchmark_mean["peak_python_memory_mb_mean"].idxmin())
        lines.extend(
            [
                f"- Fastest mean inference model: {_format_model_name(fastest_model)}",
                "- Lowest mean peak Python memory: "
                f"{_format_model_name(lightest_model)}",
            ]
        )

    lines.extend(
        [
            "- Preprocessing ablation summary: lightweight default remained the "
            "best overall practical choice",
            "- kNN note: cosine distance with brute-force search and odd "
            "k=5 avoids exact vote-count ties in the binary setting",
            "- Token cap note: 256-token embedding limit was kept because 512 "
            "tokens increased runtime on larger datasets",
        ]
    )
    return "\n".join(lines) + "\n"


def _tabular(columns: str, header: list[str], rows: list[list[str]]) -> str:
    """Build a LaTeX tabular block for report tables."""
    rendered_rows = [
        " & ".join(_escape_latex(value) for value in row) + r" \\" for row in rows
    ]
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
            *rendered_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def _escape_latex(value: object) -> str:
    """Escape a cell value for LaTeX table output."""
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_dataset_name(name: str) -> str:
    """Convert dataset slugs into report-friendly labels."""
    mapping = {
        "caffe": "Caffe",
        "incubator-mxnet": "Incubator-MXNet",
        "keras": "Keras",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
    }
    return mapping.get(name, name)


def _format_model_name(name: str) -> str:
    """Convert model ids into report-friendly labels."""
    mapping = {
        "baseline_nb_tfidf": "Baseline",
        "tfidf_lr": "TF-IDF + LogReg",
        "embedding_lr": "Embedding LogReg",
        "late_fusion_lr": "Late Fusion LogReg",
        "hybrid_lr": "Hybrid LogReg",
        "embedding_centroid": "Embedding Centroid",
        "embedding_knn": "Embedding kNN",
    }
    return mapping.get(name, name)


def _format_mode_name(name: str) -> str:
    """Convert preprocessing ids into report-friendly labels."""
    mapping = {
        "none": "Default",
        "stopwords_all": "Stopwords",
        "stopwords_keep_negation": "Stopwords + negation",
        "lemmatize": "Lemmatise",
        "stopwords_keep_negation+lemmatize": "Negation + lemmatise",
    }
    return mapping.get(name, name)


def _format_decimal(value: object, places: int = 3) -> str:
    """Format a numeric value to a fixed number of decimal places."""
    return f"{_to_float(value):.{places}f}"


def _format_percent(value: object, places: int = 1) -> str:
    """Format a percent value for LaTeX output."""
    return f"{_to_float(value):.{places}f}%"


def _format_p_value(value: object) -> str:
    """Format a p-value using scientific notation."""
    val_str = f"{_to_float(value):.4g}"
    if "e" in val_str:
        # Convert to proper standard form
        base, exp = val_str.split("e")
        return f"${base} \\times 10^{{{int(exp)}}}$"
    return f"${val_str}$"


def _as_str(value: object) -> str:
    """Convert a pandas/numpy scalar-like value into a plain string."""
    return str(value)


def _to_float(value: object) -> float:
    """Convert a pandas/numpy scalar-like value into a plain float."""
    return float(pd.Series([value]).iloc[0])


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV if it exists."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def _write_text(path: Path, content: str) -> None:
    """Write a UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


if __name__ == "__main__":
    main()
