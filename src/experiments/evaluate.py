"""Evaluation and reporting helper functions."""

import logging
import time
import tracemalloc
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.sparse import spmatrix
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import (
    ABLATION_DIMENSIONS,
    ABLATION_DIMENSIONS_SLUG,
    AVAILABLE_DATASETS,
    BENCHMARK_COLUMNS,
    MAIN_EMBEDDING_DIMENSION,
    METRIC_COLUMNS,
    PREPROCESSING_MODES,
    RESULTS_DIR,
    SEEDS,
    TEST_SIZE,
)
from features.data_load import load_dataset
from features.embedding_features import build_and_cache_embeddings_for_dim
from features.text_prep import preprocess_texts
from features.tfidf_features import build_tfidf_train_test
from models.baseline_nb_tfidf import BaselineNbTfidf
from models.centroid import CentroidClassifier
from models.knn import KnnEmbeddingClassifier
from models.linear_models import (
    EmbeddingLogisticRegressionModel,
    HybridLogisticRegressionModel,
    TfidfLogisticRegressionModel,
)
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

MAIN_COMPARISON_MODELS = (
    BASELINE_NB_TFIDF,
    TFIDF_LOGREG,
    EMBEDDING_LOGREG,
    HYBRID_LOGREG,
)


def _embedding_cache_key(preprocessing_mode: str) -> str:
    """Return the cache suffix used to keep embedding caches mode-specific."""
    return preprocessing_mode or "none"


SECONDARY_METRIC_COLUMNS = (
    "accuracy_mean",
    "accuracy_std",
    "precision_macro_mean",
    "precision_macro_std",
    "recall_macro_mean",
    "recall_macro_std",
    "f1_macro_mean",
    "f1_macro_std",
    "auc_mean",
    "auc_std",
)


def evaluate_baseline_model(
    texts: pd.Series,
    labels: pd.Series,
    model_factory,
    seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the baseline model over paired train/test splits."""
    rows = []
    for seed in SEEDS if seeds is None else seeds:
        # Keep the split seed aligned with the other models for fair pairing.
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=labels,
        )
        model = model_factory()
        (x_train_dense, x_test_dense), prep_seconds, prep_memory_mb = _benchmark_stage(
            model.prepare_features, x_train, x_test
        )
        _, fit_seconds, fit_memory_mb = _benchmark_stage(
            model.fit_features,
            x_train_dense,
            y_train,
        )
        y_pred, y_prob, predict_seconds, predict_memory_mb = _benchmark_predictions(
            model.predict_from_features,
            (x_test_dense,),
            model.predict_proba_from_features,
            (x_test_dense,),
        )
        rows.append(
            _build_result_row(
                seed,
                BASELINE_NB_TFIDF,
                y_test.to_numpy(),
                y_pred,
                y_prob,
                prep_seconds,
                prep_memory_mb,
                fit_seconds,
                fit_memory_mb,
                predict_seconds,
                predict_memory_mb,
            )
        )

    per_run_results = pd.DataFrame(rows)
    return per_run_results, _summarize_results(per_run_results)


def run_dataset_experiments(
    dataset_name: str,
    preprocessing_mode: str = "none",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full main comparison for one dataset."""
    texts, labels = _load_preprocessed_dataset(dataset_name, preprocessing_mode)
    # Build embeddings once per dataset, then reuse them across all paired runs.
    embeddings = build_and_cache_embeddings_for_dim(
        dataset_name,
        texts,
        MAIN_EMBEDDING_DIMENSION,
        cache_key=_embedding_cache_key(preprocessing_mode),
    )
    baseline_runs, _ = evaluate_baseline_model(
        texts,
        labels,
        model_factory=BaselineNbTfidf,
        seeds=SEEDS,
    )
    feature_runs = _evaluate_feature_models(texts, labels, embeddings)
    all_runs = pd.concat([baseline_runs, feature_runs], ignore_index=True)
    all_runs["embedding_dimension"] = MAIN_EMBEDDING_DIMENSION
    summary = _summarize_results(all_runs)
    summary["embedding_dimension"] = MAIN_EMBEDDING_DIMENSION
    return all_runs, summary


def run_embedding_ablation(
    dataset_name: str,
    preprocessing_mode: str = "none",
) -> pd.DataFrame:
    """Run embedding-only ablations across lower dimensions."""
    texts, labels = _load_preprocessed_dataset(dataset_name, preprocessing_mode)
    rows = []
    for dim in ABLATION_DIMENSIONS:
        # Reuse the same evaluation path so only the embedding dimension changes.
        embeddings = build_and_cache_embeddings_for_dim(
            dataset_name,
            texts,
            dim,
            cache_key=_embedding_cache_key(preprocessing_mode),
        )
        dimension_results = _evaluate_feature_models(
            texts,
            labels,
            embeddings,
            include_tfidf_lr=False,
            include_knn=False,
        )
        dimension_results["embedding_dimension"] = dim
        rows.append(dimension_results)
    return pd.concat(rows, ignore_index=True)


def build_statistics(all_results: pd.DataFrame) -> pd.DataFrame:
    """Compute paired Wilcoxon statistics against the baseline."""
    rows = []
    for dataset_name in sorted(all_results["dataset"].unique()):
        baseline_scores = _dataset_model_scores(
            all_results,
            dataset_name,
            BASELINE_NB_TFIDF,
        )
        comparison_scores = _dataset_model_scores(
            all_results, dataset_name, HYBRID_LOGREG
        )
        # Compare paired per-seed deltas rather than just comparing means.
        deltas = comparison_scores - baseline_scores
        statistic, p_value = wilcoxon(
            deltas,
            zero_method="wilcox",
            alternative="greater",
        )
        rows.append(
            {
                "dataset": dataset_name,
                "comparison_model": HYBRID_LOGREG,
                "alternative_hypothesis": f"{HYBRID_LOGREG} > {BASELINE_NB_TFIDF}",
                "baseline_f1_mean": baseline_scores.mean(),
                "comparison_f1_mean": comparison_scores.mean(),
                "f1_gain_mean": deltas.mean(),
                "f1_gain_std": deltas.std(ddof=1),
                "f1_percent_improvement": (
                    100.0 * deltas.mean() / baseline_scores.mean()
                ),
                "n_positive_deltas": int((deltas > 0).sum()),
                "n_negative_deltas": int((deltas < 0).sum()),
                "n_zero_deltas": int((deltas == 0).sum()),
                "min_delta_f1": deltas.min(),
                "max_delta_f1": deltas.max(),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def build_friedman_statistics(all_results: pd.DataFrame) -> pd.DataFrame:
    """Compute extra Friedman stats to compare models."""
    rows = []
    for dataset_name in sorted(all_results["dataset"].unique()):
        # Each seed acts as a shared block for ranking the four main models.
        pivoted = (
            all_results[
                (all_results["dataset"] == dataset_name)
                & all_results["model"].isin(MAIN_COMPARISON_MODELS)
            ]
            .pivot(index="seed", columns="model", values="f1_macro")
            .reindex(columns=MAIN_COMPARISON_MODELS)
            .sort_index()
        )
        if pivoted.isnull().any().any():
            raise ValueError(
                f"Missing model scores for Friedman test on dataset '{dataset_name}'"
            )
        statistic, p_value = friedmanchisquare(
            *[pivoted[model].to_numpy() for model in MAIN_COMPARISON_MODELS]
        )
        row = {
            "dataset": dataset_name,
            "models_compared": ", ".join(MAIN_COMPARISON_MODELS),
            "friedman_statistic": statistic,
            "friedman_p_value": p_value,
        }
        row.update(
            {
                f"{model}_mean_rank": rank
                for model, rank in pivoted.rank(
                    axis=1,
                    method="average",
                    ascending=False,
                )
                .mean()
                .items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_full_experiment_outputs(preprocessing_mode: str = "none") -> None:
    """Run and save the full experiment outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _validate_preprocessing_mode(preprocessing_mode)

    all_results_frames = []
    summary_frames = []
    ablation_frames = []
    for dataset_name in load_all_dataset_names():
        logging.info("Processing dataset: %s", dataset_name)
        all_runs, summary, ablation = _run_and_save_dataset(
            dataset_name,
            preprocessing_mode,
        )
        all_results_frames.append(all_runs)
        summary_frames.append(summary)
        ablation_frames.append(ablation)

    all_results = pd.concat(all_results_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    ablation = pd.concat(ablation_frames, ignore_index=True)
    _save_summary_outputs(
        summary,
        build_statistics(all_results),
        _summarise_ablation(ablation),
        build_friedman_statistics(all_results),
        preprocessing_mode,
    )


def save_single_dataset_outputs(
    dataset_name: str,
    preprocessing_mode: str = "none",
) -> None:
    """Run and save experiments for one dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _validate_preprocessing_mode(preprocessing_mode)

    logging.info("Processing dataset: %s", dataset_name)
    all_runs, summary, _ = _run_and_save_dataset(
        dataset_name,
        preprocessing_mode,
    )
    stats = build_statistics(all_runs)
    friedman_stats = build_friedman_statistics(all_runs)

    summary_path = _result_path(
        f"{dataset_name}_summary_{MAIN_EMBEDDING_DIMENSION}",
        preprocessing_mode,
    )
    stats_path = _result_path(
        f"{dataset_name}_wilcoxon_summary_{MAIN_EMBEDDING_DIMENSION}",
        preprocessing_mode,
    )
    friedman_path = _result_path(
        f"{dataset_name}_friedman_main_models_{MAIN_EMBEDDING_DIMENSION}",
        preprocessing_mode,
    )

    summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)
    friedman_stats.to_csv(friedman_path, index=False)
    _save_report_ready_outputs(summary, preprocessing_mode)

    logging.info("Saved summary to %s", summary_path)
    logging.info("Saved Wilcoxon stats to %s", stats_path)
    logging.info("Saved Friedman stats to %s", friedman_path)


def save_report_outputs_from_summary_csv(
    summary_path: Path,
    preprocessing_mode: str = "none",
) -> None:
    """Generate report-facing helper outputs from an existing summary CSV."""
    _save_report_ready_outputs(pd.read_csv(summary_path), preprocessing_mode)


def load_all_dataset_names() -> list[str]:
    """Return the configured dataset list."""
    return list(AVAILABLE_DATASETS)


def _run_and_save_dataset(
    dataset_name: str,
    preprocessing_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one dataset and save the per-dataset CSV outputs."""

    all_runs, summary = run_dataset_experiments(dataset_name, preprocessing_mode)
    all_runs["dataset"] = dataset_name
    summary["dataset"] = dataset_name

    dataset_results_path = _result_path(
        f"{dataset_name}_all_models_{MAIN_EMBEDDING_DIMENSION}",
        preprocessing_mode,
    )
    all_runs.to_csv(dataset_results_path, index=False)

    ablation = run_embedding_ablation(dataset_name, preprocessing_mode)
    ablation["dataset"] = dataset_name
    ablation_path = _result_path(
        f"{dataset_name}_embedding_ablation_{ABLATION_DIMENSIONS_SLUG}",
        preprocessing_mode,
    )
    ablation.to_csv(ablation_path, index=False)

    logging.info("Completed experiments for %s", dataset_name)
    logging.info("Saved per-run results to %s", dataset_results_path)
    logging.info("Saved ablation results to %s", ablation_path)
    return all_runs, summary, ablation


def _save_summary_outputs(
    summary: pd.DataFrame,
    stats: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    friedman_stats: pd.DataFrame,
    preprocessing_mode: str,
) -> None:
    """Save the top-level summary CSV outputs."""
    summary_path = _result_path(
        f"summary_{MAIN_EMBEDDING_DIMENSION}",
        preprocessing_mode,
    )
    stats_path = _result_path(
        f"wilcoxon_summary_{MAIN_EMBEDDING_DIMENSION}",
        preprocessing_mode,
    )
    ablation_path = _result_path(
        f"embedding_ablation_summary_{ABLATION_DIMENSIONS_SLUG}",
        preprocessing_mode,
    )
    friedman_path = _result_path(
        f"friedman_main_models_{MAIN_EMBEDDING_DIMENSION}",
        preprocessing_mode,
    )

    summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)
    ablation_summary.to_csv(ablation_path, index=False)
    friedman_stats.to_csv(friedman_path, index=False)
    _save_report_ready_outputs(summary, preprocessing_mode)

    logging.info("Saved summary to %s", summary_path)
    logging.info("Saved Wilcoxon stats to %s", stats_path)
    logging.info("Saved ablation summary to %s", ablation_path)
    logging.info("Saved Friedman stats to %s", friedman_path)


def _load_preprocessed_dataset(
    dataset_name: str,
    preprocessing_mode: str,
) -> tuple[pd.Series, pd.Series]:
    """Load one dataset and apply the selected preprocessing."""
    texts, labels = load_dataset(dataset_name)
    return preprocess_texts(texts, mode=preprocessing_mode), labels


def _validate_preprocessing_mode(preprocessing_mode: str) -> None:
    """Ensure the selected preprocessing mode is supported."""
    if preprocessing_mode not in PREPROCESSING_MODES:
        raise ValueError(
            f"Invalid preprocessing mode '{preprocessing_mode}'. Must be one of "
            f"{PREPROCESSING_MODES}"
        )


def _dataset_model_scores(
    all_results: pd.DataFrame,
    dataset_name: str,
    model_name: str,
) -> np.ndarray:
    """Return per-seed F1 scores for one dataset/model pair."""
    return (
        all_results[
            (all_results["dataset"] == dataset_name)
            & (all_results["model"] == model_name)
        ]
        .sort_values("seed")["f1_macro"]
        .to_numpy()
    )


def _score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | np.floating | np.ndarray]:
    """Score one prediction run."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "auc": roc_auc_score(y_true, y_prob),
    }


def _split_run_data(
    texts: pd.Series,
    labels_array: np.ndarray,
    embeddings: np.ndarray,
    seed: int,
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create one paired train/test split for all feature sets."""
    train_index, test_index = train_test_split(
        np.arange(len(texts)),
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=labels_array,
    )
    return (
        texts.iloc[train_index],
        texts.iloc[test_index],
        embeddings[train_index],
        embeddings[test_index],
        labels_array[train_index],
        labels_array[test_index],
    )


def _build_feature_models(
    include_tfidf_lr: bool = True,
    include_knn: bool = True,
) -> dict[str, object]:
    """Create all non-baseline models for one run."""
    models = {
        EMBEDDING_LOGREG: EmbeddingLogisticRegressionModel(),
        HYBRID_LOGREG: HybridLogisticRegressionModel(),
        EMBEDDING_CENTROID: CentroidClassifier(),
    }
    if include_tfidf_lr:
        models[TFIDF_LOGREG] = TfidfLogisticRegressionModel()
    if include_knn:
        models[EMBEDDING_KNN] = KnnEmbeddingClassifier()
    return models


def _evaluate_feature_models(
    texts: pd.Series,
    labels: pd.Series,
    embeddings: np.ndarray,
    include_tfidf_lr: bool = True,
    include_knn: bool = True,
) -> pd.DataFrame:
    """Evaluate TF-IDF, embedding, and hybrid models per seed."""
    rows = []
    labels_array = labels.to_numpy()

    for seed in SEEDS:
        # Split once, then reuse the same train/test views across every model.
        train_texts, test_texts, x_train_embed, x_test_embed, y_train, y_test = (
            _split_run_data(texts, labels_array, embeddings, seed)
        )
        x_train_tfidf, x_test_tfidf, tfidf_benchmark = _prepare_tfidf_features(
            train_texts,
            test_texts,
            include_tfidf_lr or HYBRID_LOGREG in MODEL_NAMES,
        )
        models = _build_feature_models(
            include_tfidf_lr=include_tfidf_lr,
            include_knn=include_knn,
        )
        predictions, benchmarks = _predict_feature_models(
            models,
            x_train_tfidf,
            x_train_embed,
            x_test_tfidf,
            x_test_embed,
            y_train,
            tfidf_benchmark,
        )
        for model_name, (y_pred, y_prob) in predictions.items():
            row = _build_result_row(
                seed,
                model_name,
                y_test,
                y_pred,
                y_prob,
                benchmarks[model_name]["feature_prep_seconds"],
                benchmarks[model_name]["feature_prep_memory_mb"],
                benchmarks[model_name]["fit_seconds"],
                benchmarks[model_name]["fit_memory_mb"],
                benchmarks[model_name]["predict_seconds"],
                benchmarks[model_name]["predict_memory_mb"],
            )
            row["runtime_seconds"] = benchmarks[model_name]["runtime_seconds"]
            row["peak_python_memory_mb"] = benchmarks[model_name][
                "peak_python_memory_mb"
            ]
            rows.append(row)

    return pd.DataFrame(rows)


def _prepare_tfidf_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
    include_tfidf: bool,
) -> tuple[spmatrix | None, spmatrix | None, dict[str, float]]:
    """Build TF-IDF features when required and return their benchmark."""
    empty_benchmark = {
        "feature_prep_seconds": 0.0,
        "feature_prep_memory_mb": 0.0,
    }
    if not include_tfidf:
        return None, None, empty_benchmark

    (x_train_tfidf, x_test_tfidf), prep_seconds, prep_memory_mb = _benchmark_stage(
        build_tfidf_train_test,
        train_texts,
        test_texts,
    )
    return (
        x_train_tfidf,
        x_test_tfidf,
        {
            "feature_prep_seconds": prep_seconds,
            "feature_prep_memory_mb": prep_memory_mb,
        },
    )


def _predict_feature_models(
    models: dict[str, object],
    x_train_tfidf,
    x_train_embed: np.ndarray,
    x_test_tfidf,
    x_test_embed: np.ndarray,
    y_train: np.ndarray,
    tfidf_benchmark: dict[str, float],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, float]]]:
    """Fit, predict, and benchmark all non-baseline models."""
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    benchmarks: dict[str, dict[str, float]] = {}

    # Embedding-only models skip TF-IDF preparation entirely.
    predictions[EMBEDDING_LOGREG], benchmarks[EMBEDDING_LOGREG] = _run_model(
        models[EMBEDDING_LOGREG].fit,
        (x_train_embed, y_train),
        models[EMBEDDING_LOGREG].predict,
        (x_test_embed,),
        models[EMBEDDING_LOGREG].predict_proba,
        (x_test_embed,),
    )
    predictions[HYBRID_LOGREG], benchmarks[HYBRID_LOGREG] = _run_model(
        models[HYBRID_LOGREG].fit,
        (x_train_tfidf, x_train_embed, y_train),
        models[HYBRID_LOGREG].predict,
        (x_test_tfidf, x_test_embed),
        models[HYBRID_LOGREG].predict_proba,
        (x_test_tfidf, x_test_embed),
        **tfidf_benchmark,
    )
    predictions[EMBEDDING_CENTROID], benchmarks[EMBEDDING_CENTROID] = _run_model(
        models[EMBEDDING_CENTROID].fit,
        (x_train_embed, y_train),
        models[EMBEDDING_CENTROID].predict,
        (x_test_embed,),
        models[EMBEDDING_CENTROID].predict_proba,
        (x_test_embed,),
    )

    if TFIDF_LOGREG in models:
        predictions[TFIDF_LOGREG], benchmarks[TFIDF_LOGREG] = _run_model(
            models[TFIDF_LOGREG].fit,
            (x_train_tfidf, y_train),
            models[TFIDF_LOGREG].predict,
            (x_test_tfidf,),
            models[TFIDF_LOGREG].predict_proba,
            (x_test_tfidf,),
            **tfidf_benchmark,
        )
        # Late fusion combines already available probabilities, so there is no
        # extra model fit stage beyond the two component classifiers.
        predictions[LATE_FUSION_LOGREG], benchmarks[LATE_FUSION_LOGREG] = (
            _run_late_fusion(
                predictions[TFIDF_LOGREG][1],
                predictions[EMBEDDING_LOGREG][1],
                benchmarks[TFIDF_LOGREG],
                benchmarks[EMBEDDING_LOGREG],
                **tfidf_benchmark,
            )
        )

    if EMBEDDING_KNN in models:
        predictions[EMBEDDING_KNN], benchmarks[EMBEDDING_KNN] = _run_model(
            models[EMBEDDING_KNN].fit,
            (x_train_embed, y_train),
            models[EMBEDDING_KNN].predict,
            (x_test_embed,),
            models[EMBEDDING_KNN].predict_proba,
            (x_test_embed,),
        )
    return predictions, benchmarks


def _run_model(
    fit_fn,
    fit_args: tuple[object, ...],
    predict_fn: Callable[..., np.ndarray],
    predict_args: tuple[object, ...],
    predict_proba_fn: Callable[..., np.ndarray],
    predict_proba_args: tuple[object, ...],
    feature_prep_seconds: float = 0.0,
    feature_prep_memory_mb: float = 0.0,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, float]]:
    """Benchmark one classifier's fit and inference stages."""
    _, fit_seconds, fit_memory_mb = _benchmark_stage(fit_fn, *fit_args)
    y_pred, y_prob, predict_seconds, predict_memory_mb = _benchmark_predictions(
        predict_fn,
        predict_args,
        predict_proba_fn,
        predict_proba_args,
    )
    return (y_pred, y_prob), _compose_stage_benchmark(
        feature_prep_seconds,
        feature_prep_memory_mb,
        fit_seconds,
        fit_memory_mb,
        predict_seconds,
        predict_memory_mb,
    )


def _run_late_fusion(
    tfidf_prob: np.ndarray,
    embedding_prob: np.ndarray,
    tfidf_benchmark: dict[str, float],
    embedding_benchmark: dict[str, float],
    feature_prep_seconds: float = 0.0,
    feature_prep_memory_mb: float = 0.0,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, float]]:
    """Benchmark late fusion using already computed model probabilities."""
    (y_fuse, p_fuse), fuse_seconds, fuse_memory_mb = _benchmark_stage(
        _late_fusion_predict,
        tfidf_prob,
        embedding_prob,
        0.5,
    )
    return (y_fuse, p_fuse), _compose_stage_benchmark(
        feature_prep_seconds,
        feature_prep_memory_mb,
        tfidf_benchmark["fit_seconds"] + embedding_benchmark["fit_seconds"],
        max(
            tfidf_benchmark["fit_memory_mb"],
            embedding_benchmark["fit_memory_mb"],
        ),
        tfidf_benchmark["predict_seconds"]
        + embedding_benchmark["predict_seconds"]
        + fuse_seconds,
        max(
            tfidf_benchmark["predict_memory_mb"],
            embedding_benchmark["predict_memory_mb"],
            fuse_memory_mb,
        ),
    )


def _benchmark_predictions(
    predict_fn: Callable[..., np.ndarray],
    predict_args: tuple[object, ...],
    predict_proba_fn: Callable[..., np.ndarray],
    predict_proba_args: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Benchmark label prediction and probability prediction together."""
    y_pred, predict_seconds_1, predict_memory_mb_1 = _benchmark_stage(
        predict_fn,
        *predict_args,
    )
    y_prob_matrix, predict_seconds_2, predict_memory_mb_2 = _benchmark_stage(
        predict_proba_fn,
        *predict_proba_args,
    )
    return (
        y_pred,
        y_prob_matrix[:, 1],
        predict_seconds_1 + predict_seconds_2,
        max(predict_memory_mb_1, predict_memory_mb_2),
    )


def _build_result_row(
    seed: int,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_prep_seconds: float,
    feature_prep_memory_mb: float,
    fit_seconds: float,
    fit_memory_mb: float,
    predict_seconds: float,
    predict_memory_mb: float,
) -> dict[str, object]:
    """Assemble one per-run output row."""
    row = {"seed": seed, "model": model_name}
    row.update(_score_predictions(y_true, y_pred, y_prob))
    row.update(
        _compose_stage_benchmark(
            feature_prep_seconds,
            feature_prep_memory_mb,
            fit_seconds,
            fit_memory_mb,
            predict_seconds,
            predict_memory_mb,
        )
    )
    return row


def _summarize_results(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean and std metrics per model."""
    summary = (
        all_runs.groupby("model")[list(METRIC_COLUMNS) + list(BENCHMARK_COLUMNS)]
        .agg(["mean", "std"])
        .round(6)
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    summary["model"] = pd.Categorical(
        summary["model"],
        categories=MODEL_NAMES,
        ordered=True,
    )
    return summary.sort_values("model").reset_index(drop=True)


def _summarise_ablation(ablation_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean and stdev metrics per dimension and model."""
    group_columns = ["embedding_dimension", "model"]
    if "dataset" in ablation_runs.columns:
        group_columns = ["dataset"] + group_columns

    summary = (
        ablation_runs.groupby(group_columns)[list(METRIC_COLUMNS)]
        .agg(["mean", "std"])
        .round(6)
        .reset_index()
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns.to_flat_index()
    ]
    return summary


def _save_report_ready_outputs(
    summary: pd.DataFrame,
    preprocessing_mode: str,
) -> None:
    """Save report-facing helper CSVs and sanity checks."""
    # These are convenience outputs for the report, not replacements for the
    # raw per-run experiment files.
    hybrid = _summary_slice(summary, HYBRID_LOGREG, "hybrid_f1_mean")
    late_fusion = _summary_slice(summary, LATE_FUSION_LOGREG, "late_fusion_f1_mean")
    comparison = hybrid.merge(late_fusion, on="dataset", how="inner")
    comparison["delta_late_minus_hybrid"] = (
        comparison["late_fusion_f1_mean"] - comparison["hybrid_f1_mean"]
    )
    comparison["within_0_01"] = comparison["delta_late_minus_hybrid"].abs() <= 0.01
    comparison.to_csv(
        _result_path("late_fusion_vs_hybrid", preprocessing_mode),
        index=False,
    )

    main_table = summary[summary["model"].isin(MAIN_COMPARISON_MODELS)][
        ["dataset", "model", "f1_macro_mean", "f1_macro_std"]
    ].copy()
    main_table["formatted_macro_f1"] = main_table.apply(
        lambda row: f"{row['f1_macro_mean']:.3f} ({row['f1_macro_std']:.3f})",
        axis=1,
    )
    (
        main_table.pivot(
            index="dataset",
            columns="model",
            values="formatted_macro_f1",
        )
        .reset_index()
        .rename_axis(columns=None)
        .to_csv(
            _result_path("main_table_macro_f1", preprocessing_mode),
            index=False,
        )
    )

    secondary_metrics = summary[summary["model"].isin(MAIN_COMPARISON_MODELS)][
        ["dataset", "model", *SECONDARY_METRIC_COLUMNS]
    ].copy()
    secondary_metrics = secondary_metrics.round(3)
    secondary_metrics.to_csv(
        _result_path("secondary_metrics_table", preprocessing_mode),
        index=False,
    )

    metric_deltas = _build_metric_deltas(summary).round(3)
    metric_deltas.to_csv(
        _result_path("metric_deltas_hybrid_vs_baseline", preprocessing_mode),
        index=False,
    )

    summary[
        [
            "dataset",
            "model",
            "feature_prep_seconds_mean",
            "fit_seconds_mean",
            "predict_seconds_mean",
            "runtime_seconds_mean",
            "feature_prep_memory_mb_mean",
            "fit_memory_mb_mean",
            "predict_memory_mb_mean",
            "peak_python_memory_mb_mean",
        ]
    ].to_csv(
        _result_path(
            f"benchmark_stage_summary_{MAIN_EMBEDDING_DIMENSION}",
            preprocessing_mode,
        ),
        index=False,
    )

    _update_preprocessing_ablation_summary(summary, preprocessing_mode)
    _write_run_notes(preprocessing_mode)


def _summary_slice(
    summary: pd.DataFrame,
    model_name: str,
    renamed_column: str,
) -> pd.DataFrame:
    """Return dataset and renamed macro-F1 mean for one model."""
    return summary[summary["model"] == model_name][["dataset", "f1_macro_mean"]].rename(
        columns={"f1_macro_mean": renamed_column}
    )


def _build_metric_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute hybrid-minus-baseline deltas across the main reported metrics."""
    baseline = (
        summary[summary["model"] == BASELINE_NB_TFIDF].set_index("dataset").sort_index()
    )
    hybrid = (
        summary[summary["model"] == HYBRID_LOGREG].set_index("dataset").sort_index()
    )

    rows = []
    for dataset_name in baseline.index:
        row = {"dataset": dataset_name}
        for metric in METRIC_COLUMNS:
            baseline_value = _to_float(baseline.loc[dataset_name, f"{metric}_mean"])
            hybrid_value = _to_float(hybrid.loc[dataset_name, f"{metric}_mean"])
            delta = hybrid_value - baseline_value
            row[f"{metric}_delta_mean"] = delta
            row[f"{metric}_percent_improvement"] = (
                100.0 * delta / baseline_value if baseline_value != 0 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _to_float(value) -> float:
    """Convert a pandas/numpy scalar into a plain float."""
    return float(np.asarray(value).item())


def _write_run_notes(preprocessing_mode: str) -> None:
    """Write one compact run-notes file for protocol and metric choices."""
    lines = [
        "Run notes",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"preprocessing_mode={preprocessing_mode}",
        f"split_seeds={SEEDS[0]}..{SEEDS[-1]}",
        f"test_size={TEST_SIZE}",
        "split_strategy=paired stratified train_test_split",
        "tfidf_fit_scope=train only per split",
        "embedding_cache=reused when dimension-specific files already exist",
        "primary_metric=macro_f1",
        "significance_tests=wilcoxon on macro_f1 only; friedman on main-model macro_f1 ranks",
        "secondary_metrics=accuracy precision_macro recall_macro auc",
        "benchmarks=feature_prep fit prediction stages",
    ]
    (RESULTS_DIR / f"run_notes{_preprocessing_mode_suffix(preprocessing_mode)}.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _update_preprocessing_ablation_summary(
    summary: pd.DataFrame,
    preprocessing_mode: str,
) -> None:
    """Write or update preprocessing ablation summary for baseline and hybrid."""
    pivoted = (
        summary[summary["model"].isin([BASELINE_NB_TFIDF, HYBRID_LOGREG])][
            ["dataset", "model", "f1_macro_mean"]
        ]
        .pivot(index="dataset", columns="model", values="f1_macro_mean")
        .reset_index()
        .rename_axis(columns=None)
    )
    pivoted["preprocessing_mode"] = preprocessing_mode
    pivoted = pivoted[
        ["preprocessing_mode", "dataset", BASELINE_NB_TFIDF, HYBRID_LOGREG]
    ]

    output_path = RESULTS_DIR / "ablation_preproc_summary.csv"
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing[existing["preprocessing_mode"] != preprocessing_mode]
    pd.concat([existing, pivoted], ignore_index=True).to_csv(output_path, index=False)


def _result_path(stem: str, preprocessing_mode: str) -> Path:
    """Build an output path with the preprocessing suffix applied."""
    return RESULTS_DIR / f"{stem}{_preprocessing_mode_suffix(preprocessing_mode)}.csv"


def _preprocessing_mode_suffix(preprocessing_mode: str) -> str:
    """Convert preprocessing mode into a filename-safe suffix."""
    if preprocessing_mode == "none":
        return ""
    sanitised_suffix = preprocessing_mode.replace(
        "+", "_plus_"
    )  # sanitise for filename safety
    return f"_{sanitised_suffix}"


def _start_benchmark() -> float:
    """Start runtime and Python-memory usage measurements for one stage."""
    tracemalloc.start()
    return time.perf_counter()


def _stop_benchmark(start_time: float) -> tuple[float, float]:
    """Stop runtime and Python-memory usage measurements for one stage."""
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return time.perf_counter() - start_time, peak_memory / (1024 * 1024)


def _benchmark_stage(func, *args):
    """Run one stage and capture its runtime and peak Python memory."""
    start_time = _start_benchmark()
    result = func(*args)
    runtime_seconds, peak_python_memory_mb = _stop_benchmark(start_time)
    return result, runtime_seconds, peak_python_memory_mb


def _compose_stage_benchmark(
    feature_prep_seconds: float,
    feature_prep_memory_mb: float,
    fit_seconds: float,
    fit_memory_mb: float,
    predict_seconds: float,
    predict_memory_mb: float,
) -> dict[str, float]:
    """Combine stage benchmarks into the saved benchmark columns."""
    return {
        "feature_prep_seconds": feature_prep_seconds,
        "feature_prep_memory_mb": feature_prep_memory_mb,
        "fit_seconds": fit_seconds,
        "fit_memory_mb": fit_memory_mb,
        "predict_seconds": predict_seconds,
        "predict_memory_mb": predict_memory_mb,
        "runtime_seconds": feature_prep_seconds + fit_seconds + predict_seconds,
        "peak_python_memory_mb": max(
            feature_prep_memory_mb,
            fit_memory_mb,
            predict_memory_mb,
        ),
    }


def _late_fusion_predict(
    tfidf_prob: np.ndarray,
    embedding_prob: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average two probability vectors for late fusion."""
    p_fuse = alpha * tfidf_prob + (1.0 - alpha) * embedding_prob
    return (p_fuse >= 0.5).astype(int), p_fuse
