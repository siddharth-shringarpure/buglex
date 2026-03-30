"""Configuration file for experiments."""

from pathlib import Path

# Paths/dirs
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
RESULTS_DIR = REPO_ROOT / "results"
EMBEDDING_CACHE_DIR = RESULTS_DIR / "embeddings"

AVAILABLE_DATASETS = (
    "caffe",
    "incubator-mxnet",
    "keras",
    "pytorch",
    "tensorflow",
)

TEST_SIZE = 0.3
N_RUNS = 30
SEEDS = list(range(N_RUNS))

METRIC_COLUMNS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "auc",
)

EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MAX_SEQ_LENGTH = 256
MAIN_EMBEDDING_DIMENSION = 768
ABLATION_DIMENSIONS = [512, 256, 128, 64]
EMBEDDING_DIMENSIONS = [MAIN_EMBEDDING_DIMENSION] + ABLATION_DIMENSIONS
ABLATION_DIMENSIONS_SLUG = "_".join(str(dimension) for dimension in ABLATION_DIMENSIONS)
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
