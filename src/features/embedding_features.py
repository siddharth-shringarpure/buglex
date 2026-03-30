"""Sentence embedding helpers with cache-aware Matryoshka truncation."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_NAME,
)


_EMBEDDING_MODEL: SentenceTransformer | None = None
_CLASSIFICATION_PREFIX: str = "classification: "


def build_and_cache_full_embeddings(
    dataset_name: str,
    texts: pd.Series,
    cache_key: str = "",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build and cache full embeddings for a given dataset.

    Args:
        dataset_name: Dataset stem for cache filenames
        texts: Preprocessed text inputs
        cache_key: Extra cache suffix

    Returns:
        Full embedding array and mapping dataframe
    """
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    full_path = _embedding_cache_path(dataset_name, "full", cache_key)
    mapping_path = _mapping_cache_path(dataset_name, cache_key)

    if full_path.exists() and mapping_path.exists():
        full_embeddings = np.load(full_path)
        mapping = pd.read_csv(mapping_path)
        return full_embeddings, mapping

    model = _load_embedding_model()

    prefixed_texts = [f"{_CLASSIFICATION_PREFIX}{text}" for text in texts.tolist()]
    full_embeddings = model.encode(
        prefixed_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )

    np.save(full_path, full_embeddings)

    mapping = pd.DataFrame(
        {
            "row_index": np.arange(len(texts)),
            "text_index": texts.index.to_numpy(),
            "model_name": EMBEDDING_MODEL_NAME,
            "task_prefix": _CLASSIFICATION_PREFIX.strip(),
        }
    )
    mapping.to_csv(mapping_path, index=False)
    return full_embeddings, mapping


def truncate_matryoshka_embeddings(
    full_embeddings: np.ndarray,
    dimension: int,
) -> np.ndarray:
    """Apply Matryoshka-style truncation and normalisation to full embeddings.

    Args:
        full_embeddings: Full embedding matrix
        dimension: Target embedding dimension

    Returns:
        Normalised truncated embedding matrix

    Raises:
        ValueError: If dimension is unsupported
    """
    if dimension not in EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Invalid embedding dimension {dimension}. Must be one of "
            f"{EMBEDDING_DIMENSIONS}"
        )

    tensor_embeddings = torch.from_numpy(full_embeddings)
    normalised = functional.layer_norm(
        tensor_embeddings,
        normalized_shape=(tensor_embeddings.shape[1],),
    )
    truncated = normalised[:, :dimension]
    scaled = functional.normalize(truncated, p=2, dim=1)
    return scaled.cpu().numpy()


def build_and_cache_embeddings_for_dim(
    dataset_name: str,
    texts: pd.Series,
    dimension: int,
    cache_key: str = "",
) -> np.ndarray:
    """Build or load embeddings for a chosen dimension.

    Args:
        dataset_name: Dataset stem for cache filenames
        texts: Preprocessed text inputs
        dimension: Target embedding dimension
        cache_key: Extra cache suffix, for example preprocessing mode

    Returns:
        Truncated embedding array
    """
    dim_path = _embedding_cache_path(dataset_name, str(dimension), cache_key)
    if dim_path.exists():
        return np.load(dim_path)

    full_embeddings, _ = build_and_cache_full_embeddings(
        dataset_name,
        texts,
        cache_key=cache_key,
    )
    truncated_embeddings = truncate_matryoshka_embeddings(full_embeddings, dimension)
    np.save(dim_path, truncated_embeddings)
    return truncated_embeddings


def _load_embedding_model() -> SentenceTransformer:
    """Load the configured sentence embedding model.

    Returns:
        SentenceTransformer model instance
    """
    global _EMBEDDING_MODEL  # pylint: disable=global-statement

    if _EMBEDDING_MODEL is None:
        if _has_local_model_snapshot():
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            _EMBEDDING_MODEL = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            _EMBEDDING_MODEL = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                trust_remote_code=True,
            )
        _EMBEDDING_MODEL.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
    return _EMBEDDING_MODEL


def _has_local_model_snapshot() -> bool:
    """Check whether the configured Hugging Face model is cached locally.

    Returns:
        True if a snapshot exists in the local Hugging Face cache
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = EMBEDDING_MODEL_NAME.replace("/", "--")
    snapshot_root = cache_root / f"models--{model_slug}" / "snapshots"
    if not snapshot_root.exists():
        return False
    return any(path.is_dir() for path in snapshot_root.iterdir())


def _embedding_cache_path(dataset_name: str, suffix: str, cache_key: str = "") -> Path:
    """Build a model-aware cache path for embeddings.

    Args:
        dataset_name: Dataset stem
        suffix: Cache suffix such as full or dimension
        cache_key: Extra cache suffix, for example preprocessing mode

    Returns:
        Cache file path
    """
    model_slug = EMBEDDING_MODEL_NAME.replace("/", "__").replace(".", "_")
    cache_key_suffix = f"_{cache_key}" if cache_key else ""
    filename = f"embeddings_{dataset_name}_{model_slug}{cache_key_suffix}_{suffix}.npy"
    return EMBEDDING_CACHE_DIR / filename


def _mapping_cache_path(dataset_name: str, cache_key: str = "") -> Path:
    """Build a model-aware cache path for row mapping.

    Args:
        dataset_name: Dataset stem
        cache_key: Extra cache suffix, for example preprocessing mode

    Returns:
        Mapping CSV path
    """
    model_slug = EMBEDDING_MODEL_NAME.replace("/", "__").replace(".", "_")
    cache_key_suffix = f"_{cache_key}" if cache_key else ""
    filename = f"embeddings_{dataset_name}_{model_slug}{cache_key_suffix}_mapping.csv"
    return EMBEDDING_CACHE_DIR / filename
