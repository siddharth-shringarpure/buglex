"""Embedding model smoke test.

Downloads the configured sentence embedding model, verifies the output
shape on a single text, and benchmarks encoding throughput on a small
synthetic batch matching the project's default batch size.
"""

import time

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
_MAX_SEQ_LENGTH = 256
_EXPECTED_DIM = 768
_BATCH_SIZE = 8
_BENCHMARK_N = 20
_TASK_PREFIX = "classification: "


def _load_model() -> SentenceTransformer:
    """Load the sentence embedding model with the project's sequence cap.

    Returns:
        Loaded SentenceTransformer model.
    """
    model = SentenceTransformer(_MODEL_NAME, trust_remote_code=True)
    model.max_seq_length = _MAX_SEQ_LENGTH
    return model


def _check_shape(model: SentenceTransformer) -> None:
    """Encode text and assert the output dimension is correct.

    Args:
        model: Loaded sentence embedding model.

    Raises:
        AssertionError: If the embedding dimension is unexpected.
    """
    embeddings = model.encode(
        [f"{_TASK_PREFIX}test performance bug report"],
        batch_size=1,
        convert_to_numpy=True,
    )
    assert embeddings.shape == (1, _EXPECTED_DIM), (
        f"Unexpected shape: {embeddings.shape}"
    )
    print(f"Shape check: {embeddings.shape} -- OK")


def _benchmark_throughput(model: SentenceTransformer) -> None:
    """Encode a small synthetic batch and report encoding throughput.

    Args:
        model: Loaded sentence embedding model.

    Raises:
        AssertionError: If the batch output shape is unexpected.
    """
    texts = [
        f"{_TASK_PREFIX}synthetic bug report number {i}" for i in range(_BENCHMARK_N)
    ]
    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - start

    assert embeddings.shape == (_BENCHMARK_N, _EXPECTED_DIM), (
        f"Unexpected batch shape: {embeddings.shape}"
    )
    throughput = len(texts) / elapsed
    ms_per_text = elapsed / len(texts) * 1000
    print(f"Batch shape: {embeddings.shape} -- OK")
    print(f"Encoded {len(texts)} texts in {elapsed:.3f}s")
    print(f"Throughput: {throughput:.1f} texts/sec | {ms_per_text:.1f} ms/text")


def main() -> None:
    """Run the embedding model shape check and throughput benchmark."""
    print(f"Loading model: {_MODEL_NAME}")
    model = _load_model()
    print("Model loaded.")
    _check_shape(model)
    _benchmark_throughput(model)
    print("Embedding smoke test passed.")


if __name__ == "__main__":
    main()
