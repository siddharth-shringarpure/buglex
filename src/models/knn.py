"""kNN embedding classifier."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KnnEmbeddingClassifier:
    """Classify embeddings with cosine k-nearest neighbors."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit kNN on dense embeddings."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(x_test)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray | list[np.ndarray]:
        """Predict class probabilities."""
        return self.model.predict_proba(x_test)
