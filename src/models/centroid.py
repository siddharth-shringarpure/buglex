"""Embedding centroid classifier."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CentroidClassifier:
    """Classify by cosine similarity to class centroids."""

    def __init__(self) -> None:
        self.centroids_: dict[int, np.ndarray] = {}
        self.classes_: np.ndarray | None = None
        self.majority_class_: int | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit class centroids."""
        classes = np.unique(y_train)
        self.classes_ = classes
        values, counts = np.unique(y_train, return_counts=True)
        self.majority_class_ = int(values[np.argmax(counts)])
        self.centroids_ = {
            int(label): x_train[y_train == label].mean(axis=0) for label in classes
        }

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict class labels by nearest centroid."""
        classes = self._get_fitted_classes()
        probabilities = self.predict_proba(x_test)
        predictions = classes[np.argmax(probabilities, axis=1)]

        # Tie-break strategy: use the training majority class.
        # Reasoning: keeps fallback deterministic; likely better than random choice.
        has_tie = (probabilities == probabilities.max(axis=1, keepdims=True)).sum(
            axis=1
        ) > 1
        if np.any(has_tie):
            predictions[has_tie] = self._get_majority_class()

        return predictions

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """Return normalised centroid similarity scores."""
        classes = self._get_fitted_classes()

        similarity_columns = []
        for label in classes:
            centroid = self.centroids_[int(label)].reshape(1, -1)
            scores = cosine_similarity(x_test, centroid).ravel()
            similarity_columns.append(scores)

        similarities = np.column_stack(similarity_columns)
        # Shift all similarities to be non-negative before row-wise normalising.
        similarities = similarities - similarities.min(axis=1, keepdims=True)
        row_sums = similarities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return similarities / row_sums

    def _get_fitted_classes(self) -> np.ndarray:
        """Return fitted classes array with guaranteed non-None type."""
        if self.classes_ is None:
            raise ValueError("CentroidClassifier must be fitted before prediction")
        return self.classes_

    def _get_majority_class(self) -> int:
        """Return the stored training majority class."""
        if self.majority_class_ is None:
            raise ValueError("CentroidClassifier must be fitted before prediction")
        return self.majority_class_
