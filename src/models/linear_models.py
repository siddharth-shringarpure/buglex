"""Linear model implementations."""

# pyright: reportArgumentType=false

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression


class TfidfLogisticRegressionModel:
    """Train logistic regression on TF-IDF features."""

    def __init__(self) -> None:
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
        )

    def fit(self, x_train: sparse.csr_matrix, y_train: np.ndarray) -> None:
        """Fit the model on sparse TF-IDF features."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test: sparse.csr_matrix) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(x_test)

    def predict_proba(self, x_test: sparse.csr_matrix) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(x_test)


class EmbeddingLogisticRegressionModel:
    """Train logistic regression on dense embedding features."""

    def __init__(self) -> None:
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on dense embeddings."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(x_test)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(x_test)


class HybridLogisticRegressionModel:
    """Train logistic regression on TF-IDF and embedding features."""

    def __init__(self) -> None:
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
        )

    def fit(
        self,
        x_train_tfidf: sparse.csr_matrix,
        x_train_embed: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Fit on concatenated sparse and embedding features."""
        hybrid_train = sparse.hstack(
            [
                x_train_tfidf,
                sparse.csr_matrix(x_train_embed),
            ],
            format="csr",
        )
        self.model.fit(hybrid_train, y_train)

    def predict(
        self,
        x_test_tfidf: sparse.csr_matrix,
        x_test_embed: np.ndarray,
    ) -> np.ndarray:
        """Predict class labels."""
        hybrid_test = sparse.hstack(
            [
                x_test_tfidf,
                sparse.csr_matrix(x_test_embed),
            ],
            format="csr",
        )
        return self.model.predict(hybrid_test)

    def predict_proba(
        self,
        x_test_tfidf: sparse.csr_matrix,
        x_test_embed: np.ndarray,
    ) -> np.ndarray:
        """Predict class probabilities."""
        hybrid_test = sparse.hstack(
            [
                x_test_tfidf,
                sparse.csr_matrix(x_test_embed),
            ],
            format="csr",
        )
        return self.model.predict_proba(hybrid_test)
