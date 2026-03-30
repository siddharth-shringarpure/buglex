"""Baseline model using TF-IDF with a Naive Bayes classifier."""

# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB


class BaselineNbTfidf(BaseEstimator, ClassifierMixin):
    """Wrap TF-IDF and GaussianNB as one sklearn-style model."""

    def __init__(
        self,
        max_features: int = 3000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectoriser: TfidfVectorizer | None = None
        self.model_: GaussianNB | None = None

    def fit(self, texts: pd.Series, labels: pd.Series) -> "BaselineNbTfidf":
        """Fit the TF-IDF plus GaussianNB baseline."""
        x_train, _ = self.prepare_features(texts, texts.iloc[:0])
        self.fit_features(x_train, labels)
        return self

    def predict(self, texts: pd.Series) -> np.ndarray:
        """Predict class labels for input texts."""
        vectoriser, model = self._get_fitted_components()
        features = vectoriser.transform(texts).toarray()
        return model.predict(features)

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """Predict class probabilities for input texts."""
        vectoriser, model = self._get_fitted_components()
        features = vectoriser.transform(texts).toarray()
        return model.predict_proba(features)

    def _get_fitted_components(self) -> tuple[TfidfVectorizer, GaussianNB]:
        """Return fitted model components."""
        if self.vectoriser is None or self.model_ is None:
            raise ValueError("BaselineNbTfidf must be fitted before prediction")
        return self.vectoriser, self.model_

    def prepare_features(
        self,
        train_texts: pd.Series,
        test_texts: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build dense TF-IDF features for one train/test split."""
        vectoriser = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )
        self.vectoriser = vectoriser
        x_train = vectoriser.fit_transform(train_texts).toarray()
        x_test = vectoriser.transform(test_texts).toarray()
        return x_train, x_test

    def fit_features(self, x_train: np.ndarray, labels: pd.Series) -> None:
        """Fit GaussianNB on precomputed dense features."""
        search = GridSearchCV(
            GaussianNB(),
            {"var_smoothing": np.logspace(-11, -7, 5)},
            cv=5,
            scoring="f1_macro",
        )
        search.fit(x_train, labels)
        self.model_ = search.best_estimator_

    def predict_from_features(self, x_test: np.ndarray) -> np.ndarray:
        """Predict labels from precomputed dense features."""
        _, model = self._get_fitted_components()
        return model.predict(x_test)

    def predict_proba_from_features(self, x_test: np.ndarray) -> np.ndarray:
        """Predict probabilities from precomputed dense features."""
        _, model = self._get_fitted_components()
        return model.predict_proba(x_test)
