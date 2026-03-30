"""TF-IDF feature helper function."""

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE


def build_tfidf_train_test(
    train_texts: pd.Series,
    test_texts: pd.Series,
) -> tuple[sparse.spmatrix, sparse.spmatrix]:
    """Build paired TF-IDF features without test leakage.

    Args:
        train_texts: Training text inputs
        test_texts: Test text inputs

    Returns:
        Train and test TF-IDF matrices
    """
    vectoriser = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
    )
    x_train = vectoriser.fit_transform(train_texts)
    x_test = vectoriser.transform(test_texts)
    return x_train, x_test
