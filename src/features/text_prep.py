"""Text pre-processing functions."""

import re
import string

import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import PREPROCESSING_MODES


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "nor",
    "nt",
    "should",
    "might",
    "could",
    "would",
}
STOPWORDS_KEEP_NEGATION = ENGLISH_STOP_WORDS.difference(NEGATION_WORDS)
LEMMATISER = WordNetLemmatizer()


def clean_text(
    text: str,
    strip_punctuation: bool = True,
    mode: str = "none",
) -> str:
    """Clean a text string with lightweight preprocessing.

    Args:
        text: Raw issue text
        strip_punctuation: Remove punctuation characters (default: True)
        mode: Preprocessing mode from config (default: "none")

    Returns:
        Cleaned text string
    """
    if mode not in PREPROCESSING_MODES:
        raise ValueError(
            f"Invalid preprocessing mode '{mode}'. Must be one of {PREPROCESSING_MODES}"
        )

    cleaned = str(text).lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.encode("ascii", "ignore").decode(
        "ascii"
    )  # remove non-ASCII, eg: emojis

    if strip_punctuation:
        cleaned = cleaned.translate(PUNCTUATION_TABLE)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned)
    tokens = cleaned.strip().split()

    if mode == "stopwords_all":
        tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    elif mode in (
        "stopwords_keep_negation",
        "stopwords_keep_negation+lemmatize",
    ):
        tokens = [token for token in tokens if token not in STOPWORDS_KEEP_NEGATION]

    if mode in ("lemmatize", "stopwords_keep_negation+lemmatize"):
        tokens = [LEMMATISER.lemmatize(token) for token in tokens]

    return " ".join(tokens)


def preprocess_texts(
    texts: pd.Series,
    strip_punctuation: bool = True,
    mode: str = "none",
) -> pd.Series:
    """Apply lightweight preprocessing to all texts.

    Args:
        texts: Raw text series
        strip_punctuation: Remove punctuation characters (default: True)
        mode: Preprocessing mode from config (default: "none")

    Returns:
        Cleaned text series
    """
    return (
        texts.fillna("")
        .astype(str)
        .apply(
            lambda text: clean_text(
                text,
                strip_punctuation=strip_punctuation,
                mode=mode,
            )
        )
    )
