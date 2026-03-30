"""Model registry."""

BASELINE_NB_TFIDF = "baseline_nb_tfidf"
TFIDF_LOGREG = "tfidf_lr"
EMBEDDING_LOGREG = "embedding_lr"
LATE_FUSION_LOGREG = "late_fusion_lr"
HYBRID_LOGREG = "hybrid_lr"
EMBEDDING_CENTROID = "embedding_centroid"
EMBEDDING_KNN = "embedding_knn"

MODEL_NAMES = (
    BASELINE_NB_TFIDF,
    TFIDF_LOGREG,
    EMBEDDING_LOGREG,
    LATE_FUSION_LOGREG,
    HYBRID_LOGREG,
    EMBEDDING_CENTROID,
    EMBEDDING_KNN,
)
