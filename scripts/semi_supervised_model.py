# scripts/semi_supervised_model.py
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import LatentDirichletAllocation

def train_hybrid_model(tfidf_features, embeddings):
    """
    Trains the hybrid semi-supervised model:
    - Isolation Forest (Anomaly Detection)
    - LDA (Topic Modeling)
    """
    print("\n🔹 Training Isolation Forest (Anomaly Detector)...")
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(embeddings)

    print("🔹 Training LDA Topic Model...")
    lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
    lda_model.fit(tfidf_features)

    print("✅ Hybrid model training complete!")
    return isolation_forest, lda_model
