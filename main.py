# main.py
import os
import pandas as pd
import numpy as np
import joblib

from scripts.data_preprocessing import clean_texts
from scripts.feature_extraction import extract_tfidf_features, extract_sentence_embeddings
from scripts.semi_supervised_model import train_hybrid_model
from scripts.visualize import visualize_topics, plot_anomaly_scores

# =============== LOAD DATASET ===============
print("🔹 Loading dataset...")
true_df = pd.read_csv("dataset/True.csv")
fake_df = pd.read_csv("dataset/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

# =============== CLEAN TEXT ===============
print("🔹 Cleaning text data...")
cleaned_texts = clean_texts(texts)

# =============== FEATURE EXTRACTION ===============
print("🔹 Extracting TF-IDF and sentence embeddings...")
tfidf_vectorizer, tfidf_features = extract_tfidf_features(cleaned_texts)
sentence_embeddings = extract_sentence_embeddings(cleaned_texts)

# =============== TRAIN HYBRID MODEL ===============
print("🔹 Training hybrid semi-supervised model...")
isolation_forest, lda_model = train_hybrid_model(tfidf_features, sentence_embeddings)

# =============== VISUALIZATIONS ===============
os.makedirs("outputs", exist_ok=True)
visualize_topics(lda_model, tfidf_vectorizer.get_feature_names_out(), save_path="outputs/lda_topics.png")
plot_anomaly_scores(isolation_forest, sentence_embeddings, save_path="outputs/anomaly_scores.png")

# =============== SAVE MODELS ===============
print("\n🔹 Saving trained models for testing...")
joblib.dump(tfidf_vectorizer, "outputs/tfidf_vectorizer.pkl")
joblib.dump(lda_model, "outputs/lda_model.pkl")
joblib.dump(isolation_forest, "outputs/isolation_forest.pkl")
print("✅ Models saved successfully in 'outputs/' folder.")

# =============== EVALUATION ===============
print("\n🔹 Evaluating anomaly detection on known labels...")
anomaly_preds = isolation_forest.predict(sentence_embeddings)
anomaly_preds = [1 if p == 1 else 0 for p in anomaly_preds]  # Convert -1 to 0 (fake), 1 to real

from sklearn.metrics import classification_report, accuracy_score
print("\nClassification Report:")
print(classification_report(labels, anomaly_preds))
print(f"Accuracy: {accuracy_score(labels, anomaly_preds) * 100:.2f}%")

print("\n✅ Training complete! All outputs saved to the 'outputs/' folder.")
