from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os

def train_lda(X_tfidf, n_topics=5):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_topics = lda.fit_transform(X_tfidf)
    return lda, lda_topics

def train_isolation_forest(X_true, contamination=0.2):
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_true)
    return iso

def evaluate_model(true_labels, pred_labels, save_path="outputs/confusion_matrix.png"):
    print("\n--- Evaluation Report ---\n")
    print(classification_report(true_labels, pred_labels, target_names=["Fake", "True"]))

    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Fake','True'], yticklabels=['Fake','True'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path)
    plt.show()

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved at {path}")
