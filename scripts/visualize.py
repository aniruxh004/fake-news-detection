# scripts/visualize.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def visualize_topics(lda_model, feature_names, save_path="outputs/lda_topics.png", n_top_words=10):
    plt.figure(figsize=(10, 6))
    for idx, topic in enumerate(lda_model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[-n_top_words:]]
        plt.barh(range(len(top_features)), topic[topic.argsort()[-n_top_words:]])
        plt.yticks(range(len(top_features)), top_features)
        plt.title(f"Topic #{idx + 1}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def plot_anomaly_scores(model, embeddings, save_path="outputs/anomaly_scores.png"):
    scores = model.decision_function(embeddings)
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=50, color="blue", alpha=0.7)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
