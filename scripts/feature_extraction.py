# scripts/feature_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

def extract_tfidf_features(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_features = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_features

def extract_sentence_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = []
    for i in tqdm(range(0, len(texts), 32), desc="Batches"):
        batch = texts[i:i+32]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)
