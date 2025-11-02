# test.py
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================
# Load trained models
# =============================
print("🔹 Loading saved models...")
tfidf_vectorizer = joblib.load("outputs/tfidf_vectorizer.pkl")
lda_model = joblib.load("outputs/lda_model.pkl")
isolation_forest = joblib.load("outputs/isolation_forest.pkl")

# =============================
# Prepare new article
# =============================
new_article = """
donald trump is the new prime minister of india
"""

# =============================
# Clean the text (same preprocessing as training)
# =============================
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text.lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

cleaned_article = clean_text(new_article)

# =============================
# Feature Extraction
# =============================
print("🔹 Extracting features...")
tfidf_features = tfidf_vectorizer.transform([cleaned_article])

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode([cleaned_article])

# =============================
# Topic Distribution (optional)
# =============================
topic_distribution = lda_model.transform(tfidf_features)
top_topic = np.argmax(topic_distribution)
print(f"\n🧩 Dominant Topic: {top_topic}")

# =============================
# Fake/Real Prediction using Isolation Forest
# =============================
prediction = isolation_forest.predict(embedding)[0]
label = "REAL (Legitimate)" if prediction == 1 else "FAKE (Anomalous)"

print(f"\n🧠 Model Prediction: {label}")
print("\n✅ Testing complete.")
