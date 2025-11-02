# test.py
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


def get_topic_keywords(lda_model, tfidf_vectorizer, topic_index, n_top_words=10):
    # 1. Get the list of all words (the vocabulary)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # 2. Get the word probability distribution for the specific topic index
    topic_weights = lda_model.components_[topic_index]

    # 3. Find the indices of the top N words (sorted descending)
    top_word_indices = topic_weights.argsort()[:-n_top_words - 1:-1]

    # 4. Map the indices to the actual words
    topic_keywords = [feature_names[i] for i in top_word_indices]

    # 5. Return the keywords as a human-readable string
    return f"Topic {topic_index}: {' / '.join(topic_keywords)}"


# Load trained models
print("🔹 Loading saved models...")
tfidf_vectorizer = joblib.load("outputs/tfidf_vectorizer.pkl")
lda_model = joblib.load("outputs/lda_model.pkl")
isolation_forest = joblib.load("outputs/isolation_forest.pkl")


# Prepare new article
new_article = """
USA bombs iran
"""


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


# Feature Extraction
print("🔹 Extracting features...")
tfidf_features = tfidf_vectorizer.transform([cleaned_article])

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode([cleaned_article])


# Topic Distribution
topic_distribution = lda_model.transform(tfidf_features)
top_topic_index = np.argmax(topic_distribution)

topic_name = get_topic_keywords(lda_model, tfidf_vectorizer, top_topic_index)
print(f"\n🧩 Dominant Topic: {topic_name}")


# Fake/Real Prediction using Isolation Forest
prediction = isolation_forest.predict(embedding)[0]
label = "REAL (Legitimate)" if prediction == 1 else "FAKE (Anomalous)"

print(f"\n Model Prediction: {label}")
print("\n Testing complete.")
