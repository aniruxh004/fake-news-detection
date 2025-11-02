# 📰 Hybrid Fake News Detection System

## 🌟 Project Overview

This project implements an advanced **Semi-Supervised Machine Learning (ML)** framework to classify news articles as **REAL** or **FAKE**. Unlike traditional supervised models, our hybrid approach is designed to be highly effective at detecting **novel (new)** and constantly evolving forms of misinformation by focusing on anomalies in the text's semantic structure.

The system combines two distinct methods: **Anomaly Detection** for prediction and **Topic Modeling** for analysis, making the results robust and highly explainable.

---

## 🔬 Methodology: The Hybrid Approach

Our detection system relies on two powerful models working together on different feature representations of the text.

### 1. Feature Representation

We use advanced Natural Language Processing (NLP) to convert text into two numerical formats:

* **Sentence Embeddings (Style/Meaning):** Created using a Sentence Transformer (like `all-MiniLM-L6-v2`). This transforms the entire article into a vector (a list of numbers) that represents its deep semantic meaning. Articles with similar meaning (even if they use different words) cluster together in the vector space.
* **TF-IDF Features (Keywords/Content):** Traditional feature scores used for topic analysis.

### 2. The Core Models

| Model | Type | Input Data | Role in Detection |
| :--- | :--- | :--- | :--- |
| **Isolation Forest** | **Semi-Supervised** (Anomaly Detector) | Sentence Embeddings | **Primary Classifier.** It is trained to identify the "boundaries of normal news." Any new article that falls outside these boundaries (i.e., is structurally or stylistically bizarre) is flagged as an **Anomaly (FAKE)**. |
| **LDA (Latent Dirichlet Allocation)** | **Unsupervised** (Topic Model) | TF-IDF Features | **Analyzer.** It automatically discovers the primary topics (e.g., 'Economy', 'US Politics') in the dataset. This provides **explainability** by showing what subject the flagged anomaly belongs to. |

---

## 🛠️ Project Structure and Setup

### Prerequisites

You need Python 3.8+ and the following libraries installed:

```bash
pip install pandas numpy scikit-learn sentence-transformers nltk joblib

File/Folder,Description
main.py,The master script that controls the entire training workflow (Data Prep -> Feature Extraction -> Model Training -> Evaluation).
test.py,"A script demonstrating how to load the saved models and make a prediction on a single, new news article."
scripts/,Contains the modular Python functions.
scripts/data_preprocessing.py,"Functions for cleaning text, removing stop words, and applying Lemmatization."
scripts/feature_extraction.py,Contains functions to generate Sentence Embeddings and TF-IDF Features.
scripts/semi_supervised_model.py,Contains the logic for initializing and training the Isolation Forest and LDA models.
dataset/,(Placeholder) Directory where your True.csv and Fake.csv dataset files should reside.
outputs/,Directory where trained models (.pkl files) and visualization plots are saved.



1.Clone the Repository:
git clone [your_repo_url]
cd fake-news-detection

2 Place your True.csv and Fake.csv files into the dataset/ folder. #already in repo

3.Train: 
python main.py  #no need already in repo 

4.Test:
python test.py
























