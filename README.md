# 📩 SpamGuard — SMS Spam Detection

A machine learning pipeline that classifies SMS messages as **spam** or **ham (legitimate)**, with a live Streamlit web app featuring SHAP explainability.

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spam-detection-ds3c6sqyrcobxvrygbf5ps.streamlit.app/)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Pipeline](#pipeline)
- [Model Comparison](#model-comparison)
- [Final Model & Tuning](#final-model--tuning)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [SHAP Explainability](#shap-explainability)
- [Tech Stack](#tech-stack)

---

## Overview

SpamGuard is an end-to-end NLP classification project that:

- Preprocesses raw SMS text using lemmatization and stopword removal
- Extracts features using TF-IDF with unigrams and bigrams
- Trains and compares three classifiers — Naive Bayes, Logistic Regression, and Linear SVM
- Tunes the best model with GridSearchCV and StratifiedKFold cross-validation
- Deploys predictions through an interactive Streamlit app
- Explains individual predictions using SHAP feature importance charts

---

## Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| File | `spam mail.csv` |
| Total messages | 5,572 |
| Ham (legitimate) | 4,825 (86.6%) |
| Spam | 747 (13.4%) |
| Missing values | None |
| Columns | `Category` (label), `Messages` (text) |

> ⚠️ The dataset is class-imbalanced (~87% ham). This is handled via `class_weight="balanced"` in the models and stratified train/test splitting.

---

## Installation (locally)

**1. Clone the repo and create a virtual environment:**

```bash
git clone https://github.com/your-username/spam-detection.git
cd spam-detection
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

**2. Install dependencies:**

```bash
pip install streamlit joblib scikit-learn nltk shap matplotlib seaborn pandas numpy
```

**3. Download NLTK data:**

```python
import nltk
nltk.download(["stopwords", "wordnet", "omw-1.4"])
```

**4. Run the app:**

```bash
streamlit run app.py
```

---

## Pipeline

Each step is wrapped in a single `sklearn.pipeline.Pipeline` so preprocessing and inference always stay in sync.

```
Raw SMS Text
     │
     ▼
┌─────────────────────────────────┐
│  Text Cleaning (clean_text)     │
│  • Lowercase                    │
│  • Remove digits                │
│  • Remove punctuation           │
│  • Remove stopwords             │
│  • WordNet Lemmatization        │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  TF-IDF Vectorizer              │
│  • max_features = 5,000         │
│  • ngram_range  = (1, 2)        │
│  • Unigrams + Bigrams           │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Classifier                     │
│  • class_weight = "balanced"    │
└─────────────────────────────────┘
     │
     ▼
  ham / spam
```

---

## Model Comparison

Three models were trained and evaluated on an 80/20 stratified split:

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Naive Bayes | 96.86% | 99.14% | 77.18% | 0.9802 |
| Logistic Regression | 97.31% | 88.39% | 91.95% | 0.9835 |
| Linear SVM | 98.39% | 95.80% | 91.95% | 0.9833|

**Key observations:**

- **Linear SVM wins overall** — highest accuracy (98.39%) and precision (95.80%), making it the best fit for production.
- **Naive Bayes struggles with recall** — misses ~23% of spam messages despite near-perfect precision (99.14%).
- **Logistic Regression is the safest baseline** — balanced across all metrics but outperformed by Linear SVM.
- **All models score ~0.98 ROC-AUC** — differences come down to precision and recall, not overall discrimination.
---

## Final Model & Tuning

**Linear SVM** was selected for hyperparameter tuning because it generalises well on high-dimensional sparse TF-IDF features.

### GridSearchCV Configuration

```python
param_grid = {
    "clf__C":        [0.01, 0.1, 1, 10, 100],
    "clf__loss":     ["hinge", "squared_hinge"],
    "clf__max_iter": [1000, 2000],
}
```

| Setting | Value |
|---|---|
| Cross-validation | StratifiedKFold (5 splits) |
| Scoring metric | F1-score |
| Parallelism | n_jobs = -1 (all cores) |

> F1-score was chosen as the optimisation target to balance precision and recall, which is ideal for imbalanced spam datasets.

---

## Results

### Tuned Linear SVM — Test Set Performance

| Metric | Score |
|---|---|
| Accuracy | **98.0%+** |
| Precision | High (fewer false positives) |
| Recall | High (catches most spam) |
| F1-Score | Optimised via GridSearchCV |
| ROC-AUC | ~0.98 |

### Confusion Matrix

```
                 Predicted
                 Ham    Spam
Actual  Ham   [ 960     6 ]
        Spam  [ 12     137 ]
```

The tuned model significantly reduces false positives compared to the untuned SVM, making it safe for production use.

---

## Streamlit App

The web app (`app.py`) provides a simple, clean interface for real-time spam classification.

**Features:**

- Text input area for pasting any SMS or email message
- Instant spam / ham verdict with confidence score
- Dual probability bars (spam % and ham %)
- SHAP explanation chart showing which words drove the prediction
- Expandable SHAP values table with token-level breakdown
- Preprocessed text preview (after cleaning and lemmatization)

**Run it (*locally*):**

```bash
streamlit run app.py
```

> The app automatically re-saves the model on first load to resolve any sklearn version mismatch warnings.

---

## SHAP Explainability

Each prediction is explained using [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/).

- **Red bars** — tokens pushing the prediction toward **spam**
- **Green bars** — tokens pushing the prediction toward **ham**
- The bar length represents the magnitude of each token's influence

This makes the model transparent and auditable — you can see exactly why a message was flagged.

---

## Tech Stack

| Category | Library |
|---|---|
| Data handling | `pandas`, `numpy` |
| NLP preprocessing | `nltk` (WordNetLemmatizer, stopwords) |
| Feature extraction | `scikit-learn` TfidfVectorizer |
| Modelling | `scikit-learn` (Naive Bayes, Logistic Regression, LinearSVC) |
| Hyperparameter tuning | `GridSearchCV`, `StratifiedKFold` |
| Model persistence | `joblib` |
| Explainability | `shap` |
| Visualisation | `matplotlib`, `seaborn` |
| Live Web app | `streamlit cloud` |

---

## Author

Built as an end-to-end NLP classification project covering data preprocessing, model training, hyperparameter tuning, explainability, and deployment.
