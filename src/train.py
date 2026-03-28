# Import Libraries and Modules
import pandas as pd
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
df = pd.read_csv("spam mail.csv")
df.head()

# Rename the column
df.rename(columns={"Masseges:": "Messages"}, inplace=True)
df.columns = ["Category", "Messages"]

# Label each category
df["label_num"] = df["Category"].map({"ham": 0, "spam": 1})

# In-depth analysis of the dataset
print(df.shape)
print(df["Category"].value_counts())
print(df.isnull().sum())  # confirm no nulls
df.head()

# Text Preprocessing
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Function to clean the text data
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # Remove punctuation
    tokens = text.split()  # Tokenize the text
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w not in stop_words
    ]  # Remove stop words and apply lemmatization
    return " ".join(tokens)


# Apply the cleaning function to the messages
df["clean_message"] = df["Messages"].apply(clean_text)
df[["Messages", "clean_message"]].head(3)

# Add a new column for message length
df["msg_length"] = df["clean_message"].apply(len)

# Prepare the data for modeling
X = df["clean_message"]
y = df["label_num"]

# Split the data into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train multiple models using Pipeline (TF-IDF + Model together)
models = {
    "Naive Bayes": Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", MultinomialNB()),
        ]
    ),
    "Logistic Regression": Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    ),
    "Linear SVM": Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LinearSVC(class_weight="balanced", max_iter=1000)),
        ]
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train_raw, y_train)  # Fit pipeline directly

    # Predictions
    y_pred = model.predict(X_test_raw)

    # Handle probability / decision scores
    if hasattr(model.named_steps["clf"], "predict_proba"):
        y_proba = model.predict_proba(X_test_raw)[:, 1]
    else:
        y_proba = model.decision_function(X_test_raw)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Store results
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }

# Display results
for model_name, metrics in results.items():
    print("---" * 10)
    print(f"{model_name}:")
    print("---" * 10)
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# Confusion Matrix and Classification Report for the models
for name, model in models.items():
    y_pred = model.predict(X_test_raw)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Perform Hyperparameter Tuning for Linear SVM
param_grid_svm = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__loss": ["hinge", "squared_hinge"],
    "clf__max_iter": [1000, 2000],
}

# Use Pipeline to ensure that TF-IDF vectorization is included in the grid search
svm_pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LinearSVC(class_weight="balanced")),
    ]
)

# Define GridSearchCV with StratifiedKFold
grid_search_svm = GridSearchCV(
    svm_pipeline,
    param_grid_svm,
    cv=StratifiedKFold(
        n_splits=5, shuffle=True, random_state=28
    ),  # Use StratifiedKFold for better class balance in CV
    scoring="f1",  # Optimize for F1-score to balance precision and recall
    n_jobs=-1,
    verbose=1,
)

grid_search_svm.fit(X_train_raw, y_train)

# Best model info
best_svm = grid_search_svm.best_estimator_
print(f"\nBest SVM Params: {grid_search_svm.best_params_}")
print(f"Best CV F1-Score: {grid_search_svm.best_score_:.4f}")

# Predictions
y_pred_svm = best_svm.predict(X_test_raw)
y_score_svm = best_svm.decision_function(X_test_raw)

# Metrics
print("\nTuned Linear SVM Performance:")
print(f"Accuracy : {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_svm):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_svm):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_score_svm):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=["ham", "spam"]))

# Plot confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(
    best_svm,
    X_test_raw,
    y_test,
)

plt.title("Confusion Matrix - Tuned Linear SVM")
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, auc

# Get decision scores (since LinearSVC doesn't have predict_proba)
y_score = best_svm.decision_function(X_test_raw)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure()
plt.plot(fpr, tpr, label=f"Linear SVM (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Linear SVM")
plt.legend(loc="lower right")

plt.show()

# Save the final tuned model (Pipeline)
joblib.dump(best_svm, "spam_classifier_pipeline.pkl")

print("Model saved successfully!")
