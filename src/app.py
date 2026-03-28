"""
Spam Detector — Streamlit App
==============================
Requirements:
    pip install streamlit joblib scikit-learn nltk shap matplotlib pandas numpy

Run:
    streamlit run app.py
"""

import re
import string
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# ── NLTK setup ────────────────────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str = "models/spam_classifier_pipeline.pkl"):
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        model = joblib.load(path)
    # Re-save so next load uses the current sklearn version (no more warning)
    joblib.dump(model, path)
    return model


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


def get_confidence(model, cleaned: str):
    """Returns (spam_probability, ham_probability)."""
    try:
        proba = model.predict_proba([cleaned])[0]
        return proba[1], proba[0]
    except Exception:
        try:
            score = float(np.ravel(model.decision_function([cleaned]))[0])
            spam_p = 1 / (1 + np.exp(-score))
            return spam_p, 1 - spam_p
        except Exception:
            return 0.5, 0.5


def compute_shap(model, cleaned: str):
    """Returns (shap_values, feature_names) or (None, None) on failure."""
    try:
        vectorizer = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
    except Exception:
        return None, None

    X_arr = vectorizer.transform([cleaned]).toarray()
    background = np.zeros((1, X_arr.shape[1]))
    feat_names = vectorizer.get_feature_names_out()

    try:
        # ✅ Fixed: use maskers.Independent + suppress residual FutureWarning
        masker = shap.maskers.Independent(background)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            explainer = shap.LinearExplainer(clf, masker)
            shap_vals = explainer.shap_values(X_arr)
    except Exception:
        try:
            explainer = shap.Explainer(clf, background)
            out = explainer(X_arr)
            shap_vals = out.values
        except Exception:
            return None, None

    # Handle binary output (list of 2 arrays)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = np.asarray(shap_vals[1])

    return np.asarray(shap_vals).ravel(), feat_names


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("📩 Spam Message Detector")
st.caption("Paste a message below and click **Predict** to classify it.")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
message = st.text_area(
    "Message", height=180, placeholder="Type or paste your message here…"
)
predict_btn = st.button("Predict", type="primary", width="stretch")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not message.strip():
        st.warning("Please enter a message first.")
        st.stop()

    try:
        model = load_model()
    except FileNotFoundError:
        st.error(
            "Model not found at `models/spam_classifier_pipeline.pkl`. "
            "Please train and save your model first."
        )
        st.stop()

    cleaned = clean_text(message)
    pred = model.predict([cleaned])[0]
    spam_p, ham_p = get_confidence(model, cleaned)
    is_spam = pred == 1

    st.divider()

    # ── Result ────────────────────────────────────────────────────────────────
    if is_spam:
        st.error(f"🚨 **SPAM** — {spam_p * 100:.1f}% confidence")
    else:
        st.success(f"✅ **HAM (Not Spam)** — {ham_p * 100:.1f}% confidence")

    # ── Confidence bars ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Spam probability", f"{spam_p * 100:.1f}%")
        st.progress(float(spam_p))
    with col2:
        st.metric("Ham probability", f"{ham_p * 100:.1f}%")
        st.progress(float(ham_p))

    st.divider()

    # ── SHAP explanation ──────────────────────────────────────────────────────
    st.subheader("🔎 What influenced this prediction?")

    shap_vals, feat_names = compute_shap(model, cleaned)

    if shap_vals is not None:
        nonzero = np.where(shap_vals != 0)[0]

        if len(nonzero) == 0:
            st.info("No recognisable features found after vectorisation.")
        else:
            top_n = min(15, len(nonzero))
            top_idx = nonzero[np.argsort(np.abs(shap_vals[nonzero]))[::-1][:top_n]]
            vals = shap_vals[top_idx]
            names = feat_names[top_idx]

            # Horizontal bar chart
            order = np.argsort(vals)
            colors = ["#ff4d6d" if v > 0 else "#22c55e" for v in vals[order]]

            fig, ax = plt.subplots(figsize=(8, max(3, top_n * 0.4)))
            ax.barh(names[order], vals[order], color=colors, height=0.6)
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.set_xlabel("SHAP value  ( + → spam,  − → ham )", fontsize=9)
            ax.tick_params(labelsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Table
            with st.expander("See full SHAP table"):
                df_shap = (
                    pd.DataFrame(
                        {
                            "Token": names,
                            "SHAP": np.round(vals, 4),
                            "Signal": ["→ Spam" if v > 0 else "→ Ham" for v in vals],
                        }
                    )
                    .sort_values("SHAP", ascending=False)
                    .reset_index(drop=True)
                )
                # ✅ Fixed: replaced use_container_width with width
                st.dataframe(df_shap, width=600, hide_index=True)
    else:
        st.warning(
            "SHAP explanation unavailable. Make sure your pipeline has "
            "`tfidf` and `clf` named steps."
        )

    # ── Cleaned text ──────────────────────────────────────────────────────────
    with st.expander("See preprocessed text"):
        st.code(cleaned or "[empty after cleaning]")
