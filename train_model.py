"""Train, compare and save the spam classifier."""
import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import load_dataset
from src.preprocess import clean_text


def build_pipeline(classifier):
    """TF-IDF turns text into numbers, then the classifier learns from them."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                  min_df=2, stop_words="english")),
        ("clf", classifier),
    ])


def spam_weights(classifier):
    """Higher number = word pushes an email towards 'spam'."""
    if hasattr(classifier, "coef_"):                       # Logistic Regression
        return classifier.coef_[0]
    return classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]   # Naive Bayes


def top_words(pipeline, n=10):
    names = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    scores = spam_weights(pipeline.named_steps["clf"])
    order = np.argsort(scores)
    return {
        "spam": [[str(names[i]), float(scores[i])] for i in order[::-1][:n]],
        "ham": [[str(names[i]), float(scores[i])] for i in order[:n]],
    }


def _evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, pos_label="spam", zero_division=0)),
        "recall": float(recall_score(y_test, pred, pos_label="spam", zero_division=0)),
        "f1": float(f1_score(y_test, pred, pos_label="spam", zero_division=0)),
        # rows = actual [ham, spam], columns = predicted [ham, spam]
        "confusion_matrix": confusion_matrix(y_test, pred, labels=["ham", "spam"]).tolist(),
    }


def train(path=config.DATASET_PATH):
    """Train two models, keep the better one (by F1) and save everything."""
    config.ensure_directories()
    df = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

    candidates = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5),
    }
    results, fitted = {}, {}
    for name, clf in candidates.items():
        model = build_pipeline(clf).fit(X_train, y_train)
        fitted[name] = model
        results[name] = _evaluate(model, X_test, y_test)

    best_name = max(results, key=lambda k: (results[k]["f1"], results[k]["accuracy"]))
    best_model = fitted[best_name]

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "best_model": best_name,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "spam_count": int((df["label"] == "spam").sum()),
        "ham_count": int((df["label"] == "ham").sum()),
        "models": results,
        "best": results[best_name],
        "top_words": top_words(best_model),
    }
    joblib.dump(best_model, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_metrics():
    if not config.METRICS_PATH.exists():
        return None
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
