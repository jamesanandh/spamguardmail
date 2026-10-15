"""Use the saved model to classify new emails."""
import joblib

from src import config
from src.preprocess import combine_text
from src.train_model import spam_weights


def load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run 'python main.py' first to train it.")
    return joblib.load(config.MODEL_PATH)


def _spam_index(model):
    return list(model.classes_).index("spam")


def predict_one(model, subject, body, threshold=0.5):
    """Return the label and spam probability (0 to 1) for one email."""
    text = combine_text(subject, body)
    prob = float(model.predict_proba([text])[0][_spam_index(model)])
    return {"label": "spam" if prob >= threshold else "ham", "spam_probability": prob}


def predict_dataframe(model, df, threshold=0.5):
    """Add 'spam_probability' and 'prediction' columns to a prepared inbox DataFrame."""
    out = df.copy()
    probs = model.predict_proba(out["text"])[:, _spam_index(model)]
    out["spam_probability"] = probs
    out["prediction"] = ["spam" if p >= threshold else "ham" for p in probs]
    return out


def explain(model, subject, body, top_n=8):
    """Words that influenced the decision most. Positive = towards spam."""
    tfidf = model.named_steps["tfidf"]
    weights = spam_weights(model.named_steps["clf"])
    names = tfidf.get_feature_names_out()
    row = tfidf.transform([combine_text(subject, body)]).tocoo()
    contributions = [(str(names[j]), float(weights[j] * v)) for j, v in zip(row.col, row.data)]
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return contributions[:top_n]
