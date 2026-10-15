"""Loading and validating CSV files (training data and inbox files)."""
import pandas as pd

from src import config
from src.preprocess import combine_text

LABEL_MAP = {
    "spam": "spam", "1": "spam", "1.0": "spam", "junk": "spam",
    "ham": "ham", "0": "ham", "0.0": "ham", "not spam": "ham", "legitimate": "ham",
}


def _normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "body" not in df.columns and "text" in df.columns:
        df["body"] = df["text"]
    if "body" not in df.columns:
        raise ValueError("The file needs a 'body' (or 'text') column containing the email text.")
    if "subject" not in df.columns:
        df["subject"] = ""
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["text"] = [combine_text(s, b) for s, b in zip(df["subject"], df["body"])]
    return df[df["text"].str.len() > 0]


def prepare_dataframe(raw):
    """Validate a *training* dataset. Requires a 'label' column."""
    df = _normalise_columns(raw)
    if "label" not in df.columns:
        raise ValueError("The file needs a 'label' column with the values spam / ham.")
    mapped = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    if mapped.isna().any():
        bad = df.loc[mapped.isna(), "label"].astype(str).unique()[:5]
        raise ValueError(f"Unknown label values: {list(bad)}. Use spam/ham (or 1/0).")
    df["label"] = mapped
    counts = df["label"].value_counts()
    if len(df) < 20 or counts.min() < 5 or len(counts) < 2:
        raise ValueError("Need at least 20 emails with 5 or more examples of both spam and ham.")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.reset_index(drop=True)


def load_dataset(path=config.DATASET_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run 'python main.py' first.")
    return prepare_dataframe(pd.read_csv(path, encoding="utf-8"))


def prepare_inbox(raw):
    """Validate an *inbox* file for scanning. The 'label' column is optional."""
    df = _normalise_columns(raw)
    if df.empty:
        raise ValueError("The file contains no email text.")
    if "label" in df.columns:
        df["true_label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    return df.reset_index(drop=True)
