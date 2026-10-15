"""Central place for folder paths and colours.

pathlib.Path is used so the paths work on Windows, macOS and Linux.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "emails.csv"
SAMPLE_INBOX_PATH = DATA_DIR / "sample_inbox.csv"
MODEL_PATH = MODEL_DIR / "spam_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
BATCH_RESULTS_PATH = OUTPUT_DIR / "last_batch_results.csv"

# Colour theme (taken from the uploaded CRM dashboard design)
COLORS = {
    "bg": "#0D1121",        # page background
    "sidebar": "#10152A",   # sidebar background
    "card": "#161C33",      # panels / cards
    "border": "#1F2747",
    "text": "#E8EAF6",
    "muted": "#8B93B5",
    "purple": "#D86BFF",    # primary gradient start
    "violet": "#8E6BF2",    # primary gradient end
    "cyan": "#22E8D2",      # secondary gradient start
    "blue": "#4A74F0",      # secondary gradient end
    "magenta": "#D81BC8",   # spam / danger accent
    "sky": "#6EC8FF",       # line chart colour
}


def ensure_directories():
    """Create the data / outputs / models folders if they do not exist."""
    for folder in (DATA_DIR, OUTPUT_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)
