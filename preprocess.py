"""Text cleaning and simple 'spam signal' detection."""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MONEY_RE = re.compile(r"[$\u20b9\u20ac\u00a3]\s?\d[\d,\.]*")
NUMBER_RE = re.compile(r"\d+")

URGENT_WORDS = [
    "urgent", "immediately", "final notice", "act now", "expires", "suspended",
    "verify", "limited time", "hurry", "winner", "claim", "free",
    "guaranteed", "no prescription", "click here", "congratulations",
]


def combine_text(subject, body):
    """Join subject and body into one string for the model."""
    return f"{subject or ''} {body or ''}".strip()


def clean_text(text):
    """Lower-case the text and replace links, money and numbers with tokens.

    Spam emails often contain links and money, so we keep that information
    as the words 'urltoken', 'moneytoken' and 'numtoken'.
    """
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_signals(subject, body):
    """Return easy-to-understand warning signs found in an email."""
    text = combine_text(subject, body)
    lower = text.lower()
    letters = [c for c in text if c.isalpha()]
    caps_ratio = (sum(c.isupper() for c in letters) / len(letters) * 100) if letters else 0.0
    return {
        "links": len(URL_RE.findall(text)),
        "exclamations": text.count("!"),
        "caps_percent": round(caps_ratio, 1),
        "money_mentions": len(MONEY_RE.findall(text)),
        "urgent_words": sorted({w for w in URGENT_WORDS if w in lower}),
    }
