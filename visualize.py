"""Save dark-themed PNG charts into the outputs folder (used by main.py)."""
import matplotlib
matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt

from src import config

C = config.COLORS


def _style(ax):
    ax.set_facecolor(C["card"])
    for spine in ax.spines.values():
        spine.set_color(C["border"])
    ax.tick_params(colors=C["muted"])
    ax.title.set_color(C["text"])


def _save(fig, name):
    path = config.OUTPUT_DIR / name
    fig.savefig(path, dpi=140, facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)
    return path


def save_confusion_matrix(metrics):
    cm = metrics["best"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.4, 4))
    fig.patch.set_facecolor(C["bg"])
    ax.imshow(cm, cmap="BuPu")
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("Predicted", color=C["muted"])
    ax.set_ylabel("Actual", color=C["muted"])
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "#0D1121")
    _style(ax)
    return _save(fig, "confusion_matrix.png")


def save_class_distribution(metrics):
    fig, ax = plt.subplots(figsize=(4.4, 4))
    fig.patch.set_facecolor(C["bg"])
    ax.pie([metrics["ham_count"], metrics["spam_count"]], labels=["Legitimate", "Spam"],
           colors=[C["cyan"], C["magenta"]], autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(width=0.4, edgecolor=C["bg"]),
           textprops=dict(color=C["text"]))
    ax.set_title("Dataset: Spam vs Legitimate", color=C["text"])
    return _save(fig, "class_distribution.png")


def save_top_words(metrics):
    spam = metrics["top_words"]["spam"][:8][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(C["bg"])
    ax.barh([w for w, _ in spam], [s for _, s in spam], color=C["purple"])
    ax.set_title("Top words that signal spam")
    _style(ax)
    return _save(fig, "top_spam_words.png")


def generate_all_charts(metrics):
    config.ensure_directories()
    return [save_confusion_matrix(metrics), save_class_distribution(metrics), save_top_words(metrics)]
