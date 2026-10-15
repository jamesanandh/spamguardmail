"""Command-line entry point.

    python main.py                      -> create data (if missing), train, evaluate, save charts
    python main.py --regenerate         -> create a brand new dataset first
    python main.py --predict "text"     -> classify one email text
"""
import argparse

from src import config
from src.data_generator import generate_dataset, generate_sample_inbox, save_dataset
from src.predict import explain, load_model, predict_one
from src.train_model import train
from src.visualize import generate_all_charts


def prepare_data(regenerate=False, samples=1200):
    config.ensure_directories()
    if regenerate or not config.DATASET_PATH.exists():
        save_dataset(generate_dataset(n=samples))
        print(f"[OK] Dataset created: {config.DATASET_PATH}")
    if regenerate or not config.SAMPLE_INBOX_PATH.exists():
        generate_sample_inbox().to_csv(config.SAMPLE_INBOX_PATH, index=False, encoding="utf-8")
        print(f"[OK] Sample inbox created: {config.SAMPLE_INBOX_PATH}")


def run_training():
    metrics = train()
    charts = generate_all_charts(metrics)
    best = metrics["best"]
    print("\n===== TRAINING SUMMARY =====")
    print(f"Emails used     : {metrics['n_samples']} "
          f"(spam {metrics['spam_count']}, legitimate {metrics['ham_count']})")
    for name, m in metrics["models"].items():
        print(f"{name:<20}: accuracy {m['accuracy']:.3f} | F1 {m['f1']:.3f}")
    print(f"Best model      : {metrics['best_model']}")
    print(f"Precision/Recall: {best['precision']:.3f} / {best['recall']:.3f}")
    print(f"Model saved     : {config.MODEL_PATH}")
    print(f"Charts saved    : {', '.join(p.name for p in charts)} (in the outputs folder)")
    print("\nNext step: streamlit run app.py")


def run_prediction(text):
    if not text.strip():
        print("[ERROR] Please provide some email text.")
        return
    model = load_model()
    result = predict_one(model, "", text)
    print(f"\nPrediction : {result['label'].upper()}")
    print(f"Spam score : {result['spam_probability'] * 100:.1f}%")
    print("Key words  :", ", ".join(f"{w} ({s:+.2f})" for w, s in explain(model, '', text, 5)))


def main():
    parser = argparse.ArgumentParser(description="Spam email prediction")
    parser.add_argument("--regenerate", action="store_true", help="create a new dataset")
    parser.add_argument("--samples", type=int, default=1200, help="emails to generate")
    parser.add_argument("--predict", type=str, help="classify one email text")
    args = parser.parse_args()

    if args.predict is not None:
        try:
            run_prediction(args.predict)
        except FileNotFoundError as error:
            print(f"[ERROR] {error}")
        return
    prepare_data(args.regenerate, args.samples)
    run_training()


if __name__ == "__main__":
    main()
