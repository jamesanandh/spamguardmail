# SpamGuard - Spam Email Prediction

A machine-learning project that classifies emails as **spam** or **legitimate (ham)**, with a dark, modern
Streamlit dashboard (navy background, purple / magenta / cyan gradients).

## Abstract
Spam emails waste time and are often used for phishing and fraud. This project trains a text classifier
(TF-IDF + Naive Bayes / Logistic Regression) on a labelled email dataset, compares the two models, keeps the
better one, and exposes it through a dashboard where users can check one email, scan a whole inbox file, and
review the model's performance.

## Problem Statement
Given the subject and body of an email, predict whether it is spam, and explain which words influenced the decision.

## Objectives
- Build a working spam classifier with measurable accuracy, precision, recall and F1 score.
- Provide an easy dashboard for single and batch predictions.
- Explain predictions (key words and warning signs) instead of giving a black-box answer.
- Allow retraining with a new or custom dataset.

## Existing System vs Proposed System
| Existing (rule-based filters) | Proposed (this project) |
|---|---|
| Fixed keyword lists written by hand | Learns patterns from labelled examples |
| Easy for spammers to bypass | Can be retrained when spam changes |
| No explanation or score | Gives a probability and the influencing words |

## Technologies Used
Python, pandas, NumPy, scikit-learn, joblib, matplotlib, Plotly, Streamlit.

## Features and Modules
| File | Purpose |
|---|---|
| `main.py` | Command-line entry: creates data, trains, evaluates, saves charts, predicts one text |
| `app.py` | Streamlit dashboard (Overview, Check Email, Batch Scan, Model Report) |
| `src/config.py` | Paths and colour theme |
| `src/data_generator.py` | Builds the synthetic dataset and sample inbox |
| `src/data_loader.py` | Loads and validates CSV files |
| `src/preprocess.py` | Text cleaning and warning-sign detection |
| `src/train_model.py` | Trains and compares models, saves the best one and its metrics |
| `src/predict.py` | Single/batch predictions and word explanations |
| `src/visualize.py` | Saves PNG charts to `outputs/` |

## Folder Structure
```
spam-email-prediction/
├── .streamlit/config.toml   # dark theme settings
├── data/                    # emails.csv, sample_inbox.csv
├── models/                  # spam_model.joblib (created when training)
├── outputs/                 # metrics.json, charts, batch results
├── src/                     # Python modules
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── GUIDE.md                 # beginner step-by-step guide + viva questions
```

## Installation (Windows 11 + VS Code)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Execution Steps
```
python main.py            # trains the model and saves charts
streamlit run app.py      # opens the dashboard in your browser
```
(The dashboard also trains the model automatically on first launch if you skip `main.py`.)

## Sample Output
```
===== TRAINING SUMMARY =====
Emails used     : 1200 (spam 479, legitimate 721)
Naive Bayes         : accuracy 0.988 | F1 0.984
Logistic Regression : accuracy 0.988 | F1 0.984
Best model      : Naive Bayes
```
Your numbers may differ slightly depending on library versions.

## Limitations
- The built-in dataset is **synthetic** (template-based), so scores are higher than real-world performance.
  Train on a real dataset (use the Retrain section) for realistic results.
- Only English text is supported; the model uses words only (no sender, headers or attachments).
- Spammers change tactics, so the model needs retraining over time.

## Future Enhancements
- Use a real public dataset (for example the Enron or SpamAssassin corpora).
- Add sender/header features and a URL-reputation check.
- Try deep-learning models and multi-language support.
- Connect to a real mailbox (IMAP) for automatic scanning.
