# Beginner Guide - Run, Test and Explain the Project

## 1. Install the software (one time)
1. **Python 3.10 - 3.12** from https://www.python.org/downloads/ - tick **"Add Python to PATH"** during install.
2. **VS Code** from https://code.visualstudio.com/ and install its **Python** extension.
3. Check in a terminal: `python --version` (should print a version number).

## 2. Open the project
1. Unzip `spam-email-prediction.zip` (right-click > Extract All).
2. In VS Code: **File > Open Folder** and choose the `spam-email-prediction` folder.
3. Open a terminal: **Terminal > New Terminal**.

## 3. Create a virtual environment (a private box for this project's packages)
```
python -m venv venv
venv\Scripts\activate
```
If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
You should see `(venv)` at the start of the terminal line. In VS Code choose this interpreter when asked.

## 4. Install packages
```
pip install -r requirements.txt
```

## 5. Train the model
```
python main.py
```
Expected: a training summary (accuracy around 0.95 - 1.00), and new files in `models/` and `outputs/`.

## 6. Start the dashboard
```
streamlit run app.py
```
A browser tab opens at http://localhost:8501. Press **Ctrl + C** in the terminal to stop it.

## 7. Test every feature
| Feature | How to test | Expected result |
|---|---|---|
| Overview | Open "Overview" | Accuracy/F1 cards, monthly line chart, donuts, bar charts |
| Spam check | "Check Email" > "Spam example" > Analyze | Magenta "SPAM DETECTED", high gauge value |
| Safe check | "Safe example" > Analyze | Cyan "LOOKS SAFE", low gauge value |
| Validation | Clear both boxes > Analyze | Warning asking for text |
| Threshold | Move the slider and Analyze again | Verdict can flip near the borderline |
| Batch scan | "Batch Scan" > "Use the sample inbox" | Table of 16 emails, accuracy card, CSV download |
| Bad file | Upload a CSV without a body/text column | Red error message |
| Model report | "Model Report" | Model comparison, confusion matrix, top words |
| Retrain | "Generate new dataset and retrain" | Success message, updated numbers |
| CLI predict | `python main.py --predict "You won $5000, click here"` | SPAM with a score |

## 8. Using a real dataset
Use any CSV with columns `subject` and `body` (or just `text`) plus `label` (`spam`/`ham` or `1`/`0`).
Upload it in **Model Report > train on your own CSV**. The old dataset is backed up to `data/emails_backup.csv`.

## 9. Common errors and fixes
| Error | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick "Add to PATH", then reopen VS Code |
| `ModuleNotFoundError: No module named 'streamlit'` | Activate the venv, then `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project's root folder (where `app.py` is) |
| Activation script blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Model not found | Run `python main.py` once |
| Page looks unstyled | Make sure the `.streamlit` folder (hidden name starting with a dot) was extracted |

## 10. How to explain it in a viva / interview
**One-minute summary:** "I built a spam detector. Emails are converted to numbers with TF-IDF, then Naive Bayes and
Logistic Regression are trained and compared; the better one by F1 score is saved. A Streamlit dashboard lets users check
one email, scan a CSV inbox, and see model performance."

**Key terms (simple English)**
- **TF-IDF:** gives higher weight to words that are special to an email, and lower weight to words everyone uses.
- **Naive Bayes:** a fast model that multiplies word probabilities per class.
- **Logistic Regression:** a model that draws a boundary between spam and ham and outputs a probability.
- **Train/test split:** 80% of data teaches the model, 20% is hidden and used only to measure it.
- **Precision:** of emails marked spam, how many truly were. **Recall:** of all spam, how many were caught.
- **F1:** a single score balancing precision and recall. **Confusion matrix:** table of correct vs wrong predictions.

**Likely questions**
1. *Why F1 and not only accuracy?* If spam is rarer than normal mail, accuracy can look high even when spam is missed.
2. *Why is accuracy so high?* The built-in data is synthetic; real emails are messier. Be honest about this limitation.
3. *What is a false positive here?* A good email wrongly marked spam - worse for users than a missed spam; raise the threshold to reduce it.
4. *How would you improve it?* Real datasets, sender/header features, URL checks, deep learning.
