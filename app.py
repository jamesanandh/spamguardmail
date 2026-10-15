"""SpamGuard - Streamlit dashboard (dark CRM-style theme).

Run with:  streamlit run app.py
"""
import json
import shutil

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_generator import generate_dataset, generate_sample_inbox, save_dataset
from src.data_loader import load_dataset, prepare_dataframe, prepare_inbox
from src.predict import explain, load_model, predict_dataframe, predict_one
from src.preprocess import extract_signals
from src.train_model import load_metrics, train

C = config.COLORS
st.set_page_config(page_title="SpamGuard Dashboard", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------------
# Styling (colours come from the uploaded CRM dashboard design)
# ----------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{
    background:
      radial-gradient(circle at 6% 0%, #3b2a5c 0%, transparent 30%),
      radial-gradient(circle at 100% 100%, #123a4a 0%, transparent 30%),
      {C['bg']};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; max-width: 1450px; }}
section[data-testid="stSidebar"] {{ background: {C['sidebar']}; border-right: 1px solid {C['border']}; }}
.brand {{ font-size: 1.35rem; font-weight: 800; color: {C['text']}; margin-bottom: 4px; }}
.brand span {{ color: {C['purple']}; }}
.side-label {{ color: {C['purple']}; font-weight: 700; font-size: .78rem; letter-spacing: .08em; margin: 18px 0 4px; }}
.page-title {{ font-size: 1.7rem; font-weight: 800; margin-bottom: 2px; }}
.page-sub {{ color: {C['muted']}; margin-bottom: 18px; }}
.kpi {{ border-radius: 18px; padding: 22px 24px; color: white; min-height: 132px; }}
.kpi.purple {{ background: linear-gradient(135deg, {C['purple']} 0%, {C['violet']} 100%); }}
.kpi.cyan {{ background: linear-gradient(135deg, {C['cyan']} 0%, {C['blue']} 100%); }}
.kpi .label {{ font-size: .95rem; opacity: .92; }}
.kpi .value {{ font-size: 2.7rem; font-weight: 800; line-height: 1.1; margin-top: 16px; }}
.kpi .unit {{ font-size: 1rem; font-weight: 500; opacity: .85; margin-left: 4px; }}
.mini {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 16px; padding: 14px 18px;
        display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; min-height: 60px; }}
.mini .name {{ font-weight: 600; }}
.mini .dot {{ width: 34px; height: 34px; border-radius: 50%; display: inline-block; margin-right: 12px; vertical-align: middle; }}
.dot.pink {{ background: linear-gradient(135deg, #ff7bb0, {C['purple']}); }}
.dot.teal {{ background: linear-gradient(135deg, {C['cyan']}, {C['blue']}); }}
.badge {{ padding: 4px 12px; border-radius: 8px; font-size: .8rem; font-weight: 700; }}
.badge.bad {{ background: #4a2d4a; color: #ff7bb0; }}
.badge.good {{ background: #0f4b52; color: {C['cyan']}; }}
.panel-title {{ font-weight: 700; font-size: 1.02rem; margin-bottom: 2px; }}
.panel-note {{ color: {C['muted']}; font-size: .85rem; margin-bottom: 6px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: {C['card']}; border-radius: 18px; border: 1px solid {C['border']}; }}
.verdict {{ border-radius: 18px; padding: 22px 26px; color: white; margin-bottom: 8px; }}
.verdict.spam {{ background: linear-gradient(135deg, {C['magenta']} 0%, #8E1BD8 100%); }}
.verdict.safe {{ background: linear-gradient(135deg, {C['cyan']} 0%, {C['blue']} 100%); }}
.verdict .big {{ font-size: 1.7rem; font-weight: 800; }}
.chip {{ display: inline-block; padding: 6px 14px; border-radius: 999px; background: #1d2547; margin: 3px 4px 3px 0; font-size: .85rem; }}
.chip.warn {{ background: #4a2d4a; color: #ffb3d1; }}
.chip.ok {{ background: #0f4b52; color: {C['cyan']}; }}
.stButton > button, .stDownloadButton > button {{
    background: linear-gradient(90deg, {C['purple']}, {C['violet']}); color: white; border: none;
    border-radius: 999px; font-weight: 600; padding: .5rem 1.4rem; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ filter: brightness(1.1); color: white; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLES = {
    "spam": ("URGENT: Verify your bank account now!!!",
             "Dear customer, we detected unusual activity on your account. Your account will be suspended "
             "within 24 hours unless you verify immediately. Click here: http://secure-bank-verify.xyz/login?id=4821 "
             "You may also claim a free $500 gift card."),
    "ham": ("Meeting notes: Phoenix sync",
            "Hi Priya, thanks for joining the Phoenix sync today. I have attached the notes. Please review the "
            "action items and let me know if I missed anything before Friday. Regards, Arun"),
}


# ----------------------------------------------------------------------------
# Data / model helpers
# ----------------------------------------------------------------------------
def ensure_ready():
    """Create the dataset and model automatically on the very first run."""
    config.ensure_directories()
    if not config.DATASET_PATH.exists():
        save_dataset(generate_dataset())
    if not config.SAMPLE_INBOX_PATH.exists():
        generate_sample_inbox().to_csv(config.SAMPLE_INBOX_PATH, index=False, encoding="utf-8")
    if not config.MODEL_PATH.exists() or not config.METRICS_PATH.exists():
        with st.spinner("First run: training the spam model..."):
            train()


def _version():
    """Changes whenever the model or dataset file changes, so caches refresh."""
    return (config.MODEL_PATH.stat().st_mtime, config.DATASET_PATH.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def get_model(version):
    return load_model()


@st.cache_data(show_spinner=False)
def get_dataset(version):
    return load_dataset()


def style_fig(fig, height=300, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["muted"]), showlegend=legend,
        legend=dict(orientation="h", y=1.14, x=1, xanchor="right"))
    fig.update_xaxes(gridcolor="#232b4d", zeroline=False)
    fig.update_yaxes(gridcolor="#232b4d", zeroline=False)
    return fig


def panel_title(title, note=""):
    st.markdown(f'<div class="panel-title">{title}</div><div class="panel-note">{note}</div>',
                unsafe_allow_html=True)


def kpi_card(label, value, unit, style):
    st.markdown(f'<div class="kpi {style}"><div class="label">{label}</div>'
                f'<div class="value">{value}<span class="unit">{unit}</span></div></div>',
                unsafe_allow_html=True)


def mini_card(name, badge, badge_style, dot):
    st.markdown(f'<div class="mini"><div><span class="dot {dot}"></span><span class="name">{name}</span></div>'
                f'<span class="badge {badge_style}">{badge}</span></div>', unsafe_allow_html=True)


def donut(labels, values, colors, center_text):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.64, sort=False,
                           marker=dict(colors=colors, line=dict(color=C["card"], width=3)),
                           textinfo="percent", textfont=dict(color="white")))
    fig.add_annotation(text=center_text, showarrow=False, font=dict(size=14, color=C["text"]))
    return style_fig(fig, height=270, legend=True)


# ----------------------------------------------------------------------------
# Page 1: Overview
# ----------------------------------------------------------------------------
def page_overview(df, metrics):
    st.markdown('<div class="page-title">Spam Email Dashboard</div>'
                '<div class="page-sub">Model performance and dataset insights</div>', unsafe_allow_html=True)
    best = metrics["best"]
    spam_n, ham_n = metrics["spam_count"], metrics["ham_count"]
    total = max(spam_n + ham_n, 1)

    left, right = st.columns([2.4, 1], gap="medium")

    with left:
        k1, k2, k3 = st.columns([1.15, 1.15, 1], gap="medium")
        with k1:
            kpi_card("Model Accuracy", f"{best['accuracy'] * 100:.1f}", "%", "purple")
        with k2:
            kpi_card("F1 Score (spam)", f"{best['f1'] * 100:.1f}", "%", "cyan")
        with k3:
            mini_card("Spam emails", f"{spam_n:,} ({spam_n / total:.0%})", "bad", "pink")
            mini_card("Legitimate", f"{ham_n:,} ({ham_n / total:.0%})", "good", "teal")

        # Line chart: spam vs legitimate per month
        with st.container(border=True):
            panel_title("Spam vs Legitimate Emails per Month", "Number of emails received each month")
            dated = df.dropna(subset=["date"]) if "date" in df.columns else pd.DataFrame()
            if dated.empty:
                st.info("Your dataset has no valid 'date' column, so the monthly trend is hidden.")
            else:
                dated = dated.assign(month=dated["date"].dt.to_period("M").dt.to_timestamp())
                monthly = dated.groupby(["month", "label"]).size().unstack(fill_value=0)
                monthly = monthly.reindex(columns=["ham", "spam"], fill_value=0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly.index, y=monthly["ham"], name="Legitimate", mode="lines+markers",
                                         line=dict(color=C["sky"], width=3, shape="spline")))
                fig.add_trace(go.Scatter(x=monthly.index, y=monthly["spam"], name="Spam", mode="lines+markers",
                                         line=dict(color=C["purple"], width=3, dash="dot", shape="spline")))
                fig.update_xaxes(tickformat="%b %Y")
                st.plotly_chart(style_fig(fig, 300), use_container_width=True)

        d1, d2 = st.columns(2, gap="medium")
        with d1:
            with st.container(border=True):
                panel_title("Spam by Category", "What kind of spam is most common")
                if "category" in df.columns:
                    cats = df[df["label"] == "spam"]["category"].value_counts()
                    palette = [C["cyan"], "#1E90FF", "#0A5CFF", "#1D7FA8", C["violet"], C["purple"]]
                    st.plotly_chart(donut(list(cats.index), list(cats.values), palette[:len(cats)],
                                          f"{int(cats.sum())}<br>spam"), use_container_width=True)
                else:
                    st.info("No 'category' column in the dataset.")
        with d2:
            with st.container(border=True):
                panel_title("Spam vs Legitimate", "Share of the whole dataset")
                st.plotly_chart(donut(["Legitimate", "Spam"], [ham_n, spam_n], [C["cyan"], C["magenta"]],
                                      f"{total:,}<br>emails"), use_container_width=True)

    with right:
        with st.container(border=True):
            panel_title("Detection Quality", "Measured on emails the model never saw")
            names = ["Accuracy", "Precision", "Recall", "F1"]
            vals = [best["accuracy"], best["precision"], best["recall"], best["f1"]]
            fig = go.Figure(go.Bar(x=names, y=[v * 100 for v in vals],
                                   text=[f"{v * 100:.1f}%" for v in vals], textposition="outside",
                                   marker=dict(color=[C["cyan"], "#1E90FF", C["violet"], C["purple"]])))
            fig.update_yaxes(range=[0, 115], showticklabels=False)
            st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True)
            st.caption("Precision = of emails flagged as spam, how many really were. "
                       "Recall = of all real spam, how much was caught.")

        with st.container(border=True):
            panel_title("Spam per Weekday", "When spam arrives most")
            if "date" in df.columns and df["date"].notna().any():
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                spam_days = df[df["label"] == "spam"]["date"].dt.day_name().value_counts().reindex(order, fill_value=0)
                fig = go.Figure(go.Bar(x=[d[:3] for d in order], y=spam_days.values,
                                       marker=dict(color=spam_days.values,
                                                   colorscale=[[0, C["blue"]], [1, C["cyan"]]])))
                st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True)
            else:
                st.info("No valid 'date' column in the dataset.")


# ----------------------------------------------------------------------------
# Page 2: Check a single email
# ----------------------------------------------------------------------------
def _load_example(kind):
    st.session_state["subject_in"], st.session_state["body_in"] = EXAMPLES[kind]


def page_check(model):
    st.markdown('<div class="page-title">Check an Email</div>'
                '<div class="page-sub">Paste an email and the model will tell you if it looks like spam</div>',
                unsafe_allow_html=True)
    st.session_state.setdefault("subject_in", "")
    st.session_state.setdefault("body_in", "")

    left, right = st.columns([1, 1.1], gap="medium")
    with left:
        with st.container(border=True):
            panel_title("Email details")
            st.text_input("Subject", key="subject_in", placeholder="e.g. You have won a prize!")
            st.text_area("Body", key="body_in", height=210, placeholder="Paste the email text here...")
            threshold = st.slider("Spam threshold (%)", 10, 90, 50,
                                  help="Lower = catches more spam but may flag more normal emails.") / 100
            b1, b2, b3 = st.columns(3)
            b1.button("Spam example", on_click=_load_example, args=("spam",))
            b2.button("Safe example", on_click=_load_example, args=("ham",))
            if b3.button("Analyze"):
                subject, body = st.session_state["subject_in"].strip(), st.session_state["body_in"].strip()
                if not subject and not body:
                    st.warning("Please enter a subject or a body first.")
                    st.session_state.pop("last_check", None)
                else:
                    st.session_state["last_check"] = (subject, body)

    with right:
        check = st.session_state.get("last_check")
        if not check:
            with st.container(border=True):
                st.info("Enter an email on the left and press Analyze. Try the example buttons first.")
            return
        subject, body = check
        result = predict_one(model, subject, body, threshold)
        prob = result["spam_probability"]
        is_spam = result["label"] == "spam"

        css, title = ("spam", "SPAM DETECTED") if is_spam else ("safe", "LOOKS SAFE")
        st.markdown(f'<div class="verdict {css}"><div class="big">{title}</div>'
                    f'Spam probability: {prob * 100:.1f}% (threshold {threshold * 100:.0f}%)</div>',
                    unsafe_allow_html=True)
        if len(f"{subject} {body}".split()) < 4:
            st.info("Very short text - results are less reliable.")
        elif 0.35 <= prob <= 0.65:
            st.warning("Borderline result - read this email carefully yourself.")

        g1, g2 = st.columns(2, gap="small")
        with g1:
            with st.container(border=True):
                panel_title("Spam score")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(color=C["text"])),
                    gauge=dict(axis=dict(range=[0, 100], tickcolor=C["muted"]),
                               bar=dict(color=C["magenta"] if is_spam else C["cyan"]),
                               bgcolor="rgba(0,0,0,0)", borderwidth=0,
                               steps=[dict(range=[0, 35], color="#0f4b52"),
                                      dict(range=[35, 65], color="#5a4a1a"),
                                      dict(range=[65, 100], color="#5a1f4a")],
                               threshold=dict(line=dict(color="white", width=3), value=threshold * 100))))
                st.plotly_chart(style_fig(fig, 230, legend=False), use_container_width=True)
        with g2:
            with st.container(border=True):
                panel_title("Warning signs")
                s = extract_signals(subject, body)
                chips = [
                    (f"{s['links']} link(s)", s["links"] > 0),
                    (f"{s['exclamations']} exclamation mark(s)", s["exclamations"] >= 2),
                    (f"{s['caps_percent']}% capital letters", s["caps_percent"] > 25),
                    (f"{s['money_mentions']} money mention(s)", s["money_mentions"] > 0),
                ]
                html = "".join(f'<span class="chip {"warn" if bad else "ok"}">{t}</span>' for t, bad in chips)
                html += "".join(f'<span class="chip warn">"{w}"</span>' for w in s["urgent_words"])
                st.markdown(html, unsafe_allow_html=True)

        with st.container(border=True):
            panel_title("Why this result?", "Words that pushed the score towards spam (magenta) or safe (cyan)")
            words = explain(model, subject, body, top_n=8)
            if words:
                words = words[::-1]
                fig = go.Figure(go.Bar(x=[w[1] for w in words], y=[w[0] for w in words], orientation="h",
                                       marker=dict(color=[C["magenta"] if w[1] > 0 else C["cyan"] for w in words])))
                st.plotly_chart(style_fig(fig, 280, legend=False), use_container_width=True)
            else:
                st.info("None of the words in this email were known to the model.")


# ----------------------------------------------------------------------------
# Page 3: Batch scan
# ----------------------------------------------------------------------------
def page_batch(model):
    st.markdown('<div class="page-title">Batch Scan</div>'
                '<div class="page-sub">Scan a whole inbox exported as a CSV file</div>', unsafe_allow_html=True)
    with st.container(border=True):
        panel_title("Choose a file", "CSV with a 'subject' and 'body' column (or a single 'text' column). "
                                     "An optional 'label' column (spam/ham) lets us measure accuracy.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        st.button("Use the sample inbox", on_click=lambda: st.session_state.update(use_sample=True))
        threshold = st.slider("Spam threshold (%)", 10, 90, 50, key="batch_threshold") / 100

    source, source_name = None, ""
    try:
        if uploaded is not None:
            try:
                source = pd.read_csv(uploaded, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded.seek(0)
                source = pd.read_csv(uploaded, encoding="latin-1")
            source_name = uploaded.name
        elif st.session_state.get("use_sample"):
            source = pd.read_csv(config.SAMPLE_INBOX_PATH, encoding="utf-8")
            source_name = "sample_inbox.csv"
    except Exception as error:   # unreadable / empty file
        st.error(f"Could not read the file: {error}")
        return
    if source is None:
        return

    try:
        inbox = prepare_inbox(source)
    except ValueError as error:
        st.error(str(error))
        return

    scanned = predict_dataframe(model, inbox, threshold)
    n_spam = int((scanned["prediction"] == "spam").sum())
    st.markdown(f"**Results for {source_name}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Emails scanned", f"{len(scanned):,}", "", "purple")
    with c2:
        kpi_card("Spam found", f"{n_spam:,}", f"({n_spam / len(scanned):.0%})", "cyan")
    with c3:
        kpi_card("Safe emails", f"{len(scanned) - n_spam:,}", "", "cyan")
    with c4:
        if "true_label" in scanned.columns and scanned["true_label"].notna().any():
            known = scanned.dropna(subset=["true_label"])
            acc = (known["true_label"] == known["prediction"]).mean() * 100
            kpi_card("Accuracy on this file", f"{acc:.1f}", "%", "purple")
        else:
            kpi_card("Accuracy", "n/a", "", "purple")

    table = pd.DataFrame({
        "Verdict": scanned["prediction"].map({"spam": "SPAM", "ham": "SAFE"}),
        "Spam score": scanned["spam_probability"] * 100,
        "Subject": scanned["subject"],
        "Body": scanned["body"].str.slice(0, 140),
    })
    st.dataframe(table, use_container_width=True, hide_index=True, column_config={
        "Spam score": st.column_config.ProgressColumn("Spam score (%)", format="%.1f", min_value=0, max_value=100)})

    export = scanned.drop(columns=["text"]).assign(spam_probability=lambda d: d["spam_probability"].round(4))
    export.to_csv(config.BATCH_RESULTS_PATH, index=False, encoding="utf-8")
    st.download_button("Download results (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name="spam_scan_results.csv", mime="text/csv")


# ----------------------------------------------------------------------------
# Page 4: Model report and retraining
# ----------------------------------------------------------------------------
def page_report(metrics):
    st.markdown('<div class="page-title">Model Report</div>'
                f'<div class="page-sub">Best model: {metrics["best_model"]} &nbsp;|&nbsp; trained {metrics["trained_at"]} '
                f'&nbsp;|&nbsp; {metrics["n_train"]} training / {metrics["n_test"]} test emails</div>',
                unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a:
        with st.container(border=True):
            panel_title("Model comparison", "Both models were tested; the best F1 score is saved")
            fig = go.Figure()
            colors = {"accuracy": C["cyan"], "precision": "#1E90FF", "recall": C["violet"], "f1": C["purple"]}
            for metric, colour in colors.items():
                fig.add_trace(go.Bar(name=metric.capitalize(), x=list(metrics["models"]),
                                     y=[m[metric] * 100 for m in metrics["models"].values()], marker_color=colour))
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 105])
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with b:
        with st.container(border=True):
            panel_title("Confusion matrix", "Rows = actual, columns = predicted")
            cm = metrics["best"]["confusion_matrix"]
            labels = ["Legitimate", "Spam"]
            fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, text=cm, texttemplate="%{text}",
                                       textfont=dict(size=22, color="white"), showscale=False,
                                       colorscale=[[0, "#1d2547"], [1, C["violet"]]]))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_fig(fig, 320, legend=False), use_container_width=True)

    c, d = st.columns(2, gap="medium")
    for col, key, colour, title in ((c, "spam", C["magenta"], "Top spam words"),
                                    (d, "ham", C["cyan"], "Top legitimate words")):
        with col:
            with st.container(border=True):
                panel_title(title, "The words the model trusts most")
                words = metrics["top_words"][key][::-1]
                fig = go.Figure(go.Bar(x=[abs(w[1]) for w in words], y=[w[0] for w in words],
                                       orientation="h", marker_color=colour))
                st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True)

    with st.container(border=True):
        panel_title("Retrain the model", "Replace the dataset and train again. The old dataset is backed up "
                                         "as data/emails_backup.csv")
        n = st.slider("Number of synthetic emails to generate", 400, 5000, 1200, step=100)
        if st.button("Generate new dataset and retrain"):
            with st.spinner("Generating data and training..."):
                backup_dataset()
                save_dataset(generate_dataset(n=n, seed=None))
                train()
            st.success("Done - the dashboard now uses the new model.")
            st.rerun()
        st.markdown("---")
        custom = st.file_uploader("...or train on your own CSV (columns: subject/body or text, plus label)",
                                  type=["csv"], key="custom_dataset")
        if custom is not None and st.button("Use my dataset and retrain"):
            try:
                raw = pd.read_csv(custom, encoding="utf-8")
                clean = prepare_dataframe(raw)       # raises ValueError if invalid
            except UnicodeDecodeError:
                st.error("Please save the CSV with UTF-8 encoding.")
            except Exception as error:
                st.error(f"Could not use this file: {error}")
            else:
                with st.spinner("Training on your dataset..."):
                    backup_dataset()
                    clean.to_csv(config.DATASET_PATH, index=False, encoding="utf-8")
                    train()
                st.success("Done - trained on your dataset.")
                st.rerun()


def backup_dataset():
    if config.DATASET_PATH.exists():
        shutil.copy(config.DATASET_PATH, config.DATA_DIR / "emails_backup.csv")


# ----------------------------------------------------------------------------
# Main layout: sidebar navigation + page router
# ----------------------------------------------------------------------------
def main():
    ensure_ready()
    metrics = load_metrics()
    version = _version()
    model, df = get_model(version), get_dataset(version)

    with st.sidebar:
        st.markdown('<div class="brand">Spam<span>Guard</span></div>', unsafe_allow_html=True)
        st.caption("Email spam prediction")
        st.markdown('<div class="side-label">NAVIGATION</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", ["📊 Overview", "✉️ Check Email", "📂 Batch Scan", "🧠 Model Report"],
                        label_visibility="collapsed")
        st.markdown('<div class="side-label">REPORT</div>', unsafe_allow_html=True)
        st.caption(f"Model: {metrics['best_model']}")
        st.caption(f"Trained on {metrics['n_samples']:,} emails")
        st.download_button("Download model report", json.dumps(metrics, indent=2),
                           file_name="spam_model_report.json", mime="application/json")

    if page.endswith("Overview"):
        page_overview(df, metrics)
    elif page.endswith("Check Email"):
        page_check(model)
    elif page.endswith("Batch Scan"):
        page_batch(model)
    else:
        page_report(metrics)


main()
