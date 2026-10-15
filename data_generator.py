"""Creates a realistic *synthetic* email dataset (no internet needed).

The emails are built from templates with random names, banks, amounts and links.
A little noise is added on purpose (spam that sounds polite, normal mail that
mentions 'free shipping') so the model does not score a perfect 100%.
"""
import random
from datetime import datetime, timedelta

import pandas as pd

from src import config

NAMES = ["Arun", "Priya", "Karthik", "Meena", "Rahul", "Divya", "Suresh", "Anita",
         "Vikram", "Lakshmi", "Sanjay", "Nisha", "David", "Sara", "Michael", "Emma"]
BANKS = ["HDFC Bank", "SBI", "ICICI Bank", "PayPal", "Amazon", "Netflix",
         "Microsoft", "Apple ID", "Google"]
PROJECTS = ["Alpha", "Phoenix", "Website Redesign", "Q3 Budget", "Mobile App",
            "Data Migration", "Annual Report", "Customer Survey"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SITES = ["TechDaily", "CodeAcademy Weekly", "DesignHub", "DataCamp Digest", "ScienceToday"]
TOPICS = ["Python programming", "machine learning", "UX design", "cloud computing", "data analysis"]

SPAM_TEMPLATES = {
    "Phishing": [
        ("Urgent: verify your {bank} account",
         "Dear customer, we detected unusual activity on your {bank} account. Your account will be suspended within 24 hours unless you verify your details immediately. Click here to confirm your password: {bad_link}"),
        ("Security alert: your {bank} password expires today",
         "Your password is about to expire. Log in now to keep your account active. Failure to update your information will result in permanent suspension. Update here: {bad_link}"),
        ("Action required: payment failed",
         "We could not process your last payment. Please update your card number and PIN immediately to avoid cancellation of your {bank} service. {bad_link}"),
    ],
    "Prize Scam": [
        ("Congratulations {name}! You have won ${amount}",
         "You have been selected as the lucky winner of our international lottery. To claim your prize of ${amount}, reply with your full name, address and bank account number. Offer expires in 48 hours!!!"),
        ("FINAL NOTICE: claim your free gift card",
         "You are our 1,000,000th visitor! Claim your free $500 gift card now. Click the link and enter your details. {bad_link}"),
    ],
    "Promotion": [
        ("{pct}% OFF everything - today only!!!",
         "Huge sale! Buy now and save {pct}% on all products. Limited time offer, free shipping on every order. Do not miss out, order today! {bad_link} To stop receiving offers reply STOP."),
        ("Best deals you will never see again",
         "Cheap watches, bags and shoes at unbeatable prices. 100% genuine, no questions asked. Buy now, hurry while stocks last! {bad_link}"),
    ],
    "Loan / Investment": [
        ("Pre-approved loan of ${amount} - no credit check",
         "Get cash in your account today. No paperwork, no credit check, guaranteed approval. Apply now and receive ${amount} within one hour. {bad_link}"),
        ("Earn ${amount} per week working from home",
         "Make easy money online with this proven system. No experience needed. Thousands of people are already earning. Join today and start earning, risk free! {bad_link}"),
        ("Invest in crypto and double your money",
         "Our secret trading bot guarantees 200% returns in 7 days. Send us your deposit today and watch your money grow. Limited slots available. {bad_link}"),
    ],
    "Pharma": [
        ("Cheap meds online - no prescription needed",
         "Buy cheap pills online without a prescription. Fast discreet delivery, up to 80% discount. Order now: {bad_link}"),
        ("Lose {kg} kg in 2 weeks guaranteed",
         "Miracle weight loss pill melts fat while you sleep. Doctors hate this trick! Order your free trial today. {bad_link}"),
    ],
}
SPAM_WEIGHTS = {"Phishing": 0.30, "Prize Scam": 0.20, "Promotion": 0.20,
                "Loan / Investment": 0.20, "Pharma": 0.10}

HAM_TEMPLATES = {
    "Work": [
        ("Meeting notes: {project} sync",
         "Hi {name}, thanks for joining the {project} sync today. I have attached the notes. Please review the action items and let me know if I missed anything before {day}. Regards, {name2}"),
        ("Re: {project} deadline",
         "Hi {name}, I checked with the team and we can move the {project} deadline to {day}. Could you update the schedule in the shared document? Thanks, {name2}"),
        ("Invoice review for {project}",
         "Hello {name}, please find the invoice for {project} attached. Let me know if the amounts look correct and I will forward it to accounts. Best, {name2}"),
        ("Interview schedule for {day}",
         "Hi {name}, the interview panel is confirmed for {day} at 11 am. I have booked the conference room. Please share the candidate resume with the panel. Thanks, {name2}"),
    ],
    "Personal": [
        ("Dinner on {day}?",
         "Hey {name}, are you free for dinner on {day}? Mom is cooking and everyone is coming over. Let me know and I will save you a seat. Love, {name2}"),
        ("Photos from the trip",
         "Hi {name}, I finally uploaded the photos from our trip. The ones at the beach turned out really well. Tell me which ones you want printed. Talk soon, {name2}"),
        ("Happy birthday!",
         "Happy birthday {name}! Hope you have a wonderful day with family and friends. Let us catch up over coffee this weekend. {name2}"),
    ],
    "Newsletter": [
        ("Your weekly digest from {site}",
         "Hello {name}, here are this week top stories and tutorials from {site}. Read the new article about {topic}. You are receiving this because you subscribed. Unsubscribe anytime in your account settings."),
        ("New {topic} course available",
         "Hi {name}, our new course on {topic} is now open for enrolment. Early members get a discount on the first month. Visit {good_link} to see the syllabus."),
    ],
    "Receipt": [
        ("Your order #{code} has shipped",
         "Hi {name}, your order #{code} has shipped and will arrive by {day}. Track your package using this link: {good_link}. Free shipping was applied to this order. Thank you for shopping with us."),
        ("Receipt for your payment of ${small}",
         "Thank you for your payment of ${small}. Your receipt is attached. If you did not make this purchase, contact support through our official website. Transaction ID {code}."),
        ("Your {site} subscription renews on {day}",
         "Hi {name}, your subscription will renew on {day} for ${small}. You can manage or cancel it from your account page at {good_link}."),
    ],
}
HAM_WEIGHTS = {"Work": 0.40, "Personal": 0.20, "Newsletter": 0.20, "Receipt": 0.20}

SPAM_NOISE_PREFIX = ["Hi {name}, hope you are doing well. ", "Hello, quick note for you. "]
HAM_NOISE_SUFFIX = [" Click here to view details.", " Reply soon, seats are limited."]


def _fill_values(rng):
    """Random values used to fill the template placeholders."""
    site = rng.choice(SITES)
    slug = site.lower().replace(" ", "")
    bank_slug = rng.choice(BANKS).lower().replace(" ", "")
    return {
        "name": rng.choice(NAMES),
        "name2": rng.choice(NAMES),
        "bank": rng.choice(BANKS),
        "project": rng.choice(PROJECTS),
        "day": rng.choice(DAYS),
        "site": site,
        "topic": rng.choice(TOPICS),
        "code": rng.randint(100000, 999999),
        "amount": f"{rng.randint(5, 500) * 100:,}",
        "small": rng.randint(5, 120),
        "pct": rng.choice([30, 40, 50, 60, 70, 80, 90]),
        "kg": rng.randint(5, 20),
        "bad_link": f"http://{bank_slug}-{rng.choice(['secure', 'verify', 'update', 'offer'])}."
                    f"{rng.choice(['xyz', 'top', 'info', 'ru'])}/login?id={rng.randint(1000, 99999)}",
        "good_link": f"https://www.{slug}.com/{rng.choice(['orders', 'courses', 'account'])}",
    }


def _make_email(is_spam, rng):
    """Build one (subject, body, category) tuple."""
    templates, weights = (SPAM_TEMPLATES, SPAM_WEIGHTS) if is_spam else (HAM_TEMPLATES, HAM_WEIGHTS)
    category = rng.choices(list(weights), weights=list(weights.values()))[0]
    subject_t, body_t = rng.choice(templates[category])
    values = _fill_values(rng)
    subject, body = subject_t.format(**values), body_t.format(**values)

    # Add a little noise so the task is not trivially easy
    if is_spam and rng.random() < 0.10:
        body = rng.choice(SPAM_NOISE_PREFIX).format(**values) + body
    if not is_spam and rng.random() < 0.06:
        body += rng.choice(HAM_NOISE_SUFFIX)
    return subject, body, category


def generate_dataset(n=1200, spam_ratio=0.4, seed=42):
    """Return a DataFrame with date, subject, body, label, category."""
    rng = random.Random(seed)
    start = datetime(2026, 1, 1)
    rows = []
    for _ in range(int(n)):
        is_spam = rng.random() < spam_ratio
        subject, body, category = _make_email(is_spam, rng)
        # Spam gradually increases over the 6 months, normal mail stays flat
        day_weights = [1 + (d / 180 if is_spam else 0) for d in range(181)]
        day = rng.choices(range(181), weights=day_weights)[0]
        date = start + timedelta(days=day, hours=rng.randint(6, 22), minutes=rng.randint(0, 59))
        label = "spam" if is_spam else "ham"
        if rng.random() < 0.01:          # ~1% label noise, like human mistakes
            label = "ham" if label == "spam" else "spam"
        rows.append({"date": date.strftime("%Y-%m-%d %H:%M"), "subject": subject,
                     "body": body, "label": label, "category": category})
    return pd.DataFrame(rows)


def generate_sample_inbox(n=16, seed=7):
    """A small file you can upload in the Batch Scan page."""
    df = generate_dataset(n=n, spam_ratio=0.5, seed=seed)
    return df[["subject", "body", "label"]]


def save_dataset(df, path=config.DATASET_PATH):
    config.ensure_directories()
    df.to_csv(path, index=False, encoding="utf-8")
    return path
