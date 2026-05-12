import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt


import sqlite3


from nltk import word_tokenize, pos_tag


nltk.download('stopwords')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv("data/sample_complaints.csv")


# ----------------------------
# DATABASE CONNECTION
# ----------------------------

conn = sqlite3.connect("complaints.db")

cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    complaint TEXT,

    predicted_priority TEXT,

    confidence REAL

)
""")

conn.commit()

# ----------------------------
# TEXT CLEANING
# ----------------------------

stop_words = set(stopwords.words('english'))

# PII Masking

def mask_pii(text):

    # Email masking
    text = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}',
        '[EMAIL]',
        text
    )

    # Phone masking
    text = re.sub(
        r'\b\d{10}\b',
        '[PHONE]',
        text
    )

    return text

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# ----------------------------
# PRIORITY LABELS
# ----------------------------

def assign_priority(text):

    text = text.lower()

    high_words = [
        "fraud", "scam", "hacked",
        "urgent", "lawsuit",
        "identity theft",
        "money deducted"
    ]

    medium_words = [
        "delay", "incorrect",
        "problem", "issue",
        "not working", "failed"
    ]

    for word in high_words:
        if word in text:
            return "High"

    for word in medium_words:
        if word in text:
            return "Medium"

    return "Low"

# ----------------------------
# PREPARE DATA
# ----------------------------

df["cleaned_text"] = df["Consumer complaint narrative"].apply(clean_text)

df["priority"] = df["cleaned_text"].apply(assign_priority)

# ----------------------------
# TRAIN MODEL
# ----------------------------

X = df["cleaned_text"]

y = df["priority"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = CountVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)

model = MultinomialNB()

model.fit(X_train_vectors, y_train)

# ----------------------------
# TF-IDF RETRIEVAL
# ----------------------------

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])

# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("Sentinel Support AI")

st.subheader("Customer Complaint Priority Classifier")

complaint = st.text_area("Enter Customer Complaint")

if st.button("Analyze Complaint"):
    masked_complaint = mask_pii(complaint)

    st.subheader("Masked Complaint")

    st.write(masked_complaint)

    # Clean complaint
    cleaned = clean_text(complaint)

    # Convert to vector
    vector = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(vector)

    confidence = model.predict_proba(vector)

    max_confidence = max(confidence[0])

    priority = prediction[0]

    # Save to database

    cursor.execute(
        """
        INSERT INTO complaints (
            complaint,
            predicted_priority,
            confidence
        )
        VALUES (?, ?, ?)
        """,
        (
            complaint,
            priority,
            float(max_confidence)
        )
    )

    conn.commit()

    # ----------------------------
    # PRIORITY DISPLAY
    # ----------------------------

    if priority == "High":
        st.error(f"Predicted Priority: {priority}")

    elif priority == "Medium":
        st.warning(f"Predicted Priority: {priority}")

    else:
        st.success(f"Predicted Priority: {priority}")

    # ----------------------------
    # IMPORTANT WORDS
    # ----------------------------

    st.subheader("Important Words")

    words = cleaned.split()

    for word in words:
        st.write("-", word)

    # ----------------------------
    # CONFIDENCE SCORES
    # ----------------------------

    # Human review warning

    if max_confidence < 0.60:

        st.warning("⚠ HUMAN REVIEW REQUIRED")

    st.subheader("Confidence Scores")

    classes = model.classes_

    for label, score in zip(classes, confidence[0]):

        st.write(f"{label}: {round(score * 100, 2)}%")

    # ----------------------------
    # SIMILAR COMPLAINTS
    # ----------------------------

    complaint_tfidf = tfidf.transform([cleaned])

    similarity = cosine_similarity(
        complaint_tfidf,
        tfidf_matrix
    )

    similar_index = similarity.argsort()[0][-3:][::-1]

    st.subheader("Similar Complaints")

    for index in similar_index:

        st.write(
            df.iloc[index]["Consumer complaint narrative"]
        )

        st.write("---")
    # POS Tagging

    st.subheader("POS Tags")

    tokens = word_tokenize(cleaned)

    pos_tags = pos_tag(tokens)

    for word, tag in pos_tags:

        st.write(f"{word} → {tag}")


# ----------------------------
# PRIORITY DISTRIBUTION CHART
# ----------------------------

st.subheader("Priority Distribution")

priority_counts = df["priority"].value_counts()

fig, ax = plt.subplots()

ax.bar(
    priority_counts.index,
    priority_counts.values
)

ax.set_xlabel("Priority")

ax.set_ylabel("Count")

ax.set_title("Complaint Priority Distribution")

st.pyplot(fig)