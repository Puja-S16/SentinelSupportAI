import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/Consumer_Complaints.csv")

print("Original Dataset Shape:")
print(df.shape)

print("\nRemoving empty complaints...")

# Keep only rows where complaint text exists
df = df[df["Consumer complaint narrative"].notna()]

print("\nNew Dataset Shape:")
print(df.shape)

# Keep only important columns
df = df[
    [
        "Complaint ID",
        "Product",
        "Issue",
        "Consumer complaint narrative"
    ]
]

print("\nFinal Columns:")
print(df.columns)

print("\nSample Complaints:")
print(df.head())

# Save cleaned dataset
df.to_csv("data/cleaned_complaints.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Take 30,000 random rows
sample_df = df.sample(n=30000, random_state=42)

# Save sample dataset
sample_df.to_csv("data/sample_complaints.csv", index=False)

print("\nSample dataset created!")
print(sample_df.shape)


# Load sample dataset
sample_df = pd.read_csv("data/sample_complaints.csv")

# Stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):

    # lowercase
    text = text.lower()

    # remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # tokenize
    words = text.split()

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Apply cleaning
sample_df["cleaned_text"] = sample_df["Consumer complaint narrative"].apply(clean_text)

# Save cleaned data
sample_df.to_csv("data/processed_complaints.csv", index=False)

print("\nText preprocessing completed!")
print(sample_df[["Consumer complaint narrative", "cleaned_text"]].head())

# Priority labeling function
def assign_priority(text):

    text = text.lower()

    high_words = [
        "fraud", "scam", "hacked", "urgent",
        "lawsuit", "identity theft", "money deducted"
    ]

    medium_words = [
        "delay", "incorrect", "problem",
        "issue", "not working", "failed"
    ]

    # High Priority
    for word in high_words:
        if word in text:
            return "High"

    # Medium Priority
    for word in medium_words:
        if word in text:
            return "Medium"

    # Default
    return "Low"

# Create priority column
sample_df["priority"] = sample_df["cleaned_text"].apply(assign_priority)

# Save updated dataset
sample_df.to_csv("data/labeled_complaints.csv", index=False)

print("\nPriority labeling completed!")

# Check label counts
print(sample_df["priority"].value_counts())



# Features and labels
X = sample_df["cleaned_text"]
y = sample_df["priority"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain/Test Split Completed!")

print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))




# Convert text into numerical vectors
vectorizer = CountVectorizer()

# Learn vocabulary from training data
X_train_vectors = vectorizer.fit_transform(X_train)

# Transform test data
X_test_vectors = vectorizer.transform(X_test)

print("\nText Vectorization Completed!")

print("Training Vector Shape:")
print(X_train_vectors.shape)

print("Testing Vector Shape:")
print(X_test_vectors.shape)





# Create model
model = MultinomialNB()

# Train model
model.fit(X_train_vectors, y_train)

# Predictions
y_pred = model.predict(X_test_vectors)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")

print("Accuracy:")
print(round(accuracy * 100, 2), "%")



# Test complaint
new_complaint = """
Please improve your app design.
"""

# Clean text
cleaned_complaint = clean_text(new_complaint)

# Convert to vector
complaint_vector = vectorizer.transform([cleaned_complaint])

# Predict
prediction = model.predict(complaint_vector)

# Confidence score
confidence = model.predict_proba(complaint_vector)

print("\nNEW COMPLAINT:")
print(new_complaint)

print("\nPredicted Priority:")
print(prediction[0])

print("\nConfidence Scores:")
print(confidence)



# Classification report
print("\nCLASSIFICATION REPORT:\n")

print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nCONFUSION MATRIX:\n")

print(confusion_matrix(y_test, y_pred))