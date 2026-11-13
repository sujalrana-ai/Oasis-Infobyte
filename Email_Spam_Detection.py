# %% [markdown]
# # Task 4: Email / SMS Spam Detection with Machine Learning
# **Objective:** Build a Natural Language Processing (NLP) binary classifier that distinguishes spam messages from legitimate (ham) messages.

# %%
# ==========================================
# 1. IMPORTS & DATASET LOADING
# ==========================================
import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Download NLTK stopwords quietly if not already present
nltk.download('stopwords', quiet=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Auto-detect whichever spam dataset file is in the folder
possible_files = [
    "spam.csv",
    "SMSSpamCollection",
    "emails.csv"
]

file_path = next((f for f in possible_files if os.path.exists(f)), None)
if file_path is None:
    raise FileNotFoundError(
        "Dataset not found! Place 'spam.csv' (from Kaggle SMS Spam Collection) "
        "in the same folder as this file."
    )

# Load dataset safely (handles latin-1 encoding and unnamed extra columns in Kaggle's spam.csv)
if file_path == "SMSSpamCollection":
    df = pd.read_csv(file_path, sep="\t", names=["label", "message"], encoding="latin-1")
else:
    df = pd.read_csv(file_path, encoding="latin-1")
    # Keep first two columns (v1 = label, v2 = message) and rename them cleanly
    df = df.iloc[:, :2].copy()
    df.columns = ["label", "message"]

print(f"Loaded file: {file_path}")
print("Initial Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

# %%
# ==========================================
# 2. DATA CLEANING & CLASS DISTRIBUTION CHECK
# ==========================================
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Standardize labels to lowercase ('ham' / 'spam') and map to binary target
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['target'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop any rows where label wasn't ham/spam
df.dropna(subset=['target'], inplace=True)
df['target'] = df['target'].astype(int)

# Class distribution counts and percentages
class_counts = df['label'].value_counts()
class_pct = df['label'].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    'Count': class_counts,
    'Percentage (%)': class_pct.round(2)
})
print("\nClass Distribution:\n", dist_df)

# Visualize class distribution
plt.figure(figsize=(7, 4))
sns.barplot(
    x=dist_df.index,
    y=dist_df['Count'],
    hue=dist_df.index,
    palette=['#2ecc71', '#e74c3c'],
    legend=False
)
plt.title("Class Distribution: Ham vs. Spam Messages", fontsize=13, fontweight='bold')
plt.xlabel("Message Category")
plt.ylabel("Number of Messages")
plt.tight_layout()
plt.show()

# %%
# ==========================================
# 3. TEXT PREPROCESSING PIPELINE
# ==========================================
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # 1. Convert to lowercase
    text = str(text).lower()
    # 2. Remove URLs, numbers, and punctuation
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenize, remove stopwords, and apply Porter stemming
    tokens = text.split()
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

df['cleaned_message'] = df['message'].apply(preprocess_text)

print("\nSample Before vs. After Preprocessing:")
for i in range(3):
    print(f"\nOriginal [{df['label'].iloc[i]}]: {df['message'].iloc[i]}")
    print(f"Cleaned  [{df['label'].iloc[i]}]: {df['cleaned_message'].iloc[i]}")

# %% [markdown]
# ### What Does TF-IDF Measure?
# **TF-IDF (Term Frequency – Inverse Document Frequency)** converts cleaned text into numerical feature vectors by weighing how important a word is to a specific message relative to the entire dataset:
# - **Term Frequency (TF):** Measures how frequently a word appears inside a single email/SMS.
# - **Inverse Document Frequency (IDF):** Penalizes common words that appear across almost all messages (such as *"call"*, *"get"*, *"know"*) and boosts rare, discriminative keywords strongly associated with spam (such as *"claim"*, *"prize"*, *"urgent"*, *"free"*).

# %%
# ==========================================
# 4. TRAIN/TEST SPLIT & TF-IDF VECTORIZATION
# ==========================================
X = df['cleaned_message']
y = df['target']

# Stratified split preserves the exact spam/ham ratio in train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(X_train_raw)
X_test = tfidf.transform(X_test_raw)

print(f"\nTraining Matrix Shape: {X_train.shape} | Testing Matrix Shape: {X_test.shape}")

# %%
# ==========================================
# 5. MODEL TRAINING & EVALUATION
# ==========================================
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42)
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4)
    })

results_df = pd.DataFrame(results).sort_values(by="F1_Score", ascending=False)
print("\nModel Evaluation Comparison:\n", results_df.to_string(index=False))

# Detailed Classification Report for Best Model
best_model_name = results_df.iloc[0]['Model']
best_preds = predictions[best_model_name]

print(f"\nDetailed Classification Report ({best_model_name}):\n")
print(classification_report(y_test, best_preds, target_names=['Ham (0)', 'Spam (1)']))

# Plot Confusion Matrix for Best Model
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Predicted Ham', 'Predicted Spam'],
    yticklabels=['Actual Ham', 'Actual Spam']
)
plt.title(f"Confusion Matrix — {best_model_name}", fontsize=13, fontweight='bold')
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Why Is Recall Particularly Important for Spam Detection?
# - **Recall** measures the proportion of actual spam messages that the model successfully catches: $\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$.
# - A **False Negative** in spam detection means a malicious phishing link, fraudulent offer, or malware-laced email slips past the filter straight into the user's primary inbox—creating a direct security and financial risk.
# - While **Precision** is also important (so legitimate work or personal emails are not wrongly blocked as False Positives), maximizing **Recall** ensures dangerous spam threats are reliably intercepted before reaching the user.

# %%
# ==========================================
# 6. (BONUS) WORDCLOUD VISUALISATIONS FOR SPAM & HAM
# ==========================================
try:
    from wordcloud import WordCloud

    ham_text = " ".join(df[df['target'] == 0]['cleaned_message'])
    spam_text = " ".join(df[df['target'] == 1]['cleaned_message'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    wc_ham = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(ham_text)
    axes[0].imshow(wc_ham, interpolation='bilinear')
    axes[0].set_title("Most Frequent Words in Legitimate (Ham) Messages", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    wc_spam = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(spam_text)
    axes[1].imshow(wc_spam, interpolation='bilinear')
    axes[1].set_title("Most Frequent Words in Spam Messages", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
except ImportError:
    print("Note: Install 'wordcloud' (`pip install wordcloud`) to display WordCloud charts.")
