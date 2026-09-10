import nbformat as nbf

notebook = nbf.v4.new_notebook()

code = """# Required pip install packages: pandas matplotlib seaborn nltk scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Download necessary NLTK datasets (run once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==========================================
# Phase 1: Data Engineer (EDA & NLP)
# ==========================================

# 1. Data Loading
print("Loading dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unneeded columns and rename
df = df.iloc[:, :2] # keep only first two columns
df.columns = ['label', 'text']

# 2. Text Cleaning Script
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_data(text):
    \"\"\"
    Cleans text data by lowercasing, removing URLs, HTML, punctuation, and digits.
    Then tokenizes, removes stopwords, and lemmatizes.
    \"\"\"
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and digits (keep only alphabets)
    text = re.sub(r'[^a-z\\s]', '', text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Stopword removal and Lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(cleaned_tokens)

print("Cleaning text data...")
df['clean_text'] = df['text'].apply(clean_text_data)

# 3. Data Visualization (EDA)
# Generate and display a Pie Chart for Spam vs. Safe (Ham)
plt.figure(figsize=(6, 6))
df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90, explode=(0, 0.1))
plt.title('Percentage of Spam vs. Safe (Ham) Messages')
plt.ylabel('')
plt.show()

# Generate and display a Bar Chart for Top 20 most frequent words in spam messages
spam_messages = df[df['label'] == 'spam']['clean_text']
all_spam_words = ' '.join(spam_messages).split()
spam_word_freq = pd.Series(all_spam_words).value_counts().head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=spam_word_freq.values, y=spam_word_freq.index, palette='Reds_r')
plt.title('Top 20 Most Frequent Words in Spam Messages')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()

# ==========================================
# Phase 2: ML Engineer (Baselines)
# ==========================================

# 1. Feature Extraction
print("Extracting features and splitting data...")
# Convert labels to binary (spam=1, ham=0)
X = df['clean_text']
y = df['label'].map({'spam': 1, 'ham': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 2. Model Training
print("Training Baseline Model (MultinomialNB)...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 3. Evaluation & Artifact Export
# Generate predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Generate, plot, and display a Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Spam Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\\nClassification Report:\\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Export the trained model AND the fitted TF-IDF vectorizer
print("Exporting model and vectorizer...")
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Process completed successfully! Artifacts saved.")
"""

notebook.cells.append(nbf.v4.new_code_cell(code))

with open('spam_classification.ipynb', 'w') as f:
    nbf.write(notebook, f)
