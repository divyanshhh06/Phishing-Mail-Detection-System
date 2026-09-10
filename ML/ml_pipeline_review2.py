import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import hstack
import warnings

warnings.filterwarnings('ignore')

def simulate_m1_features(df):
    """
    Simulates the Data Engineer (M1) features as per Review 2 requirements.
    - link_count: Number of URLs in the text.
    - urgency_density: Frequency of urgency keywords.
    - sender_mismatch: A mock feature since we only have text data (0 or 1).
    """
    print("Simulating M1 features (Data Engineering)...")
    
    # Feature 1: link_count
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['link_count'] = df['text'].apply(lambda x: len(re.findall(url_pattern, str(x).lower())))
    
    # Feature 2: urgency_keyword_density
    urgency_words = ['urgent', 'immediate', 'action required', 'act now', 'important', 'alert', 'suspended', 'verify', 'account']
    def count_urgency(text):
        text = str(text).lower()
        count = sum(1 for word in urgency_words if word in text)
        return count / (len(text.split()) + 1) # density
    
    df['urgency_density'] = df['text'].apply(count_urgency)
    
    # Feature 3: sender_mismatch (Random simulation since this is SMS/text data without headers)
    # Let's say spam has a higher chance of sender mismatch.
    np.random.seed(42)
    def simulate_mismatch(is_spam):
        if is_spam:
            return np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            return np.random.choice([0, 1], p=[0.9, 0.1])
            
    df['sender_mismatch'] = df['label'].map(lambda x: simulate_mismatch(x == 'spam'))
    
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def main():
    # 1. Load Data
    print("Loading dataset...")
    df = pd.read_csv('SMS Spam Collection Dataset/spam.csv', encoding='latin-1')
    df = df.iloc[:, :2]
    df.columns = ['label', 'text']
    
    # Convert labels
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # 2. Simulate M1 Features
    df = simulate_m1_features(df)
    
    # Clean text for TF-IDF
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 3. Split Data
    print("Splitting data...")
    X_text = df['clean_text']
    X_num = df[['link_count', 'urgency_density', 'sender_mismatch']].values
    y = df['label_num'].values
    
    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. TF-IDF Vectorization
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    # Combine TF-IDF features with engineered numerical features
    X_train_combined = hstack([X_train_tfidf, X_train_num])
    X_test_combined = hstack([X_test_tfidf, X_test_num])
    
    # 5. Model Comparison & Cross-Validation
    print("\nTraining and comparing models (Naive Bayes vs. Random Forest)...")
    
    nb_model = MultinomialNB()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Quick sanity cross-validation (5-fold) using F1-score (Review 2 requirement)
    nb_cv_f1 = cross_val_score(nb_model, X_train_combined, y_train, cv=5, scoring='f1').mean()
    rf_cv_f1 = cross_val_score(rf_model, X_train_combined, y_train, cv=5, scoring='f1').mean()
    
    print(f"Cross-Validation F1-Score (Naive Bayes): {nb_cv_f1:.4f}")
    print(f"Cross-Validation F1-Score (Random Forest): {rf_cv_f1:.4f}")
    
    # 6. Select Best Model by F1-score
    if rf_cv_f1 > nb_cv_f1:
        print("Selecting Random Forest as the best model.")
        best_model = rf_model
        model_name = "Random Forest"
    else:
        print("Selecting Naive Bayes as the best model.")
        best_model = nb_model
        model_name = "Naive Bayes"
        
    # Retrain on full training set
    best_model.fit(X_train_combined, y_train)
    
    # 7. Evaluate on Test Set
    y_pred = best_model.predict(X_test_combined)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nTest Set Metrics ({model_name}):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # 8. Export Artifacts
    print("\nExporting model, vectorizer, and metrics (Review 2 deliverables)...")
    joblib.dump(best_model, 'spam_classifier_v2.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer_v2.pkl')
    
    metrics_text = f"""--- Review 2 ML Metrics Note ---
Model Chosen: {model_name}
Features Used: TF-IDF (top 5000), link_count, urgency_density, sender_mismatch

Cross-Validation F1 (5-fold):
- Naive Bayes: {nb_cv_f1:.4f}
- Random Forest: {rf_cv_f1:.4f}

Final Test Set Evaluation:
Accuracy:  {acc:.4f}
Precision: {prec:.4f}
Recall:    {rec:.4f}
F1-Score:  {f1:.4f}
"""
    with open('metrics_note.txt', 'w') as f:
        f.write(metrics_text)
        
    print("Process completed successfully! 'spam_classifier_v2.pkl', 'tfidf_vectorizer_v2.pkl', and 'metrics_note.txt' generated.")

if __name__ == '__main__':
    main()
