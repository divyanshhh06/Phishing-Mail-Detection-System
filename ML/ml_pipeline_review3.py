import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import hstack
import warnings

warnings.filterwarnings('ignore')

def simulate_m1_features_review3(df):
    """
    Simulates the Data Engineer (M1) features as per Review 2 & 3 requirements.
    Review 2: link_count, urgency_density, sender_mismatch
    Review 3: domain-level checks (lookalike_domains, ip_based_links, url_shorteners)
    """
    print("Simulating M1 features for Review 3...")
    
    # Review 2 Features
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['link_count'] = df['text'].apply(lambda x: len(re.findall(url_pattern, str(x).lower())))
    
    urgency_words = ['urgent', 'immediate', 'action required', 'act now', 'important', 'alert', 'suspended', 'verify', 'account']
    def count_urgency(text):
        text = str(text).lower()
        count = sum(1 for word in urgency_words if word in text)
        return count / (len(text.split()) + 1)
    df['urgency_density'] = df['text'].apply(count_urgency)
    
    np.random.seed(42)
    def simulate_mismatch(is_spam):
        return np.random.choice([0, 1], p=[0.3, 0.7]) if is_spam else np.random.choice([0, 1], p=[0.9, 0.1])
    df['sender_mismatch'] = df['label'].map(lambda x: simulate_mismatch(x == 'spam'))
    
    # Review 3 Features
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd']
    df['has_url_shortener'] = df['text'].apply(lambda x: 1 if any(s in str(x).lower() for s in shorteners) else 0)
    
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    df['has_ip_link'] = df['text'].apply(lambda x: 1 if re.search(ip_pattern, str(x)) else 0)
    
    df['has_lookalike_domain'] = df['label'].map(lambda x: np.random.choice([0, 1], p=[0.4, 0.6]) if x == 'spam' else np.random.choice([0, 1], p=[0.95, 0.05]))
    
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv('SMS Spam Collection Dataset/spam.csv', encoding='latin-1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    df = df.iloc[:, :2]
    df.columns = ['label', 'text']
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    
    df = simulate_m1_features_review3(df)
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("Splitting data (including held-out test set)...")
    X_text = df['clean_text']
    numeric_features = ['link_count', 'urgency_density', 'sender_mismatch', 'has_url_shortener', 'has_ip_link', 'has_lookalike_domain']
    X_num = df[numeric_features].values
    y = df['label_num'].values
    
    # Split into train+val and held-out test set
    X_train_val_text, X_test_text, X_train_val_num, X_test_num, y_train_val, y_test = train_test_split(
        X_text, X_num, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_val_tfidf = tfidf.fit_transform(X_train_val_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    X_train_val_combined = hstack([X_train_val_tfidf, X_train_val_num])
    X_test_combined = hstack([X_test_tfidf, X_test_num])
    
    print("Performing final tuning pass (GridSearchCV on Random Forest)...")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_val_combined, y_train_val)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    print("Evaluating on held-out test set...")
    y_pred = best_model.predict(X_test_combined)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Final Model')
    plt.tight_layout()
    plt.savefig('confusion_matrix_review3.png')
    print("Saved confusion_matrix_review3.png")
    
    # Feature Importance Chart
    importances = best_model.feature_importances_
    tfidf_feature_names = tfidf.get_feature_names_out()
    all_feature_names = list(tfidf_feature_names) + numeric_features
    
    # Top 20 features
    indices = np.argsort(importances)[::-1][:20]
    top_features = [all_feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance_review3.png')
    print("Saved feature_importance_review3.png")
    
    # Results Table
    results_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [acc, prec, rec, f1]
    })
    results_table.to_csv('final_results_table_review3.csv', index=False)
    print("Saved final_results_table_review3.csv")
    
    print("Freezing the final model...")
    joblib.dump(best_model, 'frozen_model_review3.pkl')
    joblib.dump(tfidf, 'frozen_tfidf_review3.pkl')
    
    print("Review 3 tasks completed successfully.")

if __name__ == '__main__':
    main()
