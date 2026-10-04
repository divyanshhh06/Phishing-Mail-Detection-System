# ============================================
# PHISHING MAIL DETECTION - EDA & DATA PREPARATION
# Member 1: Data Engineer (EDA & NLP)
# Review 1 Deliverable
# Dataset: dataset
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import os

# Set style
plt.style.use('default')
sns.set_palette("Set2")

print("=" * 60)
print("📊 PHISHING EMAIL EDA - REVIEW 1")
print("=" * 60)

# Custom stopwords (no NLTK needed)
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'for', 'on', 'at', 'to', 'in',
    'is', 'it', 'of', 'with', 'without', 'by', 'from', 'up', 'down',
    'off', 'over', 'under', 'etc', 'i', 'you', 'we', 'they', 'them',
    'me', 'him', 'her', 'our', 'my', 'your', 'their', 'his', 'its',
    'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'yes', 'no', 'so', 'too'
}

# ============================================
# STEP 1: LOAD DATASET
# ============================================

print("\n📂 Loading dataset...")

filename = 'dataset'

# Try different extensions
possible_files = ['dataset', 'dataset.csv', 'dataset.txt', 'dataset.xlsx', 'dataset.parquet']

df = None
loaded_file = None

for file in possible_files:
    if os.path.exists(file):
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.endswith('.xlsx'):
                df = pd.read_excel(file)
            elif file.endswith('.parquet'):
                df = pd.read_parquet(file)
            else:
                # Try as CSV if no extension
                try:
                    df = pd.read_csv(file)
                except:
                    # Try as tab-separated
                    try:
                        df = pd.read_csv(file, sep='\t')
                    except:
                        continue
            loaded_file = file
            break
        except Exception as e:
            print(f"⚠️ Could not load {file}: {e}")

if df is None:
    print(f"❌ File not found: {filename}")
    print(f"📁 Current folder: {os.getcwd()}")
    print("\n📋 Files in current folder:")
    for file in os.listdir():
        print(f"   - {file}")
    exit()

print(f"✅ Loaded: {loaded_file}")
print(f"📊 Dataset shape: {df.shape}")

# ============================================
# STEP 2: EXPLORE DATASET
# ============================================

print("\n🔍 Exploring dataset...")
print(f"\n📋 Columns: {df.columns.tolist()}")
print(f"\n🔢 First 2 rows:")
print(df.head(2))

# ============================================
# STEP 3: FIND TEXT & LABEL COLUMNS
# ============================================

print("\n🔍 Identifying text and label columns...")

text_col = None
label_col = None

# Common column names for text
text_possible = ['text', 'email', 'content', 'body', 'message', 'cleaned_text', 'email_text', 'v2']
# Common column names for labels
label_possible = ['label', 'target', 'class', 'category', 'is_phishing', 'phishing', 'spam', 'v1']

for col in df.columns:
    col_lower = str(col).lower().strip()
    if col_lower in text_possible:
        text_col = col
    if col_lower in label_possible:
        label_col = col

# If not found, guess
if text_col is None:
    for col in df.columns:
        if df[col].dtype == 'object':
            text_col = col
            break

if label_col is None:
    for col in df.columns:
        if col != text_col and df[col].dtype in ['int64', 'float64', 'int32']:
            label_col = col
            break

if text_col is None:
    print("❌ Could not find text column!")
    print(f"📋 Available columns: {df.columns.tolist()}")
    print("\nPlease tell me: Which column contains the email text?")
    print("Example: 'text', 'email', 'content', 'body', 'message'")
    exit()

if label_col is None:
    print("❌ Could not find label column!")
    print(f"📋 Available columns: {df.columns.tolist()}")
    print("\nPlease tell me: Which column contains the label (0/1)?")
    print("Example: 'label', 'target', 'class', 'category', 'spam'")
    exit()

print(f"✅ Text column: '{text_col}'")
print(f"✅ Label column: '{label_col}'")

# Rename to standard names
df = df.rename(columns={text_col: 'text', label_col: 'label'})

# ============================================
# STEP 4: CLEAN LABELS
# ============================================

print("\n🧹 Cleaning labels...")
print(f"Unique label values: {df['label'].unique()}")

def clean_label(x):
    x = str(x).lower().strip()
    if x in ['spam', 'phishing', 'phish', '1', 'yes', 'true', 'y', 'positive', '1.0']:
        return 1
    elif x in ['ham', 'safe', 'legitimate', '0', 'no', 'false', 'n', 'negative', '0.0']:
        return 0
    else:
        try:
            return int(float(x))
        except:
            return 0

df['label'] = df['label'].apply(clean_label)

print(f"\n📊 Label distribution:")
print(df['label'].value_counts())

# ============================================
# STEP 5: CLEAN TEXT
# ============================================

print("\n🧹 Cleaning text...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
df['text_length'] = df['cleaned_text'].apply(len)
df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))

print("✅ Text cleaning complete!")
if len(df) > 0:
    sample = df['cleaned_text'].iloc[0][:200]
    print(f"\n📝 Sample cleaned text:\n{sample}...")

# ============================================
# STEP 6: PIE CHART
# ============================================

print("\n📊 Creating pie chart...")

plt.figure(figsize=(10, 7))
label_counts = df['label'].value_counts().sort_index()

# Make sure both labels exist
if len(label_counts) < 2:
    if 0 not in label_counts.index:
        label_counts[0] = 0
    if 1 not in label_counts.index:
        label_counts[1] = 0
    label_counts = label_counts.sort_index()

labels = ['✅ Safe', '⚠️ Phishing/Spam']
colors = ['#2ecc71', '#e74c3c']
explode = (0.02, 0.08)

plt.pie(label_counts, labels=labels, autopct='%1.1f%%', 
        startangle=90, colors=colors, explode=explode,
        textprops={'fontsize': 13, 'fontweight': 'bold'},
        shadow=True)

plt.title('📧 Email Distribution', fontsize=16, fontweight='bold')
plt.text(1.5, -0.5, f'Total: {len(df)} emails', fontsize=12, ha='center')

plt.tight_layout()
plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Pie chart saved as 'pie_chart.png'")
print(f"Safe: {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
print(f"Phishing: {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")

# ============================================
# STEP 7: TOP SPAM WORDS BAR CHART
# ============================================

print("\n📊 Creating top words bar chart...")

def get_top_words(df, n=20):
    spam_texts = df[df['label'] == 1]['cleaned_text'].str.cat(sep=' ')
    if not spam_texts or len(spam_texts.strip()) == 0:
        return []
    words = spam_texts.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return Counter(words).most_common(n)

top_words = get_top_words(df, n=20)

if top_words:
    plt.figure(figsize=(14, 7))
    words, counts = zip(*top_words)
    bars = plt.bar(words, counts, color='#e74c3c', alpha=0.75, edgecolor='black')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=10)
    
    plt.title('🔝 Top 20 Words in Phishing/Spam Emails', fontsize=16, fontweight='bold')
    plt.xlabel('Words', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('top_spam_words.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Bar chart saved as 'top_spam_words.png'")
    
    print("\n🔍 Top 10 phishing indicator words:")
    for i, (word, count) in enumerate(top_words[:10], 1):
        print(f"  {i}. '{word}' - {count} times")
else:
    print("⚠️ No spam words found to display")

# ============================================
# STEP 8: WORD CLOUDS
# ============================================

print("\n📊 Creating word clouds...")

def create_wordcloud(text_data, title, color):
    if not text_data or len(text_data.strip()) < 10:
        print(f"⚠️ Skipping {title} - not enough text")
        return
    
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap=color,
                         max_words=100,
                         random_state=42).generate(text_data)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'wordcloud_{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Word cloud saved: wordcloud_{title.replace(' ', '_')}.png")

# Phishing word cloud
spam_text = df[df['label'] == 1]['cleaned_text'].str.cat(sep=' ')
create_wordcloud(spam_text, 'Phishing_Emails', 'Reds')

# Safe word cloud
safe_text = df[df['label'] == 0]['cleaned_text'].str.cat(sep=' ')
create_wordcloud(safe_text, 'Safe_Emails', 'Greens')

# ============================================
# STEP 9: SAVE CLEAN DATA FOR ML TEAM
# ============================================

print("\n💾 Saving clean dataset for ML Engineer...")

df_ml = df[['cleaned_text', 'label']].copy()
df_ml = df_ml.rename(columns={'cleaned_text': 'text'})

# Save full dataset
output_file = 'clean_phishing_data.csv'
df_ml.to_csv(output_file, index=False)
print(f"✅ Saved: {output_file}")
print(f"📊 Shape: {df_ml.shape}")

# Save sample
if len(df_ml) > 100:
    df_ml.sample(n=min(100, len(df_ml)), random_state=42).to_csv('sample_data.csv', index=False)
    print("✅ Saved: sample_data.csv (100 rows for testing)")

# ============================================
# STEP 10: FINAL SUMMARY
# ============================================

print("\n" + "=" * 70)
print("📊 EDA COMPLETE - SUMMARY")
print("=" * 70)

total = len(df)
spam_count = df['label'].sum()
safe_count = total - spam_count

print(f"""
📁 Dataset Information:
   ├── Source: dataset
   ├── Total emails: {total:,}
   ├── Phishing/Spam: {spam_count:,} ({spam_count/total*100:.1f}%)
   └── Safe: {safe_count:,} ({safe_count/total*100:.1f}%)

🔍 Key Findings:
   ├── Top phishing words: {', '.join([w for w, _ in top_words[:5]]) if top_words else 'N/A'}
   ├── Avg phishing length: {df[df['label']==1]['text_length'].mean():.0f} chars
   └── Avg safe length: {df[df['label']==0]['text_length'].mean():.0f} chars

📁 Files Generated:
   ├── pie_chart.png
   ├── top_spam_words.png
   ├── wordcloud_Phishing_Emails.png
   ├── wordcloud_Safe_Emails.png
   ├── clean_phishing_data.csv  ← For Member 2 (ML Engineer)
   └── sample_data.csv
""")

print("=" * 70)
print("✅ Review 1 Deliverable Complete!")
print("📤 Share 'clean_phishing_data.csv' with Member 2")
print("=" * 70)

print("\n🎉 All Done!")