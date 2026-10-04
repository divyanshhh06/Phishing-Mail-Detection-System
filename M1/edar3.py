# ============================================
# PHISHING MAIL DETECTION — EDA + FEATURE ENGINEERING
# Member 1: Data Engineer (EDA & NLP)
# Review 3 (final, frozen)
# ============================================
import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path

from features import (
    build_features, clean_text, explain_row,
    save_explain_stats, FEATURE_COLS, FEATURE_LABELS,
)

plt.style.use('default')
sns.set_palette("Set2")

HERE = Path(__file__).parent                       # .../Review3
DATASET_PATH = HERE.parent / "Review1" / "dataset.csv"

print("=" * 60)
print("📊 PHISHING EMAIL EDA — REVIEW 3")
print("=" * 60)

STOP_WORDS = {
    'a','an','the','and','or','but','for','on','at','to','in','is','it',
    'of','with','without','by','from','up','down','off','over','under',
    'etc','i','you','we','they','them','me','him','her','our','my','your',
    'their','his','its','am','are','was','were','be','been','being','have',
    'has','had','do','does','did','will','would','could','should','may',
    'might','must','shall','can','yes','no','so','too',
}

# ============================================
# STEP 1: LOAD DATASET
# ============================================
print(f"\n📂 Loading dataset from:\n   {DATASET_PATH}")
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print(f"✅ Loaded. Shape: {df.shape}")
print(f"📋 Columns: {df.columns.tolist()}")

# ============================================
# STEP 2: IDENTIFY TEXT + LABEL COLUMNS
# ============================================
print("\n🔍 Identifying text/label columns...")
text_col, label_col = None, None
text_possible = ['text','email','content','body','message','cleaned_text','email_text','v2']
label_possible = ['label','target','class','category','is_phishing','phishing','spam','v1']

for col in df.columns:
    c = str(col).lower().strip()
    if c in text_possible and text_col is None:
        text_col = col
    if c in label_possible and label_col is None:
        label_col = col

if text_col is None:
    for col in df.columns:
        if df[col].dtype == 'object':
            text_col = col
            break
if label_col is None:
    for col in df.columns:
        if col != text_col and df[col].dtype in ['int64','float64','int32']:
            label_col = col
            break

if text_col is None or label_col is None:
    raise ValueError(
        f"Could not auto-detect columns. Available: {df.columns.tolist()}"
    )

print(f"✅ Text: '{text_col}'   Label: '{label_col}'")
df = df.rename(columns={text_col: 'text', label_col: 'label'})

# ============================================
# STEP 3: CLEAN LABELS
# ============================================
print("\n🧹 Cleaning labels...")
def clean_label(x):
    x = str(x).lower().strip()
    if x in ['spam','phishing','phish','1','yes','true','y','positive','1.0']:
        return 1
    if x in ['ham','safe','legitimate','0','no','false','n','negative','0.0']:
        return 0
    try:
        return int(float(x))
    except Exception:
        return 0

df['label'] = df['label'].apply(clean_label)
print(df['label'].value_counts())

# ============================================
# STEP 4: CLEAN TEXT
# ============================================
print("\n🧹 Cleaning text...")
df['cleaned_text'] = df['text'].apply(clean_text)
df['text_length'] = df['cleaned_text'].apply(len)
df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
print("✅ Done.")

# ============================================
# STEP 5: APPLY FEATURE PIPELINE FROM features.py
# ============================================
print("\n🧪 Engineering features via features.build_features()...")
_feats_df = df['text'].apply(lambda t: pd.Series(build_features(t)))
df = pd.concat([df, _feats_df], axis=1)

print("\n📊 Feature means by class:")
print(df.groupby('label')[FEATURE_COLS].mean().round(3))

# ============================================
# STEP 6: EXPLAINABILITY (frozen JSON format)
# ============================================
print("\n🔍 Building explainability column...")
_feature_means = df[FEATURE_COLS].mean()
_feature_stds = df[FEATURE_COLS].std().replace(0, 1)

df['explain_json'] = df.apply(
    lambda row: explain_row(row, _feature_means, _feature_stds, top_n=3),
    axis=1,
)

# Persist training stats so M3's live API can reuse the SAME numbers
stats_path = HERE / "explain_stats.json"
save_explain_stats(_feature_means, _feature_stds, stats_path)
print(f"✅ Saved explain stats -> {stats_path}")

if (df['label'] == 1).any():
    sample_idx = df[df['label'] == 1].index[0]
    print(f"\n🔍 Sample explain_json (row {sample_idx}):")
    print("   " + df.loc[sample_idx, 'explain_json'])

# ============================================
# STEP 7: PIE CHART
# ============================================
print("\n📊 Pie chart...")
label_counts = df['label'].value_counts().sort_index()
if 0 not in label_counts.index:
    label_counts[0] = 0
if 1 not in label_counts.index:
    label_counts[1] = 0
label_counts = label_counts.sort_index()

plt.figure(figsize=(10, 7))
plt.pie(label_counts, labels=['✅ Safe', '⚠️ Phishing/Spam'],
        autopct='%1.1f%%', startangle=90,
        colors=['#2ecc71', '#e74c3c'], explode=(0.02, 0.08),
        textprops={'fontsize': 13, 'fontweight': 'bold'}, shadow=True)
plt.title('📧 Email Distribution', fontsize=16, fontweight='bold')
plt.text(1.5, -0.5, f'Total: {len(df)} emails', fontsize=12, ha='center')
plt.tight_layout()
plt.savefig(HERE / 'pie_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ pie_chart.png")

# ============================================
# STEP 8: TOP WORDS BAR CHART
# ============================================
print("\n📊 Top spam words...")
def get_top_words(df, n=20):
    spam_texts = df[df['label'] == 1]['cleaned_text'].str.cat(sep=' ')
    if not spam_texts or len(spam_texts.strip()) == 0:
        return []
    words = [w for w in spam_texts.split() if w not in STOP_WORDS and len(w) > 2]
    return Counter(words).most_common(n)

top_words = get_top_words(df, n=20)
if top_words:
    words, counts = zip(*top_words)
    plt.figure(figsize=(14, 7))
    bars = plt.bar(words, counts, color='#e74c3c', alpha=0.75, edgecolor='black')
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=10)
    plt.title('🔝 Top 20 Words in Phishing/Spam Emails', fontsize=16, fontweight='bold')
    plt.xlabel('Words', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(HERE / 'top_spam_words.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ top_spam_words.png")

# ============================================
# STEP 9: WORD CLOUDS
# ============================================
print("\n📊 Word clouds...")
def create_wordcloud(text_data, title, color, outpath):
    if not text_data or len(text_data.strip()) < 10:
        return
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap=color, max_words=100, random_state=42).generate(text_data)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

create_wordcloud(df[df['label']==1]['cleaned_text'].str.cat(sep=' '),
                 'Phishing_Emails', 'Reds', HERE / 'wordcloud_Phishing_Emails.png')
create_wordcloud(df[df['label']==0]['cleaned_text'].str.cat(sep=' '),
                 'Safe_Emails', 'Greens', HERE / 'wordcloud_Safe_Emails.png')
print("✅ Word clouds saved.")

# ============================================
# STEP 10: SAVE CLEAN DATA FOR ML TEAM
# ============================================
print("\n💾 Saving feature CSV for M2...")
ml_columns = ['cleaned_text', 'label'] + FEATURE_COLS + ['explain_json']
df_ml = df[ml_columns].copy().rename(columns={'cleaned_text': 'text'})

out_file = HERE / 'clean_phishing_data.csv'
df_ml.to_csv(out_file, index=False)
print(f"✅ {out_file}")
print(f"   Shape: {df_ml.shape}")
print(f"   Columns: {df_ml.columns.tolist()}")

if len(df_ml) > 100:
    df_ml.sample(n=100, random_state=42).to_csv(HERE / 'sample_data.csv', index=False)
    print("✅ sample_data.csv (100 rows)")

# ============================================
# FINAL SUMMARY
# ============================================
total = len(df)
spam_count = int(df['label'].sum())
safe_count = total - spam_count
print("\n" + "=" * 70)
print("📊 EDA COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
Dataset:
  Total emails : {total:,}
  Phishing/Spam: {spam_count:,} ({spam_count/total*100:.1f}%)
  Safe         : {safe_count:,} ({safe_count/total*100:.1f}%)

Engineered features (frozen, 8 total):
  {', '.join(FEATURE_COLS)}

Explainability:
  Format: {{"triggers":[{{"feature","label","weight"}}, ...]}}
  Stats : explain_stats.json

Files written to {HERE}:
  pie_chart.png
  top_spam_words.png
  wordcloud_Phishing_Emails.png
  wordcloud_Safe_Emails.png
  clean_phishing_data.csv   <- send to M2
  sample_data.csv
  explain_stats.json        <- send to M3
""")
print("=" * 70)
print("✅ Review 3 feature engineering complete.")
print("=" * 70)