"""
EDA Step 4.4 & 4.5 — Text Content Analysis + Statistical Summary
=================================================================
Purpose: Analyze email content patterns, word frequencies, special characters,
         length distributions, and outlier detection.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import Counter
import re

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'emails.csv'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'evals', 'eda_plots'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# ---- Precompute text features ----
df['char_count'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print("=" * 70)
print("EDA STEP 4.4 — TEXT CONTENT ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Email Length Distribution
# =============================================================================
print("\n" + "-" * 70)
print("[1] EMAIL LENGTH DISTRIBUTION BY CLASS")
print("-" * 70)

for label, name in [(0, 'Ham'), (1, 'Spam')]:
    subset = df[df['spam'] == label]
    chars = subset['char_count']
    words = subset['word_count']
    print(f"\n   --- {name} (n={len(subset):,}) ---")
    print(f"   Characters:  mean={chars.mean():,.1f}  median={chars.median():,.1f}  std={chars.std():,.1f}  min={chars.min():,}  max={chars.max():,}")
    print(f"   Words:       mean={words.mean():,.1f}  median={words.median():,.1f}  std={words.std():,.1f}  min={words.min():,}  max={words.max():,}")

# Comparison
print(f"\n   --- Comparison ---")
spam_avg_words = df[df['spam']==1]['word_count'].mean()
ham_avg_words = df[df['spam']==0]['word_count'].mean()
ratio = ham_avg_words / spam_avg_words if spam_avg_words > 0 else float('inf')
print(f"   Ham avg words:    {ham_avg_words:.1f}")
print(f"   Spam avg words:   {spam_avg_words:.1f}")
print(f"   Ham/Spam ratio:   {ratio:.2f}x (ham emails are {'longer' if ratio > 1 else 'shorter'})")

# =============================================================================
# 2. Word Frequency Analysis
# =============================================================================
print("\n" + "-" * 70)
print("[2] TOP 20 MOST FREQUENT WORDS BY CLASS")
print("-" * 70)

def get_word_freq(texts: pd.Series, top_n: int = 20) -> list[tuple[str, int]]:
    """Count word frequencies from a series of text documents."""
    all_words: list[str] = []
    for text in texts.dropna():
        words = text.lower().split()
        all_words.extend(words)
    return Counter(all_words).most_common(top_n)

# Common English stop words (small set for EDA only)
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'was', 'be', 'are', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'not',
    'no', 'if', 'as', 'so', 'than', 'then', 'just', 'about', 'up',
    'out', 'all', 'more', 'also', 'very', 'what', 'when', 'where',
    'how', 'who', 'which', 'there', '-', '--', ':', '.', ',', '!', '?',
    'subject:', 're', 'fw', 'subject'
}

def get_word_freq_no_stop(texts: pd.Series, top_n: int = 20) -> list[tuple[str, int]]:
    """Count word frequencies excluding stop words."""
    all_words: list[str] = []
    for text in texts.dropna():
        words = [w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 2]
        all_words.extend(words)
    return Counter(all_words).most_common(top_n)

# With stop words (raw)
print("\n   --- Top 20 Words (with stop words) ---")
for label, name in [(1, 'SPAM'), (0, 'HAM')]:
    subset = df[df['spam'] == label]
    top_words = get_word_freq(subset['text'], 20)
    print(f"\n   {name}:")
    for rank, (word, count) in enumerate(top_words, 1):
        print(f"      {rank:>2}. {word:<20} {count:>7,}")

# Without stop words (more meaningful)
print("\n   --- Top 20 Words (stop words removed) ---")
spam_top = get_word_freq_no_stop(df[df['spam']==1]['text'], 20)
ham_top = get_word_freq_no_stop(df[df['spam']==0]['text'], 20)

print("\n   SPAM keywords:")
for rank, (word, count) in enumerate(spam_top, 1):
    print(f"      {rank:>2}. {word:<20} {count:>7,}")

print("\n   HAM keywords:")
for rank, (word, count) in enumerate(ham_top, 1):
    print(f"      {rank:>2}. {word:<20} {count:>7,}")

# =============================================================================
# 3. Spam Trigger Keywords
# =============================================================================
print("\n" + "-" * 70)
print("[3] SPAM TRIGGER KEYWORD PRESENCE")
print("-" * 70)

trigger_words = ['free', 'winner', 'click', 'urgent', 'offer', 'deal', 'discount',
                 'buy', 'cash', 'money', 'price', 'order', 'limited',
                 'act', 'now', 'guaranteed', 'credit', 'viagra', 'pills',
                 'unsubscribe', 'subscribe', 'congratulations', 'won']

print(f"\n   {'Keyword':<20} {'In Spam':<12} {'In Ham':<12} {'Spam%':<10} {'Ham%':<10} {'Ratio'}")
print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")

spam_texts = df[df['spam']==1]['text'].str.lower()
ham_texts = df[df['spam']==0]['text'].str.lower()
n_spam = len(spam_texts)
n_ham = len(ham_texts)

for word in trigger_words:
    in_spam = spam_texts.str.contains(r'\b' + word + r'\b', regex=True, na=False).sum()
    in_ham = ham_texts.str.contains(r'\b' + word + r'\b', regex=True, na=False).sum()
    spam_pct = in_spam / n_spam * 100
    ham_pct = in_ham / n_ham * 100
    ratio = spam_pct / ham_pct if ham_pct > 0 else float('inf')
    marker = "***" if ratio > 5 else "**" if ratio > 2 else ""
    print(f"   {word:<20} {in_spam:<12} {in_ham:<12} {spam_pct:<10.1f} {ham_pct:<10.1f} {ratio:>5.1f}x {marker}")

# =============================================================================
# 4. Special Character & Pattern Analysis
# =============================================================================
print("\n" + "-" * 70)
print("[4] SPECIAL CHARACTER & PATTERN ANALYSIS")
print("-" * 70)

# Compute ratios
df['special_char_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`'))
df['special_char_ratio'] = df['special_char_count'] / df['char_count']
df['upper_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
df['upper_ratio'] = df['upper_count'] / df['char_count']
df['digit_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isdigit()))
df['digit_ratio'] = df['digit_count'] / df['char_count']
df['excl_count'] = df['text'].str.count('!')

for feature, desc in [
    ('special_char_ratio', 'Special char ratio'),
    ('upper_ratio', 'Uppercase ratio'),
    ('digit_ratio', 'Digit ratio'),
    ('excl_count', 'Exclamation marks count')
]:
    spam_mean = df[df['spam']==1][feature].mean()
    ham_mean = df[df['spam']==0][feature].mean()
    ratio = spam_mean / ham_mean if ham_mean > 0 else float('inf')
    print(f"\n   {desc}:")
    print(f"      Spam mean: {spam_mean:.4f}")
    print(f"      Ham mean:  {ham_mean:.4f}")
    print(f"      Spam/Ham:  {ratio:.2f}x {'(Spam higher)' if ratio > 1 else '(Ham higher)'}")

# =============================================================================
# STEP 4.5 — STATISTICAL SUMMARY & OUTLIER DETECTION
# =============================================================================
print("\n\n" + "=" * 70)
print("EDA STEP 4.5 — STATISTICAL SUMMARY & OUTLIER DETECTION")
print("=" * 70)

# =============================================================================
# 5. Descriptive Statistics
# =============================================================================
print("\n" + "-" * 70)
print("[5] DESCRIPTIVE STATISTICS TABLE")
print("-" * 70)

stats_cols = ['char_count', 'word_count', 'special_char_ratio', 'upper_ratio', 'digit_ratio', 'excl_count']
desc = df[stats_cols].describe()
print(f"\n{desc.to_string()}")

# Per-class stats
print("\n   --- Per-Class Statistics (word_count) ---")
for label, name in [(0, 'Ham'), (1, 'Spam')]:
    subset = df[df['spam']==label]['word_count']
    print(f"\n   {name}:")
    print(f"      Mean:    {subset.mean():>10,.1f}")
    print(f"      Median:  {subset.median():>10,.1f}")
    print(f"      Std:     {subset.std():>10,.1f}")
    print(f"      Skew:    {subset.skew():>10,.2f}")
    print(f"      Q1:      {subset.quantile(0.25):>10,.1f}")
    print(f"      Q3:      {subset.quantile(0.75):>10,.1f}")
    print(f"      IQR:     {subset.quantile(0.75) - subset.quantile(0.25):>10,.1f}")

# =============================================================================
# 6. Outlier Detection
# =============================================================================
print("\n" + "-" * 70)
print("[6] OUTLIER DETECTION (IQR Method)")
print("-" * 70)

for feature, desc_name in [('char_count', 'Character Count'), ('word_count', 'Word Count')]:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    n_below = (df[feature] < lower).sum()
    n_above = (df[feature] > upper).sum()
    total_outliers = n_below + n_above

    print(f"\n   {desc_name}:")
    print(f"      Q1 = {q1:,.1f}  |  Q3 = {q3:,.1f}  |  IQR = {iqr:,.1f}")
    print(f"      Lower fence: {max(0, lower):,.1f}  |  Upper fence: {upper:,.1f}")
    print(f"      Outliers below: {n_below}  |  Outliers above: {n_above}")
    print(f"      Total outliers: {total_outliers} ({total_outliers/len(df)*100:.1f}%)")

    if n_above > 0:
        outliers = df[df[feature] > upper].sort_values(feature, ascending=False)
        print(f"      Top 3 longest emails:")
        for idx, row in outliers.head(3).iterrows():
            print(f"         [{idx}] spam={row['spam']}  {feature}={row[feature]:,.0f}  text=\"{str(row['text'])[:80]}...\"")

# Very short email threshold
print(f"\n   --- Short Email Check ---")
short_chars = df[df['char_count'] < 50]
print(f"   Emails < 50 chars:  {len(short_chars)}")
short_words = df[df['word_count'] < 10]
print(f"   Emails < 10 words:  {len(short_words)}")

print("\n" + "=" * 70)
print("STEPS 4.4 & 4.5 COMPLETE")
print("=" * 70)
print()
