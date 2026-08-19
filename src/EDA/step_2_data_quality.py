"""
EDA Step 4.2 — Data Quality Assessment
========================================
Purpose: Identify missing values, duplicates, empty emails, encoding issues.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'emails.csv'))
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("EDA STEP 4.2 — DATA QUALITY ASSESSMENT")
print("=" * 70)

# =============================================================================
# 1. Missing Values
# =============================================================================
print("\n" + "-" * 70)
print("[1] MISSING VALUES")
print("-" * 70)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
for col in df.columns:
    status = "CLEAN" if missing[col] == 0 else f"WARNING: {missing[col]} missing ({missing_pct[col]}%)"
    print(f"   {col:<20} → {status}")
print(f"\n   Total missing cells: {df.isnull().sum().sum()}")

# =============================================================================
# 2. Duplicate Rows
# =============================================================================
print("\n" + "-" * 70)
print("[2] DUPLICATE ROWS")
print("-" * 70)
n_dup = df.duplicated().sum()
n_dup_all = df.duplicated(keep=False).sum()
print(f"   Exact duplicate rows:           {n_dup}")
print(f"   Total rows involved:            {n_dup_all}")
print(f"   Unique rows:                    {len(df) - n_dup}")
print(f"   Dataset after dedup:            {len(df) - n_dup} rows")

if n_dup > 0:
    dup_by_class = df[df.duplicated(keep=False)].groupby('spam').size()
    print(f"\n   Duplicates by class:")
    for label, count in dup_by_class.items():
        label_name = "Spam" if label == 1 else "Ham"
        print(f"      {label_name} (spam={label}): {count} rows involved")

    # Show some actual duplicates
    print(f"\n   Sample duplicates (showing text that appears >1 time):")
    dup_texts = df[df.duplicated(subset='text', keep=False)].groupby('text').size()
    dup_texts = dup_texts[dup_texts > 1].sort_values(ascending=False).head(5)
    for i, (text, count) in enumerate(dup_texts.items()):
        print(f"      [{i+1}] Appears {count}x: \"{text[:100]}...\"")

# =============================================================================
# 3. Empty / Near-Empty Emails
# =============================================================================
print("\n" + "-" * 70)
print("[3] EMPTY / NEAR-EMPTY EMAILS")
print("-" * 70)

# Truly empty
n_null_text = df['text'].isna().sum()
print(f"   Null text entries:              {n_null_text}")

# Whitespace-only
n_whitespace = (df['text'].str.strip() == '').sum()
print(f"   Whitespace-only entries:        {n_whitespace}")

# Very short emails (< 20 chars)
short_threshold = 20
n_very_short = (df['text'].str.len() < short_threshold).sum()
print(f"   Emails < {short_threshold} chars:             {n_very_short}")

if n_very_short > 0:
    short_emails = df[df['text'].str.len() < short_threshold]
    print(f"\n   Shortest emails:")
    for idx, row in short_emails.head(5).iterrows():
        print(f"      [{idx}] spam={row['spam']}  len={len(row['text'])}  text=\"{row['text']}\"")

# Very short by word count (< 5 words)
word_counts = df['text'].str.split().str.len()
n_few_words = (word_counts < 5).sum()
print(f"\n   Emails < 5 words:               {n_few_words}")
if n_few_words > 0:
    few_word_emails = df[word_counts < 5]
    for idx, row in few_word_emails.head(5).iterrows():
        wc = len(str(row['text']).split())
        print(f"      [{idx}] spam={row['spam']}  words={wc}  text=\"{row['text'][:120]}\"")

# =============================================================================
# 4. Encoding & Content Issues
# =============================================================================
print("\n" + "-" * 70)
print("[4] CONTENT PATTERN CHECKS")
print("-" * 70)

# Check for HTML tags
html_pattern = df['text'].str.contains(r'<[^>]+>', regex=True, na=False)
n_html = html_pattern.sum()
print(f"   Emails with HTML tags:          {n_html} ({n_html/len(df)*100:.1f}%)")

# Check for URLs
url_pattern = df['text'].str.contains(r'http[s]?://|www\.', regex=True, na=False)
n_urls = url_pattern.sum()
print(f"   Emails with URLs:               {n_urls} ({n_urls/len(df)*100:.1f}%)")

# Check for email addresses
email_pattern = df['text'].str.contains(r'[\w.+-]+@[\w-]+\.[\w.-]+', regex=True, na=False)
n_email_addr = email_pattern.sum()
print(f"   Emails with email addresses:    {n_email_addr} ({n_email_addr/len(df)*100:.1f}%)")

# Check for numbers / dollar signs
dollar_pattern = df['text'].str.contains(r'\$', regex=True, na=False)
n_dollar = dollar_pattern.sum()
print(f"   Emails with '$' symbol:         {n_dollar} ({n_dollar/len(df)*100:.1f}%)")

# Check for excessive exclamation marks
excl_pattern = df['text'].str.contains(r'!{2,}', regex=True, na=False)
n_excl = excl_pattern.sum()
print(f"   Emails with '!!' or more:       {n_excl} ({n_excl/len(df)*100:.1f}%)")

# Check for ALL CAPS words (3+ consecutive uppercase words)
caps_pattern = df['text'].str.contains(r'\b[A-Z]{3,}\b', regex=True, na=False)
n_caps = caps_pattern.sum()
print(f"   Emails with ALL-CAPS words:     {n_caps} ({n_caps/len(df)*100:.1f}%)")

# =============================================================================
# 5. Label Integrity Check
# =============================================================================
print("\n" + "-" * 70)
print("[5] LABEL INTEGRITY CHECK")
print("-" * 70)
unique_labels = sorted(df['spam'].unique())
print(f"   Unique label values:            {unique_labels}")
print(f"   Expected:                       [0, 1]")
print(f"   Match:                          {'YES' if unique_labels == [0, 1] else 'NO - ISSUE!'}")

non_binary = df[~df['spam'].isin([0, 1])]
print(f"   Non-binary label rows:          {len(non_binary)}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4.2 — DATA QUALITY SUMMARY")
print("=" * 70)

issues = []
if df.isnull().sum().sum() > 0:
    issues.append(f"Missing values: {df.isnull().sum().sum()}")
if n_dup > 0:
    issues.append(f"Duplicates: {n_dup} rows to remove")
if n_very_short > 0:
    issues.append(f"Very short emails (<{short_threshold} chars): {n_very_short}")
if n_html > 0:
    issues.append(f"HTML content: {n_html} emails need HTML stripping")

if issues:
    print("\n   Issues found:")
    for issue in issues:
        print(f"   [!] {issue}")
else:
    print("\n   No issues found - dataset is clean!")

print(f"\n   Recommended actions:")
print(f"   1. Remove {n_dup} duplicate rows before splitting")
print(f"   2. {'Strip HTML tags from ' + str(n_html) + ' emails' if n_html > 0 else 'No HTML stripping needed'}")
print(f"   3. {'Investigate ' + str(n_very_short) + ' very short emails' if n_very_short > 0 else 'No short email issues'}")
print(f"   4. Clean URLs and email addresses during preprocessing")
print()
