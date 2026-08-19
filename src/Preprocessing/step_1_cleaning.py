"""
Preprocessing Step 1: Data Cleaning & Feature Engineering
===========================================================
Purpose: Remove duplicates, extract raw features (excl_count, word_count),
         and clean text (lowercase, remove HTML/URLs/punctuation/stopwords).
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import re
import os

# Set paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'emails.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PREPROCESSING STEP 1 — DATA CLEANING & FEATURE ENGINEERING")
print("=" * 70)

# 1. Load Data
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
initial_shape = df.shape
print(f"    Initial shape: {initial_shape[0]:,} rows × {initial_shape[1]} columns")

# 2. Remove Duplicates
print("\n[2] Removing duplicates...")
n_duplicates = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"    Removed {n_duplicates} duplicate rows.")
print(f"    Current shape: {df.shape[0]:,} rows")

# 3. Feature Extraction (Before heavy cleaning)
print("\n[3] Extracting engineered features...")
# Count exclamation marks
df['excl_count'] = df['text'].str.count('!')
print("    - Extracted 'excl_count'")

# Count words (rough estimate by splitting on spaces)
df['word_count'] = df['text'].str.split().str.len()
print("    - Extracted 'word_count'")

# 4. Text Cleaning
print("\n[4] Cleaning text data...")

# Common English stop words (expanded from EDA)
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'was', 'be', 'are', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'not',
    'no', 'if', 'as', 'so', 'than', 'then', 'just', 'about', 'up',
    'out', 'all', 'more', 'also', 'very', 'what', 'when', 'where',
    'how', 'who', 'which', 'there', 'subject', 're', 'fw', 'any', 'been', 
    'each', 'new', 'one', 'two', 'get', 'like', 'know', 'please', 'would', 
    'need', 'am', 'pm'
}

def clean_text(text: str) -> str:
    """Applies all text cleaning steps to a single string."""
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    
    # Remove "subject:" prefix (very common in this dataset)
    text = re.sub(r'^subject:\s*', ' ', text)
    
    # Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove stop words and short words (length < 2)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    
    # Rejoin with single spaces
    return " ".join(words)

# Apply cleaning
print("    - Applying text normalization (lowercase, no HTML/URLs, remove punctuation, stop words)...")
df['clean_text'] = df['text'].apply(clean_text)

# Check for empty emails after cleaning
empty_count = (df['clean_text'] == "").sum()
if empty_count > 0:
    print(f"    - WARNING: {empty_count} emails became empty after cleaning.")
    # For now, we keep them (TF-IDF will just be all zeros). 
    # Alternatively we could drop them, but they might still have value from 'excl_count'.

# 5. Save Output
print("\n[5] Saving cleaned dataset...")
output_path = os.path.join(OUTPUT_DIR, 'emails_cleaned.csv')
df.to_csv(output_path, index=False)
print(f"    Saved to: {output_path}")

print("\n" + "=" * 70)
print("STEP 1 COMPLETE")
print("=" * 70)
print(df[['spam', 'excl_count', 'word_count', 'clean_text']].head())
