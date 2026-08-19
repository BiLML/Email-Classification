"""
Preprocessing Step 3: Custom TF-IDF Vectorization & Matrix Assembly
===================================================================
Purpose: Build a custom TF-IDF vectorizer (from scratch using numpy),
         fit on the training set, transform all sets, and assemble
         the final feature matrices combining text and engineered features.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import math
import os

# Set paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')

print("=" * 70)
print("PREPROCESSING STEP 3 — CUSTOM TF-IDF VECTORIZATION")
print("=" * 70)

# 1. Load splits
print("\n[1] Loading split datasets...")
train_df = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'))
val_df = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'))
test_df = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'))

# Replace NaN in clean_text with empty string (pandas loads empty strings as NaN)
train_df['clean_text'] = train_df['clean_text'].fillna("")
val_df['clean_text'] = val_df['clean_text'].fillna("")
test_df['clean_text'] = test_df['clean_text'].fillna("")

MIN_DF = 5

print(f"\n[2] Building vocabulary from TRAIN set (min_df={MIN_DF})...")
doc_freq = {}
train_docs = [text.split() for text in train_df['clean_text']]

for doc in train_docs:
    unique_words = set(doc)
    for w in unique_words:
        doc_freq[w] = doc_freq.get(w, 0) + 1

# Filter by min_df and assign sequential indices
vocab = {}
idx = 0
for w, count in doc_freq.items():
    if count >= MIN_DF:
        vocab[w] = idx
        idx += 1
        
vocab_size = len(vocab)
print(f"    - Original unique words: {len(doc_freq):,}")
print(f"    - Vocabulary size after pruning: {vocab_size:,}")

# Calculate IDF
# Formula: IDF(t) = log( (1 + N) / (1 + df(t)) ) + 1 (matches standard smoothing)
N = len(train_df)
idf = np.zeros(vocab_size)
for w, idx in vocab.items():
    df_t = doc_freq[w]
    idf[idx] = math.log((1 + N) / (1 + df_t)) + 1

def transform_tfidf(docs, vocab, idf):
    """Transform list of documents into TF-IDF dense matrix."""
    n_docs = len(docs)
    vocab_size = len(vocab)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
    
    for i, doc in enumerate(docs):
        # Term Frequency
        tf = {}
        for w in doc:
            if w in vocab:
                tf[w] = tf.get(w, 0) + 1
                
        # Calculate TF-IDF
        for w, count in tf.items():
            idx = vocab[w]
            matrix[i, idx] = count * idf[idx]
            
        # L2 Normalization (per document)
        norm = np.linalg.norm(matrix[i])
        if norm > 0:
            matrix[i] = matrix[i] / norm
            
    return matrix

print("\n[3] Transforming datasets to TF-IDF matrices...")
print("    - Transforming Train...")
X_train_text = transform_tfidf(train_docs, vocab, idf)

print("    - Transforming Val...")
val_docs = [text.split() for text in val_df['clean_text']]
X_val_text = transform_tfidf(val_docs, vocab, idf)

print("    - Transforming Test...")
test_docs = [text.split() for text in test_df['clean_text']]
X_test_text = transform_tfidf(test_docs, vocab, idf)

print("\n[4] Assembling final feature matrices...")
# Extract numerical features
def get_numerical_features(df):
    # Normalize features using log1p transformation for counts 
    # to be on a similar scale to TF-IDF (which is between 0-1)
    excl = np.log1p(df['excl_count'].values).reshape(-1, 1)
    word = np.log1p(df['word_count'].values).reshape(-1, 1)
    return np.hstack([excl, word]).astype(np.float32)

X_train_num = get_numerical_features(train_df)
X_val_num = get_numerical_features(val_df)
X_test_num = get_numerical_features(test_df)

# Combine Text TF-IDF and Numerical Features
X_train = np.hstack([X_train_text, X_train_num])
X_val = np.hstack([X_val_text, X_val_num])
X_test = np.hstack([X_test_text, X_test_num])

y_train = train_df['spam'].values.astype(np.int8)
y_val = val_df['spam'].values.astype(np.int8)
y_test = test_df['spam'].values.astype(np.int8)

print(f"    - X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"    - X_val shape:   {X_val.shape} | y_val shape:   {y_val.shape}")
print(f"    - X_test shape:  {X_test.shape} | y_test shape:  {y_test.shape}")

print("\n[5] Saving final matrices (.npz)...")
np.savez_compressed(os.path.join(PROC_DIR, 'train_data.npz'), X=X_train, y=y_train)
np.savez_compressed(os.path.join(PROC_DIR, 'val_data.npz'), X=X_val, y=y_val)
np.savez_compressed(os.path.join(PROC_DIR, 'test_data.npz'), X=X_test, y=y_test)

# Also save the vocabulary and IDF weights for inference later
import pickle
with open(os.path.join(PROC_DIR, 'tfidf_vocab.pkl'), 'wb') as f:
    pickle.dump({'vocab': vocab, 'idf': idf}, f)
print("    - Saved train_data.npz, val_data.npz, test_data.npz, tfidf_vocab.pkl")

print("\n" + "=" * 70)
print("STEP 3 COMPLETE")
print("=" * 70)
