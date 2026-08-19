"""
Preprocessing Step 2: Stratified Train/Validation/Test Split
============================================================
Purpose: Shuffle the dataset and perform a 70/15/15 stratified split manually
         to ensure the spam/ham ratio is preserved in all sets.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os

# Set paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'emails_cleaned.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')

print("=" * 70)
print("PREPROCESSING STEP 2 — STRATIFIED DATA SPLIT")
print("=" * 70)

# 1. Load Cleaned Data
print("\n[1] Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    Loaded shape: {df.shape[0]:,} rows")

# 2. Separate Classes
print("\n[2] Separating classes for stratification...")
ham_df = df[df['spam'] == 0].copy()
spam_df = df[df['spam'] == 1].copy()

# Shuffle each class independently (important because dataset is sorted)
np.random.seed(42)  # For reproducibility
ham_df = ham_df.sample(frac=1, random_state=42).reset_index(drop=True)
spam_df = spam_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"    Ham count:  {len(ham_df):,} ({len(ham_df)/len(df)*100:.1f}%)")
print(f"    Spam count: {len(spam_df):,} ({len(spam_df)/len(df)*100:.1f}%)")

# 3. Calculate Split Indices (70 / 15 / 15)
print("\n[3] Calculating split indices...")
def get_split_indices(total_length):
    train_end = int(total_length * 0.70)
    val_end = int(total_length * 0.85)
    return train_end, val_end

ham_train_idx, ham_val_idx = get_split_indices(len(ham_df))
spam_train_idx, spam_val_idx = get_split_indices(len(spam_df))

# 4. Perform Split & Concatenate
print("\n[4] Performing stratified split...")

# Train
train_df = pd.concat([
    ham_df.iloc[:ham_train_idx],
    spam_df.iloc[:spam_train_idx]
])

# Val
val_df = pd.concat([
    ham_df.iloc[ham_train_idx:ham_val_idx],
    spam_df.iloc[spam_train_idx:spam_val_idx]
])

# Test
test_df = pd.concat([
    ham_df.iloc[ham_val_idx:],
    spam_df.iloc[spam_val_idx:]
])

# Shuffle the combined splits so ham/spam aren't clustered
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Verify Stratification
print("\n[5] Verifying stratification:")
for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    spam_ratio = split_df['spam'].mean() * 100
    print(f"    - {name:<6} | Rows: {len(split_df):>4,} | Spam ratio: {spam_ratio:.1f}%")

# 6. Save Splits
print("\n[6] Saving split datasets...")
train_path = os.path.join(OUTPUT_DIR, 'train.csv')
val_path = os.path.join(OUTPUT_DIR, 'val.csv')
test_path = os.path.join(OUTPUT_DIR, 'test.csv')

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"    Saved: {train_path}")
print(f"    Saved: {val_path}")
print(f"    Saved: {test_path}")

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)
