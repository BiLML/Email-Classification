"""
EDA Step 4.1 — Data Loading & First Look
=========================================
Purpose: Load the raw email dataset and perform initial structural inspection.
Output:  Dataset shape, column names, data types, memory usage, and sample rows.

Allowed Libraries: pandas, numpy (per WORKING_RULES.md)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os

# =============================================================================
# 1. Load Dataset
# =============================================================================
DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'data', 'emails.csv'
)
DATA_PATH = os.path.abspath(DATA_PATH)

print("=" * 70)
print("EDA STEP 4.1 — DATA LOADING & FIRST LOOK")
print("=" * 70)
print(f"\n📂 Data path: {DATA_PATH}")
print(f"   File exists: {os.path.exists(DATA_PATH)}")
print(f"   File size: {os.path.getsize(DATA_PATH) / (1024*1024):.2f} MB")

df = pd.read_csv(DATA_PATH)

# =============================================================================
# 2. Check Shape
# =============================================================================
print("\n" + "-" * 70)
print("📐 DATASET SHAPE")
print("-" * 70)
print(f"   Rows:    {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]:,}")

# =============================================================================
# 3. Check Columns
# =============================================================================
print("\n" + "-" * 70)
print("📋 COLUMN NAMES")
print("-" * 70)
for i, col in enumerate(df.columns.tolist()):
    print(f"   [{i}] '{col}'")

# =============================================================================
# 4. Check Data Types
# =============================================================================
print("\n" + "-" * 70)
print("🔧 DATA TYPES")
print("-" * 70)
for col in df.columns:
    print(f"   {col:<30} → {df[col].dtype}")

# =============================================================================
# 5. Preview Data — Head
# =============================================================================
print("\n" + "-" * 70)
print("👀 FIRST 5 ROWS (HEAD)")
print("-" * 70)
# Show full content for text columns (up to 200 chars)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', 200)
print(df.head(5).to_string(index=True))

# =============================================================================
# 6. Preview Data — Tail
# =============================================================================
print("\n" + "-" * 70)
print("👀 LAST 5 ROWS (TAIL)")
print("-" * 70)
print(df.tail(5).to_string(index=True))

# =============================================================================
# 7. Random Sample (3 rows)
# =============================================================================
print("\n" + "-" * 70)
print("🎲 RANDOM SAMPLE (3 ROWS)")
print("-" * 70)
print(df.sample(3, random_state=42).to_string(index=True))

# =============================================================================
# 8. Basic Info (df.info equivalent)
# =============================================================================
print("\n" + "-" * 70)
print("ℹ️  DATASET INFO")
print("-" * 70)
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
print(f"   Total cells:  {df.shape[0] * df.shape[1]:,}")
print()

# Non-null counts per column
print("   Non-null counts per column:")
for col in df.columns:
    non_null = df[col].notna().sum()
    null_count = df[col].isna().sum()
    pct = non_null / len(df) * 100
    status = "✅" if null_count == 0 else f"⚠️  ({null_count:,} missing)"
    print(f"   {col:<30} {non_null:>7,} / {len(df):,}  ({pct:.1f}%)  {status}")

# =============================================================================
# 9. Unique Values per Column
# =============================================================================
print("\n" + "-" * 70)
print("🔢 UNIQUE VALUES PER COLUMN")
print("-" * 70)
for col in df.columns:
    n_unique = df[col].nunique()
    print(f"   {col:<30} → {n_unique:,} unique values")
    # If few unique values (likely a label column), show value counts
    if n_unique <= 10:
        print(f"      Value counts:")
        for val, count in df[col].value_counts().items():
            pct = count / len(df) * 100
            print(f"         {val!r:<20} → {count:>7,}  ({pct:.1f}%)")

# =============================================================================
# 10. Descriptive Statistics for Numeric Columns
# =============================================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    print("\n" + "-" * 70)
    print("📊 DESCRIPTIVE STATISTICS (NUMERIC COLUMNS)")
    print("-" * 70)
    print(df[numeric_cols].describe().to_string())

# =============================================================================
# 11. Quick Text Column Inspection
# =============================================================================
text_cols = df.select_dtypes(include=['object']).columns.tolist()
if text_cols:
    print("\n" + "-" * 70)
    print("📝 TEXT COLUMN INSPECTION")
    print("-" * 70)
    for col in text_cols:
        if df[col].notna().any():
            lengths = df[col].dropna().str.len()
            word_counts = df[col].dropna().str.split().str.len()
            print(f"\n   Column: '{col}'")
            print(f"   ├── Min length:     {lengths.min():,.0f} chars")
            print(f"   ├── Max length:     {lengths.max():,.0f} chars")
            print(f"   ├── Mean length:    {lengths.mean():,.1f} chars")
            print(f"   ├── Median length:  {lengths.median():,.1f} chars")
            print(f"   ├── Min words:      {word_counts.min():,.0f}")
            print(f"   ├── Max words:      {word_counts.max():,.0f}")
            print(f"   ├── Mean words:     {word_counts.mean():,.1f}")
            print(f"   └── Median words:   {word_counts.median():,.1f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("✅ STEP 4.1 — DATA LOADING & FIRST LOOK COMPLETE")
print("=" * 70)
print(f"""
Summary:
  • Dataset:     {os.path.basename(DATA_PATH)}
  • Shape:       {df.shape[0]:,} rows × {df.shape[1]} columns
  • Memory:      {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB
  • Columns:     {df.columns.tolist()}
  • Null cells:  {df.isna().sum().sum():,}
  • Duplicates:  {df.duplicated().sum():,}
""")
