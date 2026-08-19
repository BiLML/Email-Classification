"""
EDA Step 4.3 — Class Distribution Analysis
============================================
Purpose: Analyze spam vs. ham distribution and generate visualizations.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'emails.csv'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'evals', 'eda_plots'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("EDA STEP 4.3 — CLASS DISTRIBUTION ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Class Counts
# =============================================================================
print("\n" + "-" * 70)
print("[1] CLASS DISTRIBUTION")
print("-" * 70)

class_counts = df['spam'].value_counts().sort_index()
class_pcts = (df['spam'].value_counts(normalize=True) * 100).sort_index()

labels_map = {0: 'Ham (Not Spam)', 1: 'Spam'}
for label in sorted(df['spam'].unique()):
    name = labels_map.get(label, f'Class {label}')
    print(f"   {name:<20} → {class_counts[label]:>6,} emails  ({class_pcts[label]:.1f}%)")

print(f"\n   Total:                  {len(df):>6,} emails")

# Imbalance metrics
majority = class_counts.max()
minority = class_counts.min()
imbalance_ratio = majority / minority
print(f"\n   Majority class:         {majority:,} (Ham)")
print(f"   Minority class:         {minority:,} (Spam)")
print(f"   Imbalance ratio:        {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print(f"   Assessment:             MODERATE IMBALANCE (>3:1)")
    print(f"   Impact:                 Accuracy alone is misleading")
    print(f"   Recommendation:        Use F1-Score + Recall as primary metrics")
elif imbalance_ratio > 5:
    print(f"   Assessment:             SEVERE IMBALANCE (>5:1)")
else:
    print(f"   Assessment:             ACCEPTABLE BALANCE")

print(f"\n   Stratified split:       REQUIRED (per WORKING_RULES.md)")

# =============================================================================
# 2. Visualization — Class Distribution Bar Chart
# =============================================================================
print("\n" + "-" * 70)
print("[2] GENERATING VISUALIZATIONS")
print("-" * 70)

# --- Plot 1: Bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ['#2196F3', '#F44336']
bars = axes[0].bar(
    ['Ham\n(Not Spam)', 'Spam'],
    [class_counts[0], class_counts[1]],
    color=colors,
    edgecolor='white',
    linewidth=1.5,
    width=0.5
)

# Add value labels on bars
for bar, count, pct in zip(bars, [class_counts[0], class_counts[1]], [class_pcts[0], class_pcts[1]]):
    axes[0].text(
        bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
        f'{count:,}\n({pct:.1f}%)',
        ha='center', va='bottom', fontweight='bold', fontsize=12
    )

axes[0].set_title('Email Class Distribution', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('Number of Emails', fontsize=12)
axes[0].set_ylim(0, max(class_counts) * 1.2)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Pie chart
axes[1].pie(
    [class_counts[0], class_counts[1]],
    labels=['Ham (Not Spam)', 'Spam'],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=(0, 0.05),
    shadow=True,
    textprops={'fontsize': 12}
)
axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout(pad=3.0)
plot_path = os.path.join(OUTPUT_DIR, 'step3_class_distribution.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved: {plot_path}")

# =============================================================================
# 3. Class Distribution After Dedup
# =============================================================================
print("\n" + "-" * 70)
print("[3] CLASS DISTRIBUTION AFTER REMOVING DUPLICATES")
print("-" * 70)

df_dedup = df.drop_duplicates()
class_counts_dedup = df_dedup['spam'].value_counts().sort_index()
class_pcts_dedup = (df_dedup['spam'].value_counts(normalize=True) * 100).sort_index()

for label in sorted(df_dedup['spam'].unique()):
    name = labels_map.get(label, f'Class {label}')
    before = class_counts[label]
    after = class_counts_dedup[label]
    diff = before - after
    print(f"   {name:<20} → {after:>6,} emails  ({class_pcts_dedup[label]:.1f}%)  [removed {diff}]")

print(f"\n   Total after dedup:      {len(df_dedup):>6,} emails")
new_ratio = class_counts_dedup.max() / class_counts_dedup.min()
print(f"   New imbalance ratio:    {new_ratio:.2f}:1")

print("\n" + "=" * 70)
print("STEP 4.3 COMPLETE")
print("=" * 70)
print()
