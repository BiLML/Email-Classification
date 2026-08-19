"""
EDA Step 4.6 — Visualization Dashboard
========================================
Purpose: Generate all EDA visualizations (9 plots) and save to evals/eda_plots/.
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

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'emails.csv'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'evals', 'eda_plots'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Precompute features
df['char_count'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['special_char_ratio'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`') / len(str(x)))
df['upper_ratio'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)))
df['excl_count'] = df['text'].str.count('!')

# Stop words for word frequency
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
    'subject:', 're', 'fw', 'subject', 'any', 'been', 'each', 'new',
    'one', 'two', 'get', 'like', 'know', 'please', 'would', 'need',
}


def get_top_words(texts: pd.Series, top_n: int = 20) -> tuple[list[str], list[int]]:
    """Get top N words (excluding stop words) from text series."""
    all_words: list[str] = []
    for text in texts.dropna():
        words = [w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 2]
        all_words.extend(words)
    top = Counter(all_words).most_common(top_n)
    return [w for w, _ in top], [c for _, c in top]


# Style config
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
})

SPAM_COLOR = '#F44336'
HAM_COLOR = '#2196F3'
SPAM_ALPHA = 0.6
HAM_ALPHA = 0.6

print("=" * 70)
print("EDA STEP 4.6 — GENERATING VISUALIZATION DASHBOARD")
print("=" * 70)

plots_saved = []

# =========================================================================
# PLOT 1: Class Distribution (bar + pie) — Already done in step 3
# =========================================================================

# =========================================================================
# PLOT 2: Email Length (chars) — Overlapping Histogram
# =========================================================================
print("\n   [2/9] Email Length (characters) histogram...")
fig, ax = plt.subplots(figsize=(10, 5))

# Cap at 99th percentile for better visualization
cap = int(df['char_count'].quantile(0.99))
ax.hist(df[df['spam']==0]['char_count'].clip(upper=cap), bins=80, alpha=HAM_ALPHA, 
        color=HAM_COLOR, label=f'Ham (n={len(df[df["spam"]==0]):,})', edgecolor='white', linewidth=0.5)
ax.hist(df[df['spam']==1]['char_count'].clip(upper=cap), bins=80, alpha=SPAM_ALPHA, 
        color=SPAM_COLOR, label=f'Spam (n={len(df[df["spam"]==1]):,})', edgecolor='white', linewidth=0.5)

ax.set_title('Email Length Distribution (Characters)')
ax.set_xlabel('Character Count')
ax.set_ylabel('Frequency')
ax.legend(fontsize=11, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_xlim(0, cap)

path = os.path.join(OUTPUT_DIR, 'plot2_char_length_hist.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
plots_saved.append(path)

# =========================================================================
# PLOT 3: Email Length (words) — Overlapping Histogram
# =========================================================================
print("   [3/9] Email Length (words) histogram...")
fig, ax = plt.subplots(figsize=(10, 5))

cap_w = int(df['word_count'].quantile(0.99))
ax.hist(df[df['spam']==0]['word_count'].clip(upper=cap_w), bins=80, alpha=HAM_ALPHA,
        color=HAM_COLOR, label=f'Ham (n={len(df[df["spam"]==0]):,})', edgecolor='white', linewidth=0.5)
ax.hist(df[df['spam']==1]['word_count'].clip(upper=cap_w), bins=80, alpha=SPAM_ALPHA,
        color=SPAM_COLOR, label=f'Spam (n={len(df[df["spam"]==1]):,})', edgecolor='white', linewidth=0.5)

ax.set_title('Email Length Distribution (Words)')
ax.set_xlabel('Word Count')
ax.set_ylabel('Frequency')
ax.legend(fontsize=11, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_xlim(0, cap_w)

path = os.path.join(OUTPUT_DIR, 'plot3_word_length_hist.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
plots_saved.append(path)

# =========================================================================
# PLOT 4: Email Length — Box Plot by Class
# =========================================================================
print("   [4/9] Email length box plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Character count box plot
data_box_chars = [df[df['spam']==0]['char_count'], df[df['spam']==1]['char_count']]
bp1 = axes[0].boxplot(data_box_chars, labels=['Ham', 'Spam'], patch_artist=True, 
                       showfliers=True, flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.3})
bp1['boxes'][0].set_facecolor(HAM_COLOR)
bp1['boxes'][0].set_alpha(0.5)
bp1['boxes'][1].set_facecolor(SPAM_COLOR)
bp1['boxes'][1].set_alpha(0.5)
axes[0].set_title('Character Count by Class')
axes[0].set_ylabel('Character Count')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Word count box plot
data_box_words = [df[df['spam']==0]['word_count'], df[df['spam']==1]['word_count']]
bp2 = axes[1].boxplot(data_box_words, labels=['Ham', 'Spam'], patch_artist=True,
                       showfliers=True, flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.3})
bp2['boxes'][0].set_facecolor(HAM_COLOR)
bp2['boxes'][0].set_alpha(0.5)
bp2['boxes'][1].set_facecolor(SPAM_COLOR)
bp2['boxes'][1].set_alpha(0.5)
axes[1].set_title('Word Count by Class')
axes[1].set_ylabel('Word Count')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout(pad=3.0)
path = os.path.join(OUTPUT_DIR, 'plot4_length_boxplot.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
plots_saved.append(path)

# =========================================================================
# PLOT 5: Top 20 Words — Spam (Horizontal Bar Chart)
# =========================================================================
print("   [5/9] Top 20 spam words bar chart...")
spam_words, spam_counts = get_top_words(df[df['spam']==1]['text'], 20)

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(spam_words))
ax.barh(y_pos, spam_counts[::-1], color=SPAM_COLOR, edgecolor='white', height=0.7, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(spam_words[::-1], fontsize=11)
ax.set_xlabel('Frequency')
ax.set_title('Top 20 Words in SPAM Emails (stop words removed)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add count labels
for i, (count, word) in enumerate(zip(spam_counts[::-1], spam_words[::-1])):
    ax.text(count + max(spam_counts)*0.01, i, f'{count:,}', va='center', fontsize=9, color='#333')

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'plot5_top_words_spam.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
plots_saved.append(path)

# =========================================================================
# PLOT 6: Top 20 Words — Ham (Horizontal Bar Chart)
# =========================================================================
print("   [6/9] Top 20 ham words bar chart...")
ham_words, ham_counts = get_top_words(df[df['spam']==0]['text'], 20)

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(ham_words))
ax.barh(y_pos, ham_counts[::-1], color=HAM_COLOR, edgecolor='white', height=0.7, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(ham_words[::-1], fontsize=11)
ax.set_xlabel('Frequency')
ax.set_title('Top 20 Words in HAM Emails (stop words removed)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')

for i, (count, word) in enumerate(zip(ham_counts[::-1], ham_words[::-1])):
    ax.text(count + max(ham_counts)*0.01, i, f'{count:,}', va='center', fontsize=9, color='#333')

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'plot6_top_words_ham.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
plots_saved.append(path)

# =========================================================================
# PLOT 7: Special Character Ratio — Histogram by Class
# =========================================================================
print("   [7/9] Special character ratio histogram...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Special char ratio
axes[0].hist(df[df['spam']==0]['special_char_ratio'], bins=50, alpha=HAM_ALPHA,
             color=HAM_COLOR, label='Ham', edgecolor='white', linewidth=0.5)
axes[0].hist(df[df['spam']==1]['special_char_ratio'], bins=50, alpha=SPAM_ALPHA,
             color=SPAM_COLOR, label='Spam', edgecolor='white', linewidth=0.5)
axes[0].set_title('Special Character Ratio by Class')
axes[0].set_xlabel('Special Char Ratio')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Uppercase ratio
axes[1].hist(df[df['spam']==0]['upper_ratio'], bins=50, alpha=HAM_ALPHA,
             color=HAM_COLOR, label='Ham', edgecolor='white', linewidth=0.5)
axes[1].hist(df[df['spam']==1]['upper_ratio'], bins=50, alpha=SPAM_ALPHA,
             color=SPAM_COLOR, label='Spam', edgecolor='white', linewidth=0.5)
axes[1].set_title('Uppercase Ratio by Class')
axes[1].set_xlabel('Uppercase Ratio')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout(pad=3.0)
path = os.path.join(OUTPUT_DIR, 'plot7_special_chars.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
plots_saved.append(path)

# =========================================================================
# PLOT 8 & 9: Comparative Feature Summary
# =========================================================================
print("   [8/9] Feature comparison heatmap...")

features = ['char_count', 'word_count', 'special_char_ratio', 'upper_ratio', 'excl_count']
feature_labels = ['Char Count', 'Word Count', 'Special Char %', 'Uppercase %', 'Excl. Marks']

spam_means = [df[df['spam']==1][f].mean() for f in features]
ham_means = [df[df['spam']==0][f].mean() for f in features]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(features))
width = 0.35

bars1 = ax.bar(x - width/2, ham_means, width, label='Ham', color=HAM_COLOR, alpha=0.8, edgecolor='white')
bars2 = ax.bar(x + width/2, spam_means, width, label='Spam', color=SPAM_COLOR, alpha=0.8, edgecolor='white')

ax.set_title('Feature Comparison: Ham vs Spam (Mean Values)')
ax.set_xticks(x)
ax.set_xticklabels(feature_labels, fontsize=11)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Log scale for better comparison since values differ by orders of magnitude
ax.set_yscale('log')
ax.set_ylabel('Mean Value (log scale)')

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'plot8_feature_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
plots_saved.append(path)

# =========================================================================
# PLOT 9: Correlation Matrix of Engineered Features
# =========================================================================
print("   [9/9] Correlation matrix...")
corr_features = ['char_count', 'word_count', 'special_char_ratio', 'upper_ratio', 'excl_count', 'spam']
corr_labels = ['Char Count', 'Word Count', 'Special Char %', 'Uppercase %', 'Excl. Marks', 'Spam Label']
corr_matrix = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(corr_labels, fontsize=10)

# Add correlation values as text
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, 'plot9_correlation_matrix.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
plots_saved.append(path)

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 70)
print("STEP 4.6 — VISUALIZATION DASHBOARD COMPLETE")
print("=" * 70)
print(f"\n   Output directory: {OUTPUT_DIR}")
print(f"   Plots generated: {len(plots_saved) + 1}")  # +1 for class distribution from step 3

for p in plots_saved:
    print(f"   - {os.path.basename(p)}")
print()
