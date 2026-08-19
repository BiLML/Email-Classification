# 🏆 Coding Standards & Best Practices

> Personal coding rules derived from all projects and working rules docs

---

## 🐍 Python Code Standards

### OOP Structure (ML Models)
```python
class ModelName:
    """
    Brief description of the model.
    
    Mathematical basis:
        <formula here>
    """
    
    def __init__(self, hyperparameter1: int, hyperparameter2: float):
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter2
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...
```

**Rules:**
- ✅ Type hints on ALL function parameters and return values
- ✅ Docstrings on every computational function (inputs, outputs, formula)
- ✅ OOP with `.fit()` and `.predict()` pattern
- ❌ No nested for-loops for math — use NumPy vectorization
- ❌ No high-level ML library imports (scikit-learn, xgboost) in from-scratch projects

### Vectorization Examples
```python
# ✅ CORRECT — NumPy operations
dot_product = X @ w                     # Matrix multiply
col_sums = X.sum(axis=0)               # Sum along axis
sigmoid = 1 / (1 + np.exp(-X))        # Elementwise

# ❌ WRONG — Python loops
for i in range(N):
    for j in range(M):
        result += X[i,j] * w[j]       # Extremely slow
```

### Shape Management
```python
# Always check shapes during development
print(X.shape, y.shape, w.shape)

# Fix shape mismatches
y = y.reshape(-1, 1)    # (N,) → (N,1)
y = y.flatten()          # (N,1) → (N,)
```

---

## 🌳 Git & Version Control

### Branch Strategy
```
main
├── data-prep
├── model-tree (Decision Tree, Random Forest)
├── model-ensemble (GBM)
└── evaluation
```

### Commit Checkpoints
- After successful preprocessing: `git commit -m "feat: add TF-IDF vectorizer"`
- After model training: `git commit -m "feat: implement Decision Tree from scratch"`
- Save trained models as `.pkl` files to avoid retraining

### .gitignore Must-Haves
```gitignore
# Environments
.env
.venv/
.env/

# ML artifacts
*.pkl
chroma_db/
data/temp_*.json

# Large files
*.docx
*.pdf

# IDE settings
.idea/
.vscode/

# Temp files
seed_out.txt
extracted_docx.txt
sample.txt
```

---

## 🧪 Test-Driven Development (TDD) for Math

Before training on real data, test with small dummy datasets:

```python
# Example: Test Information Gain calculation
def test_information_gain():
    y = np.array([0, 0, 1, 1, 1])          # 3 class-1, 2 class-0
    y_left = np.array([0, 0])
    y_right = np.array([1, 1, 1])
    
    ig = calculate_information_gain(y, y_left, y_right)
    assert abs(ig - 1.0) < 1e-6, f"Expected IG=1.0, got {ig}"
    print("✅ Information Gain test passed")

# Example: Test TF-IDF
def test_tfidf():
    corpus = ["the cat sat", "the dog ran", "cat dog"]
    tfidf = TFIDFVectorizer()
    matrix = tfidf.fit_transform(corpus)
    assert matrix.shape[0] == 3, "Should have 3 rows"
    assert matrix.shape[1] > 0, "Should have features"
    print("✅ TF-IDF test passed")
```

---

## 🚀 Development Workflow

```
1. 📊 EDA & Data Understanding
   └── Visualize distributions, class balance, missing values

2. 🧹 Preprocessing
   └── Clean text, handle missing values, feature engineering

3. ✂️ Train/Test Split (Stratified)
   └── 80/20 split, maintain class ratios, no leakage

4. 🏗️ Implement Models (with unit tests)
   └── Test math functions on dummy data first

5. 🏋️ Train Models
   └── Fit on training data only

6. 📏 Evaluate on Validation/Test
   └── Confusion matrix, Accuracy, Precision, Recall, F1

7. 🔧 Hyperparameter Tuning
   └── Grid search on validation set

8. 💾 Save Best Model
   └── pickle.dump() or json (for tree structures)

9. 🔗 Build Inference Pipeline
   └── Raw text → clean → TF-IDF → predict → output
```

---

## ⚠️ Failure Patterns to AVOID

| Pattern | Problem | Fix |
|---------|---------|-----|
| Fitting TF-IDF on full dataset | Data leakage | Only `.fit()` on train, `.transform()` on test |
| `max_depth=None` in trees | OOM error | Always set `max_depth` + `min_samples_split` |
| No feature subsampling in RF | Perfect overfitting | Use `max_features = sqrt(n_features)` at each split |
| High learning rate in GBM | Exploding gradients | Keep `learning_rate` < 0.3, usually 0.05–0.1 |
| Nested Python loops | Severe perf bottleneck | Use NumPy matrix operations |
| Shape `(N,)` vs `(N,1)` | Broadcasting error | Use `.reshape(-1,1)` or `.flatten()` |
| `join='inner'` in df.align() | Silent feature drop | Use `join='left'` to keep training features |
| Raw text into vectorizer | Garbage features | Always clean → lowercase → remove stopwords first |
| Duplicate data in training | Bias/overfitting | `df.drop_duplicates()` before split |

---

## 📦 Dependency Management

### Common ML Project Requirements
```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
```

### Allowed in from-scratch ML project
- `numpy` ✅
- `pandas` ✅ (data I/O only)
- `math` ✅
- `pickle` ✅ (model saving)

### NOT allowed in from-scratch project
- `scikit-learn` ❌
- `xgboost` ❌
- `lightgbm` ❌
- `tensorflow` / `torch` ❌

---

## 🐳 Python Virtual Environment (Windows/PowerShell)

```powershell
# Create venv
python -m venv .venv

# Activate
.venv/Scripts/Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Deactivate
deactivate
```

**Common issue:** `-m` not recognized → must run as `python -m venv .venv`, not `-m .env venv`

---

## 🗃️ Data Handling Best Practices

### Vietnamese Data Cleaning (Pandas)
```python
# Price column: "5.5 triệu/tháng" → 5.5 (float)
def clean_price(x: str) -> float | None:
    if 'thỏa thuận' in x or 'nghìn/m' in x:
        return None  # Remove ambiguous prices
    x = x.replace(' triệu/tháng', '')
    return pd.to_numeric(x, errors='coerce')

# Split mixed-unit price column into two datasets
df_monthly = df[df['price'].str.contains('triệu/tháng', na=False)]
df_per_m2  = df[df['price'].str.contains('nghìn/m2', na=False)]
```

### Class Imbalance
```python
# Check balance
print(df['label'].value_counts(normalize=True))

# Stratified split to maintain ratio
from custom_split import stratified_split
X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
```
