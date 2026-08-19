# 🤖 ML Domain Knowledge Base

> Covers: Machine Learning concepts, algorithms, patterns, and best practices
> Derived from: current spam classification project + TTNT house price project + ML coursework

---

## 📐 Core Concepts

### Supervised Learning Pipeline
```
Raw Data → EDA → Preprocessing & Cleaning → Vectorization → Train/Test Split
→ Model Training → Evaluation → Hyperparameter Tuning → Inference
```

### Train/Test Split Rules (CRITICAL)
- Standard split: **80% Train / 20% Test**
- Must use **Stratified Split** to maintain class distribution
- **NEVER fit transformers (e.g., TF-IDF) on the full dataset before splitting** — this causes **data leakage**
- Always `.fit()` only on training data, then `.transform()` on both train and test

---

## 🌲 Tree-Based Algorithms (From Scratch)

### Decision Trees

**Splitting Criterion — Information Gain (Entropy):**
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$
$$IG = H(S_{parent}) - \sum_{k} \frac{|S_k|}{|S|} H(S_k)$$

**Splitting Criterion — Gini Impurity:**
$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

**Class structure:**
```python
class DecisionTree:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def _best_split(self, X, y) -> tuple[int, float]: ...
    def _information_gain(self, y, y_left, y_right) -> float: ...
```

**Key hyperparameters:**
- `max_depth` — always set to prevent OOM
- `min_samples_split` — minimum samples to attempt a split
- `criterion` — `'gini'` or `'entropy'`

---

### Random Forest

**Algorithm:** Bootstrap Aggregating (Bagging) + Feature Subsampling
- Sample N bootstrap datasets (with replacement) from training data
- For each bootstrap sample, train one Decision Tree
- At each split node, consider only `sqrt(n_features)` features randomly (feature subsampling)
- Prediction = majority vote of all trees

**Common bug — Perfect Overfitting:**
- Cause: Not implementing feature subsampling at each split node
- Solution: `max_features = int(np.sqrt(n_features))` at each split

```python
class RandomForest:
    def __init__(self, n_estimators: int, max_depth: int, max_features: int): ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # For each tree: bootstrap sample → train DecisionTree with max_features
        ...
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Majority vote across all trees
        ...
```

---

### Gradient Boosting Machines (GBM)

**Core idea:** Build trees sequentially, each fitting the negative gradient (pseudo-residuals) of the previous ensemble's loss.

**Pseudo-residuals formula:**
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

For log-loss (binary classification):
$$r_i = y_i - p_i \quad \text{where } p_i = \sigma(F(x_i))$$

**Update rule:**
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
- $\eta$ = learning rate (shrinkage factor, typically 0.01–0.3)
- $h_m$ = regression tree fitted to residuals

**Common bugs:**
- Exploding gradients: always use learning rate `eta < 1.0`
- No convergence: too high `max_depth` for residual trees (use `max_depth=3–5`)

```python
class GradientBoostingClassifier:
    def __init__(self, n_estimators: int, learning_rate: float, max_depth: int): ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Initialize F_0 = log(p/(1-p))
        # For m in range(n_estimators):
        #   Compute residuals r = y - sigmoid(F)
        #   Fit regression tree h_m to residuals
        #   F += learning_rate * h_m.predict(X)
        ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

---

## 📝 Text Feature Engineering

### TF-IDF (Term Frequency - Inverse Document Frequency)

**Term Frequency:**
$$TF(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}$$

**Inverse Document Frequency:**
$$IDF(t) = \log\left(\frac{N}{df_t + 1}\right)$$
- $N$ = total number of documents
- $df_t$ = number of documents containing term $t$
- `+1` smoothing to avoid division by zero

**TF-IDF Score:**
$$TFIDF(t, d) = TF(t, d) \times IDF(t)$$

**Custom TF-IDF class pattern:**
```python
class TFIDFVectorizer:
    def fit(self, corpus: list[str]) -> None:
        # Build vocabulary, compute IDF weights from TRAINING data only
        ...
    def transform(self, corpus: list[str]) -> np.ndarray:
        # Convert texts to TF-IDF matrix using learned vocabulary + IDF
        ...
    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        return self.fit(corpus).transform(corpus)
```

**CRITICAL:** Never `.fit()` TF-IDF on test data. Only `.transform()`.

**Memory optimization:**
- Set `min_df=2` or `min_df=5` to prune rare words
- Cap vocabulary size if needed

---

## 📊 Evaluation Metrics

### Confusion Matrix
```
              Predicted Positive  Predicted Negative
Actual Positive      TP                  FN
Actual Negative      FP                  TN
```

### Core Metrics
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**For spam classification:** Prioritize **Recall** (catch all spam) over Precision (occasional false positives are acceptable).

---

## ⚠️ Common NumPy Pitfalls

### Broadcasting Errors
- **Cause:** Shape mismatch, e.g., `(N,)` vs `(N,1)` during matrix operations
- **Fix:** Use `.reshape(-1, 1)` to ensure dimensional consistency
```python
residuals = y - preds  # Both must be same shape
# If preds is (N,) and y is (N,), this is fine
# If one is (N,1), use .reshape(-1) to flatten
```

### Vectorization Rule
- **NEVER** use nested Python `for` loops for math operations on arrays
- Always use NumPy matrix operations: `np.dot()`, `np.sum(axis=)`, broadcasting
```python
# BAD — O(N²) Python loop
for i in range(N):
    for j in range(M):
        result[i] += X[i,j] * w[j]

# GOOD — vectorized
result = X @ w  # or np.dot(X, w)
```

### Type Hinting Convention
```python
def calculate_gini(y: np.ndarray) -> float:
    """
    Calculate Gini Impurity for a label array.
    
    Args:
        y: Label array of shape (N,)
    Returns:
        Gini impurity score in [0, 1]
    Formula: Gini = 1 - sum(p_i^2)
    """
    ...
```

---

## 🔍 Hyperparameter Tuning

### Grid Search (From Scratch)
```python
def grid_search(model_class, param_grid: dict, X_train, y_train, X_val, y_val):
    best_score = -np.inf
    best_params = {}
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        model = model_class(**param_dict)
        model.fit(X_train, y_train)
        score = f1_score(y_val, model.predict(X_val))
        if score > best_score:
            best_score = score
            best_params = param_dict
    return best_params, best_score
```

**Key parameters to tune for this project:**
| Model | Key Parameters |
|-------|---------------|
| Decision Tree | `max_depth`, `min_samples_split`, `criterion` |
| Random Forest | `n_estimators`, `max_depth`, `max_features`, `min_samples_split` |
| GBM | `n_estimators`, `learning_rate`, `max_depth` |

---

## 💾 Model Persistence Pattern
```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## 🏗️ Recommended Architecture (From-Scratch ML Project)

```
project/
├── data/
│   └── email.csv
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py         # Text cleaning functions
│   │   └── vectorizer.py      # Custom TF-IDF
│   ├── models/
│   │   ├── decision_tree.py   # DecisionTree class
│   │   ├── random_forest.py   # RandomForest class
│   │   └── gradient_boosting.py # GBM class
│   ├── evaluation/
│   │   └── metrics.py         # Accuracy, Precision, Recall, F1, ConfusionMatrix
│   └── utils/
│       ├── split.py           # Custom stratified train/test split
│       └── grid_search.py     # Custom hyperparameter tuner
├── notebooks/
│   └── EDA.ipynb
└── main.py                    # End-to-end inference pipeline
```

---

## 🏠 Other ML Projects Reference

### HCM Rental Price Prediction
- Vietnamese text price cleaning (tri?u/tháng → float)
- Handle `price` column with mixed units (tri?u/tháng vs nghìn/m²)
- IQR capping for outliers (preserves rows, preferable for small datasets)
- Feature engineering: `TotalSF`, `TotalBathrooms`, `HouseAge`

### Kaggle Ames Housing (Random Forest)
- 3-tier missing data strategy: domain fills → grouped median → mode/median
- Feature alignment between train/test: use `join='left'` not `join='inner'` in `.align()`
- Hold-out 20% validation with RMSE + R² metrics and residual plots

### User Behavior Classification
- XGBoost model for behavior prediction
- SMOTE (imbalanced-learn) for class imbalance
- FastAPI serving: `python -m uvicorn app.main:app --reload`
