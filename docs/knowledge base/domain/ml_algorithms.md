# 🤖 ML Domain Knowledge Base

> Covers: Machine Learning concepts, algorithms, patterns, and best practices
> Derived from: Spam Email Classification project (From-Scratch NumPy) + TTNT house price project + ML coursework

---

## 📐 Core Concepts

### Supervised Learning Pipeline
```
Raw Data → EDA → Preprocessing & Cleaning → Vectorization (TF-IDF) → Stratified Train/Val/Test Split
→ Model Training (OOP) → Evaluation (F1/Precision/Recall) → Hyperparameter Tuning → E2E Inference
```

### Train/Test Split Rules (CRITICAL)
- Standard split: **70% Train / 15% Validation / 15% Test** (or 80/20)
- Must use **Stratified Split** to maintain class distribution when dataset is imbalanced (e.g. 3.19:1 Ham vs Spam)
- **NEVER fit transformers (e.g., TF-IDF) on the full dataset before splitting** — this causes **data leakage**
- Always `.fit()` vocabulary and IDF only on training data, then `.transform()` on train, val, and test

---

## 🧠 From-Scratch Classification Algorithms (Pure NumPy)

### 1. Multinomial Naive Bayes (MNB)

**Core Principle:** Probabilistic classifier based on Bayes' Theorem with conditional independence assumption between features.

**Bayes' Theorem Formulation:**
$$P(y = c \mid \mathbf{x}) \propto P(y = c) \prod_{i=1}^{D} P(x_i \mid y = c)^{x_i}$$

**Log-Likelihood (Preventing Numerical Underflow):**
$$\log P(y = c \mid \mathbf{x}) = \log P(y = c) + \sum_{i=1}^{D} x_i \cdot \log P(w_i \mid y = c)$$

**Laplace Smoothing ($\alpha = 1.0$):**
$$\hat{\theta}_{c, i} = \frac{N_{c, i} + \alpha}{N_c + \alpha \cdot D}$$
- $N_{c, i}$: Total TF-IDF weight of word $i$ in class $c$
- $N_c$: Total TF-IDF weight of all words in class $c$
- $D$: Total number of vocabulary features

**Class structure:**
```python
class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes = None
        self.class_log_prior = None
        self.feature_log_prob = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        # Calculate class log priors and smoothed feature log probabilities
        ...
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        # log_prior + X @ log_prob.T
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]
```

---

### 2. Logistic Regression

**Core Principle:** Linear probabilistic model using the Sigmoid activation function to map linear combinations of features to probability estimates $[0, 1]$.

**Hypothesis Function:**
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{where } z = \mathbf{w}^T \mathbf{x} + b$$

**Binary Cross-Entropy Loss (Log Loss):**
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Gradient Descent Optimization:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}), \quad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$
$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$

**Numerical Stability:**
- Use `np.clip(z, -500, 500)` before computing `np.exp(-z)` to prevent overflow runtime warnings.

**Class structure:**
```python
class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 500):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        # Initialize weights (zeros), run gradient descent iterations
        ...
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

### 3. Linear Support Vector Machine (Linear SVM)

**Core Principle:** Finds the maximum-margin hyperplane separating classes with Soft-Margin Hinge Loss and L2 Regularization.

**Decision Boundary:**
$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b, \quad y \in \{-1, +1\}$$

**Hinge Loss with L2 Regularization:**
$$\mathcal{L}(\mathbf{w}, b) = \frac{\lambda}{2} \|\mathbf{w}\|^2 + \frac{1}{N} \sum_{i=1}^{N} \max\left(0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)\right)$$

**Stochastic Gradient Descent (SGD) with Subgradient:**
For sample $(\mathbf{x}_i, y_i)$:
- If margin condition violated ($y_i (\mathbf{w}^T \mathbf{x}_i + b) < 1$):
  $$\mathbf{w} \leftarrow \mathbf{w} - \eta (\lambda \mathbf{w} - y_i \mathbf{x}_i), \quad b \leftarrow b + \eta y_i$$
- If margin condition satisfied ($y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1$):
  $$\mathbf{w} \leftarrow \mathbf{w} - \eta (\lambda \mathbf{w}), \quad b \leftarrow b$$

**Class structure:**
```python
class SVM:
    def __init__(self, learning_rate: float = 0.01, lambda_param: float = 0.01, epochs: int = 500):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        # Convert {0, 1} labels to {-1, +1}, run sample-wise SGD with Hinge loss
        ...
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        approx = np.dot(X, self.weights) - self.bias
        return np.where(approx >= 0, 1, 0)
```

---

## 📝 Text Feature Engineering & Preprocessing

### Custom TF-IDF (Term Frequency - Inverse Document Frequency)

**Sublinear Term Frequency:**
$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{if } \text{count} > 0 \text{ else } 0$$

**Smooth Inverse Document Frequency:**
$$\text{IDF}(t) = \log\left(\frac{N + 1}{\text{df}(t) + 1}\right) + 1$$
- $N$ = total documents in training set
- $\text{df}(t)$ = documents containing term $t$ in training set

**L2 Vector Normalization:**
$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

**Log-Transform for Numerical Features:**
$$x_{\text{transformed}} = \log(1 + x) \quad (\text{np.log1p})$$
Applied to `excl_count` and `word_count` to align scale with $[0, 1]$ TF-IDF vectors.

**CRITICAL:** Never `.fit()` TF-IDF on test/val data. Fit vocabulary and IDF strictly on Train, then `.transform()`.

---

## 📊 Evaluation Metrics

### Confusion Matrix
```
               Predicted Positive (Spam)  Predicted Negative (Ham)
Actual Positive (Spam)       TP                       FN
Actual Negative (Ham)        FP                       TN
```

### Core Classification Metrics
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(Crucial: 0 FP = 100% Precision, no legitimate emails lost)}$$

$$\text{Recall} = \frac{TP}{TP + FN} \quad \text{(Measures proportion of spam correctly caught)}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## ⚠️ Common NumPy Pitfalls & Fixes

### Broadcasting Errors
- **Cause:** Shape mismatch, e.g., `(N,)` vs `(N,1)` during matrix operations
- **Fix:** Use `.reshape(-1, 1)` or `.flatten()` to ensure dimensional consistency

### Vectorization Rule
- **NEVER** use nested Python `for` loops for math operations on feature matrices
- Always use NumPy matrix operations: `np.dot()`, `@`, `np.sum(axis=)`, broadcasting

### Type Hinting Convention
```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute numerically stable Sigmoid activation.
    
    Args:
        z: Input array of shape (N,) or (N, D)
    Returns:
        Array of probabilities in range (0, 1)
    """
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))
```

---

## 🔍 Hyperparameter Tuning

**Key parameters to tune for from-scratch models:**
| Model | Key Parameters | Typical Search Range |
|---|---|---|
| Multinomial Naive Bayes | `alpha` (Laplace smoothing) | `0.1, 0.5, 1.0, 2.0` |
| Logistic Regression | `learning_rate`, `epochs`, `threshold` | `lr: [0.01, 0.1, 0.5]`, `epochs: [200, 500, 1000]` |
| Linear SVM | `learning_rate`, `lambda_param`, `epochs` | `lr: [0.001, 0.01]`, `lambda: [0.001, 0.01, 0.1]` |

---

## 💾 Model Persistence Pattern
```python
import pickle

# Save
with open('models/naive_bayes.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('models/naive_bayes.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## 🏗️ Project Architecture (From-Scratch ML Project)

```
project/
├── data/
│   ├── raw/                        # Raw emails.csv
│   └── processed/                  # Cleaned CSVs, .npz feature matrices, tfidf_vocab.pkl
├── src/
│   ├── EDA/                        # Step 1 to Step 5 EDA scripts
│   ├── Preprocessing/              # Step 1 cleaning, Step 2 split, Step 3 TF-IDF
│   ├── training/                   # naive_bayes.py, logistic_regression.py, svm.py
│   └── evalute/                    # train_and_evaluate.py, test_e2e.py
├── models/                         # Saved .pkl weights (naive_bayes.pkl, logistic_regression.pkl, svm.pkl)
└── docs/                           # Documentation, PLAN.md, PRESENTATION_SLIDES.md
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
