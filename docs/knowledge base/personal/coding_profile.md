# 🧠 Personal Coding Profile

> Last updated: 2026-08-05 | Source: Full conversation history flashback

---

## 👤 Identity

**Name / Handle:** Acer (Windows user, Vietnamese developer/student)
**Primary Language:** Vietnamese 🇻🇳 (also communicates in English for technical docs)
**Development OS:** Windows 11 (PowerShell environment)
**IDE / Tool:** Antigravity IDE (AI-assisted coding assistant)
**Primary Domain:** AI / Machine Learning / Full-Stack Web Development

---

## 🎓 Background & Learning Context

- Vietnamese university student studying **AI, Machine Learning, and Information Security**
- Works on multiple concurrent academic and personal projects
- Comfortable switching between **Python data science**, **full-stack web development**, and **algorithm implementation**
- Follows a **hands-on, project-based learning** approach — learns by building real systems

---

## 🛠️ Technology Stack Mastery

### Languages
| Language | Level | Context |
|----------|-------|---------|
| **Python** | Advanced | ML models, FastAPI backends, data preprocessing, cryptography |
| **TypeScript / TSX** | Intermediate | React + Vite frontend for full-stack projects |
| **SQL / PostgreSQL** | Intermediate | Database design, ORM (SQLAlchemy), Alembic migrations |
| **Prolog** | Beginner | AI logic games (Wumpus World) |
| **HTML/CSS** | Intermediate | Web UI work |

### Python Ecosystem
- **NumPy** — core linear algebra, vectorized operations (mandatory for ML-from-scratch)
- **Pandas** — data I/O, cleaning, transformation
- **Matplotlib / Seaborn** — data visualization
- **scikit-learn** — used in allowed projects (Random Forest, preprocessing)
- **XGBoost** — used in user behavior analysis project
- **imbalanced-learn** — SMOTE for class imbalance
- **FastAPI + Uvicorn** — REST API backend framework of choice
- **SQLAlchemy + Alembic** — ORM and migrations
- **ChromaDB** — vector database for semantic search
- **scikit-surprise** — Matrix Factorization (SVD) collaborative filtering

### JavaScript / TypeScript Ecosystem
- **React 18** — UI component library
- **Vite** — build tool and dev server
- **TailwindCSS v4** — utility-first CSS
- **React Router** — client-side routing
- **Axios** — HTTP client with JWT interceptors
- **Lucide React** — icon library
- **Shadcn/Radix UI** — accessible UI primitives

### DevOps / Infra
- **Docker + Docker Compose** — multi-service containerization (PostgreSQL, FastAPI, Vite)
- **Git + GitHub** — version control
- **Virtual environments** — `.venv` / `.env` Python environments (PowerShell activation)
- **Kaggle** — dataset source for ML projects

### Databases
- **PostgreSQL** — primary production database
- **ChromaDB** — vector/embedding database

---

## 📁 Known Projects (Chronological)

### 1. 🤖 Analyze User Behavior System (April 2026)
- **Stack:** FastAPI + XGBoost + scikit-learn + imbalanced-learn + React frontend
- **Purpose:** Predict/classify user behavior patterns
- **Key tech:** `python -m uvicorn app.main:app --reload`, `pip install -r requirements.txt`
- **Path:** `d:\analyze_user_behavior-main\`

### 2. 📚 LUMINA Book Recommender System (April–May 2026)
- **Stack:** FastAPI + PostgreSQL + ChromaDB + React 18 + Vite + TailwindCSS v4 + Docker
- **Purpose:** AI-powered book discovery, semantic search, and recommendation
- **Features built:**
  - JWT authentication (login/register)
  - BERT-based semantic search (ChromaDB)
  - Hybrid recommender: Content-Based + Matrix Factorization (SVD via scikit-surprise)
  - Transition Matrix session-based recommender
  - Reader/Staff/Admin roles
  - Chat/Customer Support (polling-based messaging)
  - Wishlist (Favorites) management
  - Order/Cart/Checkout + Payment tracking
  - Admin dashboard with KPIs and system logs
- **Design:** Bauhaus aesthetic, 0px border-radius, `#2563EB` blue accent, Inter font
- **Path:** `d:\OneDrive\Documents\GitHub\BOOK-RECOMMENDER-SYSTEM\`

### 3. 🔐 Information Security (ATTT) Project (April–May 2026)
- **Purpose:** Academic thesis on cryptographic algorithms
- **Algorithms implemented (from scratch in Python):**
  - **AES-128-CBC** — Symmetric block cipher, 10 rounds (SubBytes, ShiftRows, MixColumns, AddRoundKey, Key Expansion), IV prepend
  - **RSA** — Asymmetric encryption (prime generation, modular exponentiation)
  - **SHA-256** — Cryptographic hash function
  - **Digital Signature** — RSA-based signing and verification
- **Files:** `AES.py`, `RSA.py`, `SHA256.py`, `DigitalSignature.py`, `random_prime.py`
- **Path:** `d:\ATTT\`

### 4. 🗄️ Database Design Project — TKCSDL (April–May 2026)
- **Stack:** FastAPI + PostgreSQL + SQLAlchemy + Docker
- **Purpose:** Academic training management database system
- **Tables:** MON_HOC, GIANG_VIEN, SINH_VIEN, LOP_HP, KQHT (grades)
- **Architecture patterns used:**
  - Domain interfaces (ABCs) for repository layer decoupling
  - Dependency injection for database sessions
  - Validators for business rule enforcement
  - Service layer separation
- **Path:** `d:\TKCSDL\D-n-TKCSDL\`

### 5. 🏠 House Price Prediction — TTNT (April–June 2026)
- **Dataset 1:** HCM rental data (`Thue_Nha_HCM.xlsx`) — Vietnamese property scrape
- **Dataset 2:** Ames Iowa housing data (Kaggle)
- **Models:** Random Forest (from notebook), preprocessing pipelines
- **Work done:** EDA, data cleaning, feature engineering, outlier handling (IQR capping)
- **Path:** `d:\TTNT\`

### 6. 📧 Spam Email Classification — Current Project (2026-08-05)
- **Dataset:** Kaggle email CSV (`email.csv`)
- **Strict constraint:** NO scikit-learn, NO XGBoost — from-scratch with NumPy only
- **Models to implement:**
  - Decision Trees (Gini/Entropy)
  - Random Forest (Bagging + Feature Subsampling)
  - Gradient Boosting Machines (pseudo-residuals)
- **Custom implementations:** TF-IDF vectorizer, train/test split, all metrics
- **Path:** `d:\Machine Learning\máy học\`
