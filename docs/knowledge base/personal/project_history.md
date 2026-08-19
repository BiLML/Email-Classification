# 📖 Project History & Conversation Flashback

> Complete timeline of all projects and conversations with the AI assistant
> Reconstructed from brain logs — 2026-08-05

---

## Timeline Overview

| Date | Project | Key Topics |
|------|---------|-----------|
| 2026-04-23 | TKCSDL (Database) | FastAPI backend setup, Docker |
| 2026-04-25 | Analyze User Behavior | Run FastAPI + React frontend |
| 2026-04-27 | LUMINA Book Recommender | Matrix Factorization + SVD plan |
| 2026-04-28 | LUMINA | Full frontend + backend build |
| 2026-04-28 | ATTT (Security) | Implement Chapter 4 (Digital Signature) |
| 2026-04-29 | TKCSDL | Database entity analysis, relationships |
| 2026-04-30 | LUMINA | Add cancel order feature for users |
| 2026-05-03 | LUMINA | Use case diagrams (customer support + book management) |
| 2026-05-05 | LUMINA | System architecture review (all technologies) |
| 2026-05-05 | LUMINA | Write README / run guide from scratch |
| 2026-05-08 | TKCSDL | Create `.env` file for Docker deployment |
| 2026-06-06 | TTNT (House prices) | Python virtual environment setup issue |
| 2026-06-06 | TTNT | Fix swipL terminal error |
| 2026-06-10 | TTNT | Split HCM rental price data (triệu/tháng vs nghìn/m²) |
| 2026-07-01 | ML from scratch | Review ML notebooks (KNN, Naive Bayes, k-means, regression) |
| 2026-07-03 | TTNT | Analyze house price prediction notebooks |
| 2026-08-05 | Spam Classification | Plan spam email classifier (Decision Tree, RF, GBM) |

---

## Project 1: Analyze User Behavior System
**Date:** April 25, 2026
**Path:** `d:\analyze_user_behavior-main\`
**Goal:** Classify/predict user behavior using ML
**Stack:** FastAPI + XGBoost + scikit-learn + imbalanced-learn (SMOTE) + React frontend
**Key conversation:** Run backend + frontend

**Run commands:**
```powershell
# Backend
.env/Scripts/Activate.ps1
cd back-end
pip install -r requirements.txt
python -m uvicorn app.main:app --reload

# Frontend (separate terminal)
cd front-end
npm install
npm run dev
```

---

## Project 2: LUMINA Book Recommender System
**Date:** April–May 2026
**Path:** `d:\OneDrive\Documents\GitHub\BOOK-RECOMMENDER-SYSTEM\`
**Goal:** AI-powered book platform with semantic search and hybrid recommendations

**Conversations summary:**
1. **Apr 27** — Plan Matrix Factorization + SVD hybrid recommender
2. **Apr 28** — Build full frontend (React 18 + Vite + TailwindCSS v4) + backend APIs
3. **Apr 30** — Add cancel order functionality for readers
4. **May 03** — Design use case diagrams (customer care, book management)
5. **May 05** — Full system architecture review document
6. **May 05** — Complete README guide for running from scratch

**Architecture built:**
- **Backend:** FastAPI + PostgreSQL + ChromaDB + SQLAlchemy + Alembic + JWT auth
- **Frontend:** React 18 + TypeScript + Vite + TailwindCSS v4 + Axios
- **ML Models:** BERT semantic search, SVD collaborative filtering, transition matrix session recommender
- **Docker:** 3-service compose (db + backend + frontend)

**Features implemented:**
- JWT login/register with role guards (Reader, Staff, Admin)
- Semantic book search (ChromaDB + BERT embeddings)
- Hybrid recommendation = 0.5 × content_score + 0.5 × normalized_SVD_score
- Real-time polling chat (Reader ↔ Staff, every 3 seconds)
- Wishlist/Favorites with toggle + quick-add on hover
- Cart → Checkout → Payment tracking
- Admin dashboard with KPIs and logs

**Design system:**
- Bauhaus aesthetic: 0px border-radius, bold typography, `#2563EB` blue
- Inter font, Shadcn/Radix UI components

**Key bugs fixed:**
- Missing CORS middleware on backend
- Empty controllers folder (needed to add all route handlers)
- Vector DB (`chroma_db/`) exclusion from `.gitignore`
- Display limit increased from 5 → 50–100 books in Discovery view

---

## Project 3: Information Security (ATTT) Thesis
**Date:** April 28, 2026
**Path:** `d:\ATTT\`
**Goal:** Academic thesis implementing cryptographic algorithms from scratch

**Algorithms implemented:**
- **Chapter 1:** AES-128-CBC (`AES.py`)
- **Chapter 2:** RSA (`RSA.py`, `random_prime.py`)
- **Chapter 3:** SHA-256 (`SHA256.py`)
- **Chapter 4:** Digital Signature (`DigitalSignature.py`)

**Key conversation:** "tôi đã xong chương 3, làm chương 4 cho tôi" → implemented RSA-based digital signature

---

## Project 4: TKCSDL — Database Design System
**Date:** April–May 2026
**Path:** `d:\TKCSDL\D-n-TKCSDL\Đồ Án\`
**Goal:** Academic training management database system

**Entities:** MON_HOC, GIANG_VIEN, SINH_VIEN, LOP_HP, KQHT
**Relationships:** 1-N (lớp học phần → kết quả), N-N (giảng viên ↔ môn học via association table)

**Architecture patterns applied:**
- Domain interfaces (ABCs) — `IBaseRepository[T]`, `IMonHocRepository`, etc.
- Repository pattern with dependency injection (no `@staticmethod`)
- Validator extraction for business rules
- Service layer for business logic

**Key conversation:** "mô tả chi tiết thực thể, mối quan hệ, thuộc tính chức năng của từng bảng"

**Docker issue encountered:** Missing `.env.docker` file
```
env file .env.docker not found
```
**Fix:** Create `.env.docker` from `.env.docker.example`

---

## Project 5: House Price Prediction (TTNT)
**Date:** April–July 2026
**Path:** `d:\TTNT\`
**Goal:** Predict house/rental prices using ML

**Two datasets:**
1. **HCM Rental** (`Thue_Nha_HCM.xlsx`) — Vietnamese property listings
2. **Ames Iowa** (Kaggle) — classic ML benchmark

**Key preprocessing work:**
- Vietnamese price text parsing (`triệu/tháng` → float)
- Split dataset by price unit type (triệu/tháng vs nghìn/m²)
- IQR capping for outliers (better than dropping for small datasets)
- Feature engineering: `TotalSF`, `TotalBathrooms`, `HouseAge`

**Code review findings on Kaggle project:**
- Bug: `MasVnrArea` fill inside loop runs 15× unnecessarily
- Bug: `join='inner'` in `.align()` drops training features → use `join='left'`

**Environment issues:**
- June 6, 2026: Running `swipL` command → error; needed terminal fix
- June 6, 2026: Venv creation issue: `-m .env venv` (wrong) → `python -m venv .venv` (correct)

---

## Project 6: ML Coursework Notebooks
**Date:** July 2026
**Path:** `d:\Machine Learning\ML\`
**Notebooks studied:**
- `Khái niệm ML.ipynb` — ML fundamentals
- `Hồi quy tuyến tính.ipynb` — Linear Regression
- `K-lân cận.ipynb` — KNN classifier
- `Phân loại Naive Bayes.ipynb` — Naive Bayes classifier
- `Overfitting.ipynb` — Bias-variance tradeoff
- `k-mean.ipynb` — K-Means clustering

---

## Project 7: Spam Email Classification — CURRENT
**Date:** August 5, 2026
**Path:** `d:\Machine Learning\máy học\`
**Status:** Planning phase

**Constraint:** Complete from-scratch implementation using only `numpy`, `pandas`, `math`

**Plan:**
- Custom TF-IDF vectorizer
- Decision Tree (Gini/Entropy splitting)
- Random Forest (Bagging + Feature Subsampling)
- Gradient Boosting Machine (pseudo-residuals)
- Custom train/test split, all evaluation metrics

**Dataset:** `d:\Machine Learning\máy học\data\email.csv`

---

## AI Assistant Interaction Patterns

### User prefers:
- 🇻🇳 Vietnamese for requests, English for technical docs/code
- Direct, working code implementations (not simulations)
- Mathematical rigor with formulas in docstrings
- OOP patterns with `.fit()/.predict()` conventions
- Saving to Markdown files for documentation

### Common request patterns:
- "làm ... cho tôi" — implement this feature for me
- "chạy backend và frontend" — set up and run the system
- Providing context via open documents and file references
- Asking for architecture plans before implementation
