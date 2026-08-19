# 🗂️ Knowledge Base Index

> Last updated: 2026-08-05 | Author: AI-assisted flashback

This knowledge base captures all coding skills, project history, and domain expertise
derived from the complete conversation history between the user and AI assistant.

---

## 📂 Structure

```
knowledge base/
├── INDEX.md                          ← This file
│
├── personal/
│   ├── coding_profile.md             ← Who you are: tech stack, languages, identity
│   └── project_history.md            ← Full timeline of all projects & conversations
│
├── domain/
│   ├── ml_algorithms.md              ← ML algorithms, math formulas, from-scratch patterns
│   ├── fullstack_web.md              ← FastAPI, SQLAlchemy, React, Docker patterns
│   └── information_security.md      ← AES, RSA, SHA-256, Digital Signature
│
└── general/
    └── coding_standards.md           ← OOP rules, TDD, Git workflow, failure patterns
```

---

## 🚀 Quick Reference

### "What tech do I know?"
→ See [personal/coding_profile.md](personal/coding_profile.md)

### "What projects have I built?"
→ See [personal/project_history.md](personal/project_history.md)

### "How do I implement Decision Trees / GBM / TF-IDF from scratch?"
→ See [domain/ml_algorithms.md](domain/ml_algorithms.md)

### "How do I set up FastAPI + PostgreSQL + Docker?"
→ See [domain/fullstack_web.md](domain/fullstack_web.md)

### "What are the AES/RSA algorithm steps?"
→ See [domain/information_security.md](domain/information_security.md)

### "What are my coding rules and workflow?"
→ See [general/coding_standards.md](general/coding_standards.md)

---

## 📌 Key Facts to Remember

| Fact | Detail |
|------|--------|
| Current project | Spam email classification (from-scratch with NumPy only) |
| Dataset | `d:\Machine Learning\máy học\data\email.csv` |
| Forbidden libs | scikit-learn, xgboost, lightgbm — not allowed in current project |
| Python venv | `python -m venv .venv` then `.venv\Scripts\Activate.ps1` |
| LUMINA path | `d:\OneDrive\Documents\GitHub\BOOK-RECOMMENDER-SYSTEM\` |
| ATTT path | `d:\ATTT\` |
| TTNT path | `d:\TTNT\` |
| TKCSDL path | `d:\TKCSDL\D-n-TKCSDL\Đồ Án\` |

---

## 🔑 Most Critical Rules (Never Forget)

1. **No data leakage** — TF-IDF `.fit()` only on train set, `.transform()` on test
2. **Vectorize everything** — No nested Python for-loops for math
3. **Always set `max_depth`** — Unlimited tree depth causes OOM
4. **Feature subsampling in RF** — `sqrt(n_features)` per split node
5. **Learning rate in GBM** — Keep < 0.3 to prevent gradient explosion
6. **`.reshape(-1,1)`** — When NumPy shapes mismatch in matrix operations
7. **Docker gitignore** — `chroma_db/`, `.env`, `*.pkl` must always be excluded
