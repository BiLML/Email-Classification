# 🌐 Full-Stack Web Development Knowledge Base

> Derived from: LUMINA Book Recommender, TKCSDL DB project, User Behavior Analysis
> Stack: FastAPI + PostgreSQL + React + TypeScript + Docker

---

## 🏗️ Architecture Pattern (FastAPI Backend)

```
backend/
├── src/
│   ├── domain(interface)/       # Abstract Base Classes (interfaces)
│   │   ├── base_repository.py   # IBaseRepository[T] — get_all, get_by_id, create, update, delete
│   │   └── *.py                 # Entity-specific interfaces
│   ├── repositories/            # Concrete SQLAlchemy implementations
│   │   └── *.py                 # Inherit interfaces, use db session injection
│   ├── models/                  # SQLAlchemy ORM models
│   ├── schemas/                 # Pydantic request/response schemas
│   ├── services/                # Business logic layer
│   ├── controllers/             # FastAPI routers/endpoints
│   ├── validators/              # Business rule checks (existence, uniqueness)
│   └── server/
│       └── main.py              # FastAPI app setup, CORS, middleware
```

### Repository Pattern (Correct Implementation)
```python
# WRONG — Static methods cause tight coupling
class BookRepository:
    @staticmethod
    def get_all(db: Session) -> list[Book]: ...

# CORRECT — Instance method + dependency injection
class BookRepository(IBookRepository):
    def __init__(self, db: Session):
        self.db = db
    
    def get_all(self) -> list[Book]:
        return self.db.query(Book).all()
```

---

## ⚡ FastAPI Patterns

### App Setup (main.py)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### JWT Auth Pattern
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
```

### Dependency Injection (DB Session)
```python
from sqlalchemy.orm import Session
from database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/items")
def get_items(db: Session = Depends(get_db)):
    return ItemRepository(db).get_all()
```

### Running FastAPI
```powershell
# Local dev
python -m uvicorn app.main:app --reload

# With specific host/port
python -m uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🗄️ SQLAlchemy + PostgreSQL

### ORM Model Pattern
```python
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
import datetime

class Book(Base):
    __tablename__ = "books"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    author = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user_actions = relationship("UserAction", back_populates="book")
```

### Many-to-Many Relationship
```python
# Association table
giang_vien_mon_hoc = Table(
    'giang_vien_mon_hoc',
    Base.metadata,
    Column('ma_gv', String, ForeignKey('giang_vien.ma_gv'), primary_key=True),
    Column('ma_mon', String, ForeignKey('mon_hoc.ma_mon'), primary_key=True),
)

# In model
class GiangVien(Base):
    mon_hoc_list = relationship("MonHoc", secondary=giang_vien_mon_hoc)
```

### Alembic Migrations
```bash
alembic init alembic
alembic revision --autogenerate -m "add messages table"
alembic upgrade head
```

---

## 🐳 Docker Compose Pattern (3-Service Stack)

```yaml
# docker-compose.yml
version: '3.8'
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@db/${DB_NAME}
    depends_on:
      - db
    ports:
      - "8000:8000"

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  postgres_data:
```

### .env File Pattern
```env
DB_NAME=mydb
DB_USER=postgres
DB_PASSWORD=secret123
SECRET_KEY=your-jwt-secret-key
```

### Docker Commands
```powershell
docker compose up          # Start all services
docker compose up -d       # Start detached
docker compose down        # Stop all
docker compose down -v     # Stop and remove volumes
docker compose build       # Rebuild images
```

### Common Docker Issue
```
env file .env.docker not found
```
**Fix:** Create `.env.docker` file or check `docker-compose.yml` env_file reference.

---

## ⚛️ React + TypeScript Frontend

### Project Setup (Vite + React + TailwindCSS v4)
```json
// package.json key dependencies
{
  "react": "^18.0.0",
  "react-router-dom": "^6.0.0",
  "typescript": "^5.0.0",
  "vite": "^5.0.0",
  "tailwindcss": "^4.0.0",
  "axios": "^1.0.0",
  "lucide-react": "latest"
}
```

### Axios HTTP Client with JWT
```typescript
// src/api/client.ts
import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export default api;
```

### Auth Context Pattern
```typescript
// src/context/AuthContext.tsx
interface AuthContextType {
  user: User | null;
  login: (token: string, user: User) => void;
  logout: () => void;
}

export const AuthContext = createContext<AuthContextType>({} as AuthContextType);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  // JWT persistence via localStorage
  ...
}
```

### Protected Routes
```typescript
// In App.tsx
<Route path="/dashboard" element={
  user ? <Dashboard /> : <Navigate to="/login" />
} />

// Role-based guard
{user?.role === 'admin' && <AdminDashboard />}
```

### Vite Proxy Config (avoid CORS in dev)
```typescript
// vite.config.ts
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
});
```

---

## 🔍 Semantic Search (ChromaDB + BERT)

### ChromaDB Setup
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("books")
```

### Add documents
```python
collection.add(
    documents=["book text content..."],
    metadatas=[{"title": "Book Name", "author": "Author"}],
    ids=["book_1"]
)
```

### Query
```python
results = collection.query(
    query_texts=["what I'm searching for"],
    n_results=10
)
```

### .gitignore — always exclude `chroma_db/`
```gitignore
chroma_db/
*.pkl
*.docx
*.pdf
.env
.idea/
.vscode/
data/temp_*.json
```

---

## 💬 Real-Time Messaging (Polling Pattern)

> Used in LUMINA for customer support chat between Reader and Staff

```python
# Backend: POST /messages, GET /messages?conversation_id=...
# Poll every 3 seconds on frontend

useEffect(() => {
  const interval = setInterval(async () => {
    const msgs = await fetchMessages(conversationId);
    setMessages(msgs);
  }, 3000);
  return () => clearInterval(interval);
}, [conversationId]);
```

---

## 🎨 LUMINA Design System Reference

| Token | Value |
|-------|-------|
| Primary accent | `#2563EB` (blue) |
| Border radius | `0px` (Bauhaus geometric) |
| Typography | Inter font |
| Design language | Bauhaus aesthetic |
| Component library | Shadcn/Radix UI |

---

## 🔌 API Design Conventions

### REST Endpoint Naming
```
GET    /books              → list all books
GET    /books/{id}         → get single book
POST   /books              → create book
PUT    /books/{id}         → update book
DELETE /books/{id}         → delete book

POST   /auth/login         → login
POST   /auth/register      → register
GET    /auth/me            → get current user profile

POST   /favorites/toggle   → toggle favorite status
GET    /favorites/me       → get my favorites

POST   /orders             → create order
GET    /orders/me          → get my orders
PATCH  /orders/{id}/status → update order status (admin)

GET    /recommendations/{book_id}  → hybrid recommendations
POST   /search             → semantic search
```

### Pydantic Schema Pattern
```python
from pydantic import BaseModel
from datetime import datetime

class BookCreate(BaseModel):
    title: str
    author: str
    description: str | None = None

class BookResponse(BookCreate):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True
```
