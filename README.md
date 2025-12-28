## PyTutor AI Backend

Backend cho hệ thống luyện tập Python + gia sư AI (FastAPI + SQLAlchemy + Docker sandbox + Qdrant RAG).

### Cấu trúc thư mục (tóm tắt)

- `backend/app/`: cấu hình ứng dụng FastAPI (settings, db, auth, main)
- `backend/api/routers/`: các API routes (problems, submissions, ai_tutor, admin, system)
- `backend/domain/`: logic miền (AI tutor/analyzer) + ORM models
- `backend/infra/`: hạ tầng (Docker executor, scheduler, phân tích code, utils)
- `backend/migrations/`: SQL migrations (PostgreSQL)

### Chạy local (khuyến nghị)

- **Bước 1**: tạo môi trường và cài dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

- **Bước 2**: cấu hình biến môi trường (tạo `backend/.env` nếu cần).

Gợi ý: copy từ `backend/env.example` và chỉ sửa những biến bạn thật sự dùng.

```env
# TỐI THIỂU (production nên set)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pytutor
SECRET_KEY=change-me-in-production

# OPTIONAL (có default):
# - Qdrant: nếu không set -> dùng in-memory (dev)
# QDRANT_URL=
# QDRANT_API_KEY=
#
# - LLM: nếu không set -> hint/chat sẽ fallback template hoặc trả thông báo (tuỳ endpoint)
# GROQ_API_KEY=
# GROQ_MODEL=llama-3.1-8b-instant
#
# - Tắt WS terminal nếu không cần ở production
# ENABLE_WS_TERMINAL=false
```

- **Bước 3**: chạy server:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Migrations (PostgreSQL)

Repo dùng các file SQL trong `backend/migrations/`.

Các migration quan trọng cho hướng A (khóa luận):

- `004_add_learning_telemetry.sql`: tạo `student_hint_interactions` (log hints + 👍/👎) và `concept_mastery` (optional cache)
- `005_add_learning_sessions.sql`: tạo `learning_sessions` (time-to-solve) + thêm `session_id` vào `student_hint_interactions`

Apply (ví dụ):

```bash
psql "$DATABASE_URL" -f backend/migrations/004_add_learning_telemetry.sql
psql "$DATABASE_URL" -f backend/migrations/005_add_learning_sessions.sql
```

### Thesis/Analytics APIs (Direction A)

- `GET /api/ai/mastery` - Mastery theo `problem_types` (được xem là concept chuẩn)
- `GET /api/ai/path` - Learning path dựa trên mastery
- `GET /api/ai/report` - Metrics (time-to-solve, hints-per-solve, helpful rate, attempts)
- `GET /api/ai/report/export?kind=summary|sessions|hints` - Export CSV để chạy notebook/đánh giá

### Ghi chú bảo mật

- **WebSocket terminal** (`/ws/terminal`) cho phép chạy code tương tác trong Docker sandbox. Nếu không cần tính năng Terminal trên UI, nên **tắt** bằng `ENABLE_WS_TERMINAL=false` ở production.

