# Backend Architecture - PyTutor AI

## Overview
Backend được tổ chức theo 4 lớp chính để dễ maintain và dễ thay thế từng phần:

- `api/`: nơi khai báo FastAPI routers (HTTP/WebSocket)
- `app/`: wiring & cấu hình ứng dụng (settings, db, auth, FastAPI app)
- `domain/`: logic miền (AI tutor/analyzer, ORM models)
- `infra/`: hạ tầng (Docker sandbox, scheduler, các module phân tích code)

## Project Structure (thực tế)

```
backend/
├── main.py                 # Compatibility shim: uvicorn main:app -> app.main:app
├── app/
│   ├── main.py             # FastAPI app + include routers + lifecycle (startup/shutdown)
│   ├── settings.py         # Đọc env/.env và expose các cấu hình
│   ├── db.py               # SQLAlchemy engine/session + dependency get_db
│   ├── auth.py             # Auth endpoints + JWT helpers (get_current_user, admin)
│   └── ...
├── api/
│   └── routers/
│       ├── problems.py     # Danh sách/chi tiết bài + submit
│       ├── submissions.py  # Xem submissions của user
│       ├── ai_tutor.py     # Hint/chat/visualize/progress
│       ├── admin.py        # Admin endpoints
│       └── system.py       # /api/execute, /api/analyze, /api/config, WS /ws/terminal
├── domain/
│   ├── models/             # ORM models (User/Problem/TestCase/Submission/QdrantSchedule)
│   └── ai/                 # Tutor/analyzer + Qdrant RAG
├── infra/
│   ├── services/           # executor/docker_manager/scheduler
│   ├── analysis/           # AST/CFG/DFG/static/runtime analysis
│   └── utils/              # llm utils, normalize_code...
└── migrations/             # SQL migrations
```

## API Endpoints

### Authentication (`/auth`)
- `POST /auth/register` - Đăng ký user mới
- `POST /auth/login` - Đăng nhập, nhận JWT token

### Problems (`/problems`)
- `GET /problems` - Danh sách bài tập (có filter, search, pagination)
- `POST /problems/{id}/submit` - Submit code để chấm điểm

### Code Execution & Analysis (`/api`)
- `POST /api/execute` - Chạy code trong sandbox + phân tích
- `POST /api/analyze` - Chỉ phân tích code (không chạy)
- `GET /api/config` - Lấy config môi trường execution

### AI Tutor (`/api/ai`)
- `POST /api/ai/analyze` - Phân tích code chi tiết
- `POST /api/ai/hint` - Lấy gợi ý thông minh (Qdrant RAG + LLM)
- `POST /api/ai/hint/feedback` - Feedback 👍/👎 cho hint (telemetry)
- `POST /api/ai/chat` - Chat với gia sư AI
- `POST /api/ai/visualize/cfg` - Control Flow Graph
- `POST /api/ai/visualize/dfg` - Data Flow Graph
- `GET /api/ai/progress` - Theo dõi tiến độ học tập
- `GET /api/ai/mastery` - Mastery theo concept (problem_type)
- `GET /api/ai/path` - Learning path (baseline heuristic)
- `GET /api/ai/report` - Report metrics (time/hints/helpful/attempts)
- `GET /api/ai/report/export?kind=summary|sessions|hints` - Export CSV cho notebook/đánh giá
- `POST /api/ai/session/start` - Start learning session (time-to-solve)
- `POST /api/ai/session/end` - End learning session (solved/abandoned)
- `POST /api/ai/knowledge/add` - Thêm code vào knowledge base
- `GET /api/ai/knowledge/stats` - Thống kê knowledge base
- `POST /api/ai/knowledge/search` - Tìm kiếm semantic

### Admin (`/api/admin`)

#### User Management
- `GET /api/admin/users` - Danh sách users
- `PATCH /api/admin/users/{id}` - Promote/demote admin
- `DELETE /api/admin/users/{id}` - Xóa user

#### Problem Management
- `GET /api/admin/problems` - Danh sách bài tập (admin view)
- `GET /api/admin/problems/{id}` - Chi tiết bài tập
- `POST /api/admin/problems` - Tạo bài tập mới
- `PATCH /api/admin/problems/{id}` - Cập nhật bài tập
- `DELETE /api/admin/problems/{id}` - Xóa bài tập
- `POST /api/admin/problems/{id}/import-submissions` - Import submissions vào Qdrant
- `GET /api/admin/problem-types` - Danh sách loại bài tập

#### System Stats
- `GET /api/admin/stats` - Thống kê hệ thống

#### Qdrant Management
- `POST /api/admin/qdrant/chunk-submissions` - Chunk submissions vào Qdrant
- `POST /api/admin/qdrant/import` - Import file vào Qdrant
- `GET /api/admin/qdrant/stats` - Thống kê Qdrant

#### Scheduler Management
- `GET /api/admin/scheduler/config` - Lấy config scheduler
- `PATCH /api/admin/scheduler/config` - Cập nhật config
- `GET /api/admin/scheduler/schedules` - Danh sách schedules
- `POST /api/admin/scheduler/schedules` - Tạo schedule mới

## Notes

- `/health` hiện vẫn được giữ để tiện kiểm tra tình trạng service.
- `WS /ws/terminal` **đang được frontend sử dụng** (màn hình Terminal). Có thể tắt nhanh bằng env `ENABLE_WS_TERMINAL=false` ở production nếu không cần.

## Key Technologies

- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - ORM for database
- **Qdrant** - Vector database for RAG
- **Docker** - Code execution sandbox
- **JWT** - Authentication
- **Groq/LLM** - AI hint generation

## Environment Variables

```env
DATABASE_URL=sqlite:///./pytutor.db
SECRET_KEY=your-secret-key
QDRANT_URL=your-qdrant-cloud-url
QDRANT_API_KEY=your-qdrant-api-key
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama-3.1-8b-instant
```

## Performance Optimizations

1. **Loại bỏ code dư thừa** - Xóa ~40% endpoints không sử dụng
2. **Simplified imports** - Chỉ import những gì cần thiết
3. **Better error handling** - Consistent error responses
4. **Improved documentation** - Clear API docs
5. **Code organization** - Logical file structure

## Future Improvements

- [ ] Add request rate limiting
- [ ] Implement caching for common queries
- [ ] Add API versioning
- [ ] Enhance logging and monitoring
- [ ] Add comprehensive unit tests
