# nups-api

FastAPI starter configured to run on port 4000 with environment-based settings and an SQLite backend.
Python >= 3.10.x

## Setup

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Adjust values in the `.env` file if needed. `DATABASE_URL` defaults to `sqlite+aiosqlite:///./nups.db`.

## Dummy data seeding

Use `scripts/seed_dummy_data.py` to load a rich set of example users, projects, and processing runs for local testing. The script drops and recreates tables by default so you start from a clean slate.

### 실행 방법
1. (선택) 가상환경을 활성화합니다.
   ```bash
   source .venv/bin/activate
   ```
2. 필요한 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
3. 더미 데이터를 채웁니다.
   ```bash
   python -m scripts.seed_dummy_data
   ```
   - 기본적으로 모든 테이블을 재생성합니다. 기존 데이터를 유지하려면 `--keep-existing` 플래그를 추가하세요.
   - Python 3.10 이상을 권장합니다 (타입 힌트 호환).

## Development server

Run the application with auto-reload:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 4000 --reload
```

The root endpoint `/` returns a simple health payload. Sample item routes demonstrate storing and retrieving rows in SQLite.

## Docker

Build and run the API with Docker Compose from the repository root:
```bash
docker compose up --build
```
The service listens on http://localhost:4000. To stop the stack, press Ctrl+C or run `docker compose down`. The SQLite database is mounted from `nups-api/nups.db`, so the data persists between runs.

