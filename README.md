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

## Development server

Run the application with auto-reload:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 4000 --reload
```

The root endpoint `/` returns a simple health payload. Sample item routes demonstrate storing and retrieving rows in SQLite.
