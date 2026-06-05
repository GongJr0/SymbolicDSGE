# SymbolicDSGE UI Prototype

This directory contains the first-pass localhost playground for SymbolicDSGE.
It is intentionally separate from the installable Python package while the API
and frontend workflow settle.

## Backend

From the repository root:

```powershell
$env:PYTHONPATH = "sdsge-ui/backend"
uv run --extra ui python -m sdsge_ui_backend
```

The backend serves `http://127.0.0.1:8000`.

## Frontend

From `sdsge-ui/frontend`:

```bash
npm install
npm run dev
```

The frontend defaults to `http://127.0.0.1:8000` for API calls. Set
`VITE_SDSGE_API_BASE` to point it elsewhere.
