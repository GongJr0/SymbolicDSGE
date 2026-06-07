# SymbolicDSGE UI

The web playground for SymbolicDSGE. The **backend** now lives inside the
installable package at `SymbolicDSGE/ui/`; this directory holds the **frontend
source** only. The built frontend is bundled into the wheel as package data
(`SymbolicDSGE/ui/_static/`) at release time.

## Running the shipped UI

With the `ui` extra installed (`pip install 'SymbolicDSGE[ui]'`):

```bash
sdsge-ui            # launches backend + bundled frontend, opens a browser
```

Or from Python, with a solved model preloaded as the reference:

```python
solved.serve()
```

## Frontend development

The bundled assets only refresh when you rebuild. For live development, run the
backend and the Vite dev server separately.

Backend (from the repository root):

```bash
uv run --extra ui python -m SymbolicDSGE.ui --no-browser --port 8000
```

Frontend (from `sdsge-ui/frontend`):

```bash
npm install
npm run dev
```

The frontend defaults to `http://127.0.0.1:8000` for API calls; set
`VITE_SDSGE_API_BASE` to point it elsewhere.

## Building the bundle manually

```bash
python scripts/build_frontend.py
```

This runs `npm ci && npm run build` with a relative API base and copies the
output into `SymbolicDSGE/ui/_static/`. CI does this automatically during the
release (`publish.yml`); you only need it for a local build or to test packaging.
