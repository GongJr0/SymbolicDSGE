# SymbolicDSGE UI Backend

This is the first-pass localhost backend for the interactive model playground.
It is intentionally kept outside the installable library package while the UI
surface settles.

From the repository root:

```powershell
$env:PYTHONPATH = "sdsge-ui/backend"
uv run --extra ui python -m sdsge_ui_backend
```

The server binds to `127.0.0.1:8000` by default.
