# SymbolicDSGE UI Frontend

The frontend is a separate Node/Vite project so JavaScript dependencies do not
flow through Python packaging.

```bash
npm install
npm run dev
```

Set `VITE_SDSGE_API_BASE` if the backend is not running at
`http://127.0.0.1:8000`.
