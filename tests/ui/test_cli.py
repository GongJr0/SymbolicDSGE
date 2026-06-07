from pathlib import Path
from typing import Any

import uvicorn
from fastapi.testclient import TestClient

from SymbolicDSGE.ui import cli


def test_bundled_ui_serves_spa_routes_and_favicon(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    index = "<html><body>SymbolicDSGE UI</body></html>"
    favicon = "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    (tmp_path / "index.html").write_text(index, encoding="utf-8")
    (tmp_path / "favicon.svg").write_text(favicon, encoding="utf-8")
    captured: dict[str, Any] = {}

    def capture_run(app: Any, *, host: str, port: int) -> None:
        captured.update(app=app, host=host, port=port)

    monkeypatch.setattr(cli, "_STATIC_DIR", tmp_path)
    monkeypatch.setattr(uvicorn, "run", capture_run)

    cli.run_server(host="127.0.0.1", port=8765, open_browser=False)

    client = TestClient(captured["app"])
    assert client.get("/estimation").text == index
    assert client.get("/builder").text == index
    assert client.get("/favicon.ico").text == favicon
    assert client.get("/missing.js").status_code == 404
    assert client.get("/api/missing").status_code == 404
    assert client.get("/_lsp/config").status_code == 404
