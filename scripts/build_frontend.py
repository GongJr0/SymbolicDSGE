"""Build the SymbolicDSGE web frontend and stage it inside the package.

Runs an npm install + ``npm run build`` in ``sdsge-ui/frontend`` with a relative
API base (so the bundle is served same-origin), then copies the Vite ``dist``
output into ``SymbolicDSGE/ui/_static`` where it ships as package data.

Used by CI at release time and by ``local_publish.py``. Requires Node/npm; this
is a build-time dependency only and is never needed by end users.

By default it runs ``npm ci`` for a clean, reproducible install (what CI wants).
On Windows, ``npm ci`` wipes ``node_modules`` first and can fail with EPERM if a
file such as ``esbuild.exe`` is locked by a running dev server or antivirus. For
local rebuilds you can avoid that:

    python scripts/build_frontend.py --skip-install   # build against existing node_modules
    python scripts/build_frontend.py --install install  # non-destructive `npm install`
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "sdsge-ui" / "frontend"
STATIC_DIR = ROOT / "SymbolicDSGE" / "ui" / "_static"
DIST_DIR = FRONTEND_DIR / "dist"

# Empty string => api.ts falls back to relative requests (same-origin serving).
BUILD_ENV = {"VITE_SDSGE_API_BASE": ""}


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}  (cwd={FRONTEND_DIR})")
    # shell=True on Windows so the npm/npm.cmd shim resolves correctly.
    subprocess.run(
        cmd,
        cwd=FRONTEND_DIR,
        check=True,
        env={**os.environ, **BUILD_ENV},
        shell=os.name == "nt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install",
        choices=["ci", "install"],
        default="ci",
        help="npm install mode: 'ci' (clean, default) or 'install' (in-place).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip the npm install step and build against existing node_modules.",
    )
    args = parser.parse_args()

    if not FRONTEND_DIR.is_dir():
        sys.exit(f"Frontend source not found at {FRONTEND_DIR}")

    if args.skip_install:
        if not (FRONTEND_DIR / "node_modules").is_dir():
            sys.exit(
                "--skip-install was given but node_modules is missing; "
                "run an install first."
            )
        print("Skipping npm install (using existing node_modules).")
    else:
        _run(["npm", args.install])

    _run(["npm", "run", "build"])

    if not DIST_DIR.is_dir():
        sys.exit(f"Build did not produce {DIST_DIR}")

    shutil.rmtree(STATIC_DIR, ignore_errors=True)
    shutil.copytree(DIST_DIR, STATIC_DIR)
    print(f"Staged frontend -> {STATIC_DIR}")


if __name__ == "__main__":
    main()
