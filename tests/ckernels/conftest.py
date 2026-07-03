"""Shared setup for the native-kernel tests.

Several of these tests load model configs by repo-relative path (``MODELS/...``,
``tests/fixtures/...``). Under cibuildwheel the wheel is tested with
``pytest {project}/tests/ckernels`` from a *temporary* working directory, so
those relative paths would not resolve. Anchor the working directory at the repo
root for the session (restored afterwards). The tests do their file I/O at run
time -- never in a parametrize decorator -- so this session-autouse fixture takes
effect before any model is loaded.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def _chdir_repo_root():
    prev = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)
