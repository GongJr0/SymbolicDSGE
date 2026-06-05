from importlib import import_module
from typing import Any

from . import elastic_net, lasso, ols, ridge
from .result import RegressionResult
from .enums import RegressionKind, RegressionStatus

__all__ = [
    "RegressionKind",
    "RegressionResult",
    "RegressionStatus",
    "elastic_net",
    "lasso",
    "ols",
    "ridge",
    "sr",
]


def __getattr__(name: str) -> Any:
    if name == "sr":
        return import_module(f"{__name__}.sr")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
