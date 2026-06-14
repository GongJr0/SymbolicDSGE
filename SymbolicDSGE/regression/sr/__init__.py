"""Symbolic regression integration.

The submodules type-hint pysr classes via :data:`typing.TYPE_CHECKING` and only
``from pysr import ...`` inside the functions that actually need the Julia
runtime, so importing this package without the ``[sr]`` extra succeeds. The
``ImportError`` surfaces lazily at first use of a pysr-backed function.
"""

from .model_parametrizer import ModelParametrizer
from .model_defaults import make_operator_general, CustomOp, PySRParams
from .config import TemplateConfig
from .symbolic_regression import SymbolicRegressor

__all__ = [
    "ModelParametrizer",
    "make_operator_general",
    "CustomOp",
    "PySRParams",
    "TemplateConfig",
    "SymbolicRegressor",
]
