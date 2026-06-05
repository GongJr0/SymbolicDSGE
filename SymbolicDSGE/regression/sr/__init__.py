from importlib.util import find_spec

if find_spec("pysr") is None:
    raise ImportError(
        "Symbolic regression requires the 'sr' optional dependency. "
        "Install it with 'pip install SymbolicDSGE[sr]' or an equivalent "
        "environment command before importing SymbolicDSGE.regression.sr."
    )

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
