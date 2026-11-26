"""Dynare golden dumps, read from the ``.mat`` files in this directory."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import mat_struct

GOLDENS_DIR = Path(__file__).resolve().parent


def load_dynare(name: str) -> SimpleNamespace:
    """Read ``<name>.mat`` into one attribute per saved variable.

    Only storage layout is normalized: cell arrays of names become tuples,
    arrays become C-contiguous, and 3-D arrays move their trailing period axis
    to the front, so ``variance[t]`` is Dynare's ``Variance(:, :, t + 1)``.
    """
    raw = loadmat(GOLDENS_DIR / f"{name}.mat", struct_as_record=False, squeeze_me=True)
    return SimpleNamespace(
        **{
            key: _convert(value)
            for key, value in raw.items()
            if not key.startswith("__")
        }
    )


def _convert(value: object) -> object:
    if isinstance(value, mat_struct):
        return SimpleNamespace(
            **{field: _convert(getattr(value, field)) for field in value._fieldnames}
        )
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return tuple(str(item) for item in value)
        if value.ndim == 3:
            value = np.moveaxis(value, -1, 0)
        return np.ascontiguousarray(value)
    return value
