# type: ignore
from __future__ import annotations

import numpy as np
import pytest

from numba import cfunc, types

from SymbolicDSGE.kalman.validator import (
    FilterMode,
    KFValidationContext,
    validate_kf_inputs,
)

_MEAS_SIG = types.void(
    types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.CPointer(types.float64),
)


@cfunc(_MEAS_SIG)
def _ext_meas(x, p, out):
    out[0] = x[0] + p[0]
    out[1] = x[1] + p[0]


@cfunc(_MEAS_SIG)
def _ext_jac(x, p, out):
    # row-major (m=2, n=2) identity jacobian
    out[0] = 1.0
    out[1] = 0.0
    out[2] = 0.0
    out[3] = 1.0


@cfunc(_MEAS_SIG)
def _ext_meas_inf(x, p, out):
    out[0] = np.inf
    out[1] = 0.0


@cfunc(_MEAS_SIG)
def _ext_jac_nan(x, p, out):
    out[0] = 1.0
    out[1] = 0.0
    out[2] = np.nan
    out[3] = 1.0


def _linear_inputs():
    return {
        "filter_mode": FilterMode.LINEAR,
        "A": np.eye(2, dtype=np.float64),
        "B": np.ones((2, 1), dtype=np.float64),
        "Q": np.eye(1, dtype=np.float64),
        "R": np.eye(1, dtype=np.float64),
        "y": np.zeros((4, 1), dtype=np.float64),
        "C": np.ones((1, 2), dtype=np.float64),
        "d": np.zeros((1,), dtype=np.float64),
    }


def _extended_inputs():
    return {
        "filter_mode": FilterMode.EXTENDED,
        "A": np.eye(2, dtype=np.float64),
        "B": np.ones((2, 1), dtype=np.float64),
        "Q": np.eye(1, dtype=np.float64),
        "R": np.eye(2, dtype=np.float64),
        "y": np.zeros((4, 2), dtype=np.float64),
        "x0": np.array([1.0, 2.0], dtype=np.float64),
        "calib_params": np.array([0.5], dtype=np.float64),
        "meas_addr": _ext_meas.address,
        "jac_addr": _ext_jac.address,
    }


@pytest.mark.parametrize(
    ("field", "value", "exc_type", "match"),
    [
        ("A", [[1.0]], TypeError, "A must be a numpy ndarray"),
        ("B", np.ones((2,), dtype=np.float64), ValueError, "B must be 2D"),
        ("A", np.ones((2, 1), dtype=np.float64), ValueError, "A must be square"),
        ("B", np.ones((1, 1), dtype=np.float64), ValueError, "B must have n_state"),
    ],
)
def test_validate_kf_inputs_rejects_bad_shared_transition_inputs(
    field, value, exc_type, match
):
    kwargs = _linear_inputs()
    kwargs[field] = value

    with pytest.raises(exc_type, match=match):
        validate_kf_inputs(**kwargs)


@pytest.mark.parametrize(
    ("mutator", "exc_type", "match"),
    [
        (lambda kw: kw.update({"C": None}), ValueError, "Linear mode requires C and d"),
        (
            lambda kw: kw.update({"C": [[1.0, 1.0]]}),
            TypeError,
            "C must be a numpy ndarray",
        ),
        (
            lambda kw: kw.update({"C": np.ones((2,), dtype=np.float64)}),
            ValueError,
            "C must be 2D",
        ),
        (lambda kw: kw.update({"d": [0.0]}), TypeError, "d must be a numpy ndarray"),
        (
            lambda kw: kw.update({"d": np.zeros((1, 1, 1), dtype=np.float64)}),
            ValueError,
            "d must be 1D or 2D",
        ),
        (
            lambda kw: kw.update({"C": np.ones((1, 3), dtype=np.float64)}),
            ValueError,
            "C must be \\(n_obs x n_state\\)",
        ),
        (
            lambda kw: kw.update({"y": np.zeros((4, 2), dtype=np.float64)}),
            ValueError,
            "y must have n_obs=1 columns",
        ),
        (
            lambda kw: kw.update({"d": np.zeros((2,), dtype=np.float64)}),
            ValueError,
            "d must have shape \\(1,\\)",
        ),
        (
            lambda kw: kw.update({"d": np.zeros((2, 2), dtype=np.float64)}),
            ValueError,
            "d must have shape \\(1,1\\) or \\(1,1\\)",
        ),
    ],
)
def test_validate_kf_inputs_rejects_bad_linear_measurement_inputs(
    mutator, exc_type, match
):
    kwargs = _linear_inputs()
    mutator(kwargs)

    with pytest.raises(exc_type, match=match):
        validate_kf_inputs(**kwargs)


@pytest.mark.parametrize(
    ("mutator", "exc_type", "match"),
    [
        (
            lambda kw: kw.update({"meas_addr": None}),
            ValueError,
            "Extended mode requires meas_addr and jac_addr",
        ),
        (
            lambda kw: kw.update({"jac_addr": None}),
            ValueError,
            "Extended mode requires meas_addr and jac_addr",
        ),
        (
            lambda kw: kw.update({"calib_params": None}),
            ValueError,
            "calib_params must be provided",
        ),
        (
            lambda kw: kw.update({"meas_addr": _ext_meas_inf.address}),
            ValueError,
            "non-finite",
        ),
        (
            lambda kw: kw.update({"jac_addr": _ext_jac_nan.address}),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_validate_kf_inputs_rejects_bad_extended_measurement_inputs(
    mutator, exc_type, match
):
    kwargs = _extended_inputs()
    mutator(kwargs)

    with pytest.raises(exc_type, match=match):
        validate_kf_inputs(**kwargs)


def test_validate_kf_inputs_rejects_unknown_filter_mode():
    kwargs = _linear_inputs()
    kwargs["filter_mode"] = "mystery"

    with pytest.raises(ValueError, match="Unknown filter_mode"):
        validate_kf_inputs(**kwargs)


@pytest.mark.parametrize(
    ("mutator", "exc_type", "match"),
    [
        (
            lambda kw: kw.update({"Q": np.eye(2, dtype=np.float64)}),
            ValueError,
            "Q must be",
        ),
        (
            lambda kw: kw.update({"R": np.eye(2, dtype=np.float64)}),
            ValueError,
            "R must be",
        ),
        (
            lambda kw: kw.update({"x0": [0.0, 0.0]}),
            TypeError,
            "x0 must be a numpy ndarray",
        ),
        (
            lambda kw: kw.update({"x0": np.zeros((2, 1), dtype=np.float64)}),
            ValueError,
            "x0 must be 1D",
        ),
        (lambda kw: kw.update({"P0": [1.0]}), TypeError, "P0 must be a numpy ndarray"),
        (
            lambda kw: kw.update({"P0": np.zeros((2, 1), dtype=np.float64)}),
            ValueError,
            "P0 must be 2D",
        ),
        (
            lambda kw: kw.update({"Q": np.array([[1.0, 2.0]], dtype=np.float64)}),
            ValueError,
            "Q must be",
        ),
        (
            lambda kw: kw.update(
                {"P0": np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)}
            ),
            ValueError,
            "P0 must be symmetric",
        ),
        (
            lambda kw: kw.update({"Q": np.array([[-1.0]], dtype=np.float64)}),
            ValueError,
            "Q must have non-negative diagonal",
        ),
        (
            lambda kw: kw.update({"R": np.array([[-1.0]], dtype=np.float64)}),
            ValueError,
            "R must have non-negative diagonal",
        ),
        (
            lambda kw: kw.update({"P0": np.diag([1.0, -1.0]).astype(np.float64)}),
            ValueError,
            "P0 must have non-negative diagonal",
        ),
    ],
)
def test_validate_kf_inputs_rejects_bad_shared_covariance_and_state_inputs(
    mutator, exc_type, match
):
    kwargs = _linear_inputs()
    kwargs["x0"] = np.zeros((2,), dtype=np.float64)
    kwargs["P0"] = np.eye(2, dtype=np.float64)
    mutator(kwargs)

    with pytest.raises(exc_type, match=match):
        validate_kf_inputs(**kwargs)


def test_validate_kf_inputs_rejects_nonsymmetric_q_and_r_when_shapes_match():
    linear_kwargs = _linear_inputs()
    linear_kwargs["B"] = np.eye(2, dtype=np.float64)
    linear_kwargs["Q"] = np.array([[1.0, 0.3], [0.1, 1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="Q must be symmetric"):
        validate_kf_inputs(**linear_kwargs)

    extended_kwargs = _extended_inputs()
    extended_kwargs["R"] = np.array([[1.0, 0.5], [0.1, 1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="R must be symmetric"):
        validate_kf_inputs(**extended_kwargs)


def test_validate_kf_inputs_accepts_valid_linear_and_extended_inputs():
    linear = validate_kf_inputs(**_linear_inputs())
    extended = validate_kf_inputs(**_extended_inputs())

    assert isinstance(linear, KFValidationContext)
    assert linear == KFValidationContext(n_state=2, n_obs=1, n_shock=1, T=4)
    assert extended == KFValidationContext(n_state=2, n_obs=2, n_shock=1, T=4)


def test_validate_kf_inputs_extended_zero_probe_state_branch():
    # probe_state="zeros" probes the measurement cfunc at the zero state instead
    # of x0; a valid measurement still yields the expected context.
    out = validate_kf_inputs(**_extended_inputs(), probe_state="zeros")

    assert out == KFValidationContext(n_state=2, n_obs=2, n_shock=1, T=4)
