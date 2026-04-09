# type: ignore
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numba import njit

from SymbolicDSGE import _linearsolve as linearsolve


def _series_equations(fwd, cur, par):
    return np.array([fwd["x"] - par["rho"] * cur["x"]], dtype=np.complex128)


@njit
def _array_equations_numeric(fwd, cur, par):
    return np.array([fwd[0] - par[0] * cur[0]], dtype=np.complex128)


def test_normalize_parameter_input_variants_cover_success_and_errors():
    empty, names = linearsolve._normalize_parameter_input(None, parameter_names=["rho"])
    assert empty.shape == (0,)
    assert names == ["rho"]

    mapping, inferred_names = linearsolve._normalize_parameter_input(
        {"rho": 0.9, "beta": 0.99}
    )
    assert mapping == {"rho": 0.9, "beta": 0.99}
    assert inferred_names == ["rho", "beta"]

    series = pd.Series([0.9, 0.99], index=["rho", "beta"], dtype=float)
    labeled, labeled_names = linearsolve._normalize_parameter_input(series)
    assert labeled is series
    assert labeled_names == ["rho", "beta"]

    array, array_names = linearsolve._normalize_parameter_input(
        np.array([0.9], dtype=float),
        parameter_names=["rho"],
    )
    assert np.array_equal(array, np.array([0.9], dtype=float))
    assert array_names == ["rho"]

    with pytest.raises(ValueError, match="parameter_names length"):
        linearsolve._normalize_parameter_input(
            {"rho": 0.9}, parameter_names=["rho", "beta"]
        )

    with pytest.raises(ValueError, match="parameter_names length"):
        linearsolve._normalize_parameter_input(series, parameter_names=["rho"])

    with pytest.raises(ValueError, match="parameter_names is required"):
        linearsolve._normalize_parameter_input(np.array([0.9], dtype=float))


def test_normalize_named_vector_variants_cover_reordering_and_errors():
    out_mapping = linearsolve._normalize_named_vector(
        {"k": 2.0, "a": 1.0},
        ["a", "k"],
        dtype=np.float64,
    )
    assert np.array_equal(out_mapping, np.array([1.0, 2.0], dtype=np.float64))

    series = pd.Series({"k": 2.0, "a": 1.0}, dtype=float)
    out_series = linearsolve._normalize_named_vector(
        series, ["a", "k"], dtype=np.float64
    )
    assert np.array_equal(out_series, np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(KeyError, match="Missing named value"):
        linearsolve._normalize_named_vector({"a": 1.0}, ["a", "k"])

    with pytest.raises(ValueError, match="Expected vector of length 1"):
        linearsolve._normalize_named_vector(np.array([1.0, 2.0]), ["a"])


def test_linearsolve_model_constructor_handles_multiple_layouts_and_shocks():
    mdl_components = linearsolve.model(
        equations=_series_equations,
        costates=["c"],
        exo_states=["a"],
        endo_states=["k"],
        parameters={"rho": 0.9},
        shock_prefix="eps_",
    )
    assert mdl_components.n_states == 2
    assert mdl_components.n_exo_states == 1
    assert mdl_components.n_endo_states == 1
    assert mdl_components.n_costates == 1
    assert mdl_components.names["variables"] == ["a", "k", "c"]
    assert np.array_equal(mdl_components.names["shocks"], np.array(["eps_a"]))

    mdl_reordered = linearsolve.model(
        equations=_series_equations,
        variables=["k", "a", "c"],
        states=["k", "a"],
        exo_states=["a"],
        parameters=np.array([0.9], dtype=float),
        parameter_names=["rho"],
    )
    assert mdl_reordered.n_states == 2
    assert mdl_reordered.n_exo_states == 1
    assert mdl_reordered.n_endo_states == 1
    assert mdl_reordered.n_costates == 1
    assert mdl_reordered.names["variables"] == ["a", "k", "c"]
    assert mdl_reordered.names["param"] == ["rho"]


def test_linearsolve_model_rejects_mismatched_shock_names():
    with pytest.raises(Exception, match="Length of shock_names"):
        linearsolve.model(
            equations=_series_equations,
            costates=["c"],
            exo_states=["a"],
            endo_states=["k"],
            parameters={"rho": 0.9},
            shock_names=[],
        )


def test_set_ss_and_resolve_steady_state_cover_dtype_promotion_and_missing_ss():
    mdl = linearsolve.model(
        equations=_series_equations,
        variables=["x"],
        parameters={"rho": 0.9 + 0.0j},
        n_states=1,
        n_exo_states=0,
    )

    with pytest.raises(ValueError, match="must specify a steady state"):
        mdl.linear_approximation()

    mdl.set_ss(pd.Series({"x": 1.0}, dtype=float))
    assert np.array_equal(mdl.ss, np.array([1.0], dtype=np.complex128))
    assert isinstance(mdl.parameters, dict)
    assert np.iscomplexobj(np.asarray(list(mdl.parameters.values())))

    resolved = mdl._resolve_steady_state_for_approximation({"x": 2.0})
    assert np.array_equal(resolved, np.array([2.0]))


def test_numeric_approximation_helpers_match_expected_linearization():
    a, b = linearsolve._approximate_system_numeric(
        _array_equations_numeric,
        np.array([1.0], dtype=np.float64),
        np.array([0.9], dtype=np.float64),
        False,
    )
    assert np.allclose(a, np.array([[1.0]], dtype=np.float64))
    assert np.allclose(b, np.array([[0.9]], dtype=np.float64))

    a_log, b_log = linearsolve._approximate_system_numeric(
        _array_equations_numeric,
        np.array([1.0], dtype=np.float64),
        np.array([0.9], dtype=np.float64),
        True,
    )
    assert np.allclose(a_log, np.array([[1.0 / 1.1]], dtype=np.float64))
    assert np.allclose(b_log, np.array([[0.9 / 1.1]], dtype=np.float64))


def test_numeric_approximation_rejects_complex_and_nonpositive_inputs():
    mdl = linearsolve.model(
        equations=_series_equations,
        variables=["x"],
        parameters=np.array([0.9], dtype=np.float64),
        parameter_names=["rho"],
        n_states=1,
        n_exo_states=0,
    )
    mdl.set_ss(np.array([1.0], dtype=np.float64))
    setattr(mdl, "_equations_numeric", _array_equations_numeric)

    setattr(mdl, "_parameter_array", np.array([0.9 + 0.1j], dtype=np.complex128))
    assert mdl._numeric_approximation(np.array([1.0], dtype=np.float64), False) is None

    setattr(mdl, "_parameter_array", np.array([0.9], dtype=np.float64))
    assert (
        mdl._numeric_approximation(np.array([1.0 + 0.1j], dtype=np.complex128), False)
        is None
    )
    assert mdl._numeric_approximation(np.array([0.0], dtype=np.float64), True) is None


def test_log_linear_python_fallback_and_approximate_and_solve_work():
    mdl_python = linearsolve.model(
        equations=_series_equations,
        variables=["x"],
        parameters=pd.Series({"rho": 0.9 + 0.0j}, dtype=np.complex128),
        n_states=1,
        n_exo_states=0,
    )
    mdl_python.set_ss(pd.Series({"x": 1.0}, dtype=float))
    mdl_python.log_linear_approximation()
    assert mdl_python.log_linear is True
    assert mdl_python.a.shape == (1, 1)
    assert mdl_python.b.shape == (1, 1)

    mdl_numeric = linearsolve.model(
        equations=_series_equations,
        variables=["x"],
        parameters=np.array([0.9], dtype=np.float64),
        parameter_names=["rho"],
        n_states=1,
        n_exo_states=0,
    )
    mdl_numeric.set_ss(np.array([1.0], dtype=np.float64))
    setattr(mdl_numeric, "_equations_numeric", _array_equations_numeric)
    setattr(mdl_numeric, "_parameter_array", np.array([0.9], dtype=np.float64))
    mdl_numeric.approximate_and_solve(log_linear=True)
    assert mdl_numeric.log_linear is True
    assert mdl_numeric.a.shape == (1, 1)
    assert mdl_numeric.b.shape == (1, 1)
    assert hasattr(mdl_numeric, "p")


def test_klein_helpers_cover_optional_matrices_postprocess_and_default_solve():
    assert linearsolve._normalize_optional_matrix(None).shape == (0, 0)
    assert linearsolve._normalize_optional_matrix(np.array([1.0, 2.0])).shape == (2, 1)
    assert linearsolve._normalize_optional_matrix(
        np.array([], dtype=np.float64)
    ).shape == (0, 0)

    with pytest.raises(ValueError, match="both be provided or both be empty"):
        linearsolve.klein(
            np.eye(1, dtype=np.float64),
            np.eye(1, dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            None,
            1,
        )

    f0, n0, p0, l0, stab0, eig0 = linearsolve._klein_postprocess(
        np.array([[2.0 + 0.0j]], dtype=np.complex128),
        np.array([[1.0 + 0.0j]], dtype=np.complex128),
        np.array([[1.0 + 0.0j]], dtype=np.complex128),
        np.array([[1.0 + 0.0j]], dtype=np.complex128),
        np.empty((0, 0), dtype=np.float64),
        np.empty((0, 0), dtype=np.float64),
        0,
    )
    assert f0.shape == (1, 0)
    assert n0.shape == (1, 0)
    assert p0.shape == (0, 0)
    assert l0.shape == (0, 0)
    assert stab0 == 1
    assert eig0.shape == (1,)

    f1, n1, p1, l1, stab1, eig1 = linearsolve._klein_postprocess(
        np.array([[1.0, 0.2], [0.0, 1.0]], dtype=np.complex128),
        np.array([[0.8, 0.1], [0.0, 0.5]], dtype=np.complex128),
        np.array([[1.0], [0.5]], dtype=np.complex128),
        np.eye(2, dtype=np.complex128),
        np.array([[1.0]], dtype=np.float64),
        np.array([[0.1]], dtype=np.float64),
        1,
    )
    assert f1.shape == (1, 1)
    assert n1.shape == (1, 1)
    assert p1.shape == (1, 1)
    assert l1.shape == (1, 1)
    assert stab1 == 1
    assert eig1.shape == (2,)

    mdl = linearsolve.model(
        equations=_series_equations,
        variables=["x"],
        parameters=np.array([0.9], dtype=np.float64),
        parameter_names=["rho"],
        n_states=1,
        n_exo_states=0,
    )
    mdl.a = np.array([[1.0]], dtype=np.float64)
    mdl.b = np.array([[0.9]], dtype=np.float64)
    mdl.solve_klein()
    assert np.isrealobj(mdl.f)
    assert np.isrealobj(mdl.p)
