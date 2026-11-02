# type: ignore
"""Emitted kernels skip identically zero slots, and callers own the zeroing.

Which slots a kernel leaves alone is a property of its expressions, so it is
fixed at compile and every caller hands the kernel a buffer that is already
zero there. A caller that forgot would still pass against a fresh allocation,
whose pages read back as zero anyway, so the buffers here are poisoned with a
sentinel rather than merely allocated.
"""

from __future__ import annotations

import ctypes

import numpy as np
import sympy as sp
import pytest

from SymbolicDSGE.core import parse_config, solve_model
from SymbolicDSGE._ckernels.core import jacobian_eval, klein_preprocess
from SymbolicDSGE._ckernels.occbin import regime_pencil
from SymbolicDSGE._symbolic_printers import MeasurementLayout, MeasurementPrinter
from SymbolicDSGE._symbolic_printers.measurement_printer import F64Ops

#: Value a skipped slot keeps, chosen so a kernel that wrote it cannot match.
POISON = -7.5

PTR = ctypes.POINTER(ctypes.c_double)


def _stored_slots(lines):
    """Slot index of every store the printer emitted, in order."""
    return [
        int(line[line.index("[") + 1 : line.index("]")])
        for line in lines
        if line.lstrip().startswith("out[")
    ]


def _zero_slots(compiled):
    return [i for i, expr in enumerate(compiled.measurement_jacobian_eqs) if expr == 0]


def _params(solved):
    calib = solved.config.calibration.parameters
    return np.array([float(calib[name]) for name in solved.compiled.calib_params])


def _call(cfunc, state, params, out):
    cfunc.ctypes(
        state.ctypes.data_as(PTR),
        params.ctypes.data_as(PTR),
        out.ctypes.data_as(PTR),
    )


@pytest.fixture(scope="module")
def jac(compiled_post82):
    """The jacobian kernel and the flat length of the buffer it writes."""
    return (
        compiled_post82.construct_measurement_jacobian_cfunc(None),
        compiled_post82.n_obs * compiled_post82.n_var,
    )


@pytest.fixture(scope="module")
def at_post82(solved_post82):
    """The linearization point the measurement kernels are driven at."""
    return solved_post82.policy.steady_state, _params(solved_post82)


def test_an_identically_zero_expression_emits_no_store(compiled_post82):
    layout = MeasurementLayout.from_compiled(compiled_post82)
    first, second = (sp.Symbol(f"cur_{n}") for n in compiled_post82.var_names[:2])
    exprs = [first, sp.Integer(0), second, sp.Integer(0), sp.Integer(0)]

    lines = MeasurementPrinter(F64Ops()).emit(exprs, layout)

    assert len(_stored_slots(lines)) == 2


def test_a_store_keeps_the_slot_its_expression_sat_in(compiled_post82):
    # The buffer stays dense, so dropping a store must not shift the ones after
    # it; a caller reading (n_obs, n_var) reads the original positions.
    layout = MeasurementLayout.from_compiled(compiled_post82)
    first, second = (sp.Symbol(f"cur_{n}") for n in compiled_post82.var_names[:2])
    exprs = [sp.Integer(0), first, sp.Integer(0), sp.Integer(0), second]

    lines = MeasurementPrinter(F64Ops()).emit(exprs, layout)

    assert _stored_slots(lines) == [1, 4]


def test_a_zero_only_counts_when_it_is_identically_zero(compiled_post82):
    # An expression that merely evaluates to zero at some point is still a
    # function of the arguments, so it keeps its store.
    layout = MeasurementLayout.from_compiled(compiled_post82)
    sym = sp.Symbol(f"cur_{compiled_post82.var_names[0]}")
    exprs = [sym - sym, sym, sp.Integer(0)]

    lines = MeasurementPrinter(F64Ops()).emit(exprs, layout)

    assert _stored_slots(lines) == [1]


def test_the_jacobian_kernel_leaves_the_slots_it_skips_untouched(
    compiled_post82, jac, at_post82
):
    cfunc, n_out = jac
    ss, params = at_post82
    out = np.full(n_out, POISON)

    _call(cfunc, ss, params, out)

    untouched = [i for i in range(n_out) if out[i] == POISON]
    assert untouched == _zero_slots(compiled_post82)


def test_the_jacobian_kernel_ignores_what_the_buffer_held(
    compiled_post82, jac, at_post82
):
    cfunc, n_out = jac
    ss, params = at_post82
    written = [i for i in range(n_out) if i not in _zero_slots(compiled_post82)]

    from_zero = np.zeros(n_out)
    from_poison = np.full(n_out, POISON)
    _call(cfunc, ss, params, from_zero)
    _call(cfunc, ss, params, from_poison)

    np.testing.assert_array_equal(from_poison[written], from_zero[written])


def test_jacobian_eval_zeros_the_slots_the_kernel_skips(compiled_post82, at_post82):
    # The one-shot path owns its own buffer, so it is the one that has to zero.
    ss, params = at_post82
    C = jacobian_eval(
        compiled_post82.construct_measurement_jacobian_cfunc(None).address,
        ss,
        params,
        compiled_post82.n_obs,
        compiled_post82.n_var,
    )

    flat = np.ravel(C)
    assert _zero_slots(compiled_post82)
    np.testing.assert_array_equal(flat[_zero_slots(compiled_post82)], 0.0)


@pytest.fixture(scope="module")
def piecewise(rbc_occbin_test_model_path):
    """Two constraints, so the regime table runs four pencils off one arena."""
    return solve_model(parse_config(rbc_occbin_test_model_path))


def test_a_regime_pencil_does_not_inherit_the_shared_arena(piecewise):
    # The solve builds every regime through one arena that the reference Klein
    # solve has already written; this rebuilds each from a private one.
    compiled = piecewise.compiled
    policy = piecewise.policy
    params = _params(piecewise)
    func = compiled.construct_regime_pencil_func()
    reference = klein_preprocess(
        compiled.construct_objective_cfunc().address,
        policy.steady_state,
        params,
        compiled.n_var,
        compiled.n_exog,
    )

    for mask in range(1, 1 << len(compiled.constraint_names)):
        a, b, c, d, cst = regime_pencil(
            func.address(mask),
            func.rows[mask],
            policy.steady_state,
            params,
            *reference,
        )
        np.testing.assert_array_equal(policy.a[mask], a)
        np.testing.assert_array_equal(policy.b[mask], b)
        np.testing.assert_array_equal(policy.c[mask], c)
        np.testing.assert_array_equal(policy.d[mask], d)
        np.testing.assert_array_equal(policy.cst[mask], cst)
