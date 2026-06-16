"""Tests for ``SymbolicDSGE.monte_carlo.custom_op.NumpyCustomFunc``.

These tests intentionally define their target functions at module scope so
``inspect.getsource`` resolves cleanly the same way it would for a user's
notebook cell or library file.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import cloudpickle
import numpy as np
import pytest

from SymbolicDSGE.monte_carlo.custom_op import (
    CustomOpValidationError,
    NumpyCustomFunc,
    SAFE_NAMESPACE_VERSION,
    custom_operation,
)

WEIGHTS = np.array([1.0, 2.0, 3.0])
SCALE = 2.5
_NESTED = {"alpha": [1, 2, np.array([0.5, 0.5])]}


def _identity_decorator(func):
    return func


@custom_operation
def _decorated_log_diff(arr):
    return np.diff(np.log(arr + 1.0))


@_identity_decorator
def _singly_decorated(arr):
    return np.asarray(arr)


# ---- clean wrap targets --------------------------------------------------


def clean_log_diff(arr: np.ndarray) -> np.ndarray:
    return np.diff(np.log(arr + 1.0))


def uses_numpy_alias(arr):
    return np.mean(arr) * SCALE


def uses_math_only(x):
    return math.sqrt(x) + math.pi


def uses_linalg(matrix):
    return np.linalg.solve(matrix, np.ones(matrix.shape[0]))


def uses_inline_lambda(arr):
    return np.vectorize(lambda v: v + 1.0)(arr)


def uses_comprehension(arr):
    return [float(v * 2) for v in arr if v > 0]


def uses_captured_array(arr):
    return arr @ WEIGHTS


def uses_aug_assign(arr):
    total = 0.0
    for v in arr:
        total += float(v) ** 2
    return total


def uses_try_except(x):
    try:
        return 1.0 / x
    except ZeroDivisionError:
        return float("inf")


def uses_with_block(arr):
    with np.errstate(divide="ignore"):
        return np.log(arr)


def helper_inner(arr):
    return np.mean(arr) + SCALE


_helper_inner_wrapped = NumpyCustomFunc(helper_inner)


def helper_outer(arr):
    return _helper_inner_wrapped(arr) * 2.0


# ---- targets that must fail validation -----------------------------------


def banned_import(arr):
    import os  # noqa: F401

    return arr


def banned_import_from(arr):
    from os import path  # noqa: F401

    return arr


def banned_eval(s):
    return eval(s)


def banned_open(p):
    return open(p).read()


def banned_global(arr):
    global SCALE
    SCALE = 10.0
    return arr


def banned_nested_def(arr):
    def inner(v):
        return v + 1

    return [inner(v) for v in arr]


def banned_class_def(arr):
    class Helper:
        pass

    return arr


def banned_yield(arr):
    for v in arr:
        yield v + 1


async def banned_async(arr):
    return arr


def banned_attribute_load(p):
    return np.load(p)


def banned_attribute_lib(arr):
    return np.lib.format.read_array_header_1_0(arr)


@functools.lru_cache(maxsize=None)
def banned_decorated(x):
    return x + 1


def banned_unknown_global(arr):
    return UNKNOWN_THING * arr  # noqa: F821


class _HostingClass:
    @staticmethod
    def static_method(arr):
        return arr


# ---- happy-path tests ----------------------------------------------------


def test_clean_top_level_function_wraps_and_calls() -> None:
    wrapped = NumpyCustomFunc(clean_log_diff)
    out = wrapped(np.array([1.0, 2.0, 4.0, 8.0]))
    expected = np.diff(np.log(np.array([1.0, 2.0, 4.0, 8.0]) + 1.0))
    np.testing.assert_allclose(out, expected)
    assert wrapped.name == "clean_log_diff"
    assert "np.diff" in wrapped.source
    assert wrapped.safe_namespace_version == SAFE_NAMESPACE_VERSION


def test_alias_np_resolves_to_numpy_module() -> None:
    wrapped = NumpyCustomFunc(uses_numpy_alias)
    assert wrapped.captured_globals["np"] is np
    np.testing.assert_allclose(wrapped(np.array([1.0, 2.0])), 1.5 * 2.5)


def test_math_only_function_wraps() -> None:
    wrapped = NumpyCustomFunc(uses_math_only)
    assert wrapped.captured_globals["math"] is math
    assert wrapped(4.0) == pytest.approx(2.0 + math.pi)


def test_numpy_linalg_chain_is_allowed() -> None:
    wrapped = NumpyCustomFunc(uses_linalg)
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(wrapped(A), np.array([0.5, 0.25]))


def test_inline_lambda_in_body_is_allowed() -> None:
    wrapped = NumpyCustomFunc(uses_inline_lambda)
    np.testing.assert_allclose(wrapped(np.array([1.0, 2.0])), np.array([2.0, 3.0]))


def test_comprehension_scope_is_handled() -> None:
    wrapped = NumpyCustomFunc(uses_comprehension)
    assert wrapped(np.array([-1.0, 2.0, 3.0])) == [4.0, 6.0]


def test_captured_array_global_is_snapshotted() -> None:
    wrapped = NumpyCustomFunc(uses_captured_array)
    assert "WEIGHTS" in wrapped.captured_globals
    assert wrapped.captured_globals["WEIGHTS"] is WEIGHTS
    assert wrapped(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_aug_assign_target_is_bound_local() -> None:
    wrapped = NumpyCustomFunc(uses_aug_assign)
    assert wrapped(np.array([1.0, 2.0, 3.0])) == pytest.approx(14.0)


def test_try_except_is_allowed() -> None:
    wrapped = NumpyCustomFunc(uses_try_except)
    assert wrapped(2.0) == 0.5
    assert wrapped(0.0) == float("inf")


def test_with_block_is_allowed() -> None:
    wrapped = NumpyCustomFunc(uses_with_block)
    result = wrapped(np.array([1.0, 0.0]))
    assert result[0] == 0.0
    assert np.isneginf(result[1])


def test_helper_recursion_via_numpycustomfunc_helper_resolves() -> None:
    wrapped = NumpyCustomFunc(helper_outer)
    assert "_helper_inner_wrapped" in wrapped.captured_globals
    captured = wrapped.captured_globals["_helper_inner_wrapped"]
    assert isinstance(captured, NumpyCustomFunc)
    assert wrapped(np.array([1.0, 2.0])) == pytest.approx((1.5 + 2.5) * 2.0)


def test_idempotent_on_already_wrapped_input() -> None:
    once = NumpyCustomFunc(clean_log_diff)
    twice = NumpyCustomFunc(once)
    assert twice.name == once.name
    assert twice.source == once.source
    assert twice.captured_globals == once.captured_globals


# ---- structural rejections ----------------------------------------------


@pytest.mark.parametrize(
    "func, expected_fragment",
    [
        (banned_import, "import"),
        (banned_import_from, "import"),
        (banned_global, "global"),
        (banned_nested_def, "nested function"),
        (banned_class_def, "class definitions"),
        (banned_yield, "yield"),
        (banned_decorated, "decorated"),
    ],
)
def test_banned_constructs_raise_with_specific_message(
    func: Any, expected_fragment: str
) -> None:
    with pytest.raises(CustomOpValidationError, match=expected_fragment):
        NumpyCustomFunc(func)


def test_banned_async_function_rejected() -> None:
    # Async funcs aren't instances of inspect.isfunction in older lines, but
    # we still expect rejection — either by isfunction or by AST.
    with pytest.raises(CustomOpValidationError):
        NumpyCustomFunc(banned_async)


def test_eval_call_is_rejected_with_denylist_message() -> None:
    with pytest.raises(CustomOpValidationError, match="deny list"):
        NumpyCustomFunc(banned_eval)


def test_open_call_is_rejected_with_denylist_message() -> None:
    with pytest.raises(CustomOpValidationError, match="deny list"):
        NumpyCustomFunc(banned_open)


# ---- attribute deny-list -------------------------------------------------


def test_numpy_load_is_rejected_by_attribute_denylist() -> None:
    with pytest.raises(CustomOpValidationError, match=r"np\.load"):
        NumpyCustomFunc(banned_attribute_load)


def test_numpy_lib_subtree_is_rejected_by_attribute_denylist() -> None:
    with pytest.raises(CustomOpValidationError, match=r"np\.lib"):
        NumpyCustomFunc(banned_attribute_lib)


# ---- callable-shape rejections ------------------------------------------


def test_lambda_is_rejected() -> None:
    f = lambda x: x + 1  # noqa: E731
    with pytest.raises(CustomOpValidationError, match="[Ll]ambda"):
        NumpyCustomFunc(f)


def test_method_qualname_is_rejected() -> None:
    with pytest.raises(CustomOpValidationError, match="nested function or method"):
        NumpyCustomFunc(_HostingClass.static_method)


def test_closure_capturing_is_rejected() -> None:
    def outer(scale):
        def inner(x):
            return x * scale

        return inner

    with pytest.raises(CustomOpValidationError, match="closure"):
        NumpyCustomFunc(outer(2.0))


def test_functools_partial_is_rejected() -> None:
    bound = functools.partial(clean_log_diff)
    with pytest.raises(CustomOpValidationError, match="not a plain Python function"):
        NumpyCustomFunc(bound)


def test_builtin_function_is_rejected() -> None:
    with pytest.raises(CustomOpValidationError, match="not a plain Python function"):
        NumpyCustomFunc(len)


def test_non_callable_input_is_rejected() -> None:
    with pytest.raises(CustomOpValidationError, match="expected a callable"):
        NumpyCustomFunc(42)  # type: ignore[arg-type]


# ---- unknown / unsafe globals -------------------------------------------


def test_unknown_global_is_rejected_with_clear_message() -> None:
    with pytest.raises(CustomOpValidationError, match="UNKNOWN_THING"):
        NumpyCustomFunc(banned_unknown_global)


def test_module_re_binding_is_caught() -> None:
    # If a user redefined `np` to point at a *real but unsafe* module, the
    # identity check rejects it even though the name matches a safe-module key.
    import types

    fake_np = types.ModuleType("not_numpy")
    original = uses_numpy_alias.__globals__.get("np")
    uses_numpy_alias.__globals__["np"] = fake_np
    try:
        with pytest.raises(CustomOpValidationError, match="not in the safe-module set"):
            NumpyCustomFunc(uses_numpy_alias)
    finally:
        if original is not None:
            uses_numpy_alias.__globals__["np"] = original


# ---- pickle round-trip --------------------------------------------------


def test_cloudpickle_round_trip_preserves_call_behavior() -> None:
    wrapped = NumpyCustomFunc(clean_log_diff)
    raw = cloudpickle.dumps(wrapped)
    restored = cloudpickle.loads(raw)
    assert isinstance(restored, NumpyCustomFunc)
    assert restored.name == wrapped.name
    assert restored.source == wrapped.source
    np.testing.assert_allclose(
        restored(np.array([1.0, 2.0, 4.0])),
        wrapped(np.array([1.0, 2.0, 4.0])),
    )


def test_cloudpickle_round_trip_preserves_captured_globals() -> None:
    wrapped = NumpyCustomFunc(uses_captured_array)
    restored = cloudpickle.loads(cloudpickle.dumps(wrapped))
    assert "WEIGHTS" in restored.captured_globals
    np.testing.assert_allclose(
        restored.captured_globals["WEIGHTS"],
        wrapped.captured_globals["WEIGHTS"],
    )
    np.testing.assert_allclose(
        restored(np.array([1.0, 0.0, 0.0])),
        wrapped(np.array([1.0, 0.0, 0.0])),
    )


def test_cloudpickle_round_trip_preserves_helper_recursion() -> None:
    wrapped = NumpyCustomFunc(helper_outer)
    restored = cloudpickle.loads(cloudpickle.dumps(wrapped))
    inner = restored.captured_globals["_helper_inner_wrapped"]
    assert isinstance(inner, NumpyCustomFunc)
    np.testing.assert_allclose(
        restored(np.array([1.0, 2.0])),
        wrapped(np.array([1.0, 2.0])),
    )


# ---- @custom_operation decorator ----------------------------------------


def test_custom_operation_decorator_produces_numpy_custom_func() -> None:
    assert isinstance(_decorated_log_diff, NumpyCustomFunc)
    arr = np.array([1.0, 3.0, 7.0])
    np.testing.assert_allclose(_decorated_log_diff(arr), np.diff(np.log(arr + 1.0)))
    # The marker decorator stays in the captured source (intent for reviewers).
    assert "@custom_operation" in _decorated_log_diff.source


def test_custom_operation_decorated_func_round_trips() -> None:
    restored = cloudpickle.loads(cloudpickle.dumps(_decorated_log_diff))
    arr = np.array([1.0, 3.0, 7.0])
    np.testing.assert_allclose(restored(arr), _decorated_log_diff(arr))
    assert restored.source == _decorated_log_diff.source


def test_non_marker_decorator_is_still_rejected() -> None:
    with pytest.raises(CustomOpValidationError, match="decorator"):
        NumpyCustomFunc(_singly_decorated)


# --- from_source (UI / web-editor path) -----------------------------------

_FROM_SOURCE_OK = """@custom_operation
def standardize_cols(*, context, reference, dgp, rep_idx, **kwargs):
    arr = np.asarray(kwargs["x"], dtype=float)
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)
"""


def test_from_source_validates_execs_and_runs() -> None:
    func = NumpyCustomFunc.from_source(_FROM_SOURCE_OK)
    assert func.name == "standardize_cols"
    # The @custom_operation marker is preserved in the audit source.
    assert "@custom_operation" in func.source
    out = func(
        context=None, reference=None, dgp=None, rep_idx=0, x=np.array([1.0, 2, 3])
    )
    np.testing.assert_allclose(out.mean(), 0.0, atol=1e-12)


def test_from_source_round_trips_through_cloudpickle() -> None:
    func = NumpyCustomFunc.from_source(_FROM_SOURCE_OK)
    restored = cloudpickle.loads(cloudpickle.dumps(func))
    x = np.array([2.0, 4.0, 6.0])
    np.testing.assert_allclose(
        restored(context=None, reference=None, dgp=None, rep_idx=0, x=x),
        func(context=None, reference=None, dgp=None, rep_idx=0, x=x),
    )
    assert restored.source == func.source


def test_from_source_rejects_extra_statements() -> None:
    with pytest.raises(CustomOpValidationError, match="exactly one top-level"):
        NumpyCustomFunc.from_source("import os\ndef f(**kwargs):\n    return 1\n")


def test_from_source_rejects_unsafe_names() -> None:
    with pytest.raises(CustomOpValidationError, match="deny list"):
        NumpyCustomFunc.from_source('def f(**kwargs):\n    return open("x")\n')


def test_from_source_reports_syntax_errors() -> None:
    with pytest.raises(CustomOpValidationError, match="did not parse"):
        NumpyCustomFunc.from_source("def f(**kwargs)\n    return 1\n")
