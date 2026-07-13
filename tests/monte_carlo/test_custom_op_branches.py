"""Security + branch coverage for the custom-op validator.

Custom ops ride inside shareable ``.sdsge`` bundles, so the namespace validator
is the audit gate against footguns like ``eval``/``exec``/``open``/``import`` and
I/O attribute access (``np.load``, ``pd.read_csv``). The headline tests assert
those violations are rejected under *both* the numpy (transform) and pandas
(postproc) namespaces; the rest fill the validator/capture branch coverage.
"""

from __future__ import annotations

import math
from types import SimpleNamespace  # noqa: F401  (kept for parity with siblings)

import pytest

from SymbolicDSGE.monte_carlo.custom_op import (
    CustomFunc,
    CustomOpValidationError,
    NumpyCustomFunc,
    PandasCustomFunc,
    numpy_operation,
)

BOTH = [NumpyCustomFunc, PandasCustomFunc]

# --- headline: namespace violations must trip under BOTH namespaces ---------

_VIOLATIONS = [
    ("import", "def f(x):\n    import os\n    return x", "import"),
    ("from_import", "def f(x):\n    from os import getcwd\n    return x", "from"),
    ("eval", "def f(x):\n    return eval('1')", "deny list"),
    ("exec", "def f(x):\n    return exec('y = 1')", "deny list"),
    ("open", "def f(x):\n    return open('a')", "deny list"),
    ("dunder_import", "def f(x):\n    return __import__('os')", "deny list"),
    ("np_load", "def f(x):\n    return np.load('a.npy')", "deny-list"),
    ("global_stmt", "def f(x):\n    global gg\n    return x", "global"),
    ("unknown_builtin", "def f(x):\n    return hash(x)", "safe namespace"),
]


@pytest.mark.parametrize("cls", BOTH, ids=["numpy", "pandas"])
@pytest.mark.parametrize("name,src,match", _VIOLATIONS, ids=[v[0] for v in _VIOLATIONS])
def test_namespace_violation_rejected_in_both(cls, name, src, match):
    with pytest.raises(CustomOpValidationError, match=match):
        cls.from_source(src)


def test_pandas_io_reader_denied():
    with pytest.raises(CustomOpValidationError, match="deny-list"):
        PandasCustomFunc.from_source("def f(x):\n    return pd.read_csv('a.csv')")
    # numpy namespace has no pandas at all -> pd is unresolved, still rejected
    with pytest.raises(CustomOpValidationError, match="could not be resolved"):
        NumpyCustomFunc.from_source("def f(x):\n    return pd.read_csv('a.csv')")


# --- valid ops construct and call under each namespace ----------------------


def test_valid_numpy_and_pandas_ops():
    n = NumpyCustomFunc.from_source("def f(x):\n    return np.sqrt(x) + math.pi")
    assert n(4.0) == pytest.approx(2.0 + math.pi)
    assert n.namespace_kind == "numpy"
    assert repr(n).startswith("NumpyCustomFunc(")

    p = PandasCustomFunc.from_source("def g(x):\n    return int(pd.Series(x).sum())")
    assert p([1, 2, 3]) == 6
    assert p.namespace_kind == "pandas"


# --- banned constructs reachable from a sync def body -----------------------


def test_nested_async_def_rejected():
    src = "def f(x):\n    async def g():\n        return 1\n    return x"
    with pytest.raises(CustomOpValidationError, match="async"):
        NumpyCustomFunc.from_source(src)


def test_yield_from_rejected():
    with pytest.raises(CustomOpValidationError, match="yield from"):
        NumpyCustomFunc.from_source("def f(x):\n    yield from x")


# --- the validator's binding sites, all in one valid function ---------------

_BIG_VALID = """
def f(x, *args, **kwargs):
    a, b = 1, 2
    first, *rest = [1, 2, 3]
    acc = 0
    for i in range(3):
        acc += i
    else:
        acc += 10
    z: int = 5
    with np.errstate(all="ignore") as _e:
        w = np.sqrt(x)
    try:
        w = w + 1.0
    except ValueError as err:
        w = 0.0
    total = (y := a + b)
    d = {k: v for k, v in enumerate(rest)}
    g = lambda *la, **lk: la
    out = [0, 0, 0]
    out[0] = total
    return w + z + total + acc + first + len(d) + len(g(1))
"""


def test_all_binding_sites_validate():
    op = NumpyCustomFunc.from_source(_BIG_VALID)
    # exercises vararg/kwarg params, tuple + starred unpack, aug-assign,
    # for/else, ann-assign, with-as, except-as, walrus, dict-comp, lambda
    # *args/**kwargs, and subscript assignment.
    # w(=3.0) + z(5) + total(3) + acc(3+10) + first(1) + len(d)(2) + len(g(1))(1)
    assert op(4.0) == pytest.approx(28.0)


# --- from_source structural / execution failures ----------------------------


def test_from_source_syntax_error():
    with pytest.raises(CustomOpValidationError, match="did not parse"):
        NumpyCustomFunc.from_source("def f(:\n    return 1")


def test_from_source_requires_single_def():
    with pytest.raises(CustomOpValidationError, match="exactly one top-level"):
        NumpyCustomFunc.from_source("x = 1\ndef f(y):\n    return y")


def test_from_source_execution_failure():
    # default argument evaluated at def time blows up during exec
    with pytest.raises(CustomOpValidationError, match="failed to execute"):
        NumpyCustomFunc.from_source("def f(x=(1 / 0)):\n    return x")


def test_attribute_form_marker_recognized_then_exec_fails():
    # `@sd.numpy_operation` passes the marker check (Attribute branch) but `sd`
    # is undefined in the exec namespace, so it fails at execution.
    src = "@sd.numpy_operation\ndef f(x):\n    return x"
    with pytest.raises(CustomOpValidationError, match="failed to execute"):
        NumpyCustomFunc.from_source(src)


def test_base_from_source_not_implemented():
    with pytest.raises(NotImplementedError):
        CustomFunc.from_source("def f(x):\n    return x")


def test_call_form_decorator_rejected():
    # a call-expression decorator is neither a Name nor Attribute marker, so
    # _is_operation_decorator returns False and validation rejects it.
    with pytest.raises(CustomOpValidationError, match="decorated functions"):
        NumpyCustomFunc.from_source("@deco()\ndef f(x):\n    return x")


def test_plain_yield_rejected():
    with pytest.raises(CustomOpValidationError, match="yield"):
        NumpyCustomFunc.from_source("def f(x):\n    yield x")


# --- live-function capture paths (need a real module-level def) -------------


@numpy_operation
def _helper(x):
    return x + 1


def _uses_helper(x):
    return _helper(x) * 2


def test_captures_customfunc_helper():
    op = NumpyCustomFunc(_uses_helper)
    assert "_helper" in op.captured_globals
    assert op(3) == 8


_UNSAFE_OBJ = object()


def _uses_unsafe_global(x):
    return _UNSAFE_OBJ


def test_rejects_unsafe_global_type():
    with pytest.raises(CustomOpValidationError, match="unsupported type"):
        NumpyCustomFunc(_uses_unsafe_global)


_DEEP: list = []
_cur = _DEEP
for _ in range(25):
    nxt: list = []
    _cur.append(nxt)
    _cur = nxt


def _uses_deep_global(x):
    return _DEEP


def test_rejects_deeply_nested_global():
    with pytest.raises(CustomOpValidationError, match="deeply nested"):
        NumpyCustomFunc(_uses_deep_global)


def test_rejects_source_unavailable():
    ns: dict = {}
    exec("def synthetic(x):\n    return x", ns)  # noqa: S102
    with pytest.raises(CustomOpValidationError, match="Cannot extract source"):
        NumpyCustomFunc(ns["synthetic"])


# Captured globals that recurse through containers / mappings / helpers.
_TUPLE_GLOBAL = (1, 2, [3, 4])
_MAPPING_GLOBAL = {"a": 1, "b": [2, 3]}
_HELPER_IN_TUPLE = (_helper,)


def _uses_tuple(x):
    return _TUPLE_GLOBAL


def _uses_mapping(x):
    return _MAPPING_GLOBAL


def _uses_helper_container(x):
    return _HELPER_IN_TUPLE


def test_captures_recursive_container_globals():
    # tuple/list recursion, mapping key+value recursion, and a CustomFunc nested
    # inside a container all pass the recursive value validation.
    assert "_TUPLE_GLOBAL" in NumpyCustomFunc(_uses_tuple).captured_globals
    assert "_MAPPING_GLOBAL" in NumpyCustomFunc(_uses_mapping).captured_globals
    assert (
        "_HELPER_IN_TUPLE" in NumpyCustomFunc(_uses_helper_container).captured_globals
    )
