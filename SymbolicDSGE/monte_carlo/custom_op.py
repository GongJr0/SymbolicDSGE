"""Author-side hygiene wrapper for user-supplied numerical functions.

:class:`NumpyCustomFunc` is an opt-in wrapper that validates and snapshots a
plain ``def``-style numerical function at definition time so it can later be
shipped inside a ``.sdsge`` bundle alongside its source for receiver-side audit.

Authoring contract
------------------

A function passed to :class:`NumpyCustomFunc` must:

- be a plain top-level ``def`` (no lambdas, methods, nested functions,
  closures, partials, or decorated callables);
- reference only the *safe namespace* below in its body: ``numpy`` (also
  reachable as ``np``), ``math``, ``statistics``, ``operator``, a small set of
  whitelisted builtins, and other :class:`NumpyCustomFunc` instances acting as
  helpers;
- not contain ``import`` / ``import from`` / ``global`` / ``nonlocal`` / async
  constructs / ``yield`` / nested ``def`` or ``class``.

Globals referenced by the function are snapshotted into
:attr:`NumpyCustomFunc.captured_globals` and travel with the wrapper.

Threat model
------------

This is **author-side hygiene**, not a sandbox. A :class:`NumpyCustomFunc`
serialized into a ``.sdsge`` bundle and loaded on another machine executes as
arbitrary Python in the loader's interpreter. The validator catches accidental
unsafe imports / banned constructs and constrains the visible source so the
receiver-side audit prompt is meaningful — it does **not** prevent a determined
adversary who controls the bundle bytes from running code. Treat ``.sdsge``
files with custom ops the way you treat a ``.py`` file from the same sender.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import inspect
import math
import operator
import statistics
import textwrap
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

#: Bumped on any change to the safe-namespace contract below. Stored on each
#: wrapped instance so future loaders can detect cross-version drift.
SAFE_NAMESPACE_VERSION = 1


# ---- Safe namespace ------------------------------------------------------


#: Builtin names callable from a custom op body. Conservative — additions need
#: a deliberate review against the threat model.
SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        # Container / scalar constructors.
        "bool",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "set",
        "slice",
        "str",
        "tuple",
        # Numerical helpers.
        "abs",
        "divmod",
        "max",
        "min",
        "pow",
        "round",
        "sum",
        # Iteration / introspection (read-only).
        "all",
        "any",
        "enumerate",
        "filter",
        "isinstance",
        "issubclass",
        "len",
        "map",
        "range",
        "reversed",
        "sorted",
        "type",
        "zip",
        # Literals.
        "True",
        "False",
        "None",
        # Exception types — needed for sane try/except/raise inside transforms.
        "Exception",
        "ArithmeticError",
        "AttributeError",
        "FloatingPointError",
        "IndexError",
        "KeyError",
        "OverflowError",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    }
)

#: Builtin names with explicit deny messages. Anything outside SAFE_BUILTINS
#: that's also outside this set still gets rejected; this set just produces a
#: clearer error for the well-known footguns.
DENIED_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "dir",
        "callable",
        "memoryview",
        "bytearray",
        "object",
        "super",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "id",
    }
)

#: Module identities allowed as module-level references in the function body.
#: Both ``numpy`` and the conventional ``np`` alias resolve here.
SAFE_MODULES: dict[str, Any] = {
    "numpy": np,
    "np": np,
    "math": math,
    "statistics": statistics,
    "operator": operator,
}

#: Per-module attribute deny-list — reject these specific call sites even
#: though the parent module is trusted. Targets here are either I/O (load /
#: save / fromfile) or submodules whose contents would expand the trust
#: surface (lib, testing, distutils). Add to this list when a numpy function
#: turns out to expose something we'd rather not allow.
_NUMPY_DENIED: frozenset[str] = frozenset(
    {
        "compat",
        "ctypeslib",
        "distutils",
        "f2py",
        "fromfile",
        "fromstring",
        "lib",
        "load",
        "loadtxt",
        "memmap",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "testing",
        "tofile",
    }
)
DENIED_ATTRIBUTES: dict[str, frozenset[str]] = {
    "numpy": _NUMPY_DENIED,
    "np": _NUMPY_DENIED,
}


# ---- Exception -----------------------------------------------------------


class CustomOpValidationError(ValueError):
    """Raised when a function fails :class:`NumpyCustomFunc`'s wrap-time check.

    Carries the function's name in the message prefix so chained validation
    failures (helper inside helper) point at the right authoring site.
    """


# ---- Source extraction ---------------------------------------------------


def _extract_source(func: Callable[..., Any]) -> str:
    """Return the dedented source of ``func``.

    Raises :class:`CustomOpValidationError` when ``inspect.getsource`` can't
    find readable source — lambdas, partials, REPL one-liners, and synthetic
    code objects all fall here. The receiver-side audit story depends on the
    source being available, so we refuse rather than ship a wrapper whose
    bundle would have nothing to show.
    """
    try:
        raw = inspect.getsource(func)
    except (OSError, TypeError) as exc:
        name = getattr(func, "__name__", repr(func))
        raise CustomOpValidationError(
            f"Cannot extract source for {name!r}. NumpyCustomFunc requires a "
            "function whose source is retrievable via `inspect.getsource` "
            "(top-level def in a module file or a notebook cell). Lambdas, "
            "partials, and dynamically constructed code objects are not "
            "supported."
        ) from exc
    return textwrap.dedent(raw)


# ---- AST validation ------------------------------------------------------


def _is_custom_operation_decorator(node: ast.expr) -> bool:
    """Recognize the ``@custom_operation`` marker.

    A function decorated with :func:`custom_operation` carries that decorator in
    its captured source (``inspect.getsource`` includes decorator lines), so the
    validator must tolerate this one marker while still rejecting every other
    decorator. Matched by trailing name so ``@custom_operation`` and
    ``@sd.custom_operation`` both pass.
    """
    if isinstance(node, ast.Name):
        return node.id == "custom_operation"
    if isinstance(node, ast.Attribute):
        return node.attr == "custom_operation"
    return False


def _walk_attribute_chain(node: ast.Attribute) -> tuple[str, list[str]] | None:
    """Reduce ``np.linalg.solve`` to ``("np", ["linalg", "solve"])``.

    Returns ``None`` if the chain bottoms out at something other than a bare
    name (e.g. ``func().attr``), in which case the visitor will walk the
    sub-expression normally.
    """
    attrs: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return current.id, list(reversed(attrs))


class _Validator(ast.NodeVisitor):
    """Scope-aware walker enforcing the NumpyCustomFunc authoring contract.

    Records every load-context ``Name`` and attribute chain that doesn't
    resolve to a local binding; the caller validates the recorded names
    against the function's globals + the safe namespace. Structural failures
    (banned statements, nested defs, etc.) raise immediately.
    """

    def __init__(self, func_name: str) -> None:
        self._func_name = func_name
        # The function's own scope (parameters + assignment targets).
        self._function_locals: set[str] = set()
        # A stack of ephemeral scopes for comprehensions and inline lambdas
        # (Python 3 gives comprehensions their own scope; iteration variables
        # don't leak).
        self._scope_stack: list[set[str]] = []
        self._global_loads: dict[str, ast.AST] = {}
        self._attribute_loads: list[tuple[str, tuple[str, ...]]] = []

    # ----- public surface for the caller -----

    @property
    def global_loads(self) -> Mapping[str, ast.AST]:
        return self._global_loads

    @property
    def attribute_loads(self) -> list[tuple[str, tuple[str, ...]]]:
        return list(self._attribute_loads)

    def validate(self, tree: ast.FunctionDef) -> None:
        for decorator in tree.decorator_list:
            if not _is_custom_operation_decorator(decorator):
                self._fail(
                    "decorated functions cannot be wrapped (other than the "
                    "@custom_operation marker). Apply @custom_operation as the "
                    "only decorator, or remove decorators."
                )
        args = tree.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            self._function_locals.add(arg.arg)
        if args.vararg is not None:
            self._function_locals.add(args.vararg.arg)
        if args.kwarg is not None:
            self._function_locals.add(args.kwarg.arg)
        for stmt in tree.body:
            self.visit(stmt)

    # ----- helpers -----

    def _is_local(self, name: str) -> bool:
        if name in self._function_locals:
            return True
        return any(name in scope for scope in self._scope_stack)

    def _bind_local(self, name: str) -> None:
        if self._scope_stack:
            self._scope_stack[-1].add(name)
        else:
            self._function_locals.add(name)

    def _bind_targets(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self._bind_local(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._bind_targets(elt)
        elif isinstance(node, ast.Starred):
            self._bind_targets(node.value)
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            # `x.attr = ...` and `x[i] = ...` are loads on the base, then a
            # store. Visit so the base name validation still runs.
            self.visit(node)

    def _fail(self, message: str) -> None:
        raise CustomOpValidationError(f"{self._func_name!r}: {message}")

    # ----- banned constructs -----

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self._fail("`import` is not allowed inside a custom op body.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self._fail("`from ... import ...` is not allowed inside a custom op body.")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        # The entry function never reaches this visitor (it's walked manually
        # in `validate`); only nested defs do.
        self._fail(
            "nested function definitions are not allowed. Define helpers as "
            "separate top-level NumpyCustomFunc-wrapped functions."
        )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._fail("async functions are not allowed.")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._fail("class definitions inside a custom op body are not allowed.")

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self._fail("the `global` statement is not allowed.")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        self._fail("the `nonlocal` statement is not allowed.")

    def visit_Await(self, node: ast.Await) -> None:  # noqa: N802
        self._fail("`await` is not allowed.")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
        self._fail("async for is not allowed.")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:  # noqa: N802
        self._fail("async with is not allowed.")

    def visit_Yield(self, node: ast.Yield) -> None:  # noqa: N802
        self._fail("`yield` is not allowed (generators are not supported).")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:  # noqa: N802
        self._fail("`yield from` is not allowed (generators are not supported).")

    # ----- binding sites -----

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        # Walk RHS first so a self-referential binding isn't shadowed.
        self.visit(node.value)
        for target in node.targets:
            self._bind_targets(target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self.visit(node.value)
        # AugAssign reads + stores; bind to keep subsequent references local.
        self._bind_targets(node.target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is not None:
            self.visit(node.value)
        self._bind_targets(node.target)
        # Annotations are inert — typing constructs may live outside the safe
        # namespace but never execute during a call.

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self.visit(node.iter)
        self._bind_targets(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_targets(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._bind_local(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._bind_local(node.target.id)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        # Lambdas inside the body (e.g. `np.vectorize(lambda x: ...)`) are
        # allowed; they get their own scope and their args bind into it.
        self._scope_stack.append(set())
        try:
            for arg in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ):
                self._scope_stack[-1].add(arg.arg)
            if node.args.vararg is not None:
                self._scope_stack[-1].add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                self._scope_stack[-1].add(node.args.kwarg.arg)
            self.visit(node.body)
        finally:
            self._scope_stack.pop()

    # Comprehensions share scope-stack handling.
    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
    ) -> None:
        self._scope_stack.append(set())
        try:
            for gen in node.generators:
                self.visit(gen.iter)
                self._bind_targets(gen.target)
                for cond in gen.ifs:
                    self.visit(cond)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
        finally:
            self._scope_stack.pop()

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension
    visit_DictComp = _visit_comprehension

    # ----- load-context references -----

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load) and not self._is_local(node.id):
            self._global_loads.setdefault(node.id, node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        chain = _walk_attribute_chain(node)
        if chain is not None:
            root, attrs = chain
            if not self._is_local(root):
                self._global_loads.setdefault(root, node)
                self._attribute_loads.append((root, tuple(attrs)))
        else:
            # `f().attr` or similar — walk the sub-expression normally.
            self.visit(node.value)


# ---- Globals capture -----------------------------------------------------


_SAFE_SCALAR_TYPES: tuple[type, ...] = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    type(None),
)


def _capture_globals(
    func: Callable[..., Any],
    global_loads: Mapping[str, ast.AST],
    attribute_loads: list[tuple[str, tuple[str, ...]]],
    func_name: str,
) -> dict[str, Any]:
    """Resolve referenced names against the function's globals + safe namespace.

    Returns a snapshot dict keyed by name. Each value passes the safe-value
    check (numpy / scalar / immutable container / :class:`NumpyCustomFunc`
    helper). Raises :class:`CustomOpValidationError` on anything else.
    """
    captured: dict[str, Any] = {}
    func_globals = getattr(func, "__globals__", {})
    builtin_attrs = vars(_builtins)

    # Attribute-chain deny-list check (per-module).
    for root, chain in attribute_loads:
        denied = DENIED_ATTRIBUTES.get(root)
        if denied is None:
            continue
        for attr in chain:
            if attr in denied:
                raise CustomOpValidationError(
                    f"{func_name!r}: attribute access `{root}."
                    f"{'.'.join(chain)}` hits deny-list entry "
                    f"`{root}.{attr}`."
                )

    for name in global_loads:
        if name in SAFE_BUILTINS:
            continue
        if name in DENIED_BUILTINS:
            raise CustomOpValidationError(
                f"{func_name!r}: name `{name}` is in the explicit deny list "
                "of builtins. NumpyCustomFunc forbids reflective and I/O "
                "primitives by design."
            )
        if name in func_globals:
            value = func_globals[name]
            _validate_captured_value(name, value, func_name)
            captured[name] = value
            continue
        if name in builtin_attrs:
            raise CustomOpValidationError(
                f"{func_name!r}: builtin `{name}` is not in the safe namespace."
            )
        raise CustomOpValidationError(
            f"{func_name!r}: name `{name}` could not be resolved against the "
            "function's globals or the safe namespace (likely a typo or a "
            "forward-undefined reference)."
        )
    return captured


def _validate_captured_value(name: str, value: Any, func_name: str) -> None:
    """Reject globals whose runtime type isn't in the safe-value set.

    Modules are accepted only when they match one of :data:`SAFE_MODULES`
    by identity (so a re-bound ``np = some_other_module`` is caught).
    Containers are walked recursively up to a bounded depth.
    """
    if inspect.ismodule(value):
        if name not in SAFE_MODULES or SAFE_MODULES[name] is not value:
            mod_name = getattr(value, "__name__", "<unknown>")
            raise CustomOpValidationError(
                f"{func_name!r}: global `{name}` resolves to module "
                f"`{mod_name}`, which is not in the safe-module set."
            )
        return
    if isinstance(value, NumpyCustomFunc):
        return
    _validate_value_recursive(name, value, func_name)


def _validate_value_recursive(
    name: str, value: Any, func_name: str, _depth: int = 0
) -> None:
    if _depth > 20:
        raise CustomOpValidationError(
            f"{func_name!r}: captured global `{name}` is too deeply nested "
            "(>20 levels)."
        )
    if isinstance(value, _SAFE_SCALAR_TYPES):
        return
    if isinstance(value, (np.ndarray, np.generic, np.dtype)):
        return
    if isinstance(value, NumpyCustomFunc):
        return
    if isinstance(value, (tuple, list, frozenset, set)):
        for item in value:
            _validate_value_recursive(name, item, func_name, _depth + 1)
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            _validate_value_recursive(name, k, func_name, _depth + 1)
            _validate_value_recursive(name, v, func_name, _depth + 1)
        return
    raise CustomOpValidationError(
        f"{func_name!r}: captured global `{name}` has unsupported type "
        f"`{type(value).__name__}`. Allowed: numpy types, scalars, strings, "
        "bytes, immutable containers, and NumpyCustomFunc helpers."
    )


# ---- The wrapper ---------------------------------------------------------


class NumpyCustomFunc:
    """Opt-in wrapper validating a numerical function at definition time.

    Construction snapshot:

    - extracts ``inspect.getsource(func)`` (refuses if unavailable);
    - parses + walks the AST against the safe namespace contract;
    - resolves every referenced global against ``func.__globals__`` and the
      safe namespace, snapshotting accepted values into
      :attr:`captured_globals`.

    After construction the instance is callable as the original function. The
    snapshotted state survives :mod:`cloudpickle` round-trips, which is what
    lets a custom op travel inside a ``.sdsge`` bundle alongside its source.

    Calling ``NumpyCustomFunc(already_wrapped)`` returns the existing instance
    state (idempotent) so chained wrappings don't re-validate.
    """

    __slots__ = (
        "_func",
        "_source",
        "_captured_globals",
        "_name",
        "_safe_namespace_version",
    )

    # Typed mirrors of __slots__ so mypy can resolve attribute reads inside
    # the idempotent-rewrap branch and on the property getters.
    _func: Callable[..., Any]
    _source: str
    _captured_globals: dict[str, Any]
    _name: str
    _safe_namespace_version: int

    def __init__(self, func: Callable[..., Any] | NumpyCustomFunc) -> None:
        if isinstance(func, NumpyCustomFunc):
            # Idempotent — copy the validated state across.
            self._func = func._func
            self._source = func._source
            self._captured_globals = func._captured_globals
            self._name = func._name
            self._safe_namespace_version = func._safe_namespace_version
            return

        if not callable(func):
            raise CustomOpValidationError(
                "NumpyCustomFunc expected a callable, got " f"{type(func).__name__}."
            )

        func_name = getattr(func, "__name__", None) or repr(func)
        if func_name == "<lambda>":
            raise CustomOpValidationError(
                "Lambdas are not supported; define a top-level `def` function."
            )

        qualname = getattr(func, "__qualname__", func_name)
        if qualname != func_name:
            raise CustomOpValidationError(
                f"{func_name!r}: qualified name `{qualname}` indicates a "
                "nested function or method. NumpyCustomFunc requires a "
                "top-level def."
            )

        if not inspect.isfunction(func):
            raise CustomOpValidationError(
                f"{func_name!r}: not a plain Python function (got "
                f"`{type(func).__name__}`). Partials, methods, builtins, and "
                "C extensions are not supported."
            )

        if getattr(func, "__closure__", None):
            raise CustomOpValidationError(
                f"{func_name!r}: closure-capturing functions are not allowed. "
                "Promote captured values to module-level constants or pass "
                "them as arguments."
            )

        source = _extract_source(func)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise CustomOpValidationError(
                f"{func_name!r}: source did not parse — {exc.msg}."
            ) from exc

        if (
            len(tree.body) != 1
            or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != func_name
        ):
            raise CustomOpValidationError(
                f"{func_name!r}: extracted source does not contain a single "
                "matching `def` statement."
            )
        func_def = tree.body[0]

        validator = _Validator(func_name)
        validator.validate(func_def)

        captured = _capture_globals(
            func,
            validator.global_loads,
            validator.attribute_loads,
            func_name,
        )

        self._func = func
        self._source = source
        self._captured_globals = captured
        self._name = func_name
        self._safe_namespace_version = SAFE_NAMESPACE_VERSION

    # ----- read-only accessors -----

    @property
    def name(self) -> str:
        """Original function name (used in error messages)."""
        return self._name

    @property
    def source(self) -> str:
        """Author-side source text. Shown to receivers for audit at load."""
        return self._source

    @property
    def captured_globals(self) -> Mapping[str, Any]:
        """Snapshot of every global the function references at wrap time."""
        return self._captured_globals

    @property
    def safe_namespace_version(self) -> int:
        """Version of the safe-namespace contract enforced at wrap time."""
        return self._safe_namespace_version

    # ----- alternate constructor -----

    @classmethod
    def from_source(cls, source: str) -> "NumpyCustomFunc":
        """Build a wrapper from source *text* rather than a live function.

        The constructor recovers source via ``inspect.getsource``, which needs a
        real ``def`` in a module file or notebook cell — unavailable for code
        typed into a web editor and ``exec``'d. This validates the supplied text
        directly (same AST contract) and executes it in the safe namespace to
        obtain the callable, snapshotting the original text as :attr:`source` for
        receiver-side audit. A leading ``@custom_operation`` marker is accepted
        and treated as a no-op (the result is wrapped here regardless).

        Raises :class:`CustomOpValidationError` on a parse error, a structure
        other than a single top-level ``def``, a safe-namespace violation, or an
        execution failure.
        """
        text = textwrap.dedent(source)
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise CustomOpValidationError(
                f"submitted source did not parse — {exc.msg} (line {exc.lineno})."
            ) from exc
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise CustomOpValidationError(
                "submitted source must contain exactly one top-level `def` "
                "(no imports or extra statements; numpy is available as `np`)."
            )
        func_def = tree.body[0]
        func_name = func_def.name

        validator = _Validator(func_name)
        validator.validate(func_def)

        # `custom_operation` is neutralized so a decorated template execs to the
        # bare function (its own getsource-based wrap would fail on exec'd code).
        namespace: dict[str, Any] = {**SAFE_MODULES, "custom_operation": lambda f: f}
        try:
            exec(compile(tree, "<custom_op>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            raise CustomOpValidationError(
                f"{func_name!r}: failed to execute submitted source — {exc}."
            ) from exc
        func = namespace[func_name]
        captured = _capture_globals(
            func, validator.global_loads, validator.attribute_loads, func_name
        )

        instance = object.__new__(cls)
        instance._func = func
        instance._source = text
        instance._captured_globals = captured
        instance._name = func_name
        instance._safe_namespace_version = SAFE_NAMESPACE_VERSION
        return instance

    # ----- runtime surface -----

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"NumpyCustomFunc({self._name})"


def custom_operation(func: Callable[..., Any]) -> NumpyCustomFunc:
    """Decorator marking a numerical function as a shippable custom op.

    Equivalent to wrapping with :class:`NumpyCustomFunc`, but as a decorator it
    documents intent at the definition site — a reviewer reading the code sees
    the op is destined for a ``.sdsge`` bundle — and guarantees the function is
    validated and snapshotted up front rather than (auto-)wrapped later::

        @custom_operation
        def zscore(*, context, **kwargs):
            arr = context.require_data().observables
            return (arr - arr.mean(axis=0)) / arr.std(axis=0)

    The decorated name becomes a callable :class:`NumpyCustomFunc`, so it can be
    handed straight to ``transform_step`` (or any custom-op factory).
    """
    return NumpyCustomFunc(func)
