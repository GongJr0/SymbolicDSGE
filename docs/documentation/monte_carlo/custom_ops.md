---
tags:
    - doc
---

# Custom Operations

Custom Monte Carlo operations can run directly from any callable. Bundle-safe custom operations must be wrapped so the function source and accepted globals can travel with the `.sdsge` archive. Two namespaces are available:

- **`NumpyCustomFunc`** / **`numpy_operation`**: the per replication contract. Its body may reference `numpy` (as `np`), `math`, `statistics`, `operator`, and selected builtins. Use for [custom transforms](operations/transforms/custom.md) and any op that runs inside the replication loop.
- **`PandasCustomFunc`** / **`pandas_operation`**: the looser post-loop (`OpType.POSTPROC`) contract, whose namespace additionally exposes `pandas` (as `pd`) so a summary op may build a DataFrame. Use for [custom post-processing](operations/postproc/custom.md). A pandas-enabled op used inside the replication loop is rejected when the pipeline is built.

## `NumpyCustomFunc`

```python
class NumpyCustomFunc()
```

Opt-in wrapper that validates and snapshots a numerical Python function under the numpy namespace.

```python
NumpyCustomFunc(func: Callable[..., Any] | CustomFunc) -> NumpyCustomFunc
```

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Original function name. |
| source | `#!python str` | Author-side source text for receiver audit. |
| captured_globals | `#!python Mapping[str, Any]` | Snapshot of accepted globals referenced by the function. |
| safe_namespace_version | `#!python int` | Version of the safe-namespace contract used at wrap time. |
| namespace_kind | `#!python str` | Namespace validated against: `"numpy"` or `"pandas"`. |

__Methods:__

```python
NumpyCustomFunc.from_source(source: str) -> NumpyCustomFunc  # @classmethod
NumpyCustomFunc.__call__(*args: Any, **kwargs: Any) -> Any
```

`from_source` validates source text directly. This is used for code typed into a UI/editor where `inspect.getsource(...)` cannot recover a real file or notebook cell.

## `PandasCustomFunc`

```python
class PandasCustomFunc(CustomFunc)
```

The pandas sibling of `NumpyCustomFunc`: identical validation and properties, but the body may also reference `pandas` (as `pd`). Intended for `OpType.POSTPROC` summary ops (e.g. one returning a DataFrame).

```python
PandasCustomFunc(func: Callable[..., Any] | CustomFunc) -> PandasCustomFunc
PandasCustomFunc.from_source(source: str) -> PandasCustomFunc  # @classmethod
```

## `numpy_operation`

```python
numpy_operation(func: Callable[..., Any]) -> NumpyCustomFunc
```

Decorator form of `NumpyCustomFunc`. The decorated name becomes a callable `NumpyCustomFunc` and can be passed to `transform_step(...)` (or any per-replication custom-op factory).

```python
from SymbolicDSGE.monte_carlo import numpy_operation

@numpy_operation
def demean(*, context, reference, dgp, rep_idx, **kwargs):
    obs = context.require_data().observables
    return obs - obs.mean(axis=0, keepdims=True)
```

## `pandas_operation`

```python
pandas_operation(func: Callable[..., Any]) -> PandasCustomFunc
```

Decorator form of `PandasCustomFunc`, for post-loop summary ops passed to `postproc_step(...)`. The body may reference `pd`; using it on a per-replication step is rejected when the pipeline is built.

## Validation Contract

| __Allowed__ | __Description__ |
|:------------|----------------:|
| One top-level `def` | Lambdas, nested functions, methods, partials, builtins, and C-extension callables are rejected. |
| Numeric safe namespace | `numpy` as `np`, selected standard modules, selected builtins, and captured immutable/numpy globals. `pandas` as `pd` is added only under the pandas namespace. |
| Explicit source | Source must be recoverable or supplied through `from_source(...)`. |

| __Rejected__ | __Reason__ |
|:-------------|-----------:|
| Imports, `global`, `nonlocal`, async, yield, nested `def`, classes | These make the shipped function harder to audit and reproduce. |
| Closure captures | Promote values to accepted globals or pass them as kwargs. |
| Unsupported globals | The wrapper snapshots only accepted numeric/scalar/container helpers. |

???+ warning "Not a security sandbox"
    `NumpyCustomFunc` / `PandasCustomFunc` are a reproducibility and audit contract, not a sandbox. Loading a bundle with custom operations should be treated like running Python code from the bundle author.
