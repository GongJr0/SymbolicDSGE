---
tags:
    - doc
---

# Custom Operations

Custom Monte Carlo operations are authored as plain top-level functions and wrapped so the function source and accepted globals can travel with the `.sdsge` archive. The wrapper you use depends on where the operation runs:

- **`NumbaCustomFunc`** / **`custom_transform`**: the per-replication contract. The function is compiled by Numba and called from the native replication loop, so it must match a fixed two-argument array signature. Use for [custom transforms](operations/transforms/custom.md).
- **`PandasCustomFunc`** / **`pandas_operation`**: the post-loop (`OpType.POSTPROC`) contract. It stays plain Python, runs once after the loop, and its namespace additionally exposes `pandas` (as `pd`) so a summary op may build a DataFrame. Use for [custom post-processing](operations/postproc/custom.md).

Both validate the function body against a safe namespace at wrap time: `numpy` (as `np`), `math`, `statistics`, `operator`, a selected set of builtins, and captured immutable or numpy globals.

## `NumbaCustomFunc`

```python
class NumbaCustomFunc(NumpyCustomFunc)
```

```python
NumbaCustomFunc(func: Callable[..., Any] | CustomFunc) -> NumbaCustomFunc
```

A numerical function compiled for the native transform ABI. The wrapped function must accept exactly two positional parameters and return an integer status code:

```python
def my_transform(sample, output) -> int: ...
```

`sample` is the C-contiguous 2D `float64` input slice for the current replication, `output` is the C-contiguous 2D `float64` buffer the function writes into. The declared `output_shape` on the step determines the output buffer's dimensions. Returning `0` signals success; any other value marks the replication as failed for that step.

The two-argument shape is checked when the wrapper is constructed. Everything else about the body is validated by Numba when `cfunc()` is first called, so a body Numba cannot compile in nopython mode raises at that point.

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Original function name. |
| source | `#!python str` | Author-side source text for receiver audit. |
| captured_globals | `#!python Mapping[str, Any]` | Snapshot of accepted globals referenced by the function. |
| safe_namespace_version | `#!python int` | Version of the safe-namespace contract used at wrap time. |
| namespace_kind | `#!python str` | Namespace validated against: `"numpy"` or `"pandas"`. |
| address | `#!python int` | Address of the compiled native callback. Compiles on first access. |

__Methods:__

```python
@classmethod
NumbaCustomFunc.from_source(source: str) -> NumbaCustomFunc

NumbaCustomFunc.cfunc() -> Any
NumbaCustomFunc.__call__(*args: Any, **kwargs: Any) -> Any
```

`from_source` validates source text directly. This is used for code typed into a UI or editor where `inspect.getsource(...)` cannot recover a real file or notebook cell.

`cfunc` compiles the function (once, then cached) and returns the callback with the ABI the native trampoline consumes: `(input_ptr, output_ptr, input_rows, input_columns, output_rows, output_columns) -> int64`. Compilation failures are warned and re-raised carrying Numba's original diagnostic. The compiled artifacts are transient and are never pickled; a wrapper restored from a bundle recompiles on first use.

## `custom_transform`

```python
custom_transform(func: Callable[..., Any]) -> NumbaCustomFunc
```

Decorator form of `NumbaCustomFunc`. The decorated name becomes a callable `NumbaCustomFunc` and can be passed to `transform_step(...)`.

```python
from SymbolicDSGE.monte_carlo import custom_transform

@custom_transform
def demean(sample, output) -> int:
    for j in range(sample.shape[1]):
        column_mean = 0.0
        for i in range(sample.shape[0]):
            column_mean += sample[i, j]
        column_mean /= sample.shape[0]
        for i in range(sample.shape[0]):
            output[i, j] = sample[i, j] - column_mean
    return 0
```

## `PandasCustomFunc`

```python
class PandasCustomFunc(CustomFunc)
```

```python
PandasCustomFunc(func: Callable[..., Any] | CustomFunc) -> PandasCustomFunc

@classmethod
PandasCustomFunc.from_source(source: str) -> PandasCustomFunc
```

Identical validation and properties to the numpy contract, but the body may also reference `pandas` (as `pd`). Intended for `OpType.POSTPROC` summary ops, for example one returning a DataFrame. It runs in plain Python after the replication loop and is never compiled.

## `pandas_operation`

```python
pandas_operation(func: Callable[..., Any]) -> PandasCustomFunc
```

Decorator form of `PandasCustomFunc`, for post-loop ops passed to `postproc_step(...)`. The body may reference `pd`; using it on a per-replication step is rejected when the pipeline is built.

## `NumpyCustomFunc`

```python
class NumpyCustomFunc(CustomFunc)
```

The base contract `NumbaCustomFunc` extends: the same safe-namespace validation without the two-argument signature check and without native compilation. It is not exported from `SymbolicDSGE.monte_carlo`; reach it through `SymbolicDSGE.monte_carlo.custom_op` if you need the plain numpy wrapper directly.

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
    These wrappers are a reproducibility and audit contract, not a sandbox. Loading a bundle with custom operations should be treated like running Python code from the bundle author.
