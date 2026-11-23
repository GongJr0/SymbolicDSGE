---
tags:
    - doc
---

# Custom Operations

Custom Monte Carlo operations are authored as plain top-level functions and wrapped so the function source and accepted globals can travel with the `.sdsge` archive. Custom transform callables become `NumbaCustomFunc` objects and are compiled for native execution. The `custom_transform` decorator can also be used to wrap them manually. See [custom transforms](operations/transforms/custom.md) for details on how to create a custom transform step.

The wrapper validates function bodies against a safe namespace at wrap time: `numpy` (as `np`), `math`, `statistics`, `operator`, a selected set of builtins, and captured immutable or numpy globals.

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

???+ warning "Not a security sandbox"
    The wrapper is a reproducibility and audit contract, not a sandbox. Loading a bundle with custom operations should be treated like running Python code from the bundle author.
