---
tags:
    - doc
---

# Custom Operations

Custom Monte Carlo transforms can run directly from any callable. Bundle-safe custom transforms must use `NumpyCustomFunc` or the `custom_operation` decorator so the function source and accepted globals can travel with the `.sdsge` archive.

## `NumpyCustomFunc`

```python
class NumpyCustomFunc()
```

Opt-in wrapper that validates and snapshots a numerical Python function.

```python
NumpyCustomFunc(func: Callable[..., Any] | NumpyCustomFunc) -> NumpyCustomFunc
```

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Original function name. |
| source | `#!python str` | Author-side source text for receiver audit. |
| captured_globals | `#!python Mapping[str, Any]` | Snapshot of accepted globals referenced by the function. |
| safe_namespace_version | `#!python int` | Version of the safe-namespace contract used at wrap time. |

__Methods:__

```python
NumpyCustomFunc.from_source(source: str) -> NumpyCustomFunc  # @classmethod
NumpyCustomFunc.__call__(*args: Any, **kwargs: Any) -> Any
```

`from_source` validates source text directly. This is used for code typed into a UI/editor where `inspect.getsource(...)` cannot recover a real file or notebook cell.

## `custom_operation`

```python
custom_operation(func: Callable[..., Any]) -> NumpyCustomFunc
```

Decorator form of `NumpyCustomFunc`. The decorated name becomes a callable `NumpyCustomFunc` and can be passed to `transform_step(...)`.

## Validation Contract

| __Allowed__ | __Description__ |
|:------------|----------------:|
| One top-level `def` | Lambdas, nested functions, methods, partials, builtins, and C-extension callables are rejected. |
| Numeric safe namespace | `numpy` as `np`, selected standard modules, selected builtins, and captured immutable/numpy globals. |
| Explicit source | Source must be recoverable or supplied through `from_source(...)`. |

| __Rejected__ | __Reason__ |
|:-------------|-----------:|
| Imports, `global`, `nonlocal`, async, yield, nested `def`, classes | These make the shipped function harder to audit and reproduce. |
| Closure captures | Promote values to accepted globals or pass them as kwargs. |
| Unsupported globals | The wrapper snapshots only accepted numeric/scalar/container helpers. |

???+ warning "Not a security sandbox"
    `NumpyCustomFunc` is a reproducibility and audit contract, not a sandbox. Loading a bundle with custom operations should be treated like running Python code from the bundle author.

