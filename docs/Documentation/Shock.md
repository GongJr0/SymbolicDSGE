---
tags:
    - doc
---
# Shock

```python
class Shock(
    dist: Literal["norm", "t", "uni"] | rv_generic | multi_rv_generic | None = None,
    multivar: bool = False,
    seed: int | None = 0,
    dist_args: tuple = (),
    dist_kwargs: dict | None = None,
    shock_arr: ndarray | None = None,
)
```

`Shock` is a horizon independent shock specification. Pass it directly inside `SolvedModel.sim(..., shocks={...})`, or materialize it manually with `shock_generator(T)`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| dist | Distribution family (`"norm"`, `"t"`, `"uni"`) or a scipy distribution object. |
| multivar | If `True`, the generated callable expects a covariance or shape matrix and returns a two dimensional shock array. |
| seed | Random seed. Pass `None` for unseeded draws. |
| dist_args | Positional arguments passed to the distribution draw method. |
| dist_kwargs | Keyword arguments passed to the distribution draw method. Do not pass `scale`; simulation supplies the model scale. |
| shock_arr | Optional materialized shock array used by `place_shocks(...)`. |

???+ warning "Scale Is Model Supplied"
    A generator style `Shock` stores distribution shape and location parameters, but the shock standard deviation or covariance comes from the `SolvedModel` calibration at simulation time.

## `shock_generator`

```python
Shock.shock_generator(T: int) -> Callable[[float | ndarray], ndarray]
```

Build a callable for a fixed simulation horizon.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Number of simulated periods. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python Callable[[float], ndarray]` | Univariate generator accepting one shock standard deviation. |
| `#!python Callable[[ndarray], ndarray]` | Multivariate generator accepting a covariance or shape matrix. |

## `place_shocks`

```python
Shock.place_shocks(
    shock_spec: dict[int, float] | dict[tuple[int, int], float],
    T: int,
) -> ndarray
```

Return a materialized shock array with selected entries replaced by `shock_spec` values.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| shock_spec | Univariate `{time_idx: value}` or multivariate `{(time_idx, column_idx): value}` placement map. |
| T | Number of simulated periods. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python ndarray` | Shock array with specified entries replaced. |

???+ warning "Multivariate Shape Inference"
    If `shock_arr` is absent and `multivar=True`, the number of columns is inferred from the largest column index in `shock_spec`.

## Serialization

```python
Shock.to_dict() -> dict[str, Any]
Shock.from_dict(data: Mapping[str, Any]) -> Shock
```

`to_dict()` serializes generator style shocks with string distribution families. It does not serialize scipy distribution objects or materialized `shock_arr` arrays.

__Examples:__

```python
shock = Shock(dist="norm", seed=1)
generator = shock.shock_generator(T=10)

placed = Shock().place_shocks({0: 1.0, 3: -0.5}, T=10)
joint = Shock(multivar=True).place_shocks({(0, 0): 1.0, (0, 1): 2.0}, T=10)
```
