---
tags:
    - doc
---
# Shock

```python
class Shock(
    dist: Literal["norm", "t", "uni"] | rv_generic | multi_rv_generic | None = None,
    seed: int | None = 0,
    dist_args: tuple = (),
    dist_kwargs: dict | None = None,
)
```

`Shock` is a horizon independent shock specification. Pass it directly inside `SolvedModel.sim(..., shocks={...})`; the simulation supplies the horizon and the calibration.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| dist | Distribution family (`"norm"`, `"t"`, `"uni"`) or a scipy distribution object. |
| seed | Random seed. Pass `None` for unseeded draws. |
| dist_args | Positional arguments passed to the distribution draw method. |
| dist_kwargs | Keyword arguments passed to the distribution draw method. Do not pass `scale`; simulation supplies the model scale. |

???+ warning "Scale And Arity Are Model Supplied"
    A `Shock` stores distribution shape and location parameters only. The standard deviation of a single shock, and the covariance of a grouped one, come from the `SolvedModel` calibration at simulation time. Arity comes from the spec key: a key naming several shocks draws them jointly, and the same `Shock` serves either way.

## Shock specs

A simulation takes a mapping of shock names to specifications. Two forms are accepted, and both leave the horizon to the simulation:

| __Form__ | __Meaning__ |
|:---------|------------:|
| `#!python Shock(...)` | Draw this family, reseeded per Monte Carlo replication, using the calibrated scale. |
| `#!python ndarray` | Use this path verbatim, shaped `(T,)` for one shock or `(T, k)` for a grouped key. |

```python
sol.sim(T=10, shocks={"e_g": Shock(dist="norm", seed=1)})
sol.sim(T=10, shocks={"e_g,e_z": Shock(dist="norm", seed=1)})

# A deterministic impulse is just a path.
path = np.zeros(10)
path[0] = sol.config.calibration.parameters["sig_g"]
sol.sim(T=10, shocks={"e_g": path})
```

???+ warning "Callables Are Not Specs"
    A callable cannot be reseeded per replication, cannot lower to the native Monte Carlo draw, and cannot be serialized into a bundle, since only the one path it returned would travel. Pass a `Shock` to be redrawn, or an array to be used as given.

## `draw_fn`

```python
Shock.draw_fn(T: int, multivar: bool) -> Callable[[float | ndarray, int | None, ndarray | None], ndarray]
```

Resolve the distribution family once for a fixed horizon. The resolution is what a plan caches: a caller that redraws under many seeds pays for it once. Family validation is eager: an unknown family, a Student-t without `df`, or a multivariate uniform raises here rather than at draw time.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Number of simulated periods. |
| multivar | Whether the entry draws jointly. Resolution passes `len(columns) > 1` off the spec key; a direct caller states it. Required, with no default: one spec covering every shock jointly is as ordinary as one spec per shock. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python Callable[[scale, seed, factor], ndarray]` | `scale` is a standard deviation for one shock or a covariance/shape matrix for a group, `seed` varies the draw, and `factor` is an optional precomputed matrix `F` with `F @ F.T == scale` that skips refactorizing an unchanged covariance. |

???+ tip "Prefer Passing The Shock"
    `draw_fn` exists for callers that hold the resolution themselves. A spec handed to `sim` needs none of it: the resolution binds the horizon, reads the calibration, and precomputes the factor in one pass.

## Serialization

```python
Shock.to_dict() -> ShockParameters
Shock.from_dict(data: Mapping[str, Any]) -> Shock
```

`to_dict()` serializes shocks with string distribution families. A live scipy distribution object cannot be faithfully reproduced from JSON and is rejected.
