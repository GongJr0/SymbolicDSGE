---
tags:
    - doc
---
# Shock

```python
class Shock(
    dist: Literal["norm", "t", "uni"] | rv_generic | multi_rv_generic | None = None,
    seed: int | None = 0,
    dist_kwargs: dict | None = None,
)
```

`Shock` is a horizon independent shock specification. Pass it directly inside `SolvedModel.sim(..., shocks={...})`; the simulation supplies the horizon and the calibration.

__Inputs:__

| __Name__    |                                                                                                     __Description__ |
|:------------|--------------------------------------------------------------------------------------------------------------------:|
| dist        |                                      Distribution family (`"norm"`, `"t"`, `"uni"`) or a scipy distribution object. |
| seed        |                                                                        Random seed. Pass `None` for unseeded draws. |
| dist_kwargs | Keyword arguments passed to the distribution draw method. Do not pass `scale`; simulation supplies the model scale. |

???+ warning "Scale is model supplied"
    A `Shock` stores distribution shape and location parameters only. The standard deviation of a single shock, and the covariance of a grouped one, come from the `SolvedModel` calibration at simulation time.

&nbsp;

```python
Shock.joint(*keys : str) -> Shock
```

Bind one or more shock names to the `Shock` instance for a joint draw. A non-zero correlation among the group is exercised when present.
`joint` returns a new `Shock` instance and the original instance the call was made from remains usable for binding a separate shock(s).

__Inputs:__

| __Name__ |                                          __Description__ |
|:---------|---------------------------------------------------------:|
| keys     | One or more shock names to bind to the `Shock` instance. |

&nbsp;

```python
Shock.independent(*keys : str, offset_seeds: bool = True) -> list[Shock]
```

Return separate shock instances with a shared distribution specification. Binds multiple shock names at the same time but disregards any correlation the model configuration may imply. Returns a list of __new__ `Shock` instances, keeping the caller's instance usable for binding a separate shock(s). `offset_seeds` increments the seed for each instance so specifications don't return identical draws.

__Inputs:__

| __Name__     |                                          __Description__ |
|:-------------|---------------------------------------------------------:|
| keys         | One or more shock names to bind to the `Shock` instance. |
| offset_seeds |         Whether to increment the seed for each instance. |

&nbsp;

```python
class ShockPath(path: ndarray, *keys: str)
```

A pre-materialized `ndarray` path bound to `*keys` specified at the constructor. `path` column count must match the number of `*keys` and keys should be specified in the order the appear in `path`.

## Shock specs

A simulation takes a sequence of entries that name their own targets. Two forms are accepted, and both leave the horizon to the simulation:

| __Form__              |                                                                         __Meaning__ |
|:----------------------|------------------------------------------------------------------------------------:|
| `#!python Shock(...)` | Draw this family, reseeded per Monte Carlo replication, using the calibrated scale. |
| `#!python ShockPath(...)`    |  Use  `ShockPath.path` verbatim, shaped `(T,)`/`(T, 1)` for one shock or `(T, k)` for a grouped key. |

```python
sol.sim(T=10, shocks=[Shock(dist="norm", seed=1).joint("e_z")])
sol.sim(T=10, shocks=[Shock(dist="norm", seed=1).joint("e_g", "e_z")])

# IRF replicated with a simulation using a pre-materialized path
path = np.zeros(10)
path[0] = sol.config.calibration.parameters["sig_g"]
sol.sim(T=10, shocks=[ShockPath(path, "e_g")])
```

???+ warning "Mapping Specs Are Deprecated"
    A simulation still accepts the older shape, a mapping keyed from the outside:

    ```python
    sol.sim(T=10, shocks={"e_g": Shock(dist="norm", seed=1)})            # key binds the Shock
    sol.sim(T=10, shocks={("e_g", "e_z"): Shock(dist="norm", seed=1)})   # tuple key binds a group
    sol.sim(T=10, shocks={"e_g": path})                                  # key names a bare array
    ```

    `Shock`s must be unbound in the deprecated spec and `ShockPath`s are not accepted.


## Serialization

```python
Shock.to_dict() -> ShockParameters
Shock.from_dict(data: Mapping[str, Any]) -> Shock

ShockPath.to_dict() -> ShockPathParameters
ShockPath.from_dict(data: Mapping[str, Any]) -> ShockPath
```

`to_dict()` serializes shocks with string distribution families. A live scipy distribution object cannot be faithfully reproduced from JSON and is rejected.
