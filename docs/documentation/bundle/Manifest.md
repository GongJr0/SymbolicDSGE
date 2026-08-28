---
tags:
    - doc
---
# Manifest

```python
@dataclass
class Manifest()
```

`Manifest` is the schema for `manifest.json` at the root of every `.sdsge` archive. It indexes the included members, records provenance, and (optionally) carries the simulation prefill inline.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| created_by | `#!python str` | Library version string. Defaults to `"SymbolicDSGE <version>"` when produced by `BundleBuilder`. |
| created_at | `#!python str | None` | UTC ISO-8601 timestamp set at write time. |
| sdsge_version | `#!python int` | Format version the bundle was written at. Bumped on every manifest change. |
| last_breaking_version | `#!python int` | Version at which the format last broke, as of writing. A reader needs to be at least this version. |
| members | `#!python list[Member]` | Member inventory. Every archive entry has one. |
| simulation | `#!python dict[str, SimSpec] | None` | Inline simulation prefills keyed by role (no separate member). |
| checksums | `#!python dict[str, str]` | SHA-256 hex digests keyed by member path. |

__Methods:__

```python
Manifest.members_by_kind(
    kind: str,
) -> list[Member]
```

Return every member with the given `kind` (e.g. `"model_config"`, `"estimation_data"`).

```python
Manifest.model_member(
    role: str,
) -> Member | None
```

Convenience accessor. Return the `model_config` member with the given `role` (`"reference"` or `"dgp"`), or `None` if absent.

```python
Manifest.to_dict() -> dict[str, Any]
Manifest.to_json(*, indent: int | None = 2) -> str

@classmethod
Manifest.from_dict(data: Mapping[str, Any]) -> Manifest
@classmethod
Manifest.from_json(text: str) -> Manifest
```

Round-trippable JSON shape. `from_dict` / `from_json` validate the version pair and raise `ValueError` on either side of a break.

???+ warning "Forward / backward compatibility"
    Compatibility is judged against breaks, not against version equality. A reader rejects a bundle older than its own `SDSGE_LAST_BREAKING_VERSION`, and rejects one whose `last_breaking_version` exceeds its `SDSGE_FORMAT_VERSION`. A newer bundle from a bump that broke nothing reads fine.

## `Member`

```python
@dataclass
class Member()
```

One archive entry described in the manifest.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| path | `#!python str` | POSIX path inside the archive (e.g. `model/reference.yaml`). |
| kind | `#!python str` | Semantic kind. One of `MEMBER_KINDS` (see below). |
| format | `#!python str` | `"yaml"` / `"json"` / `"csv"` / `"parquet"` / `"pickle"`. Inferred from `path` extension when omitted on construction. |
| role | `#!python str | None` | `"reference"` / `"dgp"` for model members. |
| columns | `#!python list[str] | None` | Column names for tabular members (e.g. observable names on `estimation_data`). |
| options | `#!python dict[str, Any]` | Kind-specific metadata. For `model_config` this carries `compile_kwargs` / `solve_kwargs`. |

__Recognized kinds (`MEMBER_KINDS`):__

| Kind | Purpose |
| --- | --- |
| `model_config` | YAML configuration for a role. |
| `raw_data` | Raw observable file (CSV or Parquet). |
| `estimation_spec` | `EstimatorSpec.params` (`EstimatorParams`) JSON. |
| `estimation_result` | Wrapped `{"type": "mle" | "map" | "mcmc", "data": {...}}`. |
| `estimation_data` | Observed `y` matrix (CSV or Parquet). |
| `estimation_trace` | MCMC posterior columns (CSV or Parquet). |
| `mc_pipeline` | `PipelineSpec` JSON. |
| `mc_raw_model_data` | Raw model data arrays referenced by MC `raw_model_data` nodes. |
| `mc_custom_op` | Bundle-safe custom operation referenced by `transform:custom` or `postproc:custom` specs. |
| `mc_result_meta` | The run's own metadata: counts, timings, failures, and the `run_config` that reproduces it. |
| `mc_test_steps` | Every test step's meta, keyed by step name (JSON). |
| `mc_test_traces` | Every test step's trace columns in one block (CSV or Parquet). |
| `mc_regression_steps` | Every regression step's meta, keyed by step name (JSON). |
| `mc_regression_traces` | Every regression step's trace columns in one block (CSV or Parquet). |
| `mc_transform_steps` | Every transform step's meta, keyed by step name (JSON). |
| `mc_transform_trace` | One transform array: a payload or its retained rep indices (CSV or Parquet). |
| `mc_postproc_steps` | Every post-loop step's meta and inline `summary`, keyed by step name (JSON). |
| `mc_postproc_raw` | One post-loop step's bulk `Raw` array (CSV or Parquet). |

???+ note "One member per step kind, plus one per unpacked array"
    Tests and regressions pack every step's columns into a single block, qualified `{step}.{field}` and extended to `{step}.{field}.{idx}` where a column is 2-D. Transform payloads and postproc `Raw` arrays share no shape with anything, so each takes a member of its own and carries its `name` and `field` in `Member.options`.

???+ note "Kind whitelist"
    `Member.__post_init__` raises `ValueError` for any kind outside `MEMBER_KINDS`.

## `SimSpec`

```python
@dataclass
class SimSpec(Mapping)
```

Simulation prefill. Its fields reconstruct into keyword arguments of [`SolvedModel.sim`](../SolvedModel.md) via `to_sim_kwargs()`.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| T | `#!python int` | Periods to simulate. |
| x0 | `#!python list[float] | ndarray | None` | Initial state vector; zero vector when `None`. |
| observables | `#!python bool` | Include observable paths in the output. |
| shock_scale | `#!python float` | Multiplier applied to all shocks. |
| shocks | `#!python dict[str, ShockParameters] | None` | Per-key shock specs (a [`Shock.to_dict()`](../Shock.md) dict each); `None` for a deterministic run. |

???+ info "Two dict views"
    `SimSpec.to_dict()` is the JSON form written to the manifest, where shocks stay as their `Shock.to_dict()` parameter dicts. The `Mapping` view, via `dict(spec)`, `**spec`, or `spec.to_sim_kwargs()`, is the `sim` keyword form, where each shock is a live `Shock` object. No `Shock` instance is ever serialized; `sim` rebuilds it from the parameters and materializes a `T` horizon draw, so the run is reproducible under a fixed seed.

## Example

```python
from SymbolicDSGE.bundle import Manifest, Member, SimSpec

manifest = Manifest(
    created_by="experiment-1",
    members=[
        Member(
            path="model/reference.yaml",
            kind="model_config",
            role="reference",
            options={"compile_kwargs": {"linearize": False}},
        ),
        Member(
            path="estimation/spec.json",
            kind="estimation_spec",
        ),
        Member(
            path="estimation/observed.csv",
            kind="estimation_data",
            columns=["Infl", "Rate"],
        ),
    ],
    simulation={
        "reference": SimSpec(
            T=25,
            shocks={
                "u": {
                    "dist": "norm",
                    "multivar": False,
                    "seed": 42,
                    "dist_args": [],
                    "dist_kwargs": {"loc": 0.0},
                }
            },
        ),
    },
)

print(manifest.to_json())
```

## See also

- [`LoadedBundle`](LoadedBundle.md): carries the manifest at load time.
- [`sdsge-decompile`](../../portable_experiments/sdsge-decompile.md): extracts the manifest to disk.
