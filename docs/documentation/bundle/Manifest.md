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
| created_at | `#!python str \| None` | UTC ISO-8601 timestamp set at write time. |
| sdsge_version | `#!python int` | Format version. Readers reject bundles with `sdsge_version > SDSGE_FORMAT_VERSION`. |
| members | `#!python list[Member]` | Member inventory — every archive entry has one. |
| simulation | `#!python SimSpec \| None` | Inline simulation prefill (no separate member). |
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

Convenience accessor — return the `model_config` member with the given `role` (`"reference"` or `"dgp"`), or `None` if absent.

```python
Manifest.to_dict() -> dict[str, Any]
Manifest.to_json(*, indent: int | None = 2) -> str
Manifest.from_dict(data: Mapping[str, Any]) -> Manifest      # @classmethod
Manifest.from_json(text: str) -> Manifest                    # @classmethod
```

Round-trippable JSON shape. `from_dict` / `from_json` validate `sdsge_version` and raise `ValueError` when the bundle is newer than the installed library supports.

???+ warning "Forward / backward compatibility"
    Readers are forward-tolerant on older versions (a v1 reader opens v1 bundles) but strict on newer ones (a v1 reader refuses a v2 bundle). Bump `SDSGE_FORMAT_VERSION` only on breaking manifest changes.

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
| kind | `#!python str` | Semantic kind — one of `MEMBER_KINDS` (see below). |
| format | `#!python str` | `"yaml"` / `"json"` / `"csv"` / `"parquet"`. Inferred from `path` extension when omitted on construction. |
| role | `#!python str \| None` | `"reference"` / `"dgp"` for model members. |
| columns | `#!python list[str] \| None` | Column names for tabular members (e.g. observable names on `estimation_data`). |
| options | `#!python dict[str, Any]` | Kind-specific metadata. For `model_config` this carries `compile_kwargs` / `solve_kwargs`. |

__Recognized kinds (`MEMBER_KINDS`):__

| Kind | Purpose |
| --- | --- |
| `model_config` | YAML configuration for a role. |
| `raw_data` | Raw observable file (CSV or Parquet). |
| `estimation_spec` | `EstimationSpec` JSON. |
| `estimation_result` | Wrapped `{"type": "mcmc" \| "optimization", "data": {...}}`. |
| `estimation_data` | Observed `y` matrix (CSV or Parquet). |
| `estimation_trace` | MCMC posterior columns (CSV or Parquet). |
| `mc_pipeline` | `PipelineSpec` JSON. |
| `mc_result` | Trace-free MC run document (JSON). |
| `mc_trace` | MC trace columns (CSV or Parquet). |
| `mc_raw_data` | Raw-data arrays referenced by MC `raw_data` nodes. |
| `mc_custom_op` | Bundle-safe custom operation referenced by MC `custom` nodes. |

???+ note "Kind whitelist"
    `Member.__post_init__` raises `ValueError` for any kind outside `MEMBER_KINDS`. Adding a new kind requires bumping `SDSGE_FORMAT_VERSION` so older readers don't silently drop it.

## `SimSpec`

```python
@dataclass
class SimSpec()
```

Simulation prefill — the receiver clicks **Run** in the GUI to reproduce the author's intended simulation. Stored inline in `Manifest.simulation`, not as a member.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| role | `#!python str` | Active model slot — typically `"reference"`. |
| T | `#!python int` | Periods to simulate. |
| observables | `#!python bool` | Include observable paths in the output. |
| shock_scale | `#!python float` | Multiplier applied to all shocks. |
| shock_generation | `#!python ShockGeneration \| None` | RNG settings; `None` if raw shock paths are supplied instead. |
| shock_std | `#!python dict[str, float]` | Per-shock standard deviation overrides. |
| shock_corr | `#!python dict[str, float]` | Pairwise shock correlation overrides keyed by `"a,b"` syntax. |
| shocks | `#!python dict[str, list[float]] \| None` | Optional raw shock paths inline. |

???+ info "Determinism"
    Sim results are not stored. Replaying the prefill against the preloaded model reproduces the intended run because numpy `PCG64` + a fixed `ShockGeneration.seed` are deterministic.

## `ShockGeneration`

```python
@dataclass
class ShockGeneration()
```

RNG settings for replayed shock generation.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| dist | `#!python str` | Distribution name — `"norm"` / `"t"` / `"uni"`. |
| seed | `#!python int \| None` | RNG seed for reproducibility. |
| loc | `#!python float` | Location parameter. |
| df | `#!python float` | Degrees of freedom (Student's `t`). |

## Example

```python
from SymbolicDSGE.bundle import Manifest, Member, SimSpec, ShockGeneration

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
    simulation=SimSpec(
        role="reference",
        T=25,
        shock_generation=ShockGeneration(seed=42),
    ),
)

print(manifest.to_json())
```

## See also

- [`LoadedBundle`](LoadedBundle.md) — carries the manifest at load time.
- [`sdsge-decompile`](../../portable_experiments/sdsge-decompile.md) — extracts the manifest to disk.
