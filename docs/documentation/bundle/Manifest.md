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

You will never need to construct a `Manifest` or write one to disk manually. Writers generate it and loaders parse it.
However, it is available for inspection and cerries top-level metadata. The fields are documented here for reference.

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

???+ note "Simulation prefills"
    Prefills are keyed by name and a bundle can incldue multiple.
    The values unpack directly as `SolvedModel.sim(**prefill)`.

???+ warning "Forward / backward compatibility"
    The version pair is validated when a bundle is read, and an incompatible one raises `ValueError`. Compatibility is judged against breaks, not against version equality: a reader rejects a bundle older than its own `SDSGE_LAST_BREAKING_VERSION`, and rejects one whose `last_breaking_version` exceeds its `SDSGE_FORMAT_VERSION`. A newer bundle from a bump that broke nothing reads fine.

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
| kind | `#!python str` | Semantic kind, e.g. `model_config`, `estimation_data`, `mc_pipeline`. Drawn from `MEMBER_KINDS`; the builder sets it. |
| format | `#!python str` | `"yaml"` / `"json"` / `"csv"` / `"parquet"` / `"pickle"`. Inferred from the `path` extension. |
| model_name | `#!python str | None` | Name of the model associated with this member. |
| columns | `#!python list[str] | None` | Column names for tabular members (e.g. observable names on `estimation_data`). |
| options | `#!python dict[str, Any]` | Kind-specific metadata. `model_config` carries `compile_kwargs` / `solve_kwargs`; an unpacked MC array carries the `name` and `field` it came from. |

## See also

- [`LoadedBundle`](LoadedBundle.md): carries the manifest at load time.
- [`sdsge-decompile`](../../portable_experiments/sdsge-decompile.md): extracts the manifest to disk.
