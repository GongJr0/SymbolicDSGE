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
    Prefills are keyed by name and a bundle can include multiple.
    The values unpack directly as `SolvedModel.sim(**prefill)`.

???+ warning "Forward / backward compatibility"
    Each bundle on disk carries two version numbers, and validation during a read relies on them plus an additional constant.
    The version and its breaks/readability are defined through the constants below:

    | __Manifest Field__ | __Library Constant__ | __Description__ |
    |:------------------|:-------------------|----------------:|
    | `sdsge_version` | `SDSGE_FORMAT_VERSION` | The current bundle format. A mismatch of this version does not indicate unreadability. |
    | `last_breaking_version` | `SDSGE_LAST_BREAKING_VERSION` | The last version that broke backward compatibility. A reader with `SDSGE_FORMAT_VERSION` predating the bundle's `last_breaking_version` cannot read the bundle. |
    | N/A | `SDSGE_MIN_READABLE_VERSION` | The library's minimum readable version. Lets the reader declare what versions can be read regardless of breaks. Denotes the oldest bundle version that's forward compatible with the current library. |

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
