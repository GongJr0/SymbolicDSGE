---
tags:
    - guide
---

# `sdsge-decompile`

`sdsge-decompile` extracts a `.sdsge` bundle into a directory. The output layout matches the [`sdsge-compile`](sdsge-compile.md) input convention, so the result can be edited and recompiled into an equivalent bundle.

```bash
sdsge-decompile <bundle> [-o OUTPUT] [--csv] [--force]
```

## Arguments

| Argument | Description |
| --- | --- |
| `bundle` | `.sdsge` file to extract. |
| `-o`, `--output` | Output directory. Defaults to `<bundle-stem>/` next to the file. |
| `--csv` | Re-encode Parquet members as CSV in the output (useful for editing). |
| `--force` | Overwrite the output directory if it already exists. Without this flag an existing directory raises `FileExistsError`. |

## Examples

Extract verbatim — Parquet members stay Parquet:

```bash
sdsge-decompile my-experiment.sdsge
```

Decompile for editing — Parquet members are re-encoded as CSV:

```bash
sdsge-decompile my-experiment.sdsge --csv -o my-experiment/
```

## What lands where

The extractor writes each member at its compile-input authoring path, not at its archive path. For most kinds these match; the one exception is `model_config`:

| Member kind | Archive path | Extracted path |
| --- | --- | --- |
| `model_config` (role=reference) | `model/reference.yaml` | `reference.yaml` |
| `model_config` (role=dgp) | `model/dgp.yaml` | `dgp.yaml` |
| `estimation_*` | `estimation/...` | `estimation/...` |
| `mc_*` | `montecarlo/...` | `montecarlo/...` |
| `raw_data` | `data/...` | `data/...` |
| Simulation prefill (inline in manifest) | — | `simulation.json` |

???+ info "Options sidecars"
    Model `Member.options` (e.g. `compile_kwargs`/`solve_kwargs`) are extracted to `<role>.options.json` at the directory root. The compile entry point reads these sidecars on recompile, so the round-trip preserves them.

???+ note "Inline simulation prefill"
    The `SimSpec` rides inline in the manifest (it has no member of its own). On decompile it is written to `simulation.json` at the root so the recompile picks it up via the standard directory layout.

## `--csv` re-encoding

With `--csv`, every Parquet member is decoded to CSV. The extension and `manifest.json` `format` field are rewritten to reflect the new layout.

| Member | CSV header strategy |
| --- | --- |
| `estimation_data` | Observable names from `Member.columns` (semantic headers — user-friendly). |
| `estimation_trace` | Mechanical `{name}.{j}` expansion from `trace_to_csv`. |
| `mc_trace` | Same mechanical expansion. |
| `raw_data` | Existing column names from the Parquet schema. |

???+ warning "Checksum field on decompile"
    The `manifest.json` written by `sdsge-decompile` omits checksums. They were SHA-256 over the archive bytes; after extraction (and especially after `--csv` re-encoding) those bytes change. A subsequent `sdsge-compile` recomputes them. Treat the decompiled `manifest.json` as informational, not as a re-compile input — the compile pass auto-detects the layout.

## Round-trip

`compile → decompile [--csv] → edit → compile` produces a bundle equivalent on every reconstructable component:

```bash
sdsge-compile my-experiment/                  # → my-experiment.sdsge
sdsge-decompile my-experiment.sdsge --csv     # → my-experiment/ (editable)
# ... edit reference.yaml, observed.csv, estimation/spec.json, etc.
sdsge-compile my-experiment/                  # → updated my-experiment.sdsge
```

???+ info "What is preserved across the round-trip"
    Model YAML, estimation spec/result, observed data, MCMC posterior, MC pipeline/result, simulation prefill, model options (compile/solve kwargs). The `created_by` and `created_at` fields are regenerated on each compile.

## See also

- [`sdsge-compile`](sdsge-compile.md) — the inverse operation.
- [Bundle Loading Guide](../guides/bundle_loading_guide.md) — read a `.sdsge` from code instead.
- [`load_bundle`](../documentation/bundle/index.md#load_bundle) — the in-code equivalent of this CLI.
