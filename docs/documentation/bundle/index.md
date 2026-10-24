---
tags:
    - doc
---
# Bundle

The `bundle` module produces and consumes `.sdsge` archives: versioned zip files with the ability to store model configurations, estimation spec/result/data, Monte Carlo pipeline/result, and simulation prefills. Authored bundles are portable: a receiver only needs `pip install SymbolicDSGE` to reach every component the bundle carries, and `pip install "SymbolicDSGE[ui]"` to hydrate the GUI with one.

For task-oriented walkthroughs see the [Bundle Authoring Guide](../../guides/bundle_authoring_guide.md) and [Bundle Loading Guide](../../guides/bundle_loading_guide.md). For the CLI counterparts see the [Portable Experiments](../../portable_experiments/index.md) section.

???+ note "Top-level imports"
    `BundleBuilder`, `LoadedBundle`, and `load_bundle` are re-exported at `SymbolicDSGE` root. Everything else in this section lives under `SymbolicDSGE.bundle`.

## Module layout

| Class / function                    | Description                                                         |
|-------------------------------------|---------------------------------------------------------------------|
| [`BundleBuilder`](BundleBuilder.md) | Fluent assembler for the in-code authoring path.                    |
| [`load_bundle`](load_bundle.md)     | Open a `.sdsge` and reconstruct its components into Python objects. |
| [`LoadedBundle`](LoadedBundle.md)   | Container holding every component returned by `load_bundle`.        |
| [`Manifest`](Manifest.md)           | Versioned archive index for `manifest.json`.                        |
