"""Surface tests for the top-level user-facing bundle API.

Covers the :meth:`SolvedModel.to_bundle_builder` / :meth:`save_sdsge`
shortcuts and the re-exports at ``SymbolicDSGE`` root (``load_bundle`` /
``BundleBuilder`` / ``LoadedBundle``).
"""

from __future__ import annotations

from pathlib import Path

import SymbolicDSGE
from SymbolicDSGE import (
    BundleBuilder,
    load_bundle,
)
from SymbolicDSGE.bundle import LoadedBundle
from SymbolicDSGE.core.config_emitter import emit_config_yaml


def test_top_level_exports_are_importable() -> None:
    assert SymbolicDSGE.load_bundle is load_bundle
    assert SymbolicDSGE.BundleBuilder is BundleBuilder


def test_save_sdsge_round_trips_via_load_bundle(tmp_path: Path, solved_test) -> None:
    solved = solved_test
    target = solved.save_sdsge(
        tmp_path / "model.sdsge",
        compile_kwargs={},
    )
    loaded = load_bundle(target)
    assert isinstance(loaded, LoadedBundle)
    assert loaded.models["reference"] is not None
    # Re-solved model is usable.
    assert loaded.models["reference"].sim(5).X.shape[0] == 5


def test_to_bundle_builder_returns_chainable_builder(
    tmp_path: Path, solved_test
) -> None:
    solved = solved_test
    builder = solved.to_bundle_builder(compile_kwargs={}, created_by="api-test")
    assert isinstance(builder, BundleBuilder)
    target = builder.write(tmp_path / "chained.sdsge")
    loaded = load_bundle(target)
    assert loaded.manifest.created_by == "api-test"
    assert loaded.models["reference"] is not None


def test_save_sdsge_accepts_a_config_with_no_parse_history(
    tmp_path: Path, solved_test
) -> None:
    target = solved_test.save_sdsge(tmp_path / "emitted.sdsge", compile_kwargs={})
    loaded = load_bundle(target)
    member = loaded.manifest.model_member("reference")
    assert member is not None

    from SymbolicDSGE.bundle.container import BundleArchive

    archive = BundleArchive.open(target)
    assert archive.read_text(member.path) == emit_config_yaml(
        solved_test.compiled.config
    )
