"""Surface tests for the top-level user-facing bundle API.

Covers source-YAML retention on :class:`ModelConfig`, the
:meth:`SolvedModel.to_bundle_builder` / :meth:`save_sdsge` shortcuts, and the
re-exports at ``SymbolicDSGE`` root (``load_bundle`` / ``BundleBuilder`` /
``LoadedBundle``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import SymbolicDSGE
from SymbolicDSGE import (
    BundleBuilder,
    DSGESolver,
    ModelParser,
    load_bundle,
)
from SymbolicDSGE.bundle import LoadedBundle
from SymbolicDSGE.core.solved_model import SolvedModel


def _solve_test_model(test_model_path) -> SolvedModel:
    parser = ModelParser(test_model_path)
    model, kalman = parser.get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return solver.solve(compiled)


def test_top_level_exports_are_importable() -> None:
    assert SymbolicDSGE.load_bundle is load_bundle
    assert SymbolicDSGE.BundleBuilder is BundleBuilder


def test_path_based_parser_retains_source_yaml(
    test_model_path, test_model_yaml
) -> None:
    parser = ModelParser(test_model_path)
    assert parser.parsed.model.source_yaml == test_model_yaml


def test_from_string_preserves_source_text_exactly(test_model_yaml) -> None:
    # Includes the temp-file round-trip — the source_yaml field must hold the
    # caller's exact input, not whatever the temp file ended up with.
    weird_text = test_model_yaml + "\n# trailing comment\n"
    parser = ModelParser.from_string(weird_text)
    assert parser.parsed.model.source_yaml == weird_text


def test_save_sdsge_round_trips_via_load_bundle(
    tmp_path: Path, test_model_path
) -> None:
    solved = _solve_test_model(test_model_path)
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
    tmp_path: Path, test_model_path
) -> None:
    solved = _solve_test_model(test_model_path)
    builder = solved.to_bundle_builder(compile_kwargs={}, created_by="api-test")
    assert isinstance(builder, BundleBuilder)
    target = builder.write(tmp_path / "chained.sdsge")
    loaded = load_bundle(target)
    assert loaded.manifest.created_by == "api-test"
    assert loaded.models["reference"] is not None


def test_save_sdsge_yaml_text_override_takes_precedence(
    tmp_path: Path, test_model_yaml, test_model_path
) -> None:
    solved = _solve_test_model(test_model_path)
    override = test_model_yaml + "\n# explicit override marker\n"
    target = solved.save_sdsge(
        tmp_path / "override.sdsge",
        yaml_text=override,
        compile_kwargs={},
    )
    loaded = load_bundle(target)
    assert loaded.manifest.model_member("reference") is not None
    # The bundle's embedded YAML is the override, not the retained source.
    member_path = loaded.manifest.model_member("reference").path
    from SymbolicDSGE.bundle.container import BundleArchive

    archive = BundleArchive.open(target)
    assert archive.read_text(member_path) == override


def test_save_sdsge_raises_without_source_yaml(tmp_path: Path, test_model_path) -> None:
    solved = _solve_test_model(test_model_path)
    # Simulate a programmatically constructed config (no parse history).
    solved.compiled.config.source_yaml = None
    with pytest.raises(ValueError, match="source YAML"):
        solved.save_sdsge(tmp_path / "nope.sdsge")
