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

_MODEL_PATH = Path("MODELS/test.yaml")
_MODEL_YAML = _MODEL_PATH.read_text(encoding="utf-8")


def _solve_test_model() -> SolvedModel:
    parser = ModelParser(_MODEL_PATH)
    model, kalman = parser.get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile(n_state=3, n_exog=2)
    return solver.solve(compiled)


def test_top_level_exports_are_importable() -> None:
    assert SymbolicDSGE.load_bundle is load_bundle
    assert SymbolicDSGE.BundleBuilder is BundleBuilder


def test_path_based_parser_retains_source_yaml() -> None:
    parser = ModelParser(_MODEL_PATH)
    assert parser.parsed.model.source_yaml == _MODEL_YAML


def test_from_string_preserves_source_text_exactly() -> None:
    # Includes the temp-file round-trip — the source_yaml field must hold the
    # caller's exact input, not whatever the temp file ended up with.
    weird_text = _MODEL_YAML + "\n# trailing comment\n"
    parser = ModelParser.from_string(weird_text)
    assert parser.parsed.model.source_yaml == weird_text


def test_save_sdsge_round_trips_via_load_bundle(tmp_path: Path) -> None:
    solved = _solve_test_model()
    target = solved.save_sdsge(
        tmp_path / "model.sdsge",
        compile_kwargs={"n_state": 3, "n_exog": 2},
    )
    loaded = load_bundle(target)
    assert isinstance(loaded, LoadedBundle)
    assert loaded.reference is not None
    # Re-solved model is usable.
    assert loaded.reference.sim(5)["_X"].shape[0] == 6


def test_to_bundle_builder_returns_chainable_builder(tmp_path: Path) -> None:
    solved = _solve_test_model()
    builder = solved.to_bundle_builder(
        compile_kwargs={"n_state": 3, "n_exog": 2}, created_by="api-test"
    )
    assert isinstance(builder, BundleBuilder)
    target = builder.write(tmp_path / "chained.sdsge")
    loaded = load_bundle(target)
    assert loaded.manifest.created_by == "api-test"
    assert loaded.reference is not None


def test_save_sdsge_yaml_text_override_takes_precedence(tmp_path: Path) -> None:
    solved = _solve_test_model()
    override = _MODEL_YAML + "\n# explicit override marker\n"
    target = solved.save_sdsge(
        tmp_path / "override.sdsge",
        yaml_text=override,
        compile_kwargs={"n_state": 3, "n_exog": 2},
    )
    loaded = load_bundle(target)
    assert loaded.manifest.model_member("reference") is not None
    # The bundle's embedded YAML is the override, not the retained source.
    member_path = loaded.manifest.model_member("reference").path
    from SymbolicDSGE.bundle.container import BundleArchive

    archive = BundleArchive.open(target)
    assert archive.read_text(member_path) == override


def test_save_sdsge_raises_without_source_yaml(tmp_path: Path) -> None:
    solved = _solve_test_model()
    # Simulate a programmatically constructed config (no parse history).
    solved.compiled.config.source_yaml = None
    with pytest.raises(ValueError, match="source YAML"):
        solved.save_sdsge(tmp_path / "nope.sdsge")
