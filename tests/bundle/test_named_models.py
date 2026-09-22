"""Named-model bundle associations and validation at emission."""

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from SymbolicDSGE.bundle import loader
from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.bundle.container import BundleArchive
from SymbolicDSGE.bundle.manifest import Member


@pytest.fixture
def model_loader(monkeypatch):
    # Exercise bundle routing without compiling models. Each YAML payload remains
    # identifiable through the solver and estimator boundaries.
    monkeypatch.setattr(
        loader.ModelParser,
        "from_string",
        lambda text: SimpleNamespace(get_all=lambda: (text, None)),
    )

    class Solver:
        def __init__(self, model, kalman):
            self.model = model

        def compile(self, **kwargs):
            return SimpleNamespace(name=self.model, options=kwargs)

        def solve(self, compiled, **kwargs):
            return SimpleNamespace(compiled=compiled, options=kwargs)

    monkeypatch.setattr(loader, "DSGESolver", Solver)
    monkeypatch.setattr(
        loader.Estimator,
        "from_spec",
        lambda spec, *, compiled: SimpleNamespace(spec=spec, compiled=compiled),
    )


def _add_estimation(builder, name):
    source = SimpleNamespace(
        to_spec=lambda: SimpleNamespace(
            params={"observables": ["y"]}, y=np.arange(4.0).reshape(4, 1)
        )
    )
    builder.add_estimation(source, model_name=name)


@pytest.mark.parametrize("estimation_first", [False, True])
def test_named_models_and_estimation_roundtrip(
    tmp_path, model_loader, estimation_first
):
    builder = BundleBuilder()
    if estimation_first:
        _add_estimation(builder, "alternative")
    for name in ("baseline", "alternative", "third"):
        builder.add_model(
            name, name, compile_kwargs={"marker": name}, solve_kwargs={"order": 1}
        )
    if not estimation_first:
        _add_estimation(builder, "alternative")
    manifest, _ = builder.build()
    assert manifest.members_by_kind("estimation_spec")[0].model_name == "alternative"
    loaded = loader.load_bundle(builder.write(tmp_path / "named.sdsge"))
    assert set(loaded.models) == {"baseline", "alternative", "third"}
    for name, model in loaded.models.items():
        assert model.compiled.name == name
        assert model.compiled.options == {"marker": name}
        assert model.options == {"order": 1}
    assert loaded.estimation.estimator.compiled is loaded.models["alternative"].compiled
    np.testing.assert_array_equal(
        loaded.estimation.estimator.spec.y, np.arange(4.0).reshape(4, 1)
    )


@pytest.mark.parametrize("target", ["missing", ""])
@pytest.mark.parametrize("emitter", ["build", "write"])
def test_emission_rejects_missing_estimation_target(tmp_path, target, emitter):
    builder = BundleBuilder().add_model("reference", "unused")
    _add_estimation(builder, target)
    builder.manifest()  # Inspection of incomplete builders remains available.
    path = tmp_path / "existing.sdsge"
    path.write_bytes(b"existing content")
    with pytest.raises(ValueError, match="targets model"):
        builder.build() if emitter == "build" else builder.write(path)
    assert path.read_bytes() == b"existing content"


@pytest.mark.parametrize("target", ["missing", ""])
def test_loading_rejects_missing_estimation_target(model_loader, target):
    builder = BundleBuilder().add_model("reference", "reference")
    _add_estimation(builder, target)
    # Target validation must happen before any data reads.
    archive = BundleArchive(builder.manifest(), {})
    with pytest.raises(ValueError, match="not present"):
        loader._load_estimation(archive, archive.manifest, {"reference": object()})


def test_legacy_role_and_implicit_reference_load(tmp_path, model_loader):
    builder = BundleBuilder().add_model("reference", "legacy")
    _add_estimation(builder, None)
    manifest, files = builder.build()
    payload = manifest.to_dict()
    for member in payload["members"]:
        if "model_name" in member:
            member["role"] = member.pop("model_name")
    path = tmp_path / "legacy.sdsge"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(payload))
        for name, data in files.items():
            archive.writestr(name, data)
    loaded = loader.load_bundle(path)
    assert loaded.estimation.estimator.compiled is loaded.models["reference"].compiled


def test_new_model_name_takes_precedence_over_legacy_role():
    member = Member.from_dict(
        {
            "path": "model/x.yaml",
            "kind": "model_config",
            "model_name": "new",
            "role": "old",
        }
    )
    assert member.model_name == "new"
    assert "role" not in member.to_dict()
    assert member.to_dict()["model_name"] == "new"
