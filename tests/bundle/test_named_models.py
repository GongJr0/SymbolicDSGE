"""Named-model bundle associations and validation at emission."""

import pytest

from SymbolicDSGE.bundle import loader
from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.bundle.container import BundleArchive
from SymbolicDSGE.core.solved_model import (
    FirstOrderSolvedModel,
    SecondOrderSolvedModel,
)


@pytest.mark.parametrize("estimation_first", [False, True])
def test_named_models_and_estimation_roundtrip(
    tmp_path,
    parsed_test,
    parsed_post82,
    parsed_rbc_so,
    compiled_test,
    compiled_post82,
    compiled_rbc_so,
    post82_estimator,
    estimation_first,
):
    estimator = post82_estimator()
    builder = BundleBuilder()
    if estimation_first:
        builder.add_estimation(estimator, model="alternative")
    builder.add_model("baseline", parsed_test)
    builder.add_model("alternative", parsed_post82)
    builder.add_model("third", parsed_rbc_so, solve_kwargs={"order": 2})
    if not estimation_first:
        builder.add_estimation(estimator, model="alternative")

    manifest, _ = builder.build()
    assert manifest.members_by_kind("estimation_spec")[0].model_name == "alternative"

    loaded = loader.load_bundle(builder.write(tmp_path / "named.sdsge"))

    assert loaded.models
    assert loaded.estimation

    assert set(loaded.models) == {"baseline", "alternative", "third"}

    # Each name carries its own config, not whichever member the reader reached first.
    assert loaded.models["baseline"].compiled.var_names == compiled_test.var_names
    assert loaded.models["alternative"].compiled.var_names == compiled_post82.var_names
    assert loaded.models["third"].compiled.var_names == compiled_rbc_so.var_names

    # Recorded solve options pick the policy, so the order survives the bundle.
    assert isinstance(loaded.models["baseline"], FirstOrderSolvedModel)
    assert isinstance(loaded.models["alternative"], FirstOrderSolvedModel)
    assert isinstance(loaded.models["third"], SecondOrderSolvedModel)

    assert loaded.estimation.estimator.compiled is loaded.models["alternative"].compiled


@pytest.mark.parametrize("emitter", ["build", "write"])
def test_emission_rejects_missing_estimation_target(
    tmp_path, parsed_test, post82_estimator, emitter
):
    builder = BundleBuilder().add_model("reference", parsed_test)
    builder.add_estimation(post82_estimator(), model="missing")
    builder.manifest()  # Inspection of incomplete builders remains available.
    path = tmp_path / "existing.sdsge"
    path.write_bytes(b"existing content")
    with pytest.raises(ValueError, match="targets model"):
        builder.build() if emitter == "build" else builder.write(path)
    assert path.read_bytes() == b"existing content"


def test_loading_rejects_missing_estimation_target(parsed_test, post82_estimator):
    builder = BundleBuilder().add_model("reference", parsed_test)
    builder.add_estimation(post82_estimator(), model="missing")
    # Target validation must happen before any data reads: the archive carries no
    # member bytes, so a read would raise KeyError instead.
    archive = BundleArchive(builder.manifest(), {})
    with pytest.raises(ValueError, match="not present"):
        loader._load_estimation(archive, archive.manifest, {"reference": object()})
