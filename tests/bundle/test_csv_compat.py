from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.bundle.container import BundleArchive, write_bundle
from SymbolicDSGE.bundle.loader import build_from
from SymbolicDSGE.bundle.manifest import Manifest, Member
from SymbolicDSGE.bundle.parquet import (
    collapse_columns,
    csv_to_columns,
    trace_to_csv,
)
from SymbolicDSGE.estimation.spec import (
    EstimationParameterSpec,
    EstimationSpec,
    MCMCResultMeta,
)
from SymbolicDSGE.monte_carlo.spec import NodeSpec, PipelineSpec

_MODEL_YAML = Path("MODELS/test.yaml").read_text(encoding="utf-8")


# -- pure helper round-trips ------------------------------------------------


def test_csv_to_columns_infers_types_and_nulls() -> None:
    cols = csv_to_columns("x,n,label\n0.5,1,a\n,2,b\n1.5,3,\n")
    assert cols == {
        "x": [0.5, None, 1.5],
        "n": [1, 2, 3],
        "label": ["a", "b", None],
    }
    assert isinstance(cols["x"][0], float)
    assert isinstance(cols["n"][0], int)


def test_trace_to_csv_expands_2d_and_blanks_nonfinite() -> None:
    csv_bytes = trace_to_csv(
        {
            "theta": np.array([[1.0, 2.0], [3.0, np.nan]]),
            "logpost": np.array([-1.5, -2.5]),
        }
    )
    text = csv_bytes.decode("utf-8")
    assert text.splitlines() == [
        "theta.0,theta.1,logpost",
        "1.0,2.0,-1.5",
        "3.0,,-2.5",
    ]


def test_trace_to_csv_round_trips_through_csv_to_columns() -> None:
    rng = np.random.default_rng(0)
    columns = {
        "samples": rng.standard_normal((30, 3)),
        "logpost": rng.standard_normal(30),
    }
    decoded = collapse_columns(csv_to_columns(trace_to_csv(columns)))
    np.testing.assert_allclose(decoded["samples"], columns["samples"])
    np.testing.assert_allclose(decoded["logpost"], columns["logpost"])


def test_trace_to_csv_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        trace_to_csv({"a": np.zeros(3), "b": np.zeros(4)})


# -- builder writes CSV members ---------------------------------------------


def test_builder_writes_observed_with_semantic_headers() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    builder = BundleBuilder().add_estimation(
        _estimation_spec(),
        observed=matrix,
        observable_names=["gdp", "infl"],
        as_parquet=False,
    )
    manifest, files = builder.build()

    assert "estimation/observed.csv" in files
    member = next(m for m in manifest.members if m.kind == "estimation_data")
    assert member.format == "csv"
    assert member.columns == ["gdp", "infl"]

    text = files["estimation/observed.csv"].decode("utf-8")
    assert text.splitlines()[0] == "gdp,infl"


def test_builder_writes_posterior_and_mc_traces_as_csv() -> None:
    posterior = {
        "samples": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "logpost": np.array([-1.0, -2.0]),
    }
    builder = BundleBuilder().add_estimation(
        _estimation_spec(), posterior=posterior, as_parquet=False
    )
    _, files = builder.build()
    assert "estimation/posterior.csv" in files
    header = files["estimation/posterior.csv"].decode("utf-8").splitlines()[0]
    assert header == "samples.0,samples.1,logpost"


def test_builder_observable_names_length_must_match_matrix() -> None:
    builder = BundleBuilder()
    with pytest.raises(ValueError):
        builder.add_estimation(
            _estimation_spec(),
            observed=np.zeros((3, 2)),
            observable_names=["only_one"],
            as_parquet=False,
        )


# -- format-agnostic loader -------------------------------------------------


def test_csv_mode_round_trips_through_builder_and_loader(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    observed = rng.standard_normal((10, 2))
    posterior = {
        "samples": rng.standard_normal((20, 2)),
        "logpost": rng.standard_normal(20),
    }
    result_meta = MCMCResultMeta(
        param_names=["beta", "sigma"],
        accept_rate=0.25,
        n_draws=20,
        burn_in=5,
        thin=1,
    )

    target = (
        BundleBuilder(created_by="csv-test")
        .add_model("reference", _MODEL_YAML, compile_kwargs={"n_state": 3, "n_exog": 2})
        .add_estimation(
            _estimation_spec(),
            result=result_meta,
            observed=observed,
            observable_names=["Infl", "Rate"],
            posterior=posterior,
            as_parquet=False,
        )
        .add_mc(
            PipelineSpec(
                nodes=[
                    NodeSpec(
                        id="n1", step_type="simulation", name="sim", params={"T": 50}
                    )
                ]
            )
        )
        .write(tmp_path / "csv.sdsge")
    )

    loaded = build_from(target)
    assert loaded.estimation is not None
    assert loaded.estimation.observed is not None
    np.testing.assert_allclose(loaded.estimation.observed, observed)
    assert loaded.estimation.posterior is not None
    np.testing.assert_allclose(
        loaded.estimation.posterior["samples"], posterior["samples"]
    )
    np.testing.assert_allclose(
        loaded.estimation.posterior["logpost"], posterior["logpost"]
    )


def test_loader_reads_hand_built_csv_only_bundle(tmp_path: Path) -> None:
    """A hand-zipped bundle with CSV members (no Parquet) is a valid archive."""
    observed_csv = b"gdp,infl\n0.1,0.2\n0.3,0.4\n0.5,0.6\n"
    posterior_csv = trace_to_csv(
        {
            "samples": np.array([[0.9, 0.1], [0.95, 0.12], [0.98, 0.11]]),
            "logpost": np.array([-1.0, -1.1, -1.2]),
        }
    )

    spec = _estimation_spec()
    manifest = Manifest(
        created_by="hand-zipped",
        members=[
            Member(path="estimation/spec.json", kind="estimation_spec"),
            Member(
                path="estimation/observed.csv",
                kind="estimation_data",
                columns=["gdp", "infl"],
            ),
            Member(path="estimation/posterior.csv", kind="estimation_trace"),
        ],
    )
    files = {
        "estimation/spec.json": spec.to_json().encode("utf-8"),
        "estimation/observed.csv": observed_csv,
        "estimation/posterior.csv": posterior_csv,
    }
    archive_path = tmp_path / "hand.sdsge"
    write_bundle(archive_path, manifest, files)

    loaded = build_from(archive_path)
    assert loaded.estimation is not None
    assert loaded.estimation.observed is not None
    np.testing.assert_allclose(
        loaded.estimation.observed,
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    )
    assert loaded.estimation.posterior is not None
    np.testing.assert_allclose(
        loaded.estimation.posterior["samples"],
        np.array([[0.9, 0.1], [0.95, 0.12], [0.98, 0.11]]),
    )


def test_observed_csv_with_nan_round_trips_to_nan(tmp_path: Path) -> None:
    matrix = np.array([[1.0, np.nan], [2.0, 3.0]])
    target = (
        BundleBuilder()
        .add_estimation(
            _estimation_spec(),
            observed=matrix,
            observable_names=["a", "b"],
            as_parquet=False,
        )
        .write(tmp_path / "nan.sdsge")
    )
    archive = BundleArchive.open(target)
    text = archive.read_text("estimation/observed.csv")
    # Non-finite -> empty cell in row 0, column 1.
    assert text.splitlines()[1] == "1.0,"

    loaded = build_from(target)
    assert loaded.estimation is not None
    assert loaded.estimation.observed is not None
    assert np.isnan(loaded.estimation.observed[0, 1])
    np.testing.assert_allclose(
        loaded.estimation.observed[~np.isnan(loaded.estimation.observed)],
        np.array([1.0, 2.0, 3.0]),
    )


# -- helpers ----------------------------------------------------------------


def _estimation_spec() -> EstimationSpec:
    return EstimationSpec(
        method="mcmc",
        parameters=[
            EstimationParameterSpec(name="beta", initial=0.99, estimate=True),
            EstimationParameterSpec(name="sigma", initial=1.0, estimate=True),
        ],
        observables=["Infl", "Rate"],
    )
