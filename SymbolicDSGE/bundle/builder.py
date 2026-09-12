"""Assemble a ``.sdsge`` bundle from model/estimation/Monte-Carlo artifacts.

:class:`BundleBuilder` accumulates members and emits the archive. Text specs
(model YAML, estimation/MC JSON) ride as deflated text; bulk numeric data
(observed ``y``, MCMC posteriors, MC traces) flows through
:func:`SymbolicDSGE.bundle.parquet.columns_to_parquet`, and raw observable files,
whose cells may be strings, through :func:`SymbolicDSGE.bundle.parquet.to_parquet`.
This is the writer half of the container (#142) and the assembly point the
future ``sdsge-compile`` CLI calls.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, cast, Sequence

import numpy as np
from numpy.typing import NDArray

from ..estimation.results import MCMCResult

if TYPE_CHECKING:
    from ..estimation.estimator import Estimator
from ..estimation.spec import (
    MLEResultSpec,
    MAPResultSpec,
    MCMCResultMeta,
)
from ..core.shock_generators import Shock
from ..estimation.results import MLEResult, MAPResult

from ..monte_carlo.core import MCPipeline
from ..monte_carlo.spec import PipelineSpec, pipeline_meta
from ..monte_carlo.mc_constructs import MCPipelineResult, OpType
from ..monte_carlo.serialize import (
    json_safe,
    serialize_run_meta,
    serialize_datagen_result,
    serialize_filter_results,
    serialize_test_results,
    serialize_regression_results,
    serialize_transform_results,
    serialize_postproc_results,
)
from .container import write_bundle
from .manifest import Manifest, Member, MemberKind, SimSpec
from .parquet import (
    columns_to_parquet,
    csv_to_json,
    to_parquet,
    trace_to_csv,
)

NDF = NDArray[np.float64]

# Ref/DGP top level models
_MODEL_PATH = "model/{role}.yaml"

# MLE, MAP, MCMC estimation tab members
_ESTIMATION_SPEC = "estimation/spec.json"
_ESTIMATION_RESULT = "estimation/result.json"
#: Bulk-member paths take the extension as a field, so one template covers both
#: encodings. :func:`_encoding` resolves it once per call from ``as_parquet``.
_ESTIMATION_DATA = "estimation/observed.{ext}"
_ESTIMATION_POSTERIOR = "estimation/posterior.{ext}"

# Monte Carlo pipeline spec and per-rep custom members
_MC_PIPELINE = "montecarlo/pipeline.json"
_MC_FUNC = "montecarlo/func/{ref}.pkl"

_MC_DATA = "montecarlo/data/{ref}.{ext}"
# Monte Carlo result tab members
_MC_RESULT_META = "montecarlo/result/meta.json"

_MC_DATAGEN_STEPS = "montecarlo/result/datagen/datagen_steps.json"
_MC_DATAGEN = "montecarlo/result/datagen/{ref}_{field}.{ext}"

_MC_FILTER_STEPS = "montecarlo/result/filters/filter_steps.json"
_MC_FILTER = "montecarlo/result/filters/{ref}_{field}.{ext}"

_MC_TEST_STEPS = "montecarlo/result/tests/test_steps.json"
_MC_TEST = "montecarlo/result/tests/test_traces.{ext}"

_MC_REGRESSION_STEPS = "montecarlo/result/regressions/regression_steps.json"
_MC_REGRESSION = "montecarlo/result/regressions/regression_traces.{ext}"

_MC_TRANSFORM_STEPS = "montecarlo/result/transforms/transform_steps.json"
_MC_TRANSFORM = "montecarlo/result/transforms/{ref}_{field}.{ext}"

_MC_POSTPROC_STEPS = "montecarlo/result/postproc/postproc_steps.json"
_MC_POSTPROC = "montecarlo/result/postproc/{ref}_{field}.{ext}"

#: Fill for the rows a shorter column contributes to a shared block. Negative so
#: it can never be read as a rep index; no reader looks past ``n_retained``.
_PAD = -1


class _Encoding(NamedTuple):
    """How one call writes its bulk members: the path extension and the encoder.

    Resolved once from ``as_parquet`` and handed down, so no writer re-decides
    the format and no path needs a second constant for its other extension.
    """

    ext: str
    encode: Callable[[Mapping[str, Any]], bytes]


def _encoding(as_parquet: bool) -> _Encoding:
    """The encoding one ``as_parquet`` flag selects."""
    if as_parquet:
        return _Encoding("parquet", columns_to_parquet)
    return _Encoding("csv", lambda columns: trace_to_csv(dict(columns)))


def _fold(arr: NDArray[Any]) -> NDArray[Any]:
    """One array as the ``(n,)`` or ``(n, k)`` a column block holds.

    Padding is a row operation, so an array has to be folded before it can be
    padded against its neighbours; the writer would fold it anyway.
    """
    return arr if arr.ndim <= 1 else arr.reshape(-1, arr.shape[-1])


def _pad_columns(
    columns: Mapping[str, NDArray[Any]],
    fill: float = _PAD,
) -> dict[str, NDArray[Any]]:
    """Bring every column up to the tallest one's height, filling with ``fill``.

    A column block is rectangular, but ``n_retain`` is per step, so two steps in
    one kind can retain different numbers of replications. A float block can fill
    with NaN instead, which no reader can mistake for a value.
    """
    if not columns:
        return {}
    height = max(int(arr.shape[0]) for arr in columns.values())
    if height == 0:
        return {}
    out: dict[str, NDArray[Any]] = {}
    for name, arr in columns.items():
        rows = int(arr.shape[0])
        if rows == height:
            out[name] = arr
            continue
        pad = np.full((height - rows, *arr.shape[1:]), fill, dtype=arr.dtype)
        out[name] = np.concatenate([arr, pad])
    return out


def _library_version() -> str:
    try:
        return f"SymbolicDSGE {version('symbolicdsge')}"
    except PackageNotFoundError:  # pragma: no cover - source checkout without install
        return "SymbolicDSGE (unknown)"


class BundleBuilder:
    """Collect bundle members, then :meth:`write` (or :meth:`build`) the archive.

    Parameters
    ----------
    created_by : str | None
        Optional string to record in the bundle manifest as the creator. If not
        provided, the library version is used.
    """

    def __init__(self, *, created_by: str | None = None) -> None:
        self._created_by = created_by or _library_version()
        self._members: list[Member] = []
        self._files: dict[str, bytes] = {}
        self._simulation: dict[str, SimSpec] = {}

    # Models

    def add_model(
        self,
        role: str,
        yaml_text: str,
        *,
        compile_kwargs: Mapping[str, Any] | None = None,
        solve_kwargs: Mapping[str, Any] | None = None,
    ) -> BundleBuilder:
        """Add a model config (its source YAML) under ``role`` (reference/dgp).

        ``compile_kwargs``/``solve_kwargs`` are recorded so the loader rebuilds an
        identical :class:`SolvedModel`.
        """
        path = _MODEL_PATH.format(role=role)
        options: dict[str, Any] = {}
        if compile_kwargs:
            options["compile_kwargs"] = dict(compile_kwargs)
        if solve_kwargs:
            options["solve_kwargs"] = dict(solve_kwargs)
        self._add(
            Member(path=path, kind="model_config", role=role, options=options),
            yaml_text.encode("utf-8"),
        )
        return self

    # Raw data

    def add_raw_data(
        self,
        name: str,
        data: bytes | str,
        *,
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add a raw observable file.

        CSV input is converted to Parquet by default (``as_parquet``); pass
        ``as_parquet=False`` to store the CSV verbatim (still a valid, format-
        agnostic member).
        """
        if as_parquet:
            self._add(
                Member(path=f"data/{name}.parquet", kind="raw_data"),
                to_parquet(csv_to_json(data)),
            )
        else:
            text = data.encode("utf-8") if isinstance(data, str) else data
            self._add(Member(path=f"data/{name}.csv", kind="raw_data"), text)
        return self

    # Estimation

    def add_estimation(
        self,
        source: Estimator,
        *,
        result: MLEResult | MAPResult | MCMCResult | None = None,
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add the estimation tab from a live :class:`Estimator`.

        ``result`` accepts live any result object an :class:`Estimator` can produce.
        ``as_parquet`` controls whether relevant bulk data gets compressed as Parquet
        or stays user-readable as CSV.
        """
        spec = source.to_spec()

        # The observed matrix and the posterior traces are different shapes with
        # different column names, so each gets its own encoder rather than one
        # serializer that would have to hardcode a column name for both.
        observable_names = (
            None
            if spec.params["observables"] is None
            else list(spec.params["observables"])
        )
        encoding = _encoding(as_parquet)
        dpath = _ESTIMATION_DATA.format(ext=encoding.ext)
        ppath = _ESTIMATION_POSTERIOR.format(ext=encoding.ext)
        posterior_bytes = encoding.encode

        def observed_bytes(y: Any) -> bytes:
            if as_parquet:
                return columns_to_parquet({"y": y})
            return _observed_to_csv(y, observable_names)

        self._add(
            Member(path=_ESTIMATION_SPEC, kind="estimation_spec"),
            json.dumps(spec.params, indent=2).encode("utf-8"),
        )
        self._add(
            Member(path=dpath, kind="estimation_data", columns=observable_names),
            observed_bytes(spec.y),
        )
        if result is not None:
            # One name per arm: the three specs are unrelated types, and only the
            # MCMC one is a container whose bulk fields split off to their own member.
            kind: str
            payload: MCMCResultMeta | MLEResultSpec | MAPResultSpec
            match result:
                case MCMCResult():
                    mcmc_spec = result.to_spec()
                    kind = "mcmc"
                    payload = mcmc_spec.meta

                    self._add(
                        Member(path=ppath, kind="estimation_trace"),
                        posterior_bytes(
                            {
                                "samples": mcmc_spec.samples,
                                "logpost": mcmc_spec.logpost_trace,
                                "logjac": mcmc_spec.logjac_trace,
                            }
                        ),
                    )
                case MLEResult():
                    kind = "mle"
                    payload = result.to_spec()
                case MAPResult():
                    kind = "map"
                    payload = result.to_spec()
                case _:
                    raise ValueError(f"Unknown result type {type(result).__name__}.")

            result_data = json.dumps({"type": kind, "data": payload}, indent=2)
            self._add(
                Member(path=_ESTIMATION_RESULT, kind="estimation_result"),
                result_data.encode("utf-8"),
            )

        return self

    # Monte Carlo

    def add_mc(
        self,
        pipeline: MCPipeline,
        *,
        result: MCPipelineResult | None = None,
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add the MC tab from a live :class:`MCPipeline`.

        ``add_mc(pipeline)``: A live pipeline is compiled to
        its graph spec via :meth:`MCPipeline.to_spec`, and its bulk side-channels
        are shipped as their own members: ``raw_model_data`` datagen arrays as
        Parquet, and ``custom`` ops as cloudpickle blobs (each callable is
        enforced/wrapped as a :class:`NumpyCustomFunc` so its source travels for
        receiver audit).

        Optionally records a run ``result``. Its run-level config and each step
        kind's metas ride their own JSON members; the bulk traces ride separate
        members, as Parquet or, with ``as_parquet=False``, as CSV. The loader
        reads either.
        """
        encoding = _encoding(as_parquet)
        ps = pipeline.to_spec()
        self._add_mc_manifest(ps)
        self._add_mc_pipeline_traces(ps, encoding)
        self._add_mc_funcs(ps)

        if result is not None:
            self._add_mc_results(result, pipeline, encoding)

        return self

    def _add_step_metas(
        self,
        path: str,
        kind: MemberKind,
        steps: Mapping[str, tuple[Any, Mapping[str, NDArray[Any]]]],
    ) -> None:
        """Collect one step kind's metas into a single member, keyed by step name.

        The traces half is dropped here; it rides its own members. A kind with no
        steps writes nothing, so the loader reads an absent member as empty.
        """
        if not steps:
            return
        payload = {name: json_safe(meta) for name, (meta, _) in steps.items()}
        self._add(
            Member(path=path, kind=kind),
            json.dumps(payload, indent=2).encode("utf-8"),
        )

    def _add_trace_block(
        self,
        path: str,
        kind: MemberKind,
        steps: Mapping[str, tuple[Any, Mapping[str, NDArray[Any]]]],
        encoding: _Encoding,
    ) -> None:
        """Pack one step kind's traces into a single column block.

        ``path`` is already resolved, since the block is one member and its
        caller knows what to call it.

        Columns are qualified ``{step}.{field}``, which the 2-D expansion extends
        to ``{step}.{field}.{j}``. Steps that retained different numbers of
        replications are padded to the tallest; the rows past a step's own
        ``n_retained`` are never read, since its meta carries the count.
        """
        columns = _pad_columns(
            {
                f"{name}.{field}": arr
                for name, (_, traces) in steps.items()
                for field, arr in traces.items()
            }
        )
        if not columns:
            return
        self._add(Member(path=path, kind=kind), encoding.encode(columns))

    def _add_trace_arrays(
        self,
        path: str,
        kind: MemberKind,
        steps: Mapping[str, tuple[Any, Mapping[str, NDArray[Any]]]],
        encoding: _Encoding,
    ) -> None:
        """Ship one step kind's traces as a member per array.

        ``path`` is a template, since this writes one member per array and names
        each from the ``ref`` and ``field`` it is looking at.

        These are arbitrary-shape payloads that share no height with each other,
        so none of them pack. An array above 2-D is flattened to ``(-1, last)``
        and restored from the ``shape`` its step's meta records.
        """
        for name, (_, traces) in steps.items():
            for field, arr in traces.items():
                if arr.size == 0:
                    continue
                flat = arr if arr.ndim <= 1 else arr.reshape(-1, arr.shape[-1])
                columns = {f"{name}.{field}": flat}
                self._add(
                    Member(
                        path=path.format(ref=name, field=field, ext=encoding.ext),
                        kind=kind,
                        options={"name": name, "field": field},
                    ),
                    encoding.encode(columns),
                )

    def _add_mc_manifest(self, spec: PipelineSpec) -> None:

        self._add(
            Member(path=_MC_PIPELINE, kind="mc_pipeline"),
            json.dumps(pipeline_meta(spec), indent=2).encode("utf-8"),
        )

    def _add_step_arrays(
        self,
        path: str,
        kind: MemberKind,
        name: str,
        arrays: Mapping[str, NDArray[Any]],
        encoding: _Encoding,
    ) -> None:
        """Ship one step's bulk array kwargs as a single member.

        The member is already per step, so its columns are the kwarg names as
        authored. The arrays share neither rank nor height, so each folds to
        ``(-1, last)`` and the block pads to the tallest; ``options`` carries the
        original shapes, which is what a reader trims and reshapes against.
        """
        if not arrays:
            return
        shapes = {key: [int(size) for size in arr.shape] for key, arr in arrays.items()}
        columns = _pad_columns(
            {
                key: _fold(np.asarray(arr, dtype=np.float64))
                for key, arr in arrays.items()
            },
            fill=np.nan,
        )
        self._add(
            Member(path=path, kind=kind, options={"name": name, "shapes": shapes}),
            encoding.encode(columns),
        )

    def _add_mc_pipeline_traces(self, spec: PipelineSpec, encoding: _Encoding) -> None:
        """Ship every step's lifted array kwargs, one member per step."""
        for step in (*spec.replication_steps, *spec.postproc_steps):
            name = step.meta["name"]
            self._add_step_arrays(
                _MC_DATA.format(ref=name, ext=encoding.ext),
                "mc_data",
                name,
                step.arrays,
                encoding,
            )

    def _add_mc_funcs(self, spec: PipelineSpec) -> None:
        """Wrap a custom step's callable in the phase wrapper and cloudpickle it.

        Wrapping enforces the author-side contract (top-level def, safe namespace)
        and snapshots the source + captured globals, so the receiver can audit the
        op at load. Post-loop (POSTPROC) ops get the looser pandas namespace; every
        other phase gets numpy/numba.
        """
        import cloudpickle
        from ..monte_carlo.postproc import is_builtin_postproc
        from ..monte_carlo.custom_op import (
            CustomFunc,
            NumbaCustomFunc,
            PandasCustomFunc,
        )

        for step in (*spec.replication_steps, *spec.postproc_steps):
            if step.func is None:
                continue

            meta = step.meta
            if is_builtin_postproc(step.func):
                self._add(
                    Member(
                        path=_MC_FUNC.format(ref=meta["name"]),
                        kind="mc_func",
                        options={"name": meta["name"]},
                    ),
                    cloudpickle.dumps(step.func),
                )
                continue

            fn: CustomFunc
            if OpType(meta["op_type"]) == OpType.POSTPROC:
                fn = PandasCustomFunc(step.func)
            else:
                fn = NumbaCustomFunc(step.func)

            self._add(
                Member(
                    path=_MC_FUNC.format(ref=meta["name"]),
                    kind="mc_func",
                    options={"name": meta["name"]},
                ),
                cloudpickle.dumps(fn),
            )

    def _add_mc_results(
        self, result: MCPipelineResult, pipeline: MCPipeline, encoding: _Encoding
    ) -> None:
        self._add(
            Member(path=_MC_RESULT_META, kind="mc_result_meta"),
            json.dumps(json_safe(serialize_run_meta(result)), indent=2).encode("utf-8"),
        )
        datagen = serialize_datagen_result(
            result.datagen_outputs, pipeline.replication_steps[0].name
        )
        filters = serialize_filter_results(result.filter_outputs)
        tests = serialize_test_results(result.test_summaries)
        regressions = serialize_regression_results(result.regression_summaries)
        transforms = serialize_transform_results(result.transform_outputs)
        postprocs = serialize_postproc_results(result.postproc)

        self._add_step_metas(_MC_DATAGEN_STEPS, "mc_datagen_steps", datagen)
        self._add_step_metas(_MC_FILTER_STEPS, "mc_filter_steps", filters)
        self._add_step_metas(_MC_TEST_STEPS, "mc_test_steps", tests)
        self._add_step_metas(_MC_REGRESSION_STEPS, "mc_regression_steps", regressions)
        self._add_step_metas(_MC_TRANSFORM_STEPS, "mc_transform_steps", transforms)
        self._add_step_metas(_MC_POSTPROC_STEPS, "mc_postproc_steps", postprocs)

        self._add_trace_arrays(_MC_DATAGEN, "mc_datagen_trace", datagen, encoding)
        self._add_trace_arrays(_MC_FILTER, "mc_filter_trace", filters, encoding)
        self._add_trace_block(
            _MC_TEST.format(ext=encoding.ext), "mc_test_traces", tests, encoding
        )
        self._add_trace_block(
            _MC_REGRESSION.format(ext=encoding.ext),
            "mc_regression_traces",
            regressions,
            encoding,
        )
        self._add_trace_arrays(
            _MC_TRANSFORM, "mc_transform_trace", transforms, encoding
        )
        self._add_trace_arrays(_MC_POSTPROC, "mc_postproc_raw", postprocs, encoding)

    # Simulation prefill

    def set_simulation(
        self,
        role: str,
        *,
        T: int,
        shocks: Mapping[str, Shock | NDF] | None = None,
        shock_scale: float = 1.0,
        x0: Mapping[str, float] | Sequence[float] | NDF | None = None,
        observables: bool = False,
    ) -> BundleBuilder:
        """Attach a simulation prefill under ``role``, taking ``SolvedModel.sim``'s
        keywords and lowering them to the stored form.

        Each shock is a live :class:`Shock` or a raw path array, the two shapes
        ``sim`` draws from; the parameters are read off the object rather than
        hand-written. A callable cannot be stored, since only its result would
        travel and the receiver could not redraw it.
        """
        self._simulation[role] = SimSpec(
            T=int(T),
            x0=_prefill_x0(x0),
            observables=bool(observables),
            shock_scale=float(shock_scale),
            shocks=_prefill_shocks(shocks),
        )
        return self

    # Low-level passthrough

    def add_member(self, member: Member, data: bytes) -> BundleBuilder:
        """Append a pre-encoded member at its declared path.

        Public seam for callers that already hold the final member bytes, for
        example the ``sdsge-compile`` CLI copying a Parquet ``data/`` file through
        or staging a pre-split MC result + traces pair. The higher-level ``add_*`` methods would otherwise re-encode.
        """
        self._add(member, data)
        return self

    # Emit

    def manifest(self) -> Manifest:
        """Return the manifest for the current bundle state, with checksums of all members.

        Returns
        -------
        Manifest
            Object containing bundle metadata and checksums of all members.

        """
        return Manifest(
            created_by=self._created_by,
            created_at=datetime.now(timezone.utc).isoformat(),
            members=list(self._members),
            simulation=self._simulation or None,
            checksums={
                path: hashlib.sha256(data).hexdigest()
                for path, data in self._files.items()
            },
        )

    def build(self) -> tuple[Manifest, dict[str, bytes]]:
        """Build the bundle, to an in-memory mapping of member paths to their bytes, along with the manifest.

        Returns
        -------
        tuple[Manifest, dict[str, bytes]]
            Mapping of member paths to their bytes, along with the manifest containing metadata and checksums.

        """
        return self.manifest(), dict(self._files)

    def write(self, path: str | Path) -> Path:
        """Write a bundle to disk at ``path``, returning the path for convenience.

        Parameters
        ----------
        path : str | Path
            The file path where the bundle will be written.

        Returns
        -------
        Path
            The path where the bundle was written.

        """
        write_bundle(path, self.manifest(), self._files)
        return Path(path)

    def _add(self, member: Member, data: bytes) -> None:
        if member.path in self._files:
            raise ValueError(f"Duplicate bundle member path {member.path!r}.")
        self._members.append(member)
        self._files[member.path] = data


def _prefill_x0(
    x0: Mapping[str, float] | Sequence[float] | NDF | None,
) -> Mapping[str, float] | list[float] | None:
    """Lower ``sim``'s ``x0`` to its stored shape, name-keyed or positional."""
    if x0 is None:
        return None
    if isinstance(x0, Mapping):
        return {str(name): float(value) for name, value in x0.items()}
    if isinstance(x0, (Sequence, np.ndarray)) and not isinstance(x0, (str, bytes)):
        return [float(value) for value in x0]
    raise TypeError(
        f"x0 must be a mapping of variable names to values, a sequence in "
        f"declaration order, or an ndarray; got {type(x0).__name__}."
    )


def _prefill_shocks(
    shocks: Mapping[str, Shock | NDF] | None,
) -> dict[str, Any] | None:
    """Lower each shock to its stored shape: parameters, or a raw path.

    A :class:`Shock` travels as its parameters so the receiver redraws it under
    the author's seed. An array travels as itself. A callable is rejected: only
    the path it returned would survive, which is not the same run.
    """
    if shocks is None:
        return None
    lowered: dict[str, Any] = {}
    for key, shock in shocks.items():
        if isinstance(shock, Shock):
            lowered[key] = shock.to_dict()
        elif isinstance(shock, np.ndarray):
            lowered[key] = shock.tolist()
        else:
            raise TypeError(
                f"Shock {key!r} must be a Shock or a raw path array; got "
                f"{type(shock).__name__}. A callable is rejected here too: only "
                f"the path it returned would travel, and the receiver could not "
                f"redraw it."
            )
    return lowered


def _observed_to_csv(
    y: Sequence[Sequence[float]], names: Sequence[str] | None
) -> bytes:
    """Render a 2-D observed matrix as CSV with user-friendly headers.

    Uses ``names`` as the header row when provided (paired with
    ``Member.columns`` so the loader can stack semantic-header CSVs back into
    the matrix). Falls back to mechanical ``y.{j}`` headers. Round-trips
    through :func:`SymbolicDSGE.bundle.parquet.collapse_columns` the same way
    Parquet observed data does.
    """
    n = len(y)
    p = len(y[0]) if n > 0 else 0
    if names is not None and len(names) != p:
        raise ValueError(
            f"observable_names length {len(names)} does not match observed "
            f"column count {p}."
        )
    headers = list(names) if names is not None else [f"y.{j}" for j in range(p)]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(headers)
    for row in y:
        writer.writerow([_float_cell(v) for v in row])
    return out.getvalue().encode("utf-8")


def _float_cell(value: float) -> str:
    number = float(value)
    return "" if not math.isfinite(number) else repr(number)
