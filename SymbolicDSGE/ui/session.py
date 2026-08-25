from __future__ import annotations

import ast
import base64
import inspect
import io
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, cast
from uuid import uuid4

# Set non-interactive backend before any user code can import pyplot.
try:
    import matplotlib as _mpl

    _mpl.use("Agg")
except Exception:
    pass

import numpy as np
from numpy.typing import NDArray
from sympy import Symbol

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.compiled_model import CompiledModel
from SymbolicDSGE.core.config import ModelConfig
from SymbolicDSGE.core.shock_generators import Shock
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.config import KalmanConfig

from .schemas import (
    ArrayEnvelope,
    EstimationRunRequest,
    FunctionKind,
    Role,
    ShockGenerationRequest,
    ShockParamUpdate,
    WorkspaceTab,
)
from SymbolicDSGE.estimation.spec import EstimatorParams, EstimatorSpec

from .estimation import (
    build_estimation_inputs,
    estimator_spec_wire,
    serialize_estimation_result,
)
from .serializers import (
    decode_array,
    empty_model_summary,
    encode_named_arrays,
    summarize_parsed_model,
    summarize_solved_model,
)


@dataclass
class FunctionRecord:
    name: str
    kind: str
    source: str
    func: Any


@dataclass
class ModelSlot:
    role: Role
    source: str | None = None
    raw_yaml: str | None = None
    model_config: ModelConfig | None = None
    kalman_config: KalmanConfig | None = None
    solver: DSGESolver | None = None
    compiled: CompiledModel | None = None
    solved: SolvedModel | None = None


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    kind: str
    role: Role
    payload: Mapping[str, Any]


@dataclass
class TabState:
    """One tab's session state, split by who writes it.

    ``spec`` and ``result`` are the pair a ``.sdsge`` bundle stores, written
    here by the server when a run completes. ``view`` is the tab's form state,
    written only by the client. The split is what keeps GUI state out of a
    bundle: a bundle write takes ``spec``/``result`` as they stand, and the
    client cannot reach them because ``view`` is the only slot it PUTs.
    """

    spec: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    view: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        """Wire shape for this tab, omitting the slots nothing has filled."""
        out: dict[str, Any] = {}
        if self.spec is not None:
            out["spec"] = self.spec
        if self.result is not None:
            out["result"] = self.result
        if self.view is not None:
            out["view"] = self.view
        return out


@dataclass
class Workspace:
    """A session's hydration payload, in the shape the tabs repaint from.

    Populated three ways, all landing here: the bundle
    :func:`SymbolicDSGE.ui.serve.serve_from` was launched with, a run the
    server just performed, and the client's own debounced view updates. The
    frontend reads it on every load, so a refresh restores from the process
    that never went away rather than from anything stored on the client.
    """

    estimation: TabState = field(default_factory=TabState)
    mc: TabState = field(default_factory=TabState)
    #: Per-role ``SimSpec`` dicts. The Outputs tab's controls map onto the
    #: spec's own fields, so it carries no separate view.
    simulation: dict[str, Any] | None = None


class UISession:
    def __init__(
        self,
        *,
        reference: SolvedModel | None = None,
        dgp: SolvedModel | None = None,
        workspace: Workspace | None = None,
        source: str | None = None,
    ) -> None:
        self.slots: dict[Role, ModelSlot] = {
            "reference": ModelSlot(role="reference"),
            "dgp": ModelSlot(role="dgp"),
        }
        self.runs: dict[str, RunRecord] = {}
        self.functions: dict[Role, dict[str, FunctionRecord]] = {
            "reference": {},
            "dgp": {},
        }
        self.workspace: Workspace = workspace if workspace is not None else Workspace()
        # Both roles preload from the one source, so they share its label.
        if reference is not None:
            self.set_solved_model("reference", reference, source=source)
        if dgp is not None:
            self.set_solved_model("dgp", dgp, source=source)

    def summary(self) -> dict[str, Any]:
        roles: tuple[Role, Role] = ("reference", "dgp")
        return {
            "models": {role: self.model_summary(role) for role in roles},
            "runs": [
                {
                    "run_id": run.run_id,
                    "kind": run.kind,
                    "role": run.role,
                }
                for run in self.runs.values()
            ],
            "workspace": self._workspace_payload(),
        }

    def _workspace_payload(self) -> dict[str, Any]:
        """Wire shape for the workspace (omits tabs and slots nothing filled)."""
        out: dict[str, Any] = {}
        for name, tab in (
            ("estimation", self.workspace.estimation),
            ("mc", self.workspace.mc),
        ):
            if payload := tab.payload():
                out[name] = payload
        if self.workspace.simulation is not None:
            out["simulation"] = self.workspace.simulation
        return out

    def set_workspace_view(
        self, tab: WorkspaceTab, view: dict[str, Any] | None
    ) -> None:
        """Replace a tab's view with what the client last had on screen.

        The view is held verbatim: it is the GUI's own state, so a new control
        appears here without the server learning what it means. Writing it
        cannot disturb ``spec``/``result``, which only a run fills.
        """
        getattr(self.workspace, tab).view = view

    def set_solved_model(
        self, role: Role, model: SolvedModel, *, source: str | None = None
    ) -> dict[str, Any]:
        """Install an already-solved model into ``role``'s slot.

        ``source`` labels where the model came from, e.g. the bundle path
        ``sdsge-ui`` was pointed at; it is what distinguishes one preloaded
        model from another in the GUI, so an in-process model with nothing to
        cite leaves it unset rather than naming a placeholder. The YAML rides
        along on the config whenever the model was parsed rather than built,
        which is what lets the Builder tab open on the model it is serving.
        """
        slot = self._slot(role)
        slot.source = source
        slot.raw_yaml = model.config.source_yaml
        slot.model_config = model.config
        slot.kalman_config = model.kalman_config
        slot.solver = DSGESolver(model.config, cast(Any, model.kalman_config))
        slot.compiled = model.compiled
        slot.solved = model
        return self.model_summary(role)

    def load_yaml(
        self,
        *,
        role: Role,
        path: str | None = None,
        content: str | None = None,
    ) -> dict[str, Any]:
        if (path is None) == (content is None):
            raise ValueError("Provide exactly one of 'path' or 'content'.")

        source: str
        raw_yaml: str
        if path is not None:
            config_path = Path(path)
            parser = ModelParser(config_path)
            source = str(config_path)
            raw_yaml = config_path.read_text(encoding="utf-8")
        else:
            assert content is not None
            parser = self._parse_yaml_content(content)
            source = "<content>"
            raw_yaml = content

        model, kalman = parser.get_all()
        slot = self._slot(role)
        slot.source = source
        slot.raw_yaml = raw_yaml
        slot.model_config = model
        slot.kalman_config = kalman
        slot.solver = DSGESolver(model, cast(Any, kalman))
        slot.compiled = None
        slot.solved = None
        return self.model_summary(role)

    def solve_model(
        self,
        *,
        role: Role,
        compile_kwargs: Mapping[str, Any] | None = None,
        solve_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        slot = self._slot(role)
        if slot.solver is None:
            raise ValueError(f"No model is loaded for role '{role}'.")
        compiled = slot.solver.compile(**dict(compile_kwargs or {}))
        solved = slot.solver.solve(compiled, **dict(solve_kwargs or {}))
        slot.compiled = compiled
        slot.solved = solved
        return self.model_summary(role)

    def model_summary(self, role: Role) -> dict[str, Any]:
        slot = self._slot(role)
        if slot.solved is not None:
            summary = summarize_solved_model(
                role=role,
                model=slot.solved,
                source=slot.source,
            )
        elif slot.model_config is not None:
            summary = summarize_parsed_model(
                role=role,
                model=slot.model_config,
                kalman=slot.kalman_config,
                source=slot.source,
            )
        else:
            return empty_model_summary(role)
        if slot.raw_yaml is not None:
            summary["raw_yaml"] = slot.raw_yaml
        return summary

    def run_simulation(
        self,
        *,
        role: Role,
        T: int,
        observables: bool,
        shock_scale: float,
        shocks: Mapping[str, ArrayEnvelope] | None = None,
        shock_generation: ShockGenerationRequest | None = None,
        shock_params: ShockParamUpdate | None = None,
    ) -> dict[str, Any]:
        slot = self._slot(role)
        if slot.solved is None:
            raise ValueError(f"Role '{role}' does not have a solved model.")
        if shock_params is not None:
            self._apply_shock_params(slot, shock_params)
        shock_arrays = self._decode_shocks(shocks)
        generated_shocks = self._generate_shocks(
            slot=slot,
            T=T,
            generation=shock_generation,
            raw_shocks=shock_arrays,
        )
        sim = slot.solved.sim(
            T=T,
            shocks=generated_shocks,
            shock_scale=shock_scale,
            observables=observables,
        )
        run_id = str(uuid4())
        sim_dict = sim.states
        sim_dict["_X"] = sim.X
        if sim.y is not None:
            sim_dict.update(sim.observables)

        all_series = encode_named_arrays(sim_dict)
        extra = self._apply_array_functions(role, sim_dict)

        if extra:
            all_series = all_series + encode_named_arrays(extra)
        figures = self._apply_figure_functions(role, sim_dict)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "kind": "sim",
            "role": role,
            "T": T,
            "observables": observables,
            "series": all_series,
            "figures": figures,
        }
        self.runs[run_id] = RunRecord(
            run_id=run_id,
            kind="sim",
            role=role,
            payload=payload,
        )
        return payload

    def run_estimation(self, request: EstimationRunRequest) -> dict[str, Any]:
        slot = self._slot(request.role)
        if slot.solver is None:
            raise ValueError(f"No model is loaded for role '{request.role}'.")
        if slot.compiled is None:
            slot.compiled = slot.solver.compile(**dict(request.compile_kwargs))

        y = np.asarray(request.y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(
                "Observed estimation data must be a two-dimensional array."
            )
        observables = request.observables
        expected = (
            len(observables)
            if observables is not None
            else getattr(slot.compiled, "n_obs", 0)
        )
        if expected and y.shape[1] != expected:
            raise ValueError(
                f"Observed estimation data has {y.shape[1]} columns; expected {expected}."
            )

        names, theta0, priors, bounds = build_estimation_inputs(
            request.parameters,
            routine=request.routine,
        )
        # Built before the run, not after: the spec describes the estimator
        # about to be constructed, so a prior that cannot be projected says so
        # here rather than discarding a result that already cost the compute.
        spec_wire = estimator_spec_wire(
            EstimatorSpec(
                y=y.tolist(),
                params=EstimatorParams(
                    observables=observables,
                    filter_mode="linear",
                    P0=None,
                    R=None,
                    estimated_params=names,
                    priors=(
                        {name: prior.to_spec() for name, prior in priors.items()}
                        if priors is not None
                        else None
                    ),
                    ss_seed=request.ss_seed,
                    x0=None,
                    jitter=0.0,
                    symmetrize=True,
                    joseph_cov=True,
                ),
            )
        )
        kwargs = dict(request.method_kwargs)
        reserved = {
            "compiled",
            "estimated_params",
            "observables",
            "posterior_point",
            "priors",
            "routine",
            "ss_seed",
            "theta0",
            "y",
        }
        overlap = sorted(reserved.intersection(kwargs))
        if overlap:
            raise ValueError(
                f"Estimation method kwargs cannot override reserved arguments: {overlap}."
            )
        if bounds is not None and request.routine in {"mle", "map"}:
            kwargs["bounds"] = bounds

        common: dict[str, Any] = {
            "compiled": slot.compiled,
            "y": y,
            "routine": request.routine,
            "theta0": theta0,
            "observables": observables,
            "estimated_params": names,
            "priors": priors,
            "ss_seed": request.ss_seed,
            **kwargs,
        }
        solved = False
        if request.estimate_and_solve:
            result, model = slot.solver.estimate_and_solve(
                posterior_point=request.posterior_point,
                **common,
            )
            slot.solved = model
            solved = True
        else:
            result = slot.solver.estimate(**common)

        run_id = str(uuid4())
        result_wire = serialize_estimation_result(result)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "kind": "estimation",
            "role": request.role,
            "method": request.routine,
            "solved": solved,
            "result": result_wire,
        }
        self.record_run(
            run_id=run_id,
            kind="estimation",
            role=request.role,
            payload=payload,
        )
        # The bundle-bound slots, filled from the run that just produced them.
        # The client's view is untouched: it already shows this.
        self.workspace.estimation.spec = spec_wire
        self.workspace.estimation.result = result_wire
        return payload

    def submit_function(
        self,
        *,
        role: Role,
        code: str,
        kind: FunctionKind = "array",
    ) -> dict[str, Any]:
        tree = ast.parse(code)
        func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not func_defs:
            raise ValueError("No function definition found in submitted code.")
        if len(func_defs) > 1:
            raise ValueError("Submit one function at a time.")
        name = func_defs[0].name
        namespace: dict[str, Any] = {"np": np, "numpy": np}
        exec(compile(tree, "<string>", "exec"), namespace)  # noqa: S102
        func = namespace[name]
        self.functions[role][name] = FunctionRecord(
            name=name, kind=kind, source=code, func=func
        )
        return {"name": name, "kind": kind, "source": code}

    def remove_function(self, *, role: Role, name: str) -> None:
        if name not in self.functions[role]:
            raise KeyError(name)
        del self.functions[role][name]

    def list_functions(self, *, role: Role) -> list[dict[str, Any]]:
        return [
            {"name": r.name, "kind": r.kind, "source": r.source}
            for r in self.functions[role].values()
        ]

    def get_run(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.runs:
            raise KeyError(run_id)
        return dict(self.runs[run_id].payload)

    def solved_model(self, role: Role) -> SolvedModel | None:
        return self._slot(role).solved

    def record_run(
        self,
        *,
        run_id: str,
        kind: str,
        role: Role,
        payload: Mapping[str, Any],
    ) -> None:
        self.runs[run_id] = RunRecord(
            run_id=run_id,
            kind=kind,
            role=role,
            payload=payload,
        )

    def _apply_figure_functions(
        self,
        role: Role,
        sim_dict: dict[str, NDArray[np.float64]],
    ) -> list[dict[str, str]]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return [
                {
                    "name": "__error__",
                    "error": "matplotlib is not installed — run: pip install matplotlib",
                }
            ]
        except Exception as exc:
            return [{"name": "__error__", "error": f"matplotlib unavailable: {exc}"}]

        results: list[dict[str, str]] = []
        for name, record in self.functions[role].items():
            if record.kind != "figure":
                continue
            try:
                sig = inspect.signature(record.func)
                kwargs = {p: sim_dict[p] for p in sig.parameters if p in sim_dict}
                fig_result = record.func(**kwargs)
                fig = plt.gcf() if fig_result is None else fig_result
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                buf.seek(0)
                image_b64 = base64.b64encode(buf.read()).decode("ascii")
                results.append({"name": name, "image_b64": image_b64})
                plt.close(fig)
            except Exception as exc:
                results.append({"name": name, "error": str(exc)})
        return results

    def _apply_array_functions(
        self,
        role: Role,
        sim_dict: dict[str, NDArray[np.float64]],
    ) -> dict[str, NDArray[np.float64]]:
        extra: dict[str, NDArray[np.float64]] = {}
        for name, record in self.functions[role].items():
            if record.kind != "array":
                continue
            try:
                sig = inspect.signature(record.func)
                kwargs = {p: sim_dict[p] for p in sig.parameters if p in sim_dict}
                result = np.asarray(record.func(**kwargs), dtype=np.float64)
                extra[name] = result
            except Exception:
                pass
        return extra

    def _slot(self, role: Role) -> ModelSlot:
        if role not in self.slots:
            raise KeyError(role)
        return self.slots[role]

    @staticmethod
    def _parse_yaml_content(content: str) -> ModelParser:
        with NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(content)
            tmp_path = Path(handle.name)
        try:
            return ModelParser(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _decode_shocks(
        shocks: Mapping[str, ArrayEnvelope] | None,
    ) -> dict[str, NDArray[np.float64]]:
        if shocks is None:
            return {}
        return {name: decode_array(envelope) for name, envelope in shocks.items()}

    @staticmethod
    def _apply_shock_params(slot: ModelSlot, params: ShockParamUpdate) -> None:
        if slot.model_config is None:
            raise ValueError("Cannot update shock parameters before loading a model.")
        conf = slot.model_config
        for shock_name, value in params.std.items():
            shock = Symbol(shock_name)
            if shock not in conf.calibration.shock_std:
                raise ValueError(f"Unknown shock std parameter for '{shock_name}'.")
            param = conf.calibration.shock_std[shock]
            conf.calibration.parameters[param] = np.float64(value)

        for pair_key, value in params.corr.items():
            pair = _parse_corr_pair(pair_key)
            if pair not in conf.calibration.shock_corr:
                raise ValueError(
                    f"Unknown shock correlation parameter for '{pair_key}'."
                )
            param = conf.calibration.shock_corr[pair]
            conf.calibration.parameters[param] = np.float64(value)

    @staticmethod
    def _generate_shocks(
        *,
        slot: ModelSlot,
        T: int,
        generation: ShockGenerationRequest | None,
        raw_shocks: Mapping[str, NDArray[np.float64]],
    ) -> dict[str, NDArray[np.float64] | Callable[..., NDArray[np.float64]]]:
        out: dict[str, NDArray[np.float64] | Callable[..., NDArray[np.float64]]] = {
            name: value for name, value in raw_shocks.items()
        }
        if generation is None or slot.solved is None:
            return out

        conf = slot.solved.config
        # A spec is keyed by the shock, not by the variable the shock drives.
        pending = [str(shock) for shock in conf.shocks if str(shock) not in raw_shocks]
        if not pending:
            return out

        seed = generation.seed
        if generation.dist in {"norm", "t"} and len(pending) > 1:
            key = ",".join(pending)
            dist_kwargs: dict[str, Any]
            if generation.dist == "t":
                dist_kwargs = {
                    "loc": [generation.loc] * len(pending),
                    "df": generation.df,
                }
            else:
                dist_kwargs = {"mean": [generation.loc] * len(pending)}
            out[key] = Shock(
                dist=generation.dist,
                multivar=True,
                seed=seed,
                dist_kwargs=dist_kwargs,
            ).shock_generator(T)
            return out

        for i, name in enumerate(pending):
            uni_kwargs: dict[str, float] = {"loc": generation.loc}
            if generation.dist == "t":
                uni_kwargs["df"] = generation.df
            shock_seed = None if seed is None else seed + i
            out[name] = Shock(
                dist=generation.dist,
                multivar=False,
                seed=shock_seed,
                dist_kwargs=uni_kwargs,
            ).shock_generator(T)
        return out


def _parse_corr_pair(pair_key: str) -> frozenset[Symbol]:
    parts = [part.strip() for part in pair_key.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("Correlation keys must have the form 'shock_a,shock_b'.")
    return frozenset(Symbol(part) for part in parts)
