from __future__ import annotations

import ast
import base64
import inspect
import io
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, cast

# Set non-interactive backend before any user code can import pyplot.
try:
    import matplotlib as _mpl

    _mpl.use("Agg")
except Exception:
    pass

import numpy as np
from numpy.typing import NDArray

from ..core import DSGESolver, ModelParser
from ..core.compiled_model import CompiledModel
from ..core.config import ModelConfig
from ..core.solved_model.base import SolvedModel
from ..kalman.config import KalmanConfig

from .schemas import (
    EstimationRunRequest,
    FunctionKind,
    WorkspaceTab,
)
from ..bundle.manifest import SimSpec
from ..estimation.spec import EstimatorParams, EstimatorSpec

from .estimation import (
    build_estimation_inputs,
    estimator_spec_wire,
    serialize_estimation_result,
)
from .serializers import (
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
    source: str | None = None
    raw_yaml: str | None = None
    model_config: ModelConfig | None = None
    kalman_config: KalmanConfig | None = None
    solver: DSGESolver | None = None
    compiled: CompiledModel | None = None
    solved: SolvedModel | None = None


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

    estimation: dict[str, TabState] = field(default_factory=dict)
    mc: TabState = field(default_factory=TabState)
    #: Per-model simulation tabs. The Outputs tab renders only ``T`` and the
    #: observables toggle, both of them ``SimSpec`` fields, so there is no
    #: view to carry: the spec is the form.
    simulation: dict[str, TabState] = field(default_factory=dict)


class UISession:
    def __init__(
        self,
        *,
        models: Mapping[str, SolvedModel] | None = None,
        workspace: Workspace | None = None,
        source: str | None = None,
    ) -> None:
        if models is None:
            models = {}
        self.slots: dict[str, ModelSlot] = {name: ModelSlot() for name in models}
        self.functions: dict[str, dict[str, FunctionRecord]] = {
            name: {} for name in self.slots
        }
        self.workspace: Workspace = workspace if workspace is not None else Workspace()
        # Preloaded models share the source label.
        for name, model in models.items():
            self.set_solved_model(name, model, source=source)
        self.replay_bundled_simulations()

    def summary(self) -> dict[str, Any]:
        return {
            "models": {name: self.model_summary(name) for name in self.slots},
            "workspace": self._workspace_payload(),
        }

    def _workspace_payload(self) -> dict[str, Any]:
        """Wire shape for the workspace (omits tabs and slots nothing filled)."""
        out: dict[str, Any] = {}

        if mc := self.workspace.mc.payload():
            out["mc"] = mc

        for name, tab in (
            ("estimation", self.workspace.estimation),
            ("simulation", self.workspace.simulation),
        ):
            if payload := {n: t.payload() for n, t in tab.items() if t.payload()}:
                out[name] = payload
        return out

    def set_workspace_view(
        self,
        tab: WorkspaceTab,
        view: dict[str, Any] | None,
        model_name: str | None = None,
    ) -> None:
        """Replace a tab's view with what the client last had on screen.

        The view is held verbatim: it is the GUI's own state, so a new control
        appears here without the server learning what it means. Writing it
        cannot disturb ``spec``/``result``, which only a run fills.
        """
        if tab == "estimation":
            if model_name is None:
                raise ValueError("Estimation view updates require a model_name.")
            self._slot(model_name)
            state = self.workspace.estimation.setdefault(model_name, TabState())
        elif tab == "mc":
            state = self.workspace.mc
        else:
            raise ValueError(f"Unknown workspace tab {tab!r}.")
        state.view = view

    def set_solved_model(
        self, name: str, model: SolvedModel, *, source: str | None = None
    ) -> dict[str, Any]:
        """Install an already-solved model into the named slot.

        ``source`` labels where the model came from, e.g. the bundle path
        ``sdsge-ui`` was pointed at; it is what distinguishes one preloaded
        model from another in the GUI, so an in-process model with nothing to
        cite leaves it unset rather than naming a placeholder. The YAML rides
        along on the config whenever the model was parsed rather than built,
        which is what lets the Builder tab open on the model it is serving.
        """
        slot = self._ensure_slot(name)
        slot.source = source
        slot.raw_yaml = model.config.source_yaml
        slot.model_config = model.config
        slot.kalman_config = model.kalman_config
        slot.solver = DSGESolver(model.config, cast(Any, model.kalman_config))
        slot.compiled = model.compiled
        slot.solved = model
        return self.model_summary(name)

    def load_yaml(
        self,
        *,
        model_name: str,
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
        slot = self._ensure_slot(model_name)
        slot.source = source
        slot.raw_yaml = raw_yaml
        slot.model_config = model
        slot.kalman_config = kalman
        slot.solver = DSGESolver(model, cast(Any, kalman))
        slot.compiled = None
        slot.solved = None
        return self.model_summary(model_name)

    def solve_model(
        self,
        *,
        model_name: str,
        compile_kwargs: Mapping[str, Any] | None = None,
        solve_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        slot = self._slot(model_name)
        if slot.solver is None:
            raise ValueError(f"No model is loaded for '{model_name}'.")
        compiled = slot.solver.compile(**dict(compile_kwargs or {}))
        solved = slot.solver.solve(compiled, **dict(solve_kwargs or {}))
        slot.compiled = compiled
        slot.solved = solved
        return self.model_summary(model_name)

    def model_summary(self, name: str) -> dict[str, Any]:
        slot = self._slot(name)
        if slot.solved is not None:
            summary = summarize_solved_model(
                model_name=name,
                model=slot.solved,
                source=slot.source,
            )
        elif slot.model_config is not None:
            summary = summarize_parsed_model(
                model_name=name,
                model=slot.model_config,
                kalman=slot.kalman_config,
                source=slot.source,
            )
        else:
            return empty_model_summary(name)
        if slot.raw_yaml is not None:
            summary["raw_yaml"] = slot.raw_yaml
        return summary

    def _record_sim_run(
        self, *, model_name: str, sim: Any, T: int, observables: bool
    ) -> dict[str, Any]:
        """Serialize a simulation into the shape the Outputs tab renders.

        Shared by the tab's own runs and by a bundle's stored spec replayed at
        load, so both arrive there identically.
        """
        sim_dict = sim.states
        sim_dict["_X"] = sim.X
        if sim.y is not None:
            sim_dict.update(sim.observables)

        all_series = encode_named_arrays(sim_dict)
        extra, transform_errors = self._apply_array_functions(model_name, sim_dict)

        if extra:
            all_series = all_series + encode_named_arrays(extra)
        figures = self._apply_figure_functions(model_name, sim_dict)
        payload: dict[str, Any] = {
            "kind": "sim",
            "model_name": model_name,
            "T": T,
            "observables": observables,
            "series": all_series,
            "figures": figures,
            "transform_errors": transform_errors,
        }
        return payload

    def replay_bundled_simulations(self) -> None:
        """Run each stored simulation spec against the model it was stored with.

        A bundle keeps the spec rather than the output, and the Outputs tab's
        only controls (``T`` and the observables toggle) are spec fields, so
        there is nothing to prefill and no way to show that a simulation is in
        the bundle at all. Replaying it produces the output the bundle stands
        for; the spec pins the seed, so this reproduces the run rather than
        drawing a new one.

        A spec that will not replay leaves its slot empty rather than taking
        the whole session down with it: the other tabs are unaffected by it.
        """
        for model_name, tab in self.workspace.simulation.items():
            if tab.spec is None:
                continue
            try:
                if self._slot(model_name).solved is None:
                    continue
                tab.result = self.run_simulation_spec(
                    model_name, SimSpec.from_dict(tab.spec)
                )
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                print(
                    f"sdsge-ui: could not replay the '{model_name}' simulation: {exc}"
                )

    def run_simulation_spec(self, model_name: str, spec: SimSpec) -> dict[str, Any]:
        """Run a stored spec verbatim, materialized into ``sim``'s keywords."""
        slot = self._slot(model_name)
        if slot.solved is None:
            raise ValueError(f"{model_name!r} is not a solved model.")
        kwargs = spec.to_sim_kwargs()
        result = self._record_sim_run(
            model_name=model_name,
            sim=slot.solved.sim(**kwargs),
            T=int(kwargs["T"]),
            observables=bool(kwargs["observables"]),
        )
        state = self.workspace.simulation.setdefault(model_name, TabState())
        state.spec = spec.to_dict()
        state.result = result
        return result

    def run_estimation(self, request: EstimationRunRequest) -> dict[str, Any]:
        slot = self._slot(request["model_name"])
        if slot.solver is None:
            raise ValueError(f"No model is loaded for '{request['model_name']}'.")
        if slot.compiled is None:
            slot.compiled = slot.solver.compile(**dict(request["compile_kwargs"]))

        y = np.asarray(request["y"], dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(
                "Observed estimation data must be a two-dimensional array."
            )
        observables = request["observables"]
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
            request["parameters"],
            routine=request["routine"],
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
                    ss_seed=request["ss_seed"],
                    x0=None,
                    jitter=0.0,
                    symmetrize=True,
                    joseph_cov=False,
                ),
            )
        )
        kwargs = dict(request["method_kwargs"])
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
        if bounds is not None and request["routine"] in {"mle", "map"}:
            kwargs["bounds"] = bounds
        # JSON has no arrays, so a proposal covariance arrives as nested lists
        # while the sampler takes a memoryview over one.
        if kwargs.get("proposal_cov") is not None:
            kwargs["proposal_cov"] = np.asarray(
                kwargs["proposal_cov"], dtype=np.float64
            )

        common: dict[str, Any] = {
            "compiled": slot.compiled,
            "y": y,
            "routine": request["routine"],
            "theta0": theta0,
            "observables": observables,
            "estimated_params": names,
            "priors": priors,
            "ss_seed": request["ss_seed"],
            **kwargs,
        }
        solved = False
        if request["estimate_and_solve"]:
            result, model = slot.solver.estimate_and_solve(
                posterior_point=request["posterior_point"],
                **common,
            )
            slot.solved = model
            solved = True
        else:
            result = slot.solver.estimate(**common)

        result_wire = serialize_estimation_result(result)
        payload: dict[str, Any] = {
            "kind": "estimation",
            "model_name": request["model_name"],
            "routine": request["routine"],
            "solved": solved,
            "result": result_wire,
        }
        # The bundle-bound slots, filled from the run that just produced them.
        # The client's view is untouched: it already shows this.
        state = self.workspace.estimation.setdefault(request["model_name"], TabState())
        state.spec = spec_wire
        state.result = result_wire
        return payload

    def submit_function(
        self,
        *,
        model_name: str,
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
        self.functions[model_name][name] = FunctionRecord(
            name=name, kind=kind, source=code, func=func
        )
        return {"name": name, "kind": kind, "source": code}

    def remove_function(self, *, model_name: str, name: str) -> None:
        if name not in self.functions[model_name]:
            raise KeyError(name)
        del self.functions[model_name][name]

    def list_functions(self, *, model_name: str) -> list[dict[str, Any]]:
        return [
            {"name": f.name, "kind": f.kind, "source": f.source}
            for f in self.functions[model_name].values()
        ]

    def solved_model(self, name: str) -> SolvedModel | None:
        return self._slot(name).solved

    def _apply_figure_functions(
        self,
        model_name: str,
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
        for name, record in self.functions[model_name].items():
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
        model_name: str,
        sim_dict: dict[str, NDArray[np.float64]],
    ) -> tuple[dict[str, NDArray[np.float64]], list[dict[str, str]]]:
        """Run this model's transforms, reporting the ones that did not run.

        A transform that fails produces no series, which on its own reads as
        the submit having silently not worked. The commonest cause is a
        signature naming a series the run did not produce: the default
        template takes every observable, and a run with the observables
        toggle off supplies none of them.
        """
        extra: dict[str, NDArray[np.float64]] = {}
        errors: list[dict[str, str]] = []
        for name, record in self.functions[model_name].items():
            if record.kind != "array":
                continue
            try:
                sig = inspect.signature(record.func)
                # Named but unavailable, and with no default to fall back on.
                # Passing the rest would raise the same failure from deeper in,
                # with a message that does not say which series was missing.
                missing = [
                    param
                    for param, spec in sig.parameters.items()
                    if param not in sim_dict and spec.default is inspect.Parameter.empty
                ]
                if missing:
                    raise TypeError(f"this simulation produced no {', '.join(missing)}")
                kwargs = {p: sim_dict[p] for p in sig.parameters if p in sim_dict}
                extra[name] = np.asarray(record.func(**kwargs), dtype=np.float64)
            except Exception as exc:
                errors.append({"name": name, "error": str(exc)})
        return extra, errors

    def _ensure_slot(self, model_name: str) -> ModelSlot:
        if model_name not in self.slots:
            self.slots[model_name] = ModelSlot()
            self.functions[model_name] = {}
        return self.slots[model_name]

    def _slot(self, model_name: str) -> ModelSlot:
        if model_name not in self.slots:
            raise KeyError(model_name)
        return self.slots[model_name]

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
