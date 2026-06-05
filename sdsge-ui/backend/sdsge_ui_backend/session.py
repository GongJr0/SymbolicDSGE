from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from sympy import Symbol

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.compiled_model import CompiledModel
from SymbolicDSGE.core.config import ModelConfig
from SymbolicDSGE.core.shock_generators import Shock
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.config import KalmanConfig

from .schemas import ArrayEnvelope, Role, ShockGenerationRequest, ShockParamUpdate
from .serializers import (
    decode_array,
    empty_model_summary,
    encode_named_arrays,
    summarize_parsed_model,
    summarize_solved_model,
)


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


class UISession:
    def __init__(
        self,
        *,
        reference: SolvedModel | None = None,
        dgp: SolvedModel | None = None,
    ) -> None:
        self.slots: dict[Role, ModelSlot] = {
            "reference": ModelSlot(role="reference"),
            "dgp": ModelSlot(role="dgp"),
        }
        self.runs: dict[str, RunRecord] = {}
        if reference is not None:
            self.set_solved_model("reference", reference)
        if dgp is not None:
            self.set_solved_model("dgp", dgp)

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
        }

    def set_solved_model(self, role: Role, model: SolvedModel) -> dict[str, Any]:
        slot = self._slot(role)
        slot.source = "<injected>"
        slot.raw_yaml = None
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
        payload: dict[str, Any] = {
            "run_id": run_id,
            "kind": "sim",
            "role": role,
            "T": T,
            "observables": observables,
            "series": encode_named_arrays(sim),
        }
        self.runs[run_id] = RunRecord(
            run_id=run_id,
            kind="sim",
            role=role,
            payload=payload,
        )
        return payload

    def get_run(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.runs:
            raise KeyError(run_id)
        return dict(self.runs[run_id].payload)

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
        targets = [
            str(target)
            for _, target in conf.shock_map.items()
            if str(target) not in raw_shocks
        ]
        if not targets:
            return out

        seed = generation.seed
        if generation.dist in {"norm", "t"} and len(targets) > 1:
            key = ",".join(targets)
            dist_kwargs: dict[str, Any]
            if generation.dist == "t":
                dist_kwargs = {
                    "loc": [generation.loc] * len(targets),
                    "df": generation.df,
                }
            else:
                dist_kwargs = {"mean": [generation.loc] * len(targets)}
            out[key] = Shock(
                T=T,
                dist=generation.dist,
                multivar=True,
                seed=seed,
                dist_kwargs=dist_kwargs,
            ).shock_generator()
            return out

        for i, target in enumerate(targets):
            uni_kwargs: dict[str, float] = {"loc": generation.loc}
            if generation.dist == "t":
                uni_kwargs["df"] = generation.df
            shock_seed = None if seed is None else seed + i
            out[target] = Shock(
                T=T,
                dist=generation.dist,
                multivar=False,
                seed=shock_seed,
                dist_kwargs=uni_kwargs,
            ).shock_generator()
        return out


def _parse_corr_pair(pair_key: str) -> frozenset[Symbol]:
    parts = [part.strip() for part in pair_key.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("Correlation keys must have the form 'shock_a,shock_b'.")
    return frozenset(Symbol(part) for part in parts)
