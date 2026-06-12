from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from SymbolicDSGE.estimation.spec import EstimationSpec

Role = Literal["reference", "dgp"]
ShockDistribution = Literal["norm", "t", "uni"]
FunctionKind = Literal["array", "figure"]
EstimationMethod = Literal["mle", "map", "mcmc"]


class ArrayEnvelope(BaseModel):
    dtype: Literal["float64"] = "float64"
    shape: list[int]
    order: Literal["C"] = "C"
    data_b64: str


class LoadYamlRequest(BaseModel):
    role: Role = "reference"
    path: str | None = None
    content: str | None = None


class SolveModelRequest(BaseModel):
    role: Role = "reference"
    compile_kwargs: dict[str, Any] = Field(default_factory=dict)
    solve_kwargs: dict[str, Any] = Field(default_factory=dict)


class ShockGenerationRequest(BaseModel):
    dist: ShockDistribution = "norm"
    seed: int | None = 0
    loc: float = 0.0
    df: float = Field(default=5.0, gt=0.0)


class ShockParamUpdate(BaseModel):
    std: dict[str, float] = Field(default_factory=dict)
    corr: dict[str, float] = Field(default_factory=dict)


class SimRunRequest(BaseModel):
    role: Role = "reference"
    T: int = Field(gt=0)
    observables: bool = True
    shock_scale: float = 1.0
    shocks: dict[str, ArrayEnvelope] | None = None
    shock_generation: ShockGenerationRequest | None = None
    shock_params: ShockParamUpdate | None = None


class SubmitFunctionRequest(BaseModel):
    role: Role
    code: str
    kind: FunctionKind = "array"


class PriorSpec(BaseModel):
    distribution: str = "normal"
    parameters: dict[str, float | int] = Field(default_factory=dict)
    transform: str = "identity"
    transform_kwargs: dict[str, float | int] = Field(default_factory=dict)


class EstimationParameterSpec(BaseModel):
    name: str = Field(min_length=1)
    estimate: bool = False
    initial: float
    lower: float | None = None
    upper: float | None = None
    prior: PriorSpec | None = None


class EstimationRunRequest(BaseModel):
    role: Role = "reference"
    method: EstimationMethod = "mle"
    y: list[list[float]] = Field(min_length=1)
    observables: list[str] | None = None
    parameters: list[EstimationParameterSpec] = Field(min_length=1)
    method_kwargs: dict[str, Any] = Field(default_factory=dict)
    compile_kwargs: dict[str, Any] = Field(default_factory=dict)
    steady_state: list[float] | None = None
    posterior_point: str = "mean"
    estimate_and_solve: bool = False

    def to_core(self) -> EstimationSpec:
        """Convert to the pydantic-free core spec (bundle/text serialization).

        Drops UI-only fields (``role``, ``y``, ``estimate_and_solve``); ``y``
        rides a Parquet member alongside the spec in a bundle.
        """
        return EstimationSpec.from_dict(self.model_dump())
