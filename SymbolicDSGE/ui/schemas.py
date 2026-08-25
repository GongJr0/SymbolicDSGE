from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["reference", "dgp"]
ShockDistribution = Literal["norm", "t", "uni"]
FunctionKind = Literal["array", "figure"]
EstimationMethod = Literal["mle", "map", "mcmc"]
WorkspaceTab = Literal["estimation", "mc"]


class WorkspaceViewUpdate(BaseModel):
    """A tab's on-screen state, PUT by the client as it edits.

    ``view`` is opaque on purpose: it is the GUI's own shape, so a new control
    needs no field here. The bundle-bound ``spec``/``result`` slots have no
    counterpart on this model, which is what makes the client structurally
    unable to write them.
    """

    model_config = ConfigDict(extra="forbid")

    tab: WorkspaceTab
    view: dict[str, Any] | None = None


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
    """A run request from the estimation tab.

    ``routine`` arrives on the wire as ``method``, the name the frontend has
    always posted. The alias keeps that contract while freeing ``method`` inside
    ``method_kwargs`` to mean what the library means by it: the optimizer that
    :meth:`Estimator.mle` and :meth:`Estimator.map` take.
    """

    model_config = ConfigDict(populate_by_name=True)

    role: Role = "reference"
    routine: EstimationMethod = Field(default="mle", alias="method")
    y: list[list[float]] = Field(min_length=1)
    observables: list[str] | None = None
    parameters: list[EstimationParameterSpec] = Field(min_length=1)
    method_kwargs: dict[str, Any] = Field(default_factory=dict)
    compile_kwargs: dict[str, Any] = Field(default_factory=dict)
    ss_seed: list[float] | None = None
    posterior_point: str = "mean"
    estimate_and_solve: bool = False
