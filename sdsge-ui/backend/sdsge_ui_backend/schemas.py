from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["reference", "dgp"]
ShockDistribution = Literal["norm", "t", "uni"]
FunctionKind = Literal["array", "figure"]


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
