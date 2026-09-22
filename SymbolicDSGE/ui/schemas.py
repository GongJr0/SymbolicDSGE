from __future__ import annotations

from typing import Any, Literal, TypedDict

from ..estimation.spec import PriorSpec

FunctionKind = Literal["array", "figure"]
EstimationMethod = Literal["mle", "map", "mcmc"]
WorkspaceTab = Literal["estimation", "mc"]


class WorkspaceViewUpdate(TypedDict):
    """A tab's on-screen state, PUT by the client as it edits.

    ``view`` is opaque on purpose: it is the GUI's own shape, so a new control
    needs no field here. The bundle-bound ``spec``/``result`` slots have no
    counterpart on this model, which is what makes the client structurally
    unable to write them.
    """

    tab: WorkspaceTab
    view: dict[str, Any] | None


class ArrayEnvelope(TypedDict):
    shape: list[int]
    data_b64: str


class LoadYamlRequest(TypedDict):
    model_name: str
    path: str | None
    content: str | None


class SolveModelRequest(TypedDict):
    model_name: str
    compile_kwargs: dict[str, Any]
    solve_kwargs: dict[str, Any]


class SubmitFunctionRequest(TypedDict):
    model_name: str
    code: str
    kind: FunctionKind


class EstimationParameterSpec(TypedDict):
    name: str
    estimate: bool
    initial: float
    lower: float | None
    upper: float | None
    prior: PriorSpec | None


class EstimationRunRequest(TypedDict):
    model_name: str
    routine: EstimationMethod
    y: list[list[float]]
    observables: list[str] | None
    parameters: list[EstimationParameterSpec]
    method_kwargs: dict[str, Any]
    compile_kwargs: dict[str, Any]
    ss_seed: list[float] | None
    posterior_point: str
    estimate_and_solve: bool
