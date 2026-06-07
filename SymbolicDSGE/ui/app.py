from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from SymbolicDSGE.core.solved_model import SolvedModel

from .mc import (
    build_pipeline,
    mc_catalog,
    run_pipeline,
    serialize_pipeline_result,
    validate_pipeline_spec,
)
from .mc_schemas import MCPipelineSpec, MCRunRequest
from .schemas import (
    LoadYamlRequest,
    Role,
    SimRunRequest,
    SolveModelRequest,
    SubmitFunctionRequest,
)
from .session import UISession


def create_app(
    *,
    session: UISession | None = None,
    reference: SolvedModel | None = None,
    dgp: SolvedModel | None = None,
) -> FastAPI:
    ui_session = (
        session if session is not None else UISession(reference=reference, dgp=dgp)
    )
    app = FastAPI(title="SymbolicDSGE UI", version="0.1.0")
    app.state.ui_session = ui_session
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/session")
    def session_summary() -> dict[str, Any]:
        return ui_session.summary()

    @app.get("/api/mc/catalog")
    def monte_carlo_catalog() -> dict[str, Any]:
        return mc_catalog()

    @app.post("/api/mc/validate")
    def validate_monte_carlo_pipeline(request: MCPipelineSpec) -> dict[str, Any]:
        try:
            ordered = validate_pipeline_spec(
                request,
                has_reference=ui_session.solved_model("reference") is not None,
                has_dgp=ui_session.solved_model("dgp") is not None,
            )
            dgp = ui_session.solved_model("dgp")
            assert dgp is not None
            build_pipeline(ordered, dgp=dgp)
            return {"valid": True, "order": [node.id for node in ordered]}
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=_error_detail(exc)) from exc

    @app.post("/api/run/mc")
    def run_monte_carlo_pipeline(request: MCRunRequest) -> dict[str, Any]:
        try:
            result = run_pipeline(
                request.pipeline,
                reference=ui_session.solved_model("reference"),
                dgp=ui_session.solved_model("dgp"),
                n_rep=request.n_rep,
                fail_fast=request.fail_fast,
            )
            run_id = str(uuid4())
            payload = serialize_pipeline_result(result, run_id=run_id)
            ui_session.record_run(
                run_id=run_id,
                kind="mc",
                role="reference",
                payload=payload,
            )
            return payload
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=_error_detail(exc)) from exc

    @app.post("/api/model/load-yaml")
    def load_yaml(request: LoadYamlRequest) -> dict[str, Any]:
        try:
            return ui_session.load_yaml(
                role=request.role,
                path=request.path,
                content=request.content,
            )
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(exc),
            ) from exc

    @app.post("/api/model/solve")
    def solve_model(request: SolveModelRequest) -> dict[str, Any]:
        try:
            return ui_session.solve_model(
                role=request.role,
                compile_kwargs=request.compile_kwargs,
                solve_kwargs=request.solve_kwargs,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(exc),
            ) from exc

    @app.get("/api/model/{role}/summary")
    def model_summary(role: Role) -> dict[str, Any]:
        try:
            return ui_session.model_summary(role)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=_error_detail(exc),
            ) from exc

    @app.post("/api/run/sim")
    def run_simulation(request: SimRunRequest) -> dict[str, Any]:
        try:
            return ui_session.run_simulation(
                role=request.role,
                T=request.T,
                observables=request.observables,
                shock_scale=request.shock_scale,
                shocks=request.shocks,
                shock_generation=request.shock_generation,
                shock_params=request.shock_params,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(exc),
            ) from exc

    @app.get("/api/run/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return ui_session.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=_error_detail(exc),
            ) from exc

    @app.post("/api/code/submit")
    def submit_function(request: SubmitFunctionRequest) -> dict[str, Any]:
        try:
            return ui_session.submit_function(
                role=request.role,
                code=request.code,
                kind=request.kind,
            )
        except (SyntaxError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(exc),
            ) from exc

    @app.delete("/api/code/{role}/{name}")
    def remove_function(role: Role, name: str) -> dict[str, Any]:
        try:
            ui_session.remove_function(role=role, name=name)
            return {"removed": name}
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=_error_detail(exc),
            ) from exc

    @app.get("/api/code/{role}/functions")
    def list_functions(role: Role) -> list[dict[str, Any]]:
        try:
            return ui_session.list_functions(role=role)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=_error_detail(exc),
            ) from exc

    return app


def _error_detail(exc: Exception) -> dict[str, str]:
    return {"error_type": type(exc).__name__, "message": str(exc)}
