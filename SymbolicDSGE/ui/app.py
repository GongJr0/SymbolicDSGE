from __future__ import annotations

from typing import Any, Mapping, cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core.solved_model import SolvedModel
from ..monte_carlo.spec import pipeline_meta
from ..bundle.manifest import SimSpec

from .mc import (
    build_pipeline,
    mc_available_traces,
    mc_custom_op_template,
    run_pipeline,
    serialize_pipeline_result,
    validate_custom_op,
)
from .estimation import estimation_catalog
from .schemas import (
    EstimationRunRequest,
    WorkspaceViewUpdate,
    LoadYamlRequest,
    SolveModelRequest,
    SubmitFunctionRequest,
)
from .session import UISession, Workspace


def create_app(
    *,
    session: UISession | None = None,
    models: Mapping[str, SolvedModel] | None = None,
    workspace: Workspace | None = None,
    source: str | None = None,
) -> FastAPI:
    ui_session = (
        session
        if session is not None
        else UISession(models=models, workspace=workspace, source=source)
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

    @app.put("/api/session/workspace")
    def update_workspace_view(request: dict[str, Any]) -> dict[str, Any]:
        """Hold a tab's on-screen state for the life of the process.

        This is what a refresh restores from: the client posts what it has,
        and the reload reads it back out of the same process rather than out
        of anything the browser kept. Acknowledges only, since the caller is
        the one that already has the state.
        """
        try:
            ui_session.set_workspace_view(**cast(WorkspaceViewUpdate, request))
            return {"tab": request["tab"]}

        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=_error_detail(exc)) from exc

    @app.get("/api/mc/custom/template")
    def monte_carlo_custom_template() -> dict[str, str]:
        return mc_custom_op_template()

    @app.post("/api/mc/custom/validate")
    def monte_carlo_custom_validate(request: dict[str, Any]) -> dict[str, Any]:
        return validate_custom_op(
            request["code"],
            step_type=request.get("step_type", "transform:custom"),
        )

    @app.post("/api/mc/traces")
    def monte_carlo_traces(request: dict[str, Any]) -> dict[str, list[str]]:
        return mc_available_traces(request)

    @app.get("/api/estimation/catalog")
    def get_estimation_catalog() -> dict[str, Any]:
        return estimation_catalog()

    @app.post("/api/run/estimation")
    def run_estimation(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return ui_session.run_estimation(cast(EstimationRunRequest, request))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=_error_detail(exc)) from exc

    @app.post("/api/mc/validate")
    def validate_monte_carlo_pipeline(request: dict[str, Any]) -> dict[str, Any]:
        try:

            # Compile and catch.
            pipe = build_pipeline(request)
            return {
                "valid": True,
                "steps": [step.name for step in pipe.replication_steps],
                "postprocs": [pp.name for pp in pipe.postproc_steps],
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=_error_detail(exc)) from exc

    @app.post("/api/run/mc")
    def run_monte_carlo_pipeline(request: dict[str, Any]) -> dict[str, Any]:
        try:
            pipeline = build_pipeline(request["pipeline"])
            result = run_pipeline(
                pipeline,
                models={
                    name: slot.solved
                    for name, slot in ui_session.slots.items()
                    if slot.solved is not None
                },
                n_rep=int(request.get("n_rep", 100)),
                fail_fast=bool(request.get("fail_fast", True)),
                n_jobs=request.get("n_jobs"),
                verbosity=int(request.get("verbosity", 0)),
            )
            payload = serialize_pipeline_result(result)
            # Off the built pipeline, not the body that described it: the slot
            # is what a bundle stores, and only `to_spec` produces that.
            ui_session.workspace.mc.spec = dict(pipeline_meta(pipeline.to_spec()))
            ui_session.workspace.mc.result = payload
            return payload
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=_error_detail(exc)) from exc

    @app.post("/api/model/load-yaml")
    def load_yaml(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return ui_session.load_yaml(**cast(LoadYamlRequest, request))
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(exc),
            ) from exc

    @app.post("/api/model/solve")
    def solve_model(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return ui_session.solve_model(**cast(SolveModelRequest, request))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(exc),
            ) from exc

    @app.get("/api/model/{model_name}/summary")
    def model_summary(model_name: str) -> dict[str, Any]:
        try:
            return ui_session.model_summary(model_name)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=_error_detail(exc),
            ) from exc

    @app.post("/api/run/sim")
    def run_simulation(request: dict[str, Any]) -> dict[str, Any]:
        try:
            model_name = request["model_name"]
            spec = SimSpec.from_dict(request["spec"])
            return ui_session.run_simulation_spec(model_name, spec)
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(exc),
            ) from exc

    @app.post("/api/code/submit")
    def submit_function(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return ui_session.submit_function(**cast(SubmitFunctionRequest, request))
        except (SyntaxError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(exc),
            ) from exc

    @app.delete("/api/code/{model_name}/{name}")
    def remove_function(model_name: str, name: str) -> dict[str, Any]:
        try:
            ui_session.remove_function(model_name=model_name, name=name)
            return {"removed": name}
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=_error_detail(exc),
            ) from exc

    @app.get("/api/code/{model_name}/functions")
    def list_functions(model_name: str) -> list[dict[str, Any]]:
        try:
            return ui_session.list_functions(model_name=model_name)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=_error_detail(exc),
            ) from exc

    return app


def _error_detail(exc: Exception) -> dict[str, str]:
    return {"error_type": type(exc).__name__, "message": str(exc)}
