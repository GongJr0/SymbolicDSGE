import type {
  ArrayEnvelope,
  FunctionKind,
  FunctionRecord,
  EstimationCatalog,
  EstimationRunRequest,
  EstimationRunResult,
  EstimationViewState,
  MCViewState,
  ModelSummary,
  MCPipelineResult,
  MCPipelineSpec,
  MCStepType,
  SessionSummary,
  SimResult,
  SimSpecWire,
  WorkspaceTab,
} from "./types";

const API_BASE = import.meta.env.VITE_SDSGE_API_BASE ?? "http://127.0.0.1:8000";

async function requestJson<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) {
    const detail = body.detail;
    const message =
      typeof detail?.message === "string"
        ? detail.message
        : `Request failed with HTTP ${response.status}`;
    throw new Error(message);
  }
  return body as T;
}

export function getSession(): Promise<SessionSummary> {
  return requestJson<SessionSummary>("/api/session");
}

/** Hand a tab's on-screen state to the session, which outlives the page.
 *
 * The server acknowledges rather than echoing, since the caller is the one
 * that already has the state. Pass `null` to drop what it is holding.
 */
export function putWorkspaceView(
  tab: WorkspaceTab,
  view: Partial<EstimationViewState> | MCViewState | null,
  model_name?: string,
): Promise<{ tab: WorkspaceTab }> {
  return requestJson<{ tab: WorkspaceTab }>("/api/session/workspace", {
    method: "PUT",
    body: JSON.stringify({ tab, view, model_name }),
  });
}

export function loadYamlPath(model_name: string, path: string): Promise<ModelSummary> {
  return requestJson<ModelSummary>("/api/model/load-yaml", {
    method: "POST",
    body: JSON.stringify({ model_name, path }),
  });
}

export function loadYamlContent(
  model_name: string,
  content: string,
): Promise<ModelSummary> {
  return requestJson<ModelSummary>("/api/model/load-yaml", {
    method: "POST",
    body: JSON.stringify({ model_name, content }),
  });
}

export function solveModel(
  model_name: string,
  compileKwargs: Record<string, unknown> = {},
): Promise<ModelSummary> {
  return requestJson<ModelSummary>("/api/model/solve", {
    method: "POST",
    body: JSON.stringify({ model_name, compile_kwargs: compileKwargs }),
  });
}

export function runSimulation(
  model_name: string,
  spec: SimSpecWire,
): Promise<SimResult> {
  return requestJson<SimResult>("/api/run/sim", {
    method: "POST",
    body: JSON.stringify({ model_name, spec }),
  });
}

export function getEstimationCatalog(): Promise<EstimationCatalog> {
  return requestJson<EstimationCatalog>("/api/estimation/catalog");
}

export function runEstimation(
  request: EstimationRunRequest,
): Promise<EstimationRunResult> {
  return requestJson<EstimationRunResult>("/api/run/estimation", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function submitFunction(
  model_name: string,
  code: string,
  kind: FunctionKind = "array",
): Promise<FunctionRecord> {
  return requestJson<FunctionRecord>("/api/code/submit", {
    method: "POST",
    body: JSON.stringify({ model_name, code, kind }),
  });
}

export function removeFunction(
  model_name: string,
  name: string,
): Promise<{ removed: string }> {
  return requestJson<{ removed: string }>(
    `/api/code/${encodeURIComponent(model_name)}/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}

export function listFunctions(model_name: string): Promise<FunctionRecord[]> {
  return requestJson<FunctionRecord[]>(`/api/code/${encodeURIComponent(model_name)}/functions`);
}

export function getMCCustomTemplate(): Promise<{ template: string }> {
  return requestJson<{ template: string }>("/api/mc/custom/template");
}

export function validateCustomOp(
  code: string,
  stepType: MCStepType = "transform:custom",
): Promise<{ valid: boolean; name?: string; error?: string }> {
  return requestJson<{ valid: boolean; name?: string; error?: string }>(
    "/api/mc/custom/validate",
    { method: "POST", body: JSON.stringify({ code, step_type: stepType }) },
  );
}

export function fetchAvailableTraces(
  pipeline: MCPipelineSpec,
): Promise<{ traces: string[] }> {
  return requestJson<{ traces: string[] }>("/api/mc/traces", {
    method: "POST",
    body: JSON.stringify(pipeline),
  });
}

/** Compile the pipeline server-side without running it.
 *
 * `steps` comes back in execution order, since the server names them off the
 * pipeline it just built, and a 400 carries why it could not be.
 */
export function validateMCPipeline(
  pipeline: MCPipelineSpec,
): Promise<{ valid: true; steps: string[]; postprocs: string[] }> {
  return requestJson<{ valid: true; steps: string[]; postprocs: string[] }>(
    "/api/mc/validate",
    {
      method: "POST",
      body: JSON.stringify(pipeline),
    },
  );
}

export function runMCPipeline(
  pipeline: MCPipelineSpec,
  nRep: number,
  nJobs: number | null,
  failFast: boolean,
  verbosity: number,
): Promise<MCPipelineResult> {
  return requestJson<MCPipelineResult>("/api/run/mc", {
    method: "POST",
    body: JSON.stringify({
      pipeline,
      n_rep: nRep,
      n_jobs: nJobs,
      fail_fast: failFast,
      verbosity,
    }),
  });
}

export function decodeArray(envelope: ArrayEnvelope): Float64Array {
  const binary = atob(envelope.data_b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float64Array(bytes.buffer);
}
