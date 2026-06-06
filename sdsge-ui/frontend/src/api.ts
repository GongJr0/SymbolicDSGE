import type {
  ArrayEnvelope,
  FunctionKind,
  FunctionRecord,
  ModelSummary,
  Role,
  SessionSummary,
  ShockGeneration,
  ShockParamUpdate,
  SimResult,
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

export function loadYamlPath(role: Role, path: string): Promise<ModelSummary> {
  return requestJson<ModelSummary>("/api/model/load-yaml", {
    method: "POST",
    body: JSON.stringify({ role, path }),
  });
}

export function loadYamlContent(
  role: Role,
  content: string,
): Promise<ModelSummary> {
  return requestJson<ModelSummary>("/api/model/load-yaml", {
    method: "POST",
    body: JSON.stringify({ role, content }),
  });
}

export function solveModel(
  role: Role,
  compileKwargs: Record<string, unknown> = {},
): Promise<ModelSummary> {
  return requestJson<ModelSummary>("/api/model/solve", {
    method: "POST",
    body: JSON.stringify({ role, compile_kwargs: compileKwargs }),
  });
}

export function runSimulation(
  role: Role,
  T: number,
  observables: boolean,
  shocks?: Record<string, ArrayEnvelope>,
  shockGeneration?: ShockGeneration,
  shockParams?: ShockParamUpdate,
): Promise<SimResult> {
  return requestJson<SimResult>("/api/run/sim", {
    method: "POST",
    body: JSON.stringify({
      role,
      T,
      observables,
      shocks,
      shock_generation: shockGeneration,
      shock_params: shockParams,
    }),
  });
}

export function submitFunction(
  role: Role,
  code: string,
  kind: FunctionKind = "array",
): Promise<FunctionRecord> {
  return requestJson<FunctionRecord>("/api/code/submit", {
    method: "POST",
    body: JSON.stringify({ role, code, kind }),
  });
}

export function removeFunction(
  role: Role,
  name: string,
): Promise<{ removed: string }> {
  return requestJson<{ removed: string }>(
    `/api/code/${role}/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}

export function listFunctions(role: Role): Promise<FunctionRecord[]> {
  return requestJson<FunctionRecord[]>(`/api/code/${role}/functions`);
}

export function encodeArray(values: Float64Array): ArrayEnvelope {
  const bytes = new Uint8Array(values.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return {
    dtype: "float64",
    shape: [values.length],
    order: "C",
    data_b64: btoa(binary),
  };
}

export function decodeArray(envelope: ArrayEnvelope): Float64Array {
  const binary = atob(envelope.data_b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float64Array(bytes.buffer);
}
