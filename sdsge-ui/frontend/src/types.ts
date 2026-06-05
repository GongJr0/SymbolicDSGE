export type Role = "reference" | "dgp";

export interface ArrayEnvelope {
  dtype: "float64";
  shape: number[];
  order: "C";
  data_b64: string;
}

export interface NamedArray {
  name: string;
  array: ArrayEnvelope;
}

export interface ShockSpec {
  shock: string;
  target: string;
  std_param: string | null;
  std_value: number | null;
}

export interface ShockCorrSpec {
  pair: string[];
  key: string;
  corr_param: string;
  corr_value: number | null;
}

export type ShockDistribution = "norm" | "t" | "uni";

export interface ShockGeneration {
  dist: ShockDistribution;
  seed: number | null;
  loc: number;
  df: number;
}

export interface ShockParamUpdate {
  std: Record<string, number>;
  corr: Record<string, number>;
}

export interface ModelSummary {
  role: Role;
  loaded: boolean;
  solved: boolean;
  source?: string;
  raw_yaml?: string;
  name?: string;
  variables?: string[];
  observables?: string[];
  parameters?: string[];
  shock_specs?: ShockSpec[];
  shock_corr_specs?: ShockCorrSpec[];
  n_state?: number;
  n_exog?: number;
  A_shape?: number[];
  B_shape?: number[];
  has_kalman?: boolean;
}

export interface SessionSummary {
  models: Record<Role, ModelSummary>;
  runs: Array<{ run_id: string; kind: string; role: Role }>;
}

export interface SimResult {
  run_id: string;
  kind: "sim";
  role: Role;
  T: number;
  observables: boolean;
  series: NamedArray[];
}
