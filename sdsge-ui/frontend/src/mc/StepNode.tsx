import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import {
  Activity,
  Code,
  DatabaseZap,
  Filter,
  Sigma,
  TestTubeDiagonal,
  Waves,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { MCStepType } from "../types";
import { stepDefinition } from "./catalog";
import type { MCFlowNode } from "./types";

const ICONS: Record<MCStepType, LucideIcon> = {
  simulation: DatabaseZap,
  filter: Filter,
  wald: Sigma,
  ljung_box: Activity,
  jarque_bera: Activity,
  breusch_pagan: Activity,
  breusch_godfrey: Activity,
  cusum: Activity,
  cusumsq: Activity,
  chow: Activity,
  regression: TestTubeDiagonal,
  standardize: Waves,
  log: Waves,
  log_diff: Waves,
  diff: Waves,
  rolling_mean: Waves,
  rolling_std: Waves,
  rolling_var: Waves,
  payload: DatabaseZap,
  kde: Activity,
  "transform:custom": Code,
  "postproc:custom": Code,
};

export function StepNode({ data, selected }: NodeProps<MCFlowNode>) {
  const { step } = data;
  const definition = stepDefinition(step.step_type);
  const Icon = ICONS[step.step_type] ?? Activity;
  const summary = summarizeKwargs(step.kwargs);
  // Postproc ops are a terminal phase referenced by trace key, never wired into
  // the DAG (see isValidConnection). Render no handles so the UI offers no
  // grabbable connection points at all.
  const postproc = definition?.category === "postproc";
  const terminal = [
    "wald",
    "ljung_box",
    "jarque_bera",
    "breusch_pagan",
    "breusch_godfrey",
    "cusum",
    "cusumsq",
    "chow",
    "regression",
  ].includes(step.step_type);
  return (
    <div className={`mc-step-node ${step.step_type}${selected ? " selected" : ""}`}>
      {!postproc && step.step_type !== "simulation" && (
        <Handle type="target" position={Position.Left} />
      )}
      <div className="mc-step-node-heading">
        <Icon size={15} />
        <span>{definition?.title ?? step.step_type}</span>
      </div>
      <strong>{step.name}</strong>
      <span className="mc-step-node-summary">{summary}</span>
      {!postproc && !terminal && <Handle type="source" position={Position.Right} />}
    </div>
  );
}

// The first couple of kwargs, as a hint of how the step is configured. Its
// sources are the edges drawn into it, so they are not repeated here.
function summarizeKwargs(kwargs: Record<string, unknown>): string {
  const values = Object.entries(kwargs)
    .filter(([, value]) => value !== "" && value !== null && value !== undefined)
    .slice(0, 2)
    .map(([key, value]) => `${key}: ${formatValue(value)}`);
  return values.length > 0 ? values.join(" / ") : "Configure step";
}

function formatValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}
