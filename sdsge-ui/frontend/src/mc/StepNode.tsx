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
  kde: Activity,
  "transform:custom": Code,
  "postproc:custom": Code,
};

export function StepNode({ data, selected }: NodeProps<MCFlowNode>) {
  const Icon = ICONS[data.stepType] ?? Activity;
  const summary = summarizeParams(data.params);
  // Postproc ops are a terminal phase referenced by trace key, never wired into
  // the DAG (see isValidConnection). Render no handles so the UI offers no
  // grabbable connection points at all.
  const postproc = data.catalog.category === "postproc";
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
  ].includes(data.stepType);
  return (
    <div className={`mc-step-node ${data.stepType}${selected ? " selected" : ""}`}>
      {!postproc && data.stepType !== "simulation" && (
        <Handle type="target" position={Position.Left} />
      )}
      <div className="mc-step-node-heading">
        <Icon size={15} />
        <span>{data.catalog.title}</span>
      </div>
      <strong>{data.name}</strong>
      <span className="mc-step-node-summary">{summary}</span>
      {!postproc && !terminal && <Handle type="source" position={Position.Right} />}
    </div>
  );
}

function summarizeParams(params: Record<string, unknown>): string {
  const values = Object.entries(params)
    .filter(([, value]) => value !== "" && value !== null && value !== undefined)
    .slice(0, 2)
    .map(([key, value]) => `${key}: ${formatValue(value)}`);
  return values.length > 0 ? values.join(" / ") : "Configure step";
}

function formatValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}
