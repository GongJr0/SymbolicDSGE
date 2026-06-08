import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import {
  Activity,
  DatabaseZap,
  Filter,
  Sigma,
  TestTubeDiagonal,
} from "lucide-react";
import type { MCFlowNode } from "./types";

const ICONS = {
  simulation: DatabaseZap,
  filter: Filter,
  wald: Sigma,
  ljung_box: Activity,
  jarque_bera: Activity,
  regression: TestTubeDiagonal,
};

export function StepNode({ data, selected }: NodeProps<MCFlowNode>) {
  const Icon = ICONS[data.stepType];
  const summary = summarizeParams(data.params);
  const terminal = ["wald", "ljung_box", "jarque_bera", "regression"].includes(
    data.stepType,
  );
  return (
    <div className={`mc-step-node ${data.stepType}${selected ? " selected" : ""}`}>
      {data.stepType !== "simulation" && (
        <Handle type="target" position={Position.Left} />
      )}
      <div className="mc-step-node-heading">
        <Icon size={15} />
        <span>{data.catalog.title}</span>
      </div>
      <strong>{data.name}</strong>
      <span className="mc-step-node-summary">{summary}</span>
      {!terminal && <Handle type="source" position={Position.Right} />}
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
