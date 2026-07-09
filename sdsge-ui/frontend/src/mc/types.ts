import type { Node } from "@xyflow/react";
import type { MCStepCatalogItem, MCStepType } from "../types";

export interface MCNodeData extends Record<string, unknown> {
  stepType: MCStepType;
  name: string;
  params: Record<string, unknown>;
  catalog: MCStepCatalogItem;
}

export type MCFlowNode = Node<MCNodeData, "mcStep">;

// A step a source leg may read from. `kind` picks the channels the consumer can
// select: datagen -> states/observables, filter -> filter channels, transform ->
// payload (mirrors the backend's producer op-type / field compatibility).
export interface MCProducer {
  name: string;
  kind: "datagen" | "filter" | "transform";
}
