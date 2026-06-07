import type { Node } from "@xyflow/react";
import type { MCStepCatalogItem, MCStepType } from "../types";

export interface MCNodeData extends Record<string, unknown> {
  stepType: MCStepType;
  name: string;
  params: Record<string, unknown>;
  catalog: MCStepCatalogItem;
}

export type MCFlowNode = Node<MCNodeData, "mcStep">;
