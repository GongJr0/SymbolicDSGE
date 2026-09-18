import type { Node } from "@xyflow/react";
import type { MCStepSpec } from "../types";

/** What a canvas node carries: the step it is.
 *
 * Nothing sits beside it. The kind and the name are the step's own fields, the
 * catalogue entry is looked up from the kind, and every authored value has a
 * home on the step already, which `mc/fields` maps a form key onto.
 */
export interface MCNodeData extends Record<string, unknown> {
  step: MCStepSpec;
}

export type MCFlowNode = Node<MCNodeData, "mcStep">;

// A step a source leg may read from. `kind` picks the channels the consumer can
// select: datagen -> states/observables, filter -> filter channels, transform ->
// payload (mirrors the backend's producer op-type / field compatibility).
export interface MCProducer {
  name: string;
  kind: "datagen" | "filter" | "transform";
}
