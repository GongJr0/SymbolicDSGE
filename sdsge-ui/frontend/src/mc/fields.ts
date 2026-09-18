// Reading and writing one form field on a step.
//
// A step already holds every authored value: a leg's binding in `source_args`,
// the window shared across them, `n_retain` and `code` in slots of their own,
// and everything else a kwarg. The inspector edits one field at a time, so
// these two functions are the whole map between a field key and its home, and
// no flat params bag has to exist beside the step.
//
// Nothing here validates. A value that will not do is refused during lowering,
// the last point before the native kernels, which is also where a half-typed
// one belongs while the user is still typing it.

import type { MCStepSpec } from "../types";
import type { MCStepDefinition } from "./catalog";
import { asColumns } from "./catalog";

const BANDWIDTH_KEYWORDS = new Set(["andrews", "wooldridge", "auto"]);

// The wald moment selects which of two target widgets renders, and both write
// the single kwarg the step carries. Only the visible one is ever read back.
const TARGET_KEYS = new Set(["target_vector", "target_matrix"]);

interface LegSlot {
  index: number;
  role: "source_step" | "field" | "columns";
}

// Which leg a form key belongs to, by the spelling the catalogue gave it.
function legSlot(definition: MCStepDefinition, key: string): LegSlot | null {
  for (const [index, leg] of definition.legs.entries()) {
    if (key === leg.sourceKey) return { index, role: "source_step" };
    if (key === leg.fieldKey) return { index, role: "field" };
    if (key === leg.columnsKey) return { index, role: "columns" };
  }
  return null;
}

// The box holds text; a step carries a number or one of the keywords. Anything
// else is kept as typed: the user may be partway through a word, and the
// library is what refuses a bandwidth it cannot read.
function bandwidth(value: unknown): unknown {
  if (typeof value === "number") return value;
  const text = String(value ?? "").trim().toLowerCase();
  if (BANDWIDTH_KEYWORDS.has(text)) return text;
  const parsed = Number(text);
  return text !== "" && Number.isFinite(parsed) ? parsed : value;
}

export function getField(
  definition: MCStepDefinition,
  step: MCStepSpec,
  key: string,
): unknown {
  if (key === "n_retain") return step.n_retain;
  if (key === "code") return step.code ?? "";
  // One window per step, so every leg recorded the same value.
  if (key === "burn_in") return step.source_args[0]?.burn_in ?? 0;
  if (TARGET_KEYS.has(key)) return step.kwargs.target;
  const slot = legSlot(definition, key);
  if (slot === null) return step.kwargs[key];
  const source = step.source_args[slot.index];
  if (source === undefined) return undefined;
  return slot.role === "columns" ? (source.columns ?? []) : source[slot.role];
}

export function setField(
  definition: MCStepDefinition,
  step: MCStepSpec,
  key: string,
  value: unknown,
): MCStepSpec {
  if (key === "n_retain") return { ...step, n_retain: Number(value ?? -1) };
  if (key === "code") return { ...step, code: String(value ?? "") };
  if (key === "burn_in") {
    const burnIn = Number(value ?? 0) || 0;
    return {
      ...step,
      source_args: step.source_args.map((source) => ({
        ...source,
        burn_in: burnIn,
      })),
    };
  }
  const slot = legSlot(definition, key);
  if (slot === null) {
    const name = TARGET_KEYS.has(key) ? "target" : key;
    return {
      ...step,
      kwargs: {
        ...step.kwargs,
        [name]: name === "bandwidth" ? bandwidth(value) : value,
      },
    };
  }
  return {
    ...step,
    source_args: step.source_args.map((source, index) =>
      index !== slot.index
        ? source
        : slot.role === "columns"
          ? { ...source, columns: asColumns(value) }
          : { ...source, [slot.role]: String(value ?? "") },
    ),
  };
}

// A step as the catalogue's defaults describe it, which is what a node dropped
// on the canvas starts as. Every field goes through `setField`, so a new step
// and an edited one are built the same way.
export function defaultStep(
  definition: MCStepDefinition,
  name: string,
  code: string,
): MCStepSpec {
  const empty: MCStepSpec = {
    name,
    op_type: definition.opType,
    step_type: definition.step_type,
    kwargs: {},
    source_args: definition.legs.map((leg) => ({
      arg: leg.arg,
      source_step: "",
      field: "",
      columns: null,
      burn_in: 0,
    })),
    n_retain: -1,
  };
  const step = definition.fields.reduce(
    // The shock registry is a widget over `kwargs.shocks`, and an unconfigured
    // step carries no shocks at all: the server reads the key as a list when it
    // is present, so posting a null one is worse than omitting it.
    (current, field) =>
      field.type === "shock_registry"
        ? current
        : setField(definition, current, field.key, field.default),
    empty,
  );
  return code === "" ? step : { ...step, code };
}
