// The shock registry as the list a step carries.
//
// A form collects what is convenient to render, and a registry entry over
// several variables is one joint shock. The rest of the form maps onto a step
// field by field, which `mc/fields` does; this is the one widget whose shape
// differs enough from its kwarg to need a pass of its own.

import type { ShockRegistryEntry } from "../types";

interface SerializedShock {
  // The shocks this entry drives. An entry names one or more of them, which a
  // JSON object cannot be keyed by, so each entry carries its own targets and a
  // spec travels as a list.
  target: string[];
  dist: string;
  seed: number | null;
  dist_kwargs: Record<string, unknown>;
}

// One registry entry becomes one shock, joint when it selects more than one
// variable. Uniform is univariate only, so a `uni` entry takes exactly one.
//
// `loc` travels as the vector the entry holds. The library resolves `mean` and
// `loc` identically at every width and requires one value per target, so there
// is no spelling to pick and nothing to broadcast: a short vector is incomplete
// input and is refused here rather than filled in.
function shockFor(entry: ShockRegistryEntry): SerializedShock {
  const vars = entry.vars.map(String);
  const n = vars.length;
  const loc = (entry.loc ?? []).map(Number);
  const df = Number(entry.df ?? 5);
  if (entry.dist === "uni" && n > 1) {
    throw new Error(
      "A 'uni' shock is univariate; select exactly one variable per uniform " +
        "entry (use separate entries for independent uniform shocks).",
    );
  }
  if (loc.length !== n) {
    throw new Error(
      `Shock entry '${vars.join(", ")}' needs one location per variable; ` +
        `got ${loc.length} for ${n}.`,
    );
  }
  if (loc.some((value) => !Number.isFinite(value))) {
    throw new Error(
      `Shock entry '${vars.join(", ")}' has a non-numeric location.`,
    );
  }
  if (entry.dist !== "norm" && entry.dist !== "t" && entry.dist !== "uni") {
    throw new Error(`Unsupported shock distribution: ${String(entry.dist)}`);
  }
  const distKwargs: Record<string, unknown> =
    entry.dist === "t" ? { loc, df } : { loc };
  return {
    target: vars,
    dist: entry.dist,
    seed: entry.seed ?? null,
    dist_kwargs: distKwargs,
  };
}

export function shocksFromRegistry(
  registry: ShockRegistryEntry[],
): SerializedShock[] | null {
  if (registry.length === 0) return null;
  const shocks: SerializedShock[] = [];
  const seen = new Set<string>();
  for (const entry of registry) {
    const vars = entry.vars.map(String);
    if (vars.length === 0) {
      throw new Error(
        "Each shock registry entry must select at least one variable.",
      );
    }
    // A list cannot deduplicate its own entries the way an object key did, so
    // the same selection twice is caught here.
    const seenKey = vars.join("\u0000");
    if (seen.has(seenKey)) {
      throw new Error(
        `Duplicate shock entry for '${vars.join(", ")}' in the registry.`,
      );
    }
    seen.add(seenKey);
    shocks.push(shockFor(entry));
  }
  return shocks;
}
