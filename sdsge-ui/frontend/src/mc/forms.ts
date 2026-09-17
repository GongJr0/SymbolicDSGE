// Adapters from what the widgets produce to what a spec node carries.
//
// A form collects what is convenient to render: a shock registry rather than a
// shock mapping, the wald target under whichever of two conditional fields the
// selected moment uses, a bandwidth typed into one box. A node carries the
// resolved values. Nothing here validates: every value is checked again during
// lowering, the last point before the native kernels.

import type { ShockRegistryEntry } from "../types";

const BANDWIDTH_KEYWORDS = new Set(["andrews", "wooldridge", "auto"]);

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

function bandwidth(value: unknown): number | string {
  if (typeof value === "number") return value;
  const text = String(value ?? "").trim().toLowerCase();
  if (BANDWIDTH_KEYWORDS.has(text)) return text;
  const parsed = Number(text);
  if (!Number.isFinite(parsed)) {
    throw new Error(
      `bandwidth must be an integer or one of: ${[...BANDWIDTH_KEYWORDS]
        .sort()
        .join(", ")}.`,
    );
  }
  return parsed;
}

// Collapse the two conditional target fields onto the single parameter the step
// carries, and read a typed bandwidth out of its text box.
function compileWald(params: Record<string, unknown>): Record<string, unknown> {
  const out = { ...params };
  if ("target_vector" in out) {
    out.target = out.target_vector;
    delete out.target_vector;
    delete out.target_matrix;
  } else if ("target_matrix" in out) {
    out.target = out.target_matrix;
    delete out.target_matrix;
  }
  if ("bandwidth" in out) out.bandwidth = bandwidth(out.bandwidth);
  return out;
}

function compileSimulation(
  params: Record<string, unknown>,
): Record<string, unknown> {
  const out = { ...params };
  const registry = out.shock_registry;
  delete out.shock_registry;
  if (out.shocks === undefined || out.shocks === null) {
    out.shocks = Array.isArray(registry)
      ? shocksFromRegistry(registry as ShockRegistryEntry[])
      : null;
  }
  return out;
}

const COMPILERS: Record<
  string,
  (params: Record<string, unknown>) => Record<string, unknown>
> = { wald: compileWald, simulation: compileSimulation };

export function compileFormParams(
  stepType: string,
  params: Record<string, unknown>,
): Record<string, unknown> {
  const compiler = COMPILERS[stepType];
  return compiler ? compiler(params) : { ...params };
}
