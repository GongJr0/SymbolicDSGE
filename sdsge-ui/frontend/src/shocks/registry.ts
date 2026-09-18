// A shock spec as the panel reads and writes it.
//
// The library takes a simulation's shocks as a list of entries, one per shock
// family, because an entry may name several innovations at once and a JSON
// object cannot be keyed by a group. Both places that configure a simulation
// (the Outputs tab and the Monte Carlo simulation step) speak that list, so the
// conversion between it and the panel's form lives here rather than in either.
//
// Nothing here validates a path against the simulation's horizon: the length a
// path must have is `T`, which belongs to the run rather than to the registry,
// so the sim call is what refuses a mismatch.

import type {
  DrawnRegistryEntry,
  DrawnShock,
  PathRegistryEntry,
  PathShock,
  ShockDistribution,
  ShockEntry,
  ShockRegistryEntry,
} from "../types";

// One registry entry becomes one shock, joint when it selects more than one
// innovation. Uniform is univariate only, so a `uni` entry takes exactly one.
//
// `loc` travels as the vector the entry holds. The library resolves `mean` and
// `loc` identically at every width and requires one value per target, so there
// is no spelling to pick and nothing to broadcast: a short vector is incomplete
// input and is refused here rather than filled in.
function shockFor(entry: DrawnRegistryEntry): DrawnShock {
  const target = entry.target.map(String);
  const n = target.length;
  const loc = (entry.loc ?? []).map(Number);
  const df = Number(entry.df ?? 5);
  if (entry.dist === "uni" && n > 1) {
    throw new Error(
      "A 'uni' shock is univariate; select exactly one innovation per uniform " +
        "entry (use separate entries for independent uniform shocks).",
    );
  }
  if (loc.length !== n) {
    throw new Error(
      `Shock entry '${target.join(", ")}' needs one location per innovation; ` +
        `got ${loc.length} for ${n}.`,
    );
  }
  if (loc.some((value) => !Number.isFinite(value))) {
    throw new Error(`Shock entry '${target.join(", ")}' has a non-numeric location.`);
  }
  if (entry.dist !== "norm" && entry.dist !== "t" && entry.dist !== "uni") {
    throw new Error(`Unsupported shock distribution: ${String(entry.dist)}`);
  }
  const distKwargs: Record<string, unknown> =
    entry.dist === "t" ? { loc, df } : { loc };
  return {
    target,
    dist: entry.dist,
    seed: entry.seed ?? null,
    dist_kwargs: distKwargs,
  };
}

// A supplied path travels as it stands. Its width is what pairs a column with
// an innovation, so a row that disagrees with the selection would drive the
// wrong one and is refused rather than padded.
function pathFor(entry: PathRegistryEntry): PathShock {
  const target = entry.target.map(String);
  const label = target.join(", ");
  if (entry.path.length === 0) {
    throw new Error(`Shock path for '${label}' covers no periods.`);
  }
  const ragged = entry.path.findIndex((row) => row.length !== target.length);
  if (ragged !== -1) {
    throw new Error(
      `Shock path for '${label}' needs one value per innovation; row ` +
        `${ragged + 1} has ${entry.path[ragged].length} for ${target.length}.`,
    );
  }
  if (entry.path.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error(`Shock path for '${label}' has a non-numeric value.`);
  }
  return { target, path: entry.path.map((row) => row.map(Number)) };
}

export function shocksFromRegistry(
  registry: ShockRegistryEntry[],
): ShockEntry[] | null {
  if (registry.length === 0) return null;
  const shocks: ShockEntry[] = [];
  const seen = new Set<string>();
  for (const entry of registry) {
    const target = entry.target.map(String);
    if (target.length === 0) {
      throw new Error(
        "Each shock registry entry must select at least one innovation.",
      );
    }
    // A list cannot deduplicate its own entries the way an object key did, so
    // the same selection twice is caught here.
    const seenKey = target.join(" ");
    if (seen.has(seenKey)) {
      throw new Error(
        `Duplicate shock entry for '${target.join(", ")}' in the registry.`,
      );
    }
    seen.add(seenKey);
    shocks.push(entry.kind === "path" ? pathFor(entry) : shockFor(entry));
  }
  return shocks;
}

// The registry a spec's shocks describe, in spec order. Both kinds become
// entries the panel renders and edits, so an index into this list is an index
// into the spec and there is no second numbering to keep aligned.
export function registryFromShocks(shocks: unknown): ShockRegistryEntry[] {
  if (!Array.isArray(shocks)) return [];
  return shocks.map((value) =>
    entryFromSpec((value ?? {}) as Record<string, unknown>),
  );
}

// Invert the pair above: recover the form entry from a serialized spec entry.
// The kinds are told apart by what the entry declares, since `path` is present
// on one and absent on the other, which is the test the library applies too.
function entryFromSpec(dict: Record<string, unknown>): ShockRegistryEntry {
  const target = Array.isArray(dict.target) ? dict.target.map(String) : [];
  if ("path" in dict) {
    return { kind: "path", target, path: pathMatrix(dict.path) };
  }
  const dist = asDist(dict.dist);
  const kwargs = (dict.dist_kwargs ?? {}) as Record<string, unknown>;
  // The library reads `mean` and `loc` identically and prefers `mean` when a
  // spec carries both, so this reads them in that order and keeps the whole
  // vector. A bundle-authored entry may name a different location per
  // innovation, and collapsing that to one number would silently rewrite the
  // spec on the next edit.
  const declared = "mean" in kwargs ? kwargs.mean : kwargs.loc;
  return {
    kind: "drawn",
    target,
    dist,
    loc: locVector(declared, target.length),
    df: dist === "t" ? Number(kwargs.df ?? 5) : 5,
    seed: dict.seed === null || dict.seed === undefined ? null : Number(dict.seed),
  };
}

// A serialized path as rows. The spec's shape is (T, width), so a flat list is
// read as the width-1 column it describes rather than as a single period.
function pathMatrix(value: unknown): number[][] {
  if (!Array.isArray(value)) return [];
  return value.map((row) => (Array.isArray(row) ? row.map(Number) : [Number(row)]));
}

// A declared location as one value per target. An omitted location takes the
// library's own default of zeros; a scalar is a width-1 vector and nothing
// more. A length that disagrees with the target count is incomplete input and
// is carried through as given, for `shockFor` to refuse at compile time.
function locVector(declared: unknown, width: number): number[] {
  if (declared === undefined || declared === null) {
    return Array.from({ length: width }, () => 0);
  }
  if (Array.isArray(declared)) return declared.map(Number);
  return [Number(declared)];
}

function asDist(value: unknown): ShockDistribution {
  return value === "t" || value === "uni" ? value : "norm";
}

// One period per line, one value per innovation within it, separated by any
// whitespace, comma, or semicolon. A tab-separated block pasted out of a
// spreadsheet is already in this form, which is the point: the textarea stands
// in for a real grid, and the parse should not have to change when one arrives.
export function parsePathText(text: string): number[][] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== "")
    .map((line) => line.split(/[\s,;]+/).filter(Boolean).map(Number));
}

/** The inverse of `parsePathText`, for loading an entry back into the form. */
export function formatPathText(path: number[][]): string {
  return path.map((row) => row.join(" ")).join("\n");
}
