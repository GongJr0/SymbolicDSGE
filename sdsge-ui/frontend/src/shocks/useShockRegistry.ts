// The editing state around a simulation's shock list.
//
// Thin on purpose: the spec is the state. This holds the registry the panel
// renders, compiles edits back into the spec, and keeps the message from an
// edit the compiler refused, which is the one thing with nowhere else to live
// because a rejected edit never reaches the spec.

import { useState } from "react";
import type { ShockEntry, ShockRegistryEntry } from "../types";
import { registryFromShocks, shocksFromRegistry } from "./registry";

export interface ShockRegistryState {
  entries: ShockRegistryEntry[];
  /** The last rejected entry set's message, empty while the list compiles. */
  error: string;
  setRegistry: (entries: ShockRegistryEntry[]) => void;
}

/** Drive one shock list. */
export function useShockRegistry(
  shocks: unknown,
  onChange: (shocks: ShockEntry[] | null) => void,
): ShockRegistryState {
  const [error, setError] = useState("");

  // The registry compiles here rather than on the way out: the editor commits
  // an entry only once it is complete, so this is the first moment the entries
  // can become the list the spec holds. It replaces any serialized one, which
  // is what the user is editing away from.
  const setRegistry = (entries: ShockRegistryEntry[]) => {
    let compiled: ShockEntry[] | null;
    try {
      compiled = shocksFromRegistry(entries);
    } catch (problem: unknown) {
      setError(problem instanceof Error ? problem.message : String(problem));
      return;
    }
    setError("");
    onChange(compiled);
  };

  return { entries: registryFromShocks(shocks), error, setRegistry };
}
