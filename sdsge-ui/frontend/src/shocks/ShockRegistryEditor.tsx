// The shock panel, shared by every place a simulation's shocks are configured.
//
// The checklist is the target model's declared innovations (offered as options,
// never assumed); one entry is a free-form shock over the chosen subset. The
// kind is a property of the entry, not of the panel: an entry either draws from
// a family or supplies an array, which is the same split the spec carries and
// the same one `registryFromShocks` reads back. Client-side validation blocks an
// innovation already claimed by another entry and a joint uniform shock. Path
// length is not checked here; only the run knows `T`.

import { Check, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import type {
  Role,
  ShockDistribution,
  ShockRegistryEntry,
} from "../types";
import { formatPathText, parsePathText } from "./registry";

const DIST_LABEL: Record<ShockDistribution, string> = {
  norm: "Normal",
  t: "Student-t",
  uni: "Uniform",
};

type EntryKind = ShockRegistryEntry["kind"];

export function ShockRegistryEditor({
  role,
  shockNames,
  entries,
  onChange,
}: {
  role: Role;
  shockNames: string[];
  entries: ShockRegistryEntry[];
  onChange: (entries: ShockRegistryEntry[]) => void;
}) {
  const [kind, setKind] = useState<EntryKind>("drawn");
  const [selected, setSelected] = useState<string[]>([]);
  const [dist, setDist] = useState<ShockDistribution>("norm");
  // Keyed by innovation, not positional: `commitEntry` reorders the selection
  // into the model's own order, and a positional list would hand each location
  // to the wrong shock.
  const [locs, setLocs] = useState<Record<string, string>>({});
  const [df, setDf] = useState("5");
  const [seed, setSeed] = useState("");
  // The path as typed. It lives as text only while the form is open; an entry
  // holds the parsed matrix, which is what a grid editor would produce too.
  const [pathText, setPathText] = useState("");
  const [error, setError] = useState("");
  // Index of the entry being edited (null while composing a fresh one). Indices
  // are the spec's own, since both kinds are entries. The entry under edit is
  // excluded from the "used" set so its own innovations stay selectable, and
  // Save replaces it in place instead of appending.
  const [editIndex, setEditIndex] = useState<number | null>(null);

  // Every entry claims its innovations, whichever kind it is.
  const usedNames = new Set(
    entries.flatMap((entry, index) => (index === editIndex ? [] : entry.target)),
  );
  const locFor = (name: string) => locs[name] ?? "0";
  const multivarUni = kind === "drawn" && dist === "uni" && selected.length > 1;
  const multivarJoint = kind === "drawn" && dist !== "uni" && selected.length > 1;

  function resetForm() {
    setEditIndex(null);
    setKind("drawn");
    setSelected([]);
    setDist("norm");
    setLocs({});
    setDf("5");
    setSeed("");
    setPathText("");
    setError("");
  }

  function toggleVar(name: string) {
    setError("");
    setSelected((current) =>
      current.includes(name)
        ? current.filter((item) => item !== name)
        : [...current, name],
    );
  }

  // Load an existing entry back into the form for editing.
  function startEdit(index: number) {
    const entry = entries[index];
    setEditIndex(index);
    setKind(entry.kind);
    setSelected(entry.target);
    setError("");
    if (entry.kind === "path") {
      setPathText(formatPathText(entry.path));
      return;
    }
    setDist(entry.dist);
    setLocs(
      Object.fromEntries(
        entry.target.map((name, position) => [name, String(entry.loc[position] ?? 0)]),
      ),
    );
    setDf(String(entry.df));
    setSeed(entry.seed === null ? "" : String(entry.seed));
  }

  // Build the entry this form describes, or report why it cannot be built. The
  // compiler in `shocksFromRegistry` refuses the same things, so this is the
  // early half of one rule rather than a second one.
  function entryFromForm(target: string[]): ShockRegistryEntry | null {
    if (kind === "path") {
      const path = parsePathText(pathText);
      if (path.length === 0) {
        setError("A supplied path needs at least one period.");
        return null;
      }
      const ragged = path.findIndex((row) => row.length !== target.length);
      if (ragged !== -1) {
        setError(
          `Row ${ragged + 1} has ${path[ragged].length} value(s); ` +
            `${target.length} selected.`,
        );
        return null;
      }
      if (path.some((row) => row.some((value) => !Number.isFinite(value)))) {
        setError("The path contains a value that is not a number.");
        return null;
      }
      return { kind: "path", target, path };
    }
    if (dist === "uni" && target.length > 1) {
      setError("A uniform shock is univariate; select exactly one innovation.");
      return null;
    }
    // One location per innovation, in the selection's order. An untouched box
    // is the form's own default of zero; anything else that is not a number is
    // an error rather than a silent zero.
    const loc: number[] = [];
    for (const name of target) {
      const raw = locFor(name).trim();
      const value = raw === "" ? 0 : Number(raw);
      if (!Number.isFinite(value)) {
        setError(`Location for '${name}' is not a number.`);
        return null;
      }
      loc.push(value);
    }
    return {
      kind: "drawn",
      target,
      dist,
      loc,
      df: Number(df) || 5,
      seed: seed.trim() === "" ? null : Number(seed),
    };
  }

  function commitEntry() {
    if (selected.length === 0) {
      setError("Select at least one innovation.");
      return;
    }
    const clash = selected.find((name) => usedNames.has(name));
    if (clash !== undefined) {
      setError(`'${clash}' is already used in another shock entry.`);
      return;
    }
    // Order the selection by the model's own shock order for a stable identity.
    const target = shockNames.filter((name) => selected.includes(name));
    const entry = entryFromForm(target);
    if (entry === null) return;
    onChange(
      editIndex === null
        ? [...entries, entry]
        : entries.map((current, index) => (index === editIndex ? entry : current)),
    );
    resetForm();
  }

  function removeEntry(index: number) {
    onChange(entries.filter((_, position) => position !== index));
    // Keep the edit target consistent with the shrunk list.
    if (editIndex === index) resetForm();
    else if (editIndex !== null && index < editIndex) setEditIndex(editIndex - 1);
  }

  return (
    <div className="mc-shock-registry">
      <div className="mc-shock-registry-head">
        <span className="mc-shock-registry-label">Shocks</span>
        <span className="mc-shock-registry-target">from {role}</span>
      </div>
      {entries.length > 0 ? (
        <ul className="mc-shock-list">
          {entries.map((entry, index) => (
            <li
              key={`${entry.kind}:${index}`}
              className={`mc-shock-entry${editIndex === index ? " editing" : ""}${
                entry.kind === "path" ? " mc-shock-entry-path" : ""
              }`}
            >
              <button
                className="mc-shock-entry-select"
                title="Edit shock"
                disabled={shockNames.length === 0}
                onClick={() => startEdit(index)}
              >
                <div className="mc-shock-entry-body">
                  <strong>
                    {entry.target.length > 0 ? entry.target.join(", ") : "(unnamed)"}
                    {entry.target.length > 1 && (
                      <span className="mc-shock-badge">joint</span>
                    )}
                    {entry.kind === "path" && (
                      <span className="mc-shock-badge">path</span>
                    )}
                  </strong>
                  <span>{describeEntry(entry)}</span>
                </div>
              </button>
              <button
                className="icon-button"
                title="Remove shock"
                onClick={() => removeEntry(index)}
              >
                <Trash2 size={13} />
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mc-shock-empty">
          No shocks configured; this simulation runs deterministically.
        </p>
      )}
      {shockNames.length === 0 ? (
        <p className="mc-shock-empty">
          Load and solve the {role} model to choose its declared shocks.
        </p>
      ) : (
        <div className="mc-shock-form">
          <div className="mc-shock-checklist">
            {shockNames.map((name) => {
              const used = usedNames.has(name);
              return (
                <label
                  key={name}
                  className={`mc-shock-check${used ? " used" : ""}`}
                  title={used ? "Already used in another shock entry." : undefined}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(name)}
                    disabled={used}
                    onChange={() => toggleVar(name)}
                  />
                  <span>{name}</span>
                </label>
              );
            })}
          </div>
          <div className="mc-shock-fields">
            <label>
              Source
              <select
                value={kind}
                onChange={(event) => {
                  setKind(event.target.value as EntryKind);
                  setError("");
                }}
              >
                <option value="drawn">Drawn</option>
                <option value="path">Supplied path</option>
              </select>
            </label>
            {kind === "drawn" && (
              <>
                <label>
                  Distribution
                  <select
                    value={dist}
                    onChange={(event) => {
                      setDist(event.target.value as ShockDistribution);
                      setError("");
                    }}
                  >
                    <option value="norm">Normal</option>
                    <option value="t">Student-t</option>
                    <option value="uni">Uniform</option>
                  </select>
                </label>
                {shockNames
                  .filter((name) => selected.includes(name))
                  .map((name) => (
                    <label key={`loc:${name}`}>
                      {selected.length > 1 ? `Location (${name})` : "Location"}
                      <input
                        type="number"
                        value={locFor(name)}
                        onChange={(event) =>
                          setLocs((current) => ({
                            ...current,
                            [name]: event.target.value,
                          }))
                        }
                      />
                    </label>
                  ))}
                {dist === "t" && (
                  <label>
                    Degrees of freedom
                    <input
                      type="number"
                      value={df}
                      onChange={(event) => setDf(event.target.value)}
                    />
                  </label>
                )}
                <label>
                  Seed
                  <input
                    type="number"
                    placeholder="none"
                    value={seed}
                    onChange={(event) => setSeed(event.target.value)}
                  />
                </label>
              </>
            )}
          </div>
          {kind === "path" && (
            <label className="mc-shock-path">
              Path
              <textarea
                className="shock-input"
                value={pathText}
                placeholder={selected.length > 1 ? "0 0\n1 0\n0 0" : "0\n1\n0"}
                onChange={(event) => {
                  setPathText(event.target.value);
                  setError("");
                }}
              />
            </label>
          )}
          {kind === "path" && (
            <span className="mc-shock-hint">
              One period per line, one value per selected innovation. The run
              refuses a path that does not cover its horizon.
            </span>
          )}
          {multivarUni && (
            <span className="mc-shock-hint">
              A uniform shock is univariate; select exactly one innovation.
            </span>
          )}
          {multivarJoint && (
            <span className="mc-shock-hint">
              This is one joint (multivar) shock over {selected.length}{" "}
              innovations. Add a separate entry per innovation if you want them
              independent.
            </span>
          )}
          {error !== "" && <span className="status error mc-shock-error">{error}</span>}
          <div className="mc-shock-actions">
            <button
              className="secondary mc-shock-add"
              onClick={commitEntry}
              disabled={selected.length === 0}
            >
              {editIndex === null ? <Plus size={13} /> : <Check size={13} />}
              {editIndex === null ? "Add shock" : "Save shock"}
            </button>
            {editIndex !== null && (
              <button className="secondary" onClick={resetForm}>
                Cancel
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function describeEntry(entry: ShockRegistryEntry): string {
  if (entry.kind === "path") {
    const periods = entry.path.length;
    return `Supplied over ${periods} ${periods === 1 ? "period" : "periods"}`;
  }
  const parts = [DIST_LABEL[entry.dist], `loc ${entry.loc.join(", ")}`];
  if (entry.dist === "t") parts.push(`df ${entry.df}`);
  parts.push(entry.seed === null ? "seed none" : `seed ${entry.seed}`);
  return parts.join(", ");
}
