import Editor from "@monaco-editor/react";
import type * as Monaco from "monaco-editor";
import { Check, Plus, TriangleAlert, Trash2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import { validateCustomOp } from "../api";
import { registerPythonLsp } from "../lsp/registerPythonLsp";
import type {
  MCFieldSpec,
  MCStepSpec,
  MCStepType,
  Role,
  ShockDistribution,
  ShockRegistryEntry,
} from "../types";
import { stepDefinition } from "./catalog";
import { getField, setField } from "./fields";
import { shocksFromRegistry } from "./forms";
import type { MCFlowNode, MCProducer } from "./types";

// The two data-step channels; every other catalog source-field option is a
// filter channel. A transform producer instead exposes a single "payload".
const DATA_CHANNELS = ["states", "observables"];

// A binding's source field key: `source` or a `<leg>_source`. Its channel
// select follows immediately in the catalog (`field` / `<leg>_field`), then an
// optional columns list.
function isSourceKey(key: string): boolean {
  return key === "source" || key.endsWith("_source");
}

// The channels a consumer may read from a producer of the given kind. Driven by
// the catalog's own source-field options (INPUT_SOURCES) so the filter channel
// set can't drift, with "payload" added for transform producers.
function channelOptionsFor(
  kind: MCProducer["kind"] | undefined,
  catalogOptions: string[],
): string[] {
  if (kind === "datagen") {
    return catalogOptions.filter((option) => DATA_CHANNELS.includes(option));
  }
  if (kind === "filter") {
    return catalogOptions.filter((option) => !DATA_CHANNELS.includes(option));
  }
  if (kind === "transform") return ["payload"];
  // No producer selected yet: offer everything so the field stays editable.
  return [...catalogOptions, "payload"];
}

export function StepInspector({
  node,
  onChange,
  onDelete,
  theme,
  producers,
  availableTraces,
  exogByRole,
}: {
  node: MCFlowNode | null;
  onChange: (node: MCFlowNode) => void;
  onDelete: (id: string) => void;
  theme: "light" | "dark";
  producers: MCProducer[];
  availableTraces: string[];
  exogByRole: Record<Role, string[]>;
}) {
  // Discarded shock paths, keyed by step id so switching steps cannot offer one
  // step's paths back to another. Session-lived and deliberately not on the
  // node: a restored path belongs in the compiled spec, an undo buffer does
  // not. Declared above the early return, since a hook cannot be conditional.
  const [discardedPaths, setDiscardedPaths] = useState<Record<string, unknown[]>>(
    {},
  );
  // The registry now compiles as it is edited, so a rejected entry set has to
  // say so here; before, the same message surfaced only when the pipeline was
  // posted. Declared above the early return, since a hook cannot be conditional.
  const [registryError, setRegistryError] = useState("");

  if (node === null) {
    return (
      <div className="mc-empty">
        <span>Select a step to edit its configuration.</span>
      </div>
    );
  }

  const stashedPaths = discardedPaths[node.id] ?? [];
  const step = node.data.step;
  const definition = stepDefinition(step.step_type);

  // Every edit replaces the node's step; `mc/fields` is what knows whether a
  // form key is a leg, a slot of the step's own, or a kwarg.
  const putStep = (next: MCStepSpec) =>
    onChange({ ...node, data: { ...node.data, step: next } });

  const isPathEntry = (entry: unknown) =>
    entry !== null && typeof entry === "object" && "path" in entry;

  // The undo expires on the first registry edit. Anything the user adds over a
  // variable the discarded path drove would collide with it on restore, so the
  // offer is withdrawn rather than allowed to rebuild a spec the backend
  // refuses.
  const dropStash = () =>
    setDiscardedPaths((current) => {
      if (!(node.id in current)) return current;
      const next = { ...current };
      delete next[node.id];
      return next;
    });

  const updateParam = (key: string, value: unknown) => {
    if (definition === undefined) return;
    putStep(setField(definition, step, key, value));
  };

  // Apply several field changes in one node update (a source-leg change may set
  // both the producer and its channel together).
  const updateParams = (patch: Record<string, unknown>) => {
    if (definition === undefined) return;
    putStep(
      Object.entries(patch).reduce(
        (current, [key, value]) => setField(definition, current, key, value),
        step,
      ),
    );
  };

  // The registry compiles here rather than on the way out: a step carries
  // shocks, and the editor commits an entry only once it is complete, so this
  // is the first moment the entries can become the list the step holds. It
  // replaces any bundle-serialized one, which is what the user is editing away
  // from. Reachable only once the step carries no supplied paths, since the
  // panel is read-only while any remain, so there is nothing left to drop.
  const setRegistry = (entries: ShockRegistryEntry[]) => {
    let shocks: unknown;
    try {
      shocks = shocksFromRegistry(entries);
    } catch (error: unknown) {
      setRegistryError(error instanceof Error ? error.message : String(error));
      return;
    }
    setRegistryError("");
    dropStash();
    putStep({ ...step, kwargs: { ...step.kwargs, shocks } });
  };

  // Drop the supplied paths and keep every drawn entry, which the panel then
  // renders as an editable registry. The arrays are held in session state so the
  // step can take them back until an edit makes that meaningless.
  const discardPaths = () => {
    const shocks = step.kwargs.shocks;
    if (!Array.isArray(shocks)) return;
    const paths = shocks.filter(isPathEntry);
    if (paths.length === 0) return;
    setDiscardedPaths((current) => ({ ...current, [node.id]: paths }));
    putStep({
      ...step,
      kwargs: {
        ...step.kwargs,
        shocks: shocks.filter((entry) => !isPathEntry(entry)),
      },
    });
  };

  // Put the stashed paths back, after the drawn entries: the native draw keys on
  // `(seed, entry index)`, and prepending them would move every drawn entry's
  // stream. Restoring re-freezes the panel, which is the state discard left.
  const restorePaths = () => {
    if (stashedPaths.length === 0) return;
    const shocks = step.kwargs.shocks;
    const kept = Array.isArray(shocks) ? shocks : [];
    putStep({
      ...step,
      kwargs: { ...step.kwargs, shocks: [...kept, ...stashedPaths] },
    });
    dropStash();
  };

  const isCustom =
    step.step_type === "transform:custom" ||
    step.step_type === "postproc:custom";

  // The leg widget reads three fields by key, so it is handed those values
  // rather than a bag it would have to be trusted not to rummage through.
  const legValues = (keys: string[]): Record<string, unknown> =>
    definition === undefined
      ? {}
      : Object.fromEntries(
          keys.map((key) => [key, getField(definition, step, key)]),
        );

  // The wald moment gates which of its two target widgets renders; every other
  // kind declares no condition and shows all of its fields.
  const visibleFields = (definition?.fields ?? []).filter(
    (field) =>
      field.when.length === 0 ||
      field.when.includes(String(step.kwargs.kind ?? "")),
  );

  // Render the step's fields, folding each source binding (a `<leg>_source`
  // text field, its `<leg>_field` channel select, and an optional columns list)
  // into a single source-leg widget. The catalog emits those three
  // consecutively, so we consume them together and render the rest generically.
  const renderFields = (fields: MCFieldSpec[]): ReactNode[] => {
    const items: ReactNode[] = [];
    for (let i = 0; i < fields.length; i++) {
      const field = fields[i];
      const key = `${node.id}:${field.key}`;
      if (field.type === "shock_registry") {
        const targetRole = String(step.kwargs.target ?? "dgp") as Role;
        items.push(
          <ShockRegistryEditor
            key={key}
            target={targetRole}
            exogVars={exogByRole[targetRole] ?? []}
            rows={shockRows(step.kwargs.shocks)}
            restorableCount={stashedPaths.length}
            onChange={setRegistry}
            onDiscardPaths={discardPaths}
            onRestorePaths={restorePaths}
          />,
        );
        continue;
      }
      const channel = fields[i + 1];
      if (isSourceKey(field.key) && field.type === "text" && channel?.type === "select") {
        const columns = fields[i + 2];
        const hasColumns = columns?.type === "number_list";
        items.push(
          <SourceLeg
            key={key}
            sourceField={field}
            channelField={channel}
            columnsField={hasColumns ? columns : undefined}
            values={legValues([
              field.key,
              channel.key,
              ...(hasColumns ? [columns.key] : []),
            ])}
            producers={producers}
            onUpdate={updateParams}
          />,
        );
        i += hasColumns ? 2 : 1;
        continue;
      }
      items.push(
        <FieldEditor
          key={key}
          field={field}
          value={
            definition === undefined
              ? field.default
              : (getField(definition, step, field.key) ?? field.default)
          }
          availableTraces={availableTraces}
          onChange={(value) => updateParam(field.key, value)}
        />,
      );
    }
    return items;
  };

  return (
    <div className="mc-inspector">
      <div className="mc-inspector-title">
        <div>
          <strong>{definition?.title ?? step.step_type}</strong>
          <span>{definition?.description ?? ""}</span>
        </div>
        <button
          className="icon-button"
          onClick={() => onDelete(node.id)}
          title="Delete step"
        >
          <Trash2 size={15} />
        </button>
      </div>
      <label>
        Step name
        <input
          value={step.name}
          onChange={(event) => putStep({ ...step, name: event.target.value })}
        />
      </label>
      {definition?.category !== "postproc" && (
        <label>
          Retained samples
          <input
            type="number"
            min={-1}
            value={step.n_retain}
            onChange={(event) => updateParam("n_retain", Number(event.target.value))}
          />
        </label>
      )}
      {isCustom ? (
        <>
          {step.step_type === "transform:custom" && (
            <div className="mc-inspector-fields">{renderFields(visibleFields)}</div>
          )}
          <CustomOpEditor
            nodeId={node.id}
            stepType={step.step_type}
            code={step.code ?? ""}
            theme={theme}
            onChange={(value) => updateParam("code", value)}
          />
        </>
      ) : (
        <div className="mc-inspector-fields">{renderFields(visibleFields)}</div>
      )}
      {registryError !== "" && (
        <p className="mc-inspector-error">{registryError}</p>
      )}
    </div>
  );
}

const DIST_LABEL: Record<ShockDistribution, string> = {
  norm: "Normal",
  t: "Student-t",
  uni: "Uniform",
};

// Bespoke shock panel for the simulation step. The checklist is the target
// model's exogenous variables (offered as options, never assumed); one entry is
// a free-form shock over the chosen subset. Client-side validation blocks a
// variable already claimed by another entry and a joint uniform shock.
function ShockRegistryEditor({
  target,
  exogVars,
  rows,
  restorableCount,
  onChange,
  onDiscardPaths,
  onRestorePaths,
}: {
  target: Role;
  exogVars: string[];
  rows: ShockRow[];
  restorableCount: number;
  onChange: (entries: ShockRegistryEntry[]) => void;
  onDiscardPaths: () => void;
  onRestorePaths: () => void;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [dist, setDist] = useState<ShockDistribution>("norm");
  // Keyed by variable, not positional: `commitEntry` reorders the selection
  // into the model's own variable order, and a positional list would hand each
  // location to the wrong shock.
  const [locs, setLocs] = useState<Record<string, string>>({});
  const [df, setDf] = useState("5");
  const [seed, setSeed] = useState("");
  const [error, setError] = useState("");
  // Index of the entry being edited (null while composing a fresh one). The
  // entry under edit is excluded from the "used" set so its own variables stay
  // selectable, and Save replaces it in place instead of appending.
  const [editIndex, setEditIndex] = useState<number | null>(null);

  // A path claims its variables as firmly as a drawn entry does, so both are
  // read off the same rows.
  const usedVars = new Set(
    rows.flatMap((row) =>
      row.kind === "path"
        ? row.vars
        : row.entryIndex === editIndex
          ? []
          : row.entry.vars,
    ),
  );
  const entries = rows.flatMap((row) => (row.kind === "drawn" ? [row.entry] : []));
  const frozen = rows.some((row) => row.kind === "path");
  const locFor = (name: string) => locs[name] ?? "0";
  const multivarUni = dist === "uni" && selected.length > 1;
  const multivarJoint = dist !== "uni" && selected.length > 1;

  function resetForm() {
    setEditIndex(null);
    setSelected([]);
    setDist("norm");
    setLocs({});
    setDf("5");
    setSeed("");
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
    setSelected(entry.vars);
    setDist(entry.dist);
    setLocs(
      Object.fromEntries(
        entry.vars.map((name, position) => [name, String(entry.loc[position] ?? 0)]),
      ),
    );
    setDf(String(entry.df));
    setSeed(entry.seed === null ? "" : String(entry.seed));
    setError("");
  }

  function commitEntry() {
    if (selected.length === 0) {
      setError("Select at least one exogenous variable.");
      return;
    }
    const clash = selected.find((name) => usedVars.has(name));
    if (clash !== undefined) {
      setError(`'${clash}' is already used in another shock entry.`);
      return;
    }
    if (dist === "uni" && selected.length > 1) {
      setError("A uniform shock is univariate; select exactly one variable.");
      return;
    }
    // Order the key by the model's variable order for a stable identity.
    const vars = exogVars.filter((name) => selected.includes(name));
    // One location per variable, in that same order. An untouched box is the
    // form's own default of zero; anything else that is not a number is an
    // error rather than a silent zero.
    const loc: number[] = [];
    for (const name of vars) {
      const raw = locFor(name).trim();
      const value = raw === "" ? 0 : Number(raw);
      if (!Number.isFinite(value)) {
        setError(`Location for '${name}' is not a number.`);
        return;
      }
      loc.push(value);
    }
    const entry: ShockRegistryEntry = {
      vars,
      dist,
      loc,
      df: Number(df) || 5,
      seed: seed.trim() === "" ? null : Number(seed),
    };
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
        <span className="mc-shock-registry-target">from {target}</span>
      </div>
      {frozen && (
        <div className="mc-shock-frozen">
          <span>
            A supplied path is data this panel cannot author, so the whole list
            is read-only while one is present.
          </span>
          <button className="secondary" onClick={onDiscardPaths}>
            Discard paths
          </button>
        </div>
      )}
      {!frozen && restorableCount > 0 && (
        <div className="mc-shock-frozen">
          <span>
            {restorableCount} discarded shock{" "}
            {restorableCount === 1 ? "path is" : "paths are"} still held for this
            session. Editing an entry gives {restorableCount === 1 ? "it" : "them"}{" "}
            up.
          </span>
          <button className="secondary" onClick={onRestorePaths}>
            Restore {restorableCount === 1 ? "path" : "paths"}
          </button>
        </div>
      )}
      {rows.length > 0 ? (
        <ul className="mc-shock-list">
          {rows.map((row) =>
            row.kind === "path" ? (
              // A path is a spec the step carries like any other, so it is a row
              // like any other. Its array is not something a form can author,
              // which is all that sets the row apart.
              <li
                key={`path:${row.index}`}
                className="mc-shock-entry mc-shock-entry-path"
              >
                <div className="mc-shock-entry-body">
                  <strong>
                    {row.vars.length > 0 ? row.vars.join(", ") : "(unnamed)"}
                    {row.vars.length > 1 && (
                      <span className="mc-shock-badge">joint</span>
                    )}
                    <span className="mc-shock-badge">path</span>
                  </strong>
                  <span>
                    Supplied over {row.periods}{" "}
                    {row.periods === 1 ? "period" : "periods"}
                  </span>
                </div>
              </li>
            ) : (
              <li
                key={`drawn:${row.index}`}
                className={`mc-shock-entry${
                  editIndex === row.entryIndex ? " editing" : ""
                }`}
              >
                <button
                  className="mc-shock-entry-select"
                  title={frozen ? "Discard the supplied paths to edit" : "Edit shock"}
                  disabled={frozen || exogVars.length === 0}
                  onClick={() => startEdit(row.entryIndex)}
                >
                  <div className="mc-shock-entry-body">
                    <strong>
                      {row.entry.vars.join(", ")}
                      {row.entry.vars.length > 1 && (
                        <span className="mc-shock-badge">joint</span>
                      )}
                    </strong>
                    <span>{describeEntry(row.entry)}</span>
                  </div>
                </button>
                <button
                  className="icon-button"
                  title={frozen ? "Discard the supplied paths to edit" : "Remove shock"}
                  disabled={frozen}
                  onClick={() => removeEntry(row.entryIndex)}
                >
                  <Trash2 size={13} />
                </button>
              </li>
            ),
          )}
        </ul>
      ) : (
        <p className="mc-shock-empty">
          No shocks configured; this simulation runs deterministically.
        </p>
      )}
      {frozen ? null : exogVars.length === 0 ? (
        <p className="mc-shock-empty">
          Load and solve the {target} model to choose its exogenous shocks.
        </p>
      ) : (
        <div className="mc-shock-form">
          <div className="mc-shock-checklist">
            {exogVars.map((name) => {
              const used = usedVars.has(name);
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
            {exogVars
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
          </div>
          {multivarUni && (
            <span className="mc-shock-hint">
              A uniform shock is univariate; select exactly one variable.
            </span>
          )}
          {multivarJoint && (
            <span className="mc-shock-hint">
              This is one joint (multivar) shock over {selected.length} variables.
              Add a separate entry per variable if you want them independent.
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
  const parts = [DIST_LABEL[entry.dist], `loc ${entry.loc.join(", ")}`];
  if (entry.dist === "t") parts.push(`df ${entry.df}`);
  parts.push(entry.seed === null ? "seed none" : `seed ${entry.seed}`);
  return parts.join(", ");
}

// Read the registry the panel renders, literally, from the step params. A
// GUI-authored step carries `shock_registry`; a serialized step carries the
// compiled `shocks` list, which we reconstruct one entry per element with no
// invented info (`target` alone gives the variables and whether it is joint).
// The registry as the step's compiled shocks describe it. There is no stored
// draft to prefer: the editor holds the in-progress entry itself and commits a
// complete one, so the step is the only record.
/** One row of the shock panel: a drawn family, or a path supplied for it.
 *
 * Both are shock specs a step carries, so both are rows. `index` is the entry's
 * place in `kwargs.shocks`, which is what an action on one row has to address;
 * `entryIndex` is its place among the drawn ones, which is the list the editor
 * form emits.
 */
type ShockRow =
  | { kind: "drawn"; index: number; entryIndex: number; entry: ShockRegistryEntry }
  | { kind: "path"; index: number; vars: string[]; periods: number };

// The rows a step's shocks describe. A path carries an array the panel has no
// way to render, so it reports the variables it drives and how long it is; the
// two entry kinds are told apart by what the entry declares, since `path` is
// present on one and absent on the other.
function shockRows(shocks: unknown): ShockRow[] {
  if (!Array.isArray(shocks)) return [];
  const rows: ShockRow[] = [];
  let entryIndex = 0;
  shocks.forEach((value, index) => {
    const entry = (value ?? {}) as Record<string, unknown>;
    if ("path" in entry) {
      rows.push({
        kind: "path",
        index,
        vars: Array.isArray(entry.target) ? entry.target.map(String) : [],
        periods: Array.isArray(entry.path) ? entry.path.length : 0,
      });
      return;
    }
    rows.push({
      kind: "drawn",
      index,
      entryIndex: entryIndex++,
      entry: entryFromShock(entry),
    });
  });
  return rows;
}

// Invert `shockFor`: recover the registry entry from a serialized Shock dict.
// The library reads `mean` and `loc` identically and prefers `mean` when a spec
// carries both, so this reads them in that order and keeps the whole vector. A
// bundle-authored entry may name a different location per variable, and
// collapsing that to one number would silently rewrite the spec on the next
// edit.
function entryFromShock(dict: Record<string, unknown>): ShockRegistryEntry {
  const vars = Array.isArray(dict.target) ? dict.target.map(String) : [];
  const dist = asDist(dict.dist);
  const kwargs = (dict.dist_kwargs ?? {}) as Record<string, unknown>;
  const declared = "mean" in kwargs ? kwargs.mean : kwargs.loc;
  return {
    vars,
    dist,
    loc: locVector(declared, vars.length),
    df: dist === "t" ? Number(kwargs.df ?? 5) : 5,
    seed: dict.seed === null || dict.seed === undefined ? null : Number(dict.seed),
  };
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

// The supplied-path entries of a serialized spec, which the registry cannot
// represent. Their presence freezes the panel: the drawn entries stay visible
// and uneditable until the paths are discarded, which is what keeps a user from
// authoring a second entry over a variable a path already drives.
function asDist(value: unknown): ShockDistribution {
  return value === "t" || value === "uni" ? value : "norm";
}

// One source binding: which producer step to read (a dropdown of the pipeline's
// producers) and which of its channels (kind-aware: datagen -> states/observables,
// filter -> filter channels, transform -> payload), plus an optional column list.
function SourceLeg({
  sourceField,
  channelField,
  columnsField,
  values,
  producers,
  onUpdate,
}: {
  sourceField: MCFieldSpec;
  channelField: MCFieldSpec;
  columnsField?: MCFieldSpec;
  values: Record<string, unknown>;
  producers: MCProducer[];
  onUpdate: (patch: Record<string, unknown>) => void;
}) {
  const sourceValue = String(values[sourceField.key] ?? sourceField.default ?? "");
  const channelValue = String(values[channelField.key] ?? channelField.default ?? "");
  const selected = producers.find((producer) => producer.name === sourceValue);
  const channels = channelOptionsFor(selected?.kind, channelField.options);
  // Keep a stale/unknown selection visible instead of silently dropping it.
  const channelOptions =
    channelValue && !channels.includes(channelValue)
      ? [channelValue, ...channels]
      : channels;
  const producerNames = producers.map((producer) => producer.name);
  const sourceOptions =
    sourceValue && !producerNames.includes(sourceValue)
      ? [sourceValue, ...producerNames]
      : producerNames;

  const onSourceChange = (value: string) => {
    const kind = producers.find((producer) => producer.name === value)?.kind;
    const nextChannels = channelOptionsFor(kind, channelField.options);
    const patch: Record<string, unknown> = { [sourceField.key]: value };
    // If the current channel isn't valid for the newly chosen producer's kind,
    // snap it to the first valid channel so the leg stays consistent.
    if (nextChannels.length > 0 && !nextChannels.includes(channelValue)) {
      patch[channelField.key] = nextChannels[0];
    }
    onUpdate(patch);
  };

  return (
    <div className="mc-source-leg">
      <label>
        {sourceField.label}
        <select value={sourceValue} onChange={(event) => onSourceChange(event.target.value)}>
          <option value="">— select step —</option>
          {sourceOptions.map((name) => {
            const kind = producers.find((producer) => producer.name === name)?.kind;
            return (
              <option key={name} value={name}>
                {kind ? `${name} · ${kind}` : name}
              </option>
            );
          })}
        </select>
      </label>
      <label>
        {channelField.label}
        <select
          value={channelValue}
          onChange={(event) => onUpdate({ [channelField.key]: event.target.value })}
        >
          {channelOptions.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
      {columnsField && (
        <DraftListEditor
          field={columnsField}
          value={values[columnsField.key] ?? columnsField.default}
          onChange={(value) => onUpdate({ [columnsField.key]: value })}
        />
      )}
    </div>
  );
}

function CustomOpEditor({
  nodeId,
  stepType,
  code,
  theme,
  onChange,
}: {
  nodeId: string;
  stepType: MCStepType;
  code: string;
  theme: "light" | "dark";
  onChange: (value: string) => void;
}) {
  const [status, setStatus] = useState<{ ok: boolean; message: string } | null>(null);
  const [busy, setBusy] = useState(false);

  async function validate() {
    setBusy(true);
    try {
      const result = await validateCustomOp(code, stepType);
      setStatus(
        result.valid
          ? { ok: true, message: `Valid op: ${result.name ?? ""}` }
          : { ok: false, message: result.error ?? "Invalid op." },
      );
    } catch (error) {
      setStatus({
        ok: false,
        message: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mc-custom-editor">
      <div className="mc-custom-editor-wrap">
        <Editor
          height="100%"
          language="python"
          path={`sdsge-mc-custom://${nodeId}.py`}
          theme={theme === "dark" ? "vs-dark" : "light"}
          value={code}
          onChange={(value) => onChange(value ?? "")}
          onMount={(_editor: Monaco.editor.IStandaloneCodeEditor, monaco) => {
            void registerPythonLsp(monaco);
          }}
          options={{
            automaticLayout: true,
            fontSize: 13,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            tabSize: 4,
            wordWrap: "on",
          }}
        />
      </div>
      <div className="mc-custom-editor-footer">
        <button
          className="secondary"
          disabled={busy || code.trim() === ""}
          onClick={() => void validate()}
        >
          <Check size={14} />
          Validate op
        </button>
        {status !== null && (
          <span
            className={
              status.ok ? "status mc-custom-status" : "status error mc-custom-status"
            }
          >
            {status.ok ? <Check size={13} /> : <TriangleAlert size={13} />}
            {status.message}
          </span>
        )}
      </div>
    </div>
  );
}

function FieldEditor({
  field,
  value,
  availableTraces,
  onChange,
}: {
  field: MCFieldSpec;
  value: unknown;
  availableTraces: string[];
  onChange: (value: unknown) => void;
}) {
  if (field.type === "trace") {
    const current = String(value ?? "");
    // Offer the pipeline's producible traces; keep a stale selection visible.
    const options =
      current && !availableTraces.includes(current)
        ? [current, ...availableTraces]
        : availableTraces;
    return (
      <label>
        {field.label}
        <select value={current} onChange={(event) => onChange(event.target.value)}>
          <option value="">— select trace —</option>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
    );
  }
  if (field.type === "boolean") {
    return (
      <label className="switch-row">
        <span>{field.label}</span>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => onChange(event.target.checked)}
        />
      </label>
    );
  }
  if (field.type === "select") {
    return (
      <label>
        {field.label}
        <select
          value={String(value ?? "")}
          onChange={(event) => onChange(event.target.value)}
        >
          {field.options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
    );
  }
  if (field.type === "number") {
    return (
      <label>
        {field.label}
        <input
          type="number"
          min={field.minimum ?? undefined}
          value={value === null ? "" : String(value ?? "")}
          onChange={(event) =>
            onChange(event.target.value === "" ? "" : Number(event.target.value))
          }
        />
      </label>
    );
  }
  if (field.type === "number_list" || field.type === "text_list") {
    return (
      <DraftListEditor field={field} value={value} onChange={onChange} />
    );
  }
  if (field.type === "number_matrix") {
    return (
      <DraftMatrixEditor field={field} value={value} onChange={onChange} />
    );
  }
  return (
    <label>
      {field.label}
      <input
        value={String(value ?? "")}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function DraftListEditor({
  field,
  value,
  onChange,
}: {
  field: MCFieldSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const [draft, setDraft] = useState(
    Array.isArray(value) ? value.join(", ") : String(value ?? ""),
  );
  const commit = () =>
    onChange(field.type === "number_list" ? parseNumberList(draft) : parseTextList(draft));
  return (
    <label>
      {field.label}
      <input
        type="text"
        inputMode={field.type === "number_list" ? "decimal" : undefined}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            commit();
            event.currentTarget.blur();
          }
        }}
        placeholder={field.type === "number_list" ? "0, 1" : "name_a, name_b"}
      />
    </label>
  );
}

function DraftMatrixEditor({
  field,
  value,
  onChange,
}: {
  field: MCFieldSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const [draft, setDraft] = useState(formatMatrix(value));
  return (
    <label className="mc-matrix-field">
      {field.label}
      <textarea
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => onChange(parseNumberMatrix(draft))}
        placeholder={"1, 0\n0, 1"}
      />
    </label>
  );
}

function parseNumberList(value: string): number[] {
  return value
    .split(/[\s,;]+/)
    .filter(Boolean)
    .map(Number)
    .filter(Number.isFinite);
}

function parseTextList(value: string): string[] {
  return value
    .split(/[\s,;]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseNumberMatrix(value: string): number[][] {
  return value
    .split(/[;\n]+/)
    .map(parseNumberList)
    .filter((row) => row.length > 0);
}

function formatMatrix(value: unknown): string {
  if (!Array.isArray(value)) return "";
  return value
    .map((row) => (Array.isArray(row) ? row.join(", ") : String(row)))
    .join("\n");
}
