import Editor from "@monaco-editor/react";
import type * as Monaco from "monaco-editor";
import { Check, Plus, TriangleAlert, Trash2 } from "lucide-react";
import { useState } from "react";
import { validateCustomOp } from "../api";
import { registerPythonLsp } from "../lsp/registerPythonLsp";
import type {
  MCFieldSpec,
  MCStepType,
  Role,
  ShockDistribution,
  ShockRegistryEntry,
} from "../types";
import type { MCFlowNode } from "./types";

// Input legs whose value is a dependency source (a "select" with INPUT_SOURCES
// options). A leg set to "payload" reads a producer's output by key.
const SOURCE_LEG_KEYS = new Set([
  "source",
  "residual_source",
  "y_source",
  "X_source",
  "x_source",
]);

// Source leg -> the param naming that leg's payload producer (mirrors the
// backend's _LEG_TO_PAYLOAD_KEY).
const LEG_PAYLOAD_KEY: Record<string, string> = {
  source: "payload_key",
  residual_source: "residual_payload_key",
  y_source: "y_payload_key",
  X_source: "x_payload_key",
  x_source: "x_payload_key",
};

export function StepInspector({
  node,
  onChange,
  onDelete,
  theme,
  payloadProducers,
  availableTraces,
  exogByRole,
}: {
  node: MCFlowNode | null;
  onChange: (node: MCFlowNode) => void;
  onDelete: (id: string) => void;
  theme: "light" | "dark";
  payloadProducers: string[];
  availableTraces: string[];
  exogByRole: Record<Role, string[]>;
}) {
  if (node === null) {
    return (
      <div className="mc-empty">
        <span>Select a step to edit its configuration.</span>
      </div>
    );
  }

  const updateParam = (key: string, value: unknown) => {
    onChange({
      ...node,
      data: {
        ...node.data,
        params: { ...node.data.params, [key]: value },
      },
    });
  };

  // Writing the registry replaces any bundle-serialized `shocks` map so the
  // backend compiles from the user's explicit entries, not a stale compiled form.
  const setRegistry = (entries: ShockRegistryEntry[]) => {
    const params: Record<string, unknown> = {
      ...node.data.params,
      shock_registry: entries,
    };
    delete params.shocks;
    onChange({ ...node, data: { ...node.data, params } });
  };

  const isCustom =
    node.data.stepType === "transform:custom" ||
    node.data.stepType === "postproc:custom";
  const producers = payloadProducers.filter((name) => name !== node.data.name);

  return (
    <div className="mc-inspector">
      <div className="mc-inspector-title">
        <div>
          <strong>{node.data.catalog.title}</strong>
          <span>{node.data.catalog.description}</span>
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
          value={node.data.name}
          onChange={(event) =>
            onChange({
              ...node,
              data: { ...node.data, name: event.target.value },
            })
          }
        />
      </label>
      {isCustom ? (
        <CustomOpEditor
          nodeId={node.id}
          stepType={node.data.stepType}
          code={String(node.data.params.code ?? "")}
          theme={theme}
          onChange={(value) => updateParam("code", value)}
        />
      ) : (
        <div className="mc-inspector-fields">
          {node.data.catalog.fields
            .filter((field) => fieldVisible(field, node.data.params))
            .map((field) => {
              const key = `${node.id}:${field.key}`;
              if (field.type === "shock_registry") {
                const targetRole = String(
                  node.data.params.target ?? "dgp",
                ) as Role;
                return (
                  <ShockRegistryEditor
                    key={key}
                    target={targetRole}
                    exogVars={exogByRole[targetRole] ?? []}
                    entries={registryFromParams(node.data.params)}
                    onChange={setRegistry}
                  />
                );
              }
              if (field.type === "select" && SOURCE_LEG_KEYS.has(field.key)) {
                const payloadKey = LEG_PAYLOAD_KEY[field.key];
                return (
                  <SourceLegEditor
                    key={key}
                    field={field}
                    channel={String(node.data.params[field.key] ?? field.default ?? "")}
                    producer={String(node.data.params[payloadKey] ?? "")}
                    producers={producers}
                    onChannelChange={(value) => updateParam(field.key, value)}
                    onProducerChange={(value) => updateParam(payloadKey, value)}
                  />
                );
              }
              return (
                <FieldEditor
                  key={key}
                  field={field}
                  value={node.data.params[field.key] ?? field.default}
                  availableTraces={availableTraces}
                  onChange={(value) => updateParam(field.key, value)}
                />
              );
            })}
        </div>
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
  entries,
  onChange,
}: {
  target: Role;
  exogVars: string[];
  entries: ShockRegistryEntry[];
  onChange: (entries: ShockRegistryEntry[]) => void;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [dist, setDist] = useState<ShockDistribution>("norm");
  const [loc, setLoc] = useState("0");
  const [df, setDf] = useState("5");
  const [seed, setSeed] = useState("");
  const [error, setError] = useState("");
  // Index of the entry being edited (null while composing a fresh one). The
  // entry under edit is excluded from the "used" set so its own variables stay
  // selectable, and Save replaces it in place instead of appending.
  const [editIndex, setEditIndex] = useState<number | null>(null);

  const usedVars = new Set(
    entries.flatMap((entry, index) => (index === editIndex ? [] : entry.vars)),
  );
  const multivarUni = dist === "uni" && selected.length > 1;
  const multivarJoint = dist !== "uni" && selected.length > 1;

  function resetForm() {
    setEditIndex(null);
    setSelected([]);
    setDist("norm");
    setLoc("0");
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
    setLoc(String(entry.loc));
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
    const entry: ShockRegistryEntry = {
      vars,
      dist,
      loc: Number(loc) || 0,
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
      {entries.length > 0 ? (
        <ul className="mc-shock-list">
          {entries.map((entry, index) => (
            <li
              key={entry.vars.join(",")}
              className={`mc-shock-entry${editIndex === index ? " editing" : ""}`}
            >
              <button
                className="mc-shock-entry-select"
                title="Edit shock"
                disabled={exogVars.length === 0}
                onClick={() => startEdit(index)}
              >
                <div className="mc-shock-entry-body">
                  <strong>
                    {entry.vars.join(", ")}
                    {entry.vars.length > 1 && (
                      <span className="mc-shock-badge">joint</span>
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
      {exogVars.length === 0 ? (
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
            <label>
              Location
              <input
                type="number"
                value={loc}
                onChange={(event) => setLoc(event.target.value)}
              />
            </label>
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
  const parts = [DIST_LABEL[entry.dist], `loc ${entry.loc}`];
  if (entry.dist === "t") parts.push(`df ${entry.df}`);
  parts.push(entry.seed === null ? "seed none" : `seed ${entry.seed}`);
  return parts.join(", ");
}

// Read the registry the panel renders, literally, from the step params. A
// GUI-authored step carries `shock_registry`; a bundle-serialized step carries
// the compiled `shocks` map, which we reconstruct one entry per key with no
// invented info (the key alone gives the variables and joint/multivar status).
function registryFromParams(
  params: Record<string, unknown>,
): ShockRegistryEntry[] {
  const registry = params.shock_registry;
  if (Array.isArray(registry) && registry.length > 0) {
    return registry.map(normalizeEntry);
  }
  const shocks = params.shocks;
  if (shocks !== null && typeof shocks === "object" && !Array.isArray(shocks)) {
    return Object.entries(shocks as Record<string, unknown>).map(([key, value]) =>
      entryFromShock(key, (value ?? {}) as Record<string, unknown>),
    );
  }
  return [];
}

function normalizeEntry(raw: unknown): ShockRegistryEntry {
  const entry = (raw ?? {}) as Record<string, unknown>;
  return {
    vars: Array.isArray(entry.vars) ? entry.vars.map(String) : [],
    dist: asDist(entry.dist),
    loc: Number(entry.loc ?? 0),
    df: Number(entry.df ?? 5),
    seed:
      entry.seed === null || entry.seed === undefined ? null : Number(entry.seed),
  };
}

// Invert `_shock_for`: recover the registry entry from a serialized Shock dict.
// `loc` lives under `mean`/`loc` (scalar or per-variable list) by dist and shape.
function entryFromShock(
  key: string,
  dict: Record<string, unknown>,
): ShockRegistryEntry {
  const vars = key
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
  const dist = asDist(dict.dist);
  const multivar = Boolean(dict.multivar) || vars.length > 1;
  const kwargs = (dict.dist_kwargs ?? {}) as Record<string, unknown>;
  const first = (value: unknown) =>
    Array.isArray(value) ? Number(value[0] ?? 0) : Number(value ?? 0);
  const loc =
    dist === "norm" && multivar ? first(kwargs.mean) : first(kwargs.loc);
  return {
    vars,
    dist,
    loc,
    df: dist === "t" ? Number(kwargs.df ?? 5) : 5,
    seed: dict.seed === null || dict.seed === undefined ? null : Number(dict.seed),
  };
}

function asDist(value: unknown): ShockDistribution {
  return value === "t" || value === "uni" ? value : "norm";
}

function SourceLegEditor({
  field,
  channel,
  producer,
  producers,
  onChannelChange,
  onProducerChange,
}: {
  field: MCFieldSpec;
  channel: string;
  producer: string;
  producers: string[];
  onChannelChange: (value: string) => void;
  onProducerChange: (value: string) => void;
}) {
  // "payload" is always selectable; picking it reveals a producer key dropdown.
  const options = field.options.includes("payload")
    ? field.options
    : [...field.options, "payload"];
  return (
    <div className="mc-source-leg">
      <label>
        {field.label}
        <select value={channel} onChange={(event) => onChannelChange(event.target.value)}>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
      {channel === "payload" && (
        <label className="mc-payload-producer">
          Payload from
          <select
            value={producer}
            onChange={(event) => onProducerChange(event.target.value)}
          >
            <option value="">— select producer —</option>
            {producers.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
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

function fieldVisible(
  field: MCFieldSpec,
  params: Record<string, unknown>,
): boolean {
  return field.when.length === 0 || field.when.includes(String(params.kind ?? ""));
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
