import Editor from "@monaco-editor/react";
import type * as Monaco from "monaco-editor";
import { Check, TriangleAlert, Trash2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import { validateCustomOp } from "../api";
import { registerPythonLsp } from "../lsp/registerPythonLsp";
import type { MCFieldSpec, MCStepSpec, MCStepType } from "../types";
import { ShockRegistryEditor } from "../shocks/ShockRegistryEditor";
import { useShockRegistry } from "../shocks/useShockRegistry";
import { stepDefinition, usesModelTarget } from "./catalog";
import { getField, setField } from "./fields";
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
  // Fields are available only after an eligible producer is selected.
  return [];
}

export function StepInspector({
  node,
  onChange,
  onDelete,
  theme,
  producers,
  availableTraces,
  shockNamesByRole,
  modelNames,
}: {
  node: MCFlowNode | null;
  onChange: (node: MCFlowNode) => void;
  onDelete: (id: string) => void;
  theme: "light" | "dark";
  producers: MCProducer[];
  availableTraces: string[];
  shockNamesByRole: Record<string, string[]>;
  modelNames: string[];
}) {
  // The shock panel's editing state. Declared above the early return, since a
  // hook cannot be conditional.
  const registry = useShockRegistry(
    node?.data.step.kwargs.shocks,
    (shocks) => {
      if (node === null) return;
      const current = node.data.step;
      onChange({
        ...node,
        data: {
          ...node.data,
          step: { ...current, kwargs: { ...current.kwargs, shocks } },
        },
      });
    },
  );

  if (node === null) {
    return (
      <div className="mc-empty">
        <span>Select a step to edit its configuration.</span>
      </div>
    );
  }

  const step = node.data.step;
  const definition = stepDefinition(step.step_type);

  // Every edit replaces the node's step; `mc/fields` is what knows whether a
  // form key is a leg, a slot of the step's own, or a kwarg.
  const putStep = (next: MCStepSpec) =>
    onChange({ ...node, data: { ...node.data, step: next } });

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
      if (field.key === "target" && usesModelTarget(step.step_type)) {
        const current = String(step.kwargs.target ?? "");
        const options = current && !modelNames.includes(current)
          ? [current, ...modelNames]
          : modelNames;
        items.push(
          <label key={key}>
            {field.label}
            <select value={current} onChange={(event) => updateParam(field.key, event.target.value || null)}>
              <option value="">Select a model</option>
              {options.map((name) => (
                <option key={name} value={name}>
                  {name}{modelNames.includes(name) ? "" : " (missing)"}
                </option>
              ))}
            </select>
          </label>,
        );
        continue;
      }
      if (field.type === "shock_registry") {
        const targetRole = String(step.kwargs.target ?? "") as string;
        items.push(
          <ShockRegistryEditor
            key={key}
            role={targetRole}
            shockNames={shockNamesByRole[targetRole] ?? []}
            entries={registry.entries}
            onChange={registry.setRegistry}
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
      {registry.error !== "" && (
        <span className="status error shock-registry-error">
          {registry.error}
        </span>
      )}
    </div>
  );
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
          disabled={selected === undefined}
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
