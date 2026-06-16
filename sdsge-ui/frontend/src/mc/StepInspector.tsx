import Editor from "@monaco-editor/react";
import type * as Monaco from "monaco-editor";
import { Check, TriangleAlert, Trash2 } from "lucide-react";
import { useState } from "react";
import { validateCustomOp } from "../api";
import { registerPythonLsp } from "../lsp/registerPythonLsp";
import type { MCFieldSpec } from "../types";
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
}: {
  node: MCFlowNode | null;
  onChange: (node: MCFlowNode) => void;
  onDelete: (id: string) => void;
  theme: "light" | "dark";
  payloadProducers: string[];
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

  const isCustom = node.data.stepType === "custom";
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
                  onChange={(value) => updateParam(field.key, value)}
                />
              );
            })}
        </div>
      )}
    </div>
  );
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
  code,
  theme,
  onChange,
}: {
  nodeId: string;
  code: string;
  theme: "light" | "dark";
  onChange: (value: string) => void;
}) {
  const [status, setStatus] = useState<{ ok: boolean; message: string } | null>(null);
  const [busy, setBusy] = useState(false);

  async function validate() {
    setBusy(true);
    try {
      const result = await validateCustomOp(code);
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
  onChange,
}: {
  field: MCFieldSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
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
