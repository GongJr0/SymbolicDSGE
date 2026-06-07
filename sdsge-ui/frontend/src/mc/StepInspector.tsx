import { Trash2 } from "lucide-react";
import { useState } from "react";
import type { MCFieldSpec } from "../types";
import type { MCFlowNode } from "./types";

export function StepInspector({
  node,
  onChange,
  onDelete,
}: {
  node: MCFlowNode | null;
  onChange: (node: MCFlowNode) => void;
  onDelete: (id: string) => void;
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
      <div className="mc-inspector-fields">
        {node.data.catalog.fields
          .filter((field) => fieldVisible(field, node.data.params))
          .map((field) => (
            <FieldEditor
              key={`${node.id}:${field.key}`}
              field={field}
              value={node.data.params[field.key] ?? field.default}
              onChange={(value) => updateParam(field.key, value)}
            />
          ))}
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
