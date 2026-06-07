import Editor from "@monaco-editor/react";
import type * as Monaco from "monaco-editor";
import { Send, X } from "lucide-react";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { listFunctions, removeFunction, submitFunction } from "./api";
import { registerPythonLsp } from "./lsp/registerPythonLsp";
import type { FunctionKind, FunctionRecord, ModelSummary, Role } from "./types";

function handleEditorMount(_editor: Monaco.editor.IStandaloneCodeEditor, monaco: typeof Monaco) {
  void registerPythonLsp(monaco);
}

export interface CodePanelHandle {
  resetTemplate: () => void;
}

function makeTemplate(model: ModelSummary, kind: FunctionKind): string {
  const vars = model.variables ?? [];
  const obs = model.observables ?? [];
  const allArgs = [...new Set([...vars, ...obs])];
  const argStr = allArgs.join(", ");
  const firstArg = allArgs[0];

  if (kind === "figure") {
    const body = firstArg
      ? `    ax.plot(${firstArg}, label="${firstArg}")
    ax.legend()
    return fig`
      : `    ax.set_title("Figure")
    return fig`;
    return `import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


# Each argument is a 1-D array (length T) of simulation results.
# Return a Figure to register as a plot in the Outputs tab.
def plot(${argStr}) -> Figure:
    fig, ax = plt.subplots()
${body}
`;
  }

  const returnExpr = firstArg ? `${firstArg}.copy()` : "np.zeros(100)";
  return `import numpy as np
from numpy import ndarray


# Each argument is a 1-D array (length T) of simulation results.
# Return a 1-D array of length T to register as a new series.
def transform(${argStr}) -> ndarray:
    return ${returnExpr}
`;
}

export const CodePanel = forwardRef<
  CodePanelHandle,
  { role: Role; activeModel: ModelSummary; theme: "light" | "dark"; kind: FunctionKind }
>(function CodePanel({ role, activeModel, theme, kind }, ref) {
  const initialTemplate = makeTemplate(activeModel, kind);
  const prevTemplateRef = useRef<string>(initialTemplate);
  const activeModelRef = useRef(activeModel);
  activeModelRef.current = activeModel;

  const [code, setCode] = useState(initialTemplate);
  const [functions, setFunctions] = useState<FunctionRecord[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useImperativeHandle(
    ref,
    () => ({
      resetTemplate() {
        const template = makeTemplate(activeModelRef.current, kind);
        prevTemplateRef.current = template;
        setCode(template);
      },
    }),
    [kind],
  );

  const varKey = [
    ...(activeModel.variables ?? []),
    ...(activeModel.observables ?? []),
  ].join("|");

  useEffect(() => {
    const newTemplate = makeTemplate(activeModel, kind);
    if (newTemplate === prevTemplateRef.current) return;
    setCode((current) =>
      current === prevTemplateRef.current ? newTemplate : current,
    );
    prevTemplateRef.current = newTemplate;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [varKey, kind]);

  const refreshFunctions = useCallback(async () => {
    try {
      const all = await listFunctions(role);
      setFunctions(all.filter((f) => f.kind === kind));
    } catch {
      // backend might not be running yet
    }
  }, [role, kind]);

  useEffect(() => {
    void refreshFunctions();
  }, [refreshFunctions]);

  async function handleSubmit() {
    setBusy(true);
    setError("");
    try {
      await submitFunction(role, code, kind);
      await refreshFunctions();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove(name: string) {
    try {
      await removeFunction(role, name);
      setFunctions((prev) => prev.filter((f) => f.name !== name));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="code-panel">
      <div className="code-editor-wrap">
        <Editor
          height="100%"
          language="python"
          path={`sdsge-code://${role}-${kind}.py`}
          theme={theme === "dark" ? "vs-dark" : "light"}
          value={code}
          onChange={(v) => setCode(v ?? "")}
          onMount={handleEditorMount}
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
      <div className="code-panel-footer">
        <div className="code-fn-row">
          <div className="code-fn-list">
            {functions.length === 0 ? (
              <span className="muted code-fn-empty">No functions submitted</span>
            ) : (
              functions.map((fn) => (
                <span key={fn.name} className="code-fn-chip">
                  {fn.name}
                  <button
                    className="code-fn-remove"
                    onClick={() => void handleRemove(fn.name)}
                    title={`Remove ${fn.name}`}
                  >
                    <X size={11} />
                  </button>
                </span>
              ))
            )}
          </div>
          <button
            className="code-submit-btn"
            disabled={busy || code.trim() === ""}
            onClick={() => void handleSubmit()}
            title="Submit function"
          >
            <Send size={14} />
            Submit
          </button>
        </div>
        {error && <span className="status error code-error">{error}</span>}
      </div>
    </div>
  );
});
