import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Database, Moon, Play, RefreshCw, Sun, Upload, Zap } from "lucide-react";
import Editor from "@monaco-editor/react";
import { configureMonacoYaml } from "monaco-yaml";
import { useEffect, useMemo, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { Line } from "react-chartjs-2";
import {
  decodeArray,
  encodeArray,
  getSession,
  loadYamlContent,
  loadYamlPath,
  runSimulation,
  solveModel,
} from "./api";
import {
  symbolicDsgeConfigModelPath,
  symbolicDsgeConfigSchema,
  symbolicDsgeSchemaUri,
} from "./configSchema";
import type {
  ModelSummary,
  Role,
  SessionSummary,
  ShockCorrSpec,
  ShockDistribution,
  ShockGeneration,
  ShockParamUpdate,
  ShockSpec,
  SimResult,
} from "./types";

ChartJS.register(
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
);

const SERIES_COLORS = [
  "#2563eb",
  "#16a34a",
  "#dc2626",
  "#9333ea",
  "#d97706",
  "#0891b2",
  "#be123c",
  "#4f46e5",
  "#65a30d",
  "#c2410c",
];
let yamlLanguageConfigured = false;
type View = "builder" | "spec" | "graph";
type ShockMode = "raw" | "generated";

export default function App() {
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [role, setRole] = useState<Role>("reference");
  const [path, setPath] = useState("MODELS/POST82.yaml");
  const [content, setContent] = useState("");
  const [linearize, setLinearize] = useState(false);
  const [simT, setSimT] = useState(100);
  const [includeObs, setIncludeObs] = useState(true);
  const [result, setResult] = useState<SimResult | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [shockInputs, setShockInputs] = useState<Record<string, string>>({});
  const [shockMode, setShockMode] = useState<ShockMode>("generated");
  const [shockDist, setShockDist] = useState<ShockDistribution>("norm");
  const [shockSeed, setShockSeed] = useState("0");
  const [shockLoc, setShockLoc] = useState("0");
  const [shockDf, setShockDf] = useState("5");
  const [shockStdParams, setShockStdParams] = useState<Record<string, string>>({});
  const [shockCorrParams, setShockCorrParams] = useState<Record<string, string>>({});
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [view, setView] = useState<View>(initialView);
  const [message, setMessage] = useState("Backend not checked yet.");
  const [busy, setBusy] = useState(false);

  const activeModel = session?.models[role] ?? { role, loaded: false, solved: false };
  const shockSpecs = useMemo(
    () => activeModel.shock_specs ?? [],
    [activeModel.shock_specs],
  );
  const shockCorrSpecs = useMemo(
    () => activeModel.shock_corr_specs ?? [],
    [activeModel.shock_corr_specs],
  );
  const graphSeries = useMemo(
    () =>
      result?.series.filter(
        (item) => item.name !== "_X" && item.array.shape.length === 1,
      ) ?? [],
    [result],
  );

  async function refreshSession() {
    const next = await getSession();
    setSession(next);
    setMessage("Session refreshed.");
  }

  function navigate(next: View) {
    const path = `/${next}`;
    window.history.pushState({}, "", path);
    setView(next);
  }

  async function runAction(action: () => Promise<unknown>, done: string) {
    setBusy(true);
    try {
      await action();
      setMessage(done);
      await refreshSession();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refreshSession().catch((error: unknown) => {
      setMessage(error instanceof Error ? error.message : String(error));
    });
  }, []);

  useEffect(() => {
    const onPopState = () => setView(initialView());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (result === null) return;
    setSelected(graphSeries.map((series) => series.name));
  }, [result, graphSeries]);

  useEffect(() => {
    setShockInputs((current) => {
      const next: Record<string, string> = {};
      for (const spec of shockSpecs) {
        next[spec.target] = current[spec.target] ?? "";
      }
      return next;
    });
  }, [shockSpecs]);

  useEffect(() => {
    const next: Record<string, string> = {};
    for (const spec of shockSpecs) {
      next[spec.shock] = String(spec.std_value ?? 1.0);
    }
    setShockStdParams(next);
  }, [shockSpecs]);

  useEffect(() => {
    const next: Record<string, string> = {};
    for (const spec of shockCorrSpecs) {
      next[spec.key] = String(spec.corr_value ?? 0.0);
    }
    setShockCorrParams(next);
  }, [shockCorrSpecs]);

  useEffect(() => {
    if (content.trim() === "" && activeModel.raw_yaml !== undefined) {
      setContent(activeModel.raw_yaml);
    }
  }, [activeModel.raw_yaml, content]);

  const chartData = useMemo(() => {
    const series = graphSeries.filter((item) => selected.includes(item.name));
    const maxLen = Math.max(
      0,
      ...series.map((item) => decodeArray(item.array).length),
    );
    return {
      labels: Array.from({ length: maxLen }, (_, i) => String(i)),
      datasets: series.map((item) => {
        const color = colorForSeries(item.name);
        return {
          label: item.name,
          data: Array.from(decodeArray(item.array)),
          borderColor: color,
          backgroundColor: color,
          pointRadius: 0,
          borderWidth: 1.8,
        };
      }),
    };
  }, [graphSeries, selected]);

  function buildShockPayload(): Record<string, ReturnType<typeof encodeArray>> {
    const payload: Record<string, ReturnType<typeof encodeArray>> = {};
    for (const spec of shockSpecs) {
      const raw = shockInputs[spec.target]?.trim();
      if (!raw) continue;
      const values = raw
        .split(/[\s,;]+/)
        .filter(Boolean)
        .map((value) => Number(value));
      if (values.some((value) => !Number.isFinite(value))) {
        throw new Error(`Shock path for ${spec.target} contains a non-numeric value.`);
      }
      if (values.length !== simT) {
        throw new Error(
          `Shock path for ${spec.target} has ${values.length} values; expected ${simT}.`,
        );
      }
      payload[spec.target] = encodeArray(new Float64Array(values));
    }
    return payload;
  }

  function buildShockParams(): ShockParamUpdate {
    const std: Record<string, number> = {};
    const corr: Record<string, number> = {};
    for (const spec of shockSpecs) {
      std[spec.shock] = parseFinite(
        shockStdParams[spec.shock] ?? String(spec.std_value ?? 1.0),
        `std for ${spec.shock}`,
      );
    }
    for (const spec of shockCorrSpecs) {
      corr[spec.key] = parseFinite(
        shockCorrParams[spec.key] ?? String(spec.corr_value ?? 0.0),
        `corr for ${spec.key}`,
      );
    }
    return { std, corr };
  }

  function buildShockGeneration(): ShockGeneration {
    return {
      dist: shockDist,
      seed: shockSeed.trim() === "" ? null : parseInteger(shockSeed, "seed"),
      loc: parseFinite(shockLoc, "loc"),
      df: parseFinite(shockDf, "df"),
    };
  }

  return (
    <main className={`app theme-${theme}`}>
      <aside className="sidebar">
        <div className="brand">
          <Database size={22} />
          <div>
            <h1>SymbolicDSGE</h1>
            <span>localhost playground</span>
          </div>
        </div>

        <button
          className="secondary"
          onClick={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
        >
          {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
          {theme === "light" ? "Dark" : "Light"}
        </button>

        <label>
          Role
          <select value={role} onChange={(event) => setRole(event.target.value as Role)}>
            <option value="reference">reference</option>
            <option value="dgp">dgp</option>
          </select>
        </label>

        <label>
          YAML Path
          <input value={path} onChange={(event) => setPath(event.target.value)} />
        </label>

        <div className="button-row">
          <button
            disabled={busy || path.trim() === ""}
            onClick={() =>
              runAction(
                async () => {
                  const loaded = await loadYamlPath(role, path.trim());
                  if (loaded.raw_yaml !== undefined) {
                    setContent(loaded.raw_yaml);
                  }
                },
                "YAML loaded from path.",
              )
            }
          >
            <Upload size={16} />
            Load
          </button>
          <button disabled={busy} onClick={() => runAction(refreshSession, "Ready.")}>
            <RefreshCw size={16} />
            Sync
          </button>
        </div>

        <label className="switch-row">
          <span>Linearize</span>
          <input
            type="checkbox"
            checked={linearize}
            onChange={(event) => setLinearize(event.target.checked)}
          />
        </label>

        <button
          disabled={busy || !activeModel.loaded}
          onClick={() =>
            runAction(
              () => solveModel(role, { linearize }),
              linearize ? "Model linearized and solved." : "Model solved.",
            )
          }
        >
          <Play size={16} />
          Solve
        </button>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <ModelStatus model={activeModel} />
          <span className={message.includes("failed") ? "status error" : "status"}>
            {message}
          </span>
        </header>

        <nav className="view-tabs">
          <button
            className={view === "builder" ? "active" : ""}
            onClick={() => navigate("builder")}
          >
            Builder
          </button>
          <button
            className={view === "spec" ? "active" : ""}
            onClick={() => navigate("spec")}
          >
            Spec
          </button>
          <button
            className={view === "graph" ? "active" : ""}
            onClick={() => navigate("graph")}
          >
            Graph
          </button>
        </nav>

        {view === "builder" ? (
          <BuilderView
            role={role}
            busy={busy}
            theme={theme}
            content={content}
            setContent={setContent}
            beforeEditorMount={configureYamlEditor}
            loadContentAction={() =>
              runAction(
                () => loadYamlContent(role, content),
                "YAML loaded from content.",
              )
            }
            syncAction={() => runAction(refreshSession, "Ready.")}
          />
        ) : view === "spec" ? (
          <SpecView
            activeModel={activeModel}
            shockSpecs={shockSpecs}
            shockCorrSpecs={shockCorrSpecs}
            shockMode={shockMode}
            setShockMode={setShockMode}
            shockDist={shockDist}
            setShockDist={setShockDist}
            shockSeed={shockSeed}
            setShockSeed={setShockSeed}
            shockLoc={shockLoc}
            setShockLoc={setShockLoc}
            shockDf={shockDf}
            setShockDf={setShockDf}
            shockInputs={shockInputs}
            setShockInputs={setShockInputs}
            shockStdParams={shockStdParams}
            setShockStdParams={setShockStdParams}
            shockCorrParams={shockCorrParams}
            setShockCorrParams={setShockCorrParams}
          />
        ) : (
          <GraphView
            busy={busy}
            activeModel={activeModel}
            simT={simT}
            setSimT={setSimT}
            includeObs={includeObs}
            setIncludeObs={setIncludeObs}
            runSimulationAction={() =>
              runAction(async () => {
                const sim = await runSimulation(
                  role,
                  simT,
                  includeObs,
                  shockMode === "raw" ? buildShockPayload() : undefined,
                  shockMode === "generated" ? buildShockGeneration() : undefined,
                  buildShockParams(),
                );
                setResult(sim);
              }, "Simulation complete.")
            }
            result={result}
            graphSeries={graphSeries}
            selected={selected}
            setSelected={setSelected}
            chartData={chartData}
          />
        )}
      </section>
    </main>
  );
}

function BuilderView({
  role,
  busy,
  theme,
  content,
  setContent,
  beforeEditorMount,
  loadContentAction,
  syncAction,
}: {
  role: Role;
  busy: boolean;
  theme: "light" | "dark";
  content: string;
  setContent: Dispatch<SetStateAction<string>>;
  beforeEditorMount: Parameters<typeof Editor>[0]["beforeMount"];
  loadContentAction: () => void;
  syncAction: () => void;
}) {
  return (
    <section className="builder-panel">
      <div className="builder-toolbar">
        <div>
          <h3>Config Builder</h3>
          <p>Target role: {role}</p>
        </div>
        <div className="builder-actions">
          <button disabled={busy || content.trim() === ""} onClick={loadContentAction}>
            <Upload size={16} />
            Load
          </button>
          <button className="secondary" disabled={busy} onClick={syncAction}>
            <RefreshCw size={16} />
            Sync
          </button>
        </div>
      </div>

      <div className="builder-editor">
        <span>YAML Content</span>
        <div className="monaco-shell">
          <Editor
            height="460px"
            language="yaml"
            path={symbolicDsgeConfigModelPath}
            theme={theme === "dark" ? "vs-dark" : "light"}
            value={content}
            beforeMount={beforeEditorMount}
            onChange={(value) => setContent(value ?? "")}
            options={{
              automaticLayout: true,
              fontSize: 13,
              minimap: { enabled: false },
              quickSuggestions: {
                comments: false,
                other: true,
                strings: true,
              },
              suggestOnTriggerCharacters: true,
              scrollBeyondLastLine: false,
              tabSize: 2,
              tabCompletion: "on",
              wordWrap: "on",
            }}
          />
        </div>
      </div>
    </section>
  );
}

function SpecView({
  activeModel,
  shockSpecs,
  shockCorrSpecs,
  shockMode,
  setShockMode,
  shockDist,
  setShockDist,
  shockSeed,
  setShockSeed,
  shockLoc,
  setShockLoc,
  shockDf,
  setShockDf,
  shockInputs,
  setShockInputs,
  shockStdParams,
  setShockStdParams,
  shockCorrParams,
  setShockCorrParams,
}: {
  activeModel: ModelSummary;
  shockSpecs: ShockSpec[];
  shockCorrSpecs: ShockCorrSpec[];
  shockMode: ShockMode;
  setShockMode: Dispatch<SetStateAction<ShockMode>>;
  shockDist: ShockDistribution;
  setShockDist: Dispatch<SetStateAction<ShockDistribution>>;
  shockSeed: string;
  setShockSeed: Dispatch<SetStateAction<string>>;
  shockLoc: string;
  setShockLoc: Dispatch<SetStateAction<string>>;
  shockDf: string;
  setShockDf: Dispatch<SetStateAction<string>>;
  shockInputs: Record<string, string>;
  setShockInputs: Dispatch<SetStateAction<Record<string, string>>>;
  shockStdParams: Record<string, string>;
  setShockStdParams: Dispatch<SetStateAction<Record<string, string>>>;
  shockCorrParams: Record<string, string>;
  setShockCorrParams: Dispatch<SetStateAction<Record<string, string>>>;
}) {
  return (
    <>
      <section className="summary-grid">
        <SummaryBlock title="Variables" values={activeModel.variables ?? []} />
        <SummaryBlock title="Observables" values={activeModel.observables ?? []} />
        <SummaryBlock
          title="State Layout"
          values={[
            `n_state: ${activeModel.n_state ?? "pending"}`,
            `n_exog: ${activeModel.n_exog ?? "pending"}`,
          ]}
        />
      </section>

      <section className="shock-panel">
        <div className="panel-heading">
          <Zap size={16} />
          <h3>Shocks</h3>
        </div>

        <div className="shock-controls">
          <label>
            Source
            <select
              value={shockMode}
              onChange={(event) => setShockMode(event.target.value as ShockMode)}
            >
              <option value="generated">Shock</option>
              <option value="raw">Raw</option>
            </select>
          </label>

          {shockMode === "generated" && (
            <>
              <label>
                Distribution
                <select
                  value={shockDist}
                  onChange={(event) =>
                    setShockDist(event.target.value as ShockDistribution)
                  }
                >
                  <option value="norm">normal</option>
                  <option value="t">student-t</option>
                  <option value="uni">uniform</option>
                </select>
              </label>
              <label>
                Seed
                <input
                  value={shockSeed}
                  onChange={(event) => setShockSeed(event.target.value)}
                />
              </label>
              <label>
                Loc
                <input
                  value={shockLoc}
                  onChange={(event) => setShockLoc(event.target.value)}
                />
              </label>
              <label>
                DF
                <input
                  value={shockDf}
                  disabled={shockDist !== "t"}
                  onChange={(event) => setShockDf(event.target.value)}
                />
              </label>
            </>
          )}
        </div>

        {shockMode === "raw" ? (
          <div className="param-grid">
            {shockSpecs.length === 0 ? (
              <span className="muted">none</span>
            ) : (
              shockSpecs.map((spec) => (
                <label key={spec.target}>
                  {spec.shock} {"->"} {spec.target}
                  <textarea
                    className="shock-input"
                    value={shockInputs[spec.target] ?? ""}
                    onChange={(event) =>
                      setShockInputs((current) => ({
                        ...current,
                        [spec.target]: event.target.value,
                      }))
                    }
                    placeholder="0 0 1 0"
                  />
                </label>
              ))
            )}
          </div>
        ) : (
          <div className="param-grid">
            {shockSpecs.map((spec) => (
              <label key={spec.shock}>
                {spec.shock} std {spec.std_param ? `(${spec.std_param})` : ""}
                <input
                  value={shockStdParams[spec.shock] ?? ""}
                  onChange={(event) =>
                    setShockStdParams((current) => ({
                      ...current,
                      [spec.shock]: event.target.value,
                    }))
                  }
                />
              </label>
            ))}
            {shockCorrSpecs.map((spec) => (
              <label key={spec.key}>
                {spec.key} corr ({spec.corr_param})
                <input
                  value={shockCorrParams[spec.key] ?? ""}
                  onChange={(event) =>
                    setShockCorrParams((current) => ({
                      ...current,
                      [spec.key]: event.target.value,
                    }))
                  }
                />
              </label>
            ))}
          </div>
        )}
      </section>
    </>
  );
}

function GraphView({
  busy,
  activeModel,
  simT,
  setSimT,
  includeObs,
  setIncludeObs,
  runSimulationAction,
  result,
  graphSeries,
  selected,
  setSelected,
  chartData,
}: {
  busy: boolean;
  activeModel: ModelSummary;
  simT: number;
  setSimT: Dispatch<SetStateAction<number>>;
  includeObs: boolean;
  setIncludeObs: Dispatch<SetStateAction<boolean>>;
  runSimulationAction: () => void;
  result: SimResult | null;
  graphSeries: SimResult["series"];
  selected: string[];
  setSelected: Dispatch<SetStateAction<string[]>>;
  chartData: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      borderColor: string;
      backgroundColor: string;
      pointRadius: number;
      borderWidth: number;
    }[];
  };
}) {
  return (
    <>
      <section className="run-panel">
        <label>
          T
          <input
            type="number"
            min={1}
            value={simT}
            onChange={(event) => setSimT(Number(event.target.value))}
          />
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={includeObs}
            onChange={(event) => setIncludeObs(event.target.checked)}
          />
          observables
        </label>
        <button disabled={busy || !activeModel.solved} onClick={runSimulationAction}>
          <Play size={16} />
          Simulate
        </button>
      </section>

      {result !== null && (
        <section className="results">
          <div className="series-list">
            {graphSeries.map((item) => (
              <label key={item.name} className="series-toggle">
                <input
                  type="checkbox"
                  checked={selected.includes(item.name)}
                  onChange={(event) => {
                    setSelected((current) =>
                      event.target.checked
                        ? [...current, item.name]
                        : current.filter((name) => name !== item.name),
                    );
                  }}
                />
                {item.name}
              </label>
            ))}
          </div>
          <div className="chart-wrap">
            <Line
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: { mode: "nearest", intersect: false },
                plugins: { legend: { position: "bottom" } },
                scales: {
                  x: { ticks: { maxTicksLimit: 12 } },
                  y: { beginAtZero: false },
                },
              }}
            />
          </div>
        </section>
      )}
    </>
  );
}

function initialView(): View {
  if (window.location.pathname.endsWith("/builder")) return "builder";
  if (window.location.pathname.endsWith("/graph")) return "graph";
  return "spec";
}

function colorForSeries(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i += 1) {
    hash = (hash * 31 + name.charCodeAt(i)) | 0;
  }
  return SERIES_COLORS[Math.abs(hash) % SERIES_COLORS.length];
}

function configureYamlEditor(monaco: Parameters<typeof configureMonacoYaml>[0]) {
  if (yamlLanguageConfigured) return;
  configureMonacoYaml(monaco, {
    completion: true,
    enableSchemaRequest: false,
    format: { enable: true },
    hover: true,
    validate: true,
    schemas: [
      {
        uri: symbolicDsgeSchemaUri,
        fileMatch: [symbolicDsgeConfigModelPath, "**/*.yaml", "**/*.yml"],
        schema: symbolicDsgeConfigSchema,
      },
    ],
  });
  yamlLanguageConfigured = true;
}

function parseFinite(value: string | undefined, label: string): number {
  const out = Number(value);
  if (!Number.isFinite(out)) {
    throw new Error(`${label} must be numeric.`);
  }
  return out;
}

function parseInteger(value: string, label: string): number {
  const out = Number(value);
  if (!Number.isInteger(out)) {
    throw new Error(`${label} must be an integer.`);
  }
  return out;
}

function ModelStatus({ model }: { model: ModelSummary }) {
  return (
    <div>
      <h2>{model.name ?? model.role}</h2>
      <p>
        {model.loaded ? "loaded" : "empty"} / {model.solved ? "solved" : "unsolved"}
        {model.source ? ` / ${model.source}` : ""}
      </p>
    </div>
  );
}

function SummaryBlock({ title, values }: { title: string; values: string[] }) {
  return (
    <section className="summary-block">
      <h3>{title}</h3>
      <div>
        {values.length === 0 ? (
          <span className="muted">none</span>
        ) : (
          values.map((value) => <span key={value}>{value}</span>)
        )}
      </div>
    </section>
  );
}
