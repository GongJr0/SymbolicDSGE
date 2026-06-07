import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";
import {
  Database,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  RefreshCw,
  RotateCcw,
  Sun,
  Upload,
} from "lucide-react";
import Editor from "@monaco-editor/react";
import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, Dispatch, PointerEvent, SetStateAction } from "react";
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
} from "./configSchema";
import { configureSymbolicDsgeYaml } from "./monacoWorkers";
import { CodePanel } from "./CodePanel";
import type { CodePanelHandle } from "./CodePanel";
import { OutputWorkspace } from "./OutputWorkspace";
import { EstimationView } from "./EstimationView";
import { PanelWorkspace } from "./PanelWorkspace";
import type { PanelDef } from "./PanelWorkspace";
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
  Filler,
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
const MCPipelineView = lazy(() => import("./mc/MCPipelineView"));

type View = "builder" | "spec" | "outputs" | "estimation" | "mc";
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
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [view, setView] = useState<View>(initialView);
  const [message, setMessage] = useState("");
  const [messageIsError, setMessageIsError] = useState(false);
  const [busy, setBusy] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(320);
  const [mcMounted, setMcMounted] = useState(() => initialView() === "mc");

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
  }

  function showMessage(text: string, isError = false) {
    setMessage(text);
    setMessageIsError(isError);
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
      await refreshSession();
      showMessage(done);
    } catch (error) {
      showMessage(error instanceof Error ? error.message : String(error), true);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refreshSession().catch((error: unknown) => {
      showMessage(error instanceof Error ? error.message : String(error), true);
    });
  }, []);

  // Populate the builder with yaml from the backend when the page is refreshed
  // (content state starts empty; session carries the last-loaded yaml)
  useEffect(() => {
    if (session === null) return;
    const yaml = session.models[role]?.raw_yaml;
    if (yaml) setContent((c) => (c === "" ? yaml : c));
  }, [session, role]);

  useEffect(() => {
    if (message === "" || messageIsError) return;
    const timeout = window.setTimeout(() => setMessage(""), 3500);
    return () => window.clearTimeout(timeout);
  }, [message, messageIsError]);

  useEffect(() => {
    const onPopState = () => setView(initialView());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (view === "mc") setMcMounted(true);
  }, [view]);

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

  const chartData = useMemo(() => {
    const series = graphSeries.filter((item) => selected.includes(item.name));
    let maxLen = 0;
    const datasets = series.map((item) => {
      const decoded = decodeArray(item.array);
      if (decoded.length > maxLen) maxLen = decoded.length;
      const color = colorForSeries(item.name);
      return {
        label: item.name,
        data: Array.from(decoded),
        borderColor: color,
        backgroundColor: color,
        pointRadius: 0,
        borderWidth: 1.8,
      };
    });
    return {
      labels: Array.from({ length: maxLen }, (_, i) => String(i)),
      datasets,
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

  function startSidebarResize(event: PointerEvent<HTMLDivElement>) {
    const startX = event.clientX;
    const startWidth = sidebarWidth;

    function move(pointerEvent: globalThis.PointerEvent) {
      setSidebarWidth(
        Math.min(480, Math.max(240, startWidth + pointerEvent.clientX - startX)),
      );
    }

    function stop() {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    }

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }

  return (
    <main
      className={`app theme-${theme}`}
      style={
        {
          "--sidebar-width": `${sidebarCollapsed ? 58 : sidebarWidth}px`,
        } as CSSProperties
      }
    >
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="brand">
          <Database size={22} />
          <div>
            <h1>SymbolicDSGE</h1>
            <span>localhost playground</span>
          </div>
          <button
            className="icon-button sidebar-fold"
            onClick={() => setSidebarCollapsed((current) => !current)}
            title={sidebarCollapsed ? "Expand sidebar" : "Fold sidebar"}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </button>
        </div>

        <button
          className="secondary sidebar-theme"
          onClick={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
          title={theme === "light" ? "Use dark theme" : "Use light theme"}
        >
          {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
          <span>{theme === "light" ? "Dark" : "Light"}</span>
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
          <button
            disabled={busy}
            onClick={() => runAction(async () => undefined, "Session refreshed.")}
          >
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
      <div
        className={`sidebar-resizer ${sidebarCollapsed ? "disabled" : ""}`}
        onPointerDown={sidebarCollapsed ? undefined : startSidebarResize}
        title="Resize sidebar"
      />

      <section className="workspace">
        <header className="topbar">
          <ModelStatus model={activeModel} />
          {message !== "" && (
            <span className={messageIsError ? "status error" : "status"}>{message}</span>
          )}
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
            className={view === "outputs" ? "active" : ""}
            onClick={() => navigate("outputs")}
          >
            Outputs
          </button>
          <button
            className={view === "estimation" ? "active" : ""}
            onClick={() => navigate("estimation")}
          >
            Estimation
          </button>
          <button
            className={view === "mc" ? "active" : ""}
            onClick={() => navigate("mc")}
          >
            MC Pipeline
          </button>
        </nav>

        <BuilderView
          hidden={view !== "builder"}
          role={role}
          busy={busy}
          theme={theme}
          content={content}
          setContent={setContent}
          loadContentAction={() =>
            runAction(
              () => loadYamlContent(role, content),
              "YAML loaded from content.",
            )
          }
          syncAction={() =>
            runAction(async () => undefined, "Session refreshed.")
          }
        />
        <SpecView
          hidden={view !== "spec"}
          role={role}
          theme={theme}
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
        <OutputsView
          hidden={view !== "outputs"}
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
          theme={theme}
        />
        <EstimationView
          hidden={view !== "estimation"}
          role={role}
          model={activeModel}
          onSessionRefresh={refreshSession}
        />
        {mcMounted && (
          <Suspense
            fallback={
              <div className="panel-view">
                <span className="muted">Loading pipeline builder...</span>
              </div>
            }
          >
            <MCPipelineView hidden={view !== "mc"} session={session} />
          </Suspense>
        )}
      </section>
    </main>
  );
}

function BuilderView({
  hidden,
  role,
  busy,
  theme,
  content,
  setContent,
  loadContentAction,
  syncAction,
}: {
  hidden?: boolean;
  role: Role;
  busy: boolean;
  theme: "light" | "dark";
  content: string;
  setContent: Dispatch<SetStateAction<string>>;
  loadContentAction: () => void;
  syncAction: () => void;
}) {
  const panels: PanelDef[] = [
    {
      id: "editor",
      title: "Config Builder",
      badge: role,
      noPadding: true,
      headerActions: (
        <>
          <button disabled={busy || content.trim() === ""} onClick={loadContentAction}>
            <Upload size={16} />
            Load
          </button>
          <button className="secondary" disabled={busy} onClick={syncAction}>
            <RefreshCw size={16} />
            Sync
          </button>
        </>
      ),
      content: (
        <div className="monaco-shell">
          <Editor
            beforeMount={configureSymbolicDsgeYaml}
            height="100%"
            language="yaml"
            path={symbolicDsgeConfigModelPath}
            theme={theme === "dark" ? "vs-dark" : "light"}
            value={content}
            onChange={(value) => setContent(value ?? "")}
            options={{
              automaticLayout: true,
              fontSize: 13,
              minimap: { enabled: false },
              quickSuggestions: { comments: false, other: true, strings: true },
              suggestOnTriggerCharacters: true,
              scrollBeyondLastLine: false,
              tabSize: 2,
              tabCompletion: "on",
              wordWrap: "on",
            }}
          />
        </div>
      ),
    },
  ];

  return (
    <div className="panel-view" style={hidden ? { display: "none" } : undefined}>
      <PanelWorkspace panels={panels} defaultLayout="vertical" />
    </div>
  );
}

function SpecView({
  hidden,
  role,
  theme,
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
  hidden?: boolean;
  role: Role;
  theme: "light" | "dark";
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
  const overviewPanels: PanelDef[] = [
    {
      id: "summary",
      title: "Model",
      badge: activeModel.name ?? activeModel.role,
      defaultHeight: 200,
      content: (
        <div className="summary-grid">
          <SummaryBlock title="Variables" values={activeModel.variables ?? []} />
          <SummaryBlock title="Observables" values={activeModel.observables ?? []} />
          <SummaryBlock
            title="State Layout"
            values={[
              `n_state: ${activeModel.n_state ?? "pending"}`,
              `n_exog: ${activeModel.n_exog ?? "pending"}`,
            ]}
          />
        </div>
      ),
    },
    {
      id: "shocks",
      title: "Shocks",
      defaultHeight: 200,
      scrollable: true,
      content: (
        <>
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
                <label className="shock-num">
                  Seed
                  <input
                    value={shockSeed}
                    onChange={(event) => setShockSeed(event.target.value)}
                  />
                </label>
                <label className="shock-num">
                  Loc
                  <input
                    value={shockLoc}
                    onChange={(event) => setShockLoc(event.target.value)}
                  />
                </label>
                <label className="shock-num">
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
        </>
      ),
    },
  ];

  const arrayPanelRef = useRef<CodePanelHandle>(null);
  const figurePanelRef = useRef<CodePanelHandle>(null);

  const codePanels: PanelDef[] = [
    {
      id: "code-array",
      title: "Transform",
      badge: "Python",
      noPadding: true,
      headerActions: (
        <button
          className="icon-button"
          onClick={() => arrayPanelRef.current?.resetTemplate()}
          title="Reset to default template"
        >
          <RotateCcw size={15} />
        </button>
      ),
      content: (
        <CodePanel ref={arrayPanelRef} kind="array" role={role} activeModel={activeModel} theme={theme} />
      ),
    },
    {
      id: "code-figure",
      title: "Plot",
      badge: "Python",
      noPadding: true,
      headerActions: (
        <button
          className="icon-button"
          onClick={() => figurePanelRef.current?.resetTemplate()}
          title="Reset to default template"
        >
          <RotateCcw size={15} />
        </button>
      ),
      content: (
        <CodePanel ref={figurePanelRef} kind="figure" role={role} activeModel={activeModel} theme={theme} />
      ),
    },
  ];

  return (
    <div className="spec-layout" style={hidden ? { display: "none" } : undefined}>
      <div className="spec-overview-row">
        <PanelWorkspace panels={overviewPanels} defaultLayout="horizontal" defaultSplit={50} />
      </div>
      <div className="spec-code-section">
        <PanelWorkspace panels={codePanels} defaultLayout="horizontal" defaultSplit={50} fillHeight />
      </div>
    </div>
  );
}

function OutputsView({
  hidden,
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
  theme,
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
  theme: "light" | "dark";
  hidden?: boolean;
}) {
  return (
    <div className="panel-view" style={hidden ? { display: "none" } : undefined}>
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
        <label className="switch-row">
          <span>Observables</span>
          <input
            type="checkbox"
            checked={includeObs}
            onChange={(event) => setIncludeObs(event.target.checked)}
          />
        </label>
        <button disabled={busy || !activeModel.solved} onClick={runSimulationAction}>
          <Play size={16} />
          Simulate
        </button>
      </section>

      {result !== null && (
        <OutputWorkspace
          result={result}
          graphSeries={graphSeries}
          selected={selected}
          setSelected={setSelected}
          chartData={chartData}
          theme={theme}
        />
      )}
    </div>
  );
}

function initialView(): View {
  if (window.location.pathname.endsWith("/builder")) return "builder";
  if (window.location.pathname.endsWith("/estimation")) return "estimation";
  if (window.location.pathname.endsWith("/mc")) return "mc";
  if (
    window.location.pathname.endsWith("/outputs") ||
    window.location.pathname.endsWith("/graph")
  ) {
    return "outputs";
  }
  return "spec";
}

function colorForSeries(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i += 1) {
    hash = (hash * 31 + name.charCodeAt(i)) | 0;
  }
  return SERIES_COLORS[Math.abs(hash) % SERIES_COLORS.length];
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
