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
import { ShockRegistryEditor } from "./shocks/ShockRegistryEditor";
import { useShockRegistry } from "./shocks/useShockRegistry";
import type { ShockRegistryState } from "./shocks/useShockRegistry";
import type {
    ModelSummary,
    SessionSummary,
    ShockEntry,
    SimResult,
    SimSpecWire,
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

export default function App() {
    const [session, setSession] = useState<SessionSummary | null>(null);
    const [role, setRole] = useState<string>("");
    const [modelName, setModelName] = useState("");
    const [path, setPath] = useState("MODELS/POST82.yaml");
    const [content, setContent] = useState("");
    const [linearize, setLinearize] = useState(false);
    const [simT, setSimT] = useState(100);
    const [shockScale, setShockScale] = useState(1);
    const [includeObs, setIncludeObs] = useState(true);
    // Initial levels as typed, keyed by variable. Text rather than numbers so a
    // half-entered value stays as written, and blank stays distinct from zero: a
    // variable the dict omits starts at its steady state, which zero would not.
    const [x0Text, setX0Text] = useState<Record<string, string>>({});
    const [result, setResult] = useState<SimResult | null>(null);
    const [selected, setSelected] = useState<string[]>([]);
    const [simShocks, setSimShocks] = useState<ShockEntry[] | null>(null);
    const [theme, setTheme] = useState<"light" | "dark">("dark");
    const [view, setView] = useState<View>(initialView);
    const [message, setMessage] = useState("");
    const [messageIsError, setMessageIsError] = useState(false);
    const [busy, setBusy] = useState(false);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [sidebarWidth, setSidebarWidth] = useState(320);
    const [mcMounted, setMcMounted] = useState(() => initialView() === "mc");

    const activeModel = session?.models[role] ?? { model_name: role, loaded: false, solved: false };
    const shockNames = useMemo(
        () => activeModel.shocks ?? [],
        [activeModel.shocks],
    );
    const variables = useMemo(
        () => activeModel.variables ?? [],
        [activeModel.variables],
    );
    // The simulation's shocks, edited through the same panel the Monte Carlo
    // simulation step uses. The list is component state rather than anything the
    // session holds, which is what the tab's shock form was too.
    const registry = useShockRegistry(simShocks, setSimShocks);
    // An entry names one model's declared innovations, and the other model need
    // not declare them, so switching role drops the list rather than carrying a
    // spec the new model cannot run.
    useEffect(() => {
        setSimShocks(null);
        setX0Text({});
    }, [role]);
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
        setRole((current) =>
            Object.hasOwn(next.models, current)
                ? current
                : Object.keys(next.models)[0] ?? "",
        );
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

    // Seed the Outputs tab from the session: this role's last simulation, which
    // is either a run made here or a bundle's stored spec replayed at load.
    // Once per role, so the refresh after each run does not reset the controls
    // under the user.
    const outputsSeeded = useRef<string | null>(null);
    useEffect(() => {
        if (session === null || outputsSeeded.current === role) return;
        outputsSeeded.current = role;
        const simulation = session.workspace.simulation?.[role];
        if (simulation === undefined) return;
        if (simulation.spec !== undefined) {
            setSimT(simulation.spec.T);
            setIncludeObs(simulation.spec.observables);
        }
        setResult(simulation.result ?? null);
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

    // Only the variables someone typed a level for travel. The rest are omitted,
    // which the library reads as the steady state, and an all-blank grid posts
    // null so the run starts from the steady state throughout. Reading the model's
    // variable list rather than the form's keys leaves a level typed against an
    // earlier model inert instead of sending an unknown name.
    function buildX0(): Record<string, number> | null {
        const levels: Record<string, number> = {};
        for (const name of variables) {
            const raw = (x0Text[name] ?? "").trim();
            if (raw === "") continue;
            const value = Number(raw);
            if (!Number.isFinite(value)) {
                throw new Error(`Initial level for ${name} is not a number.`);
            }
            levels[name] = value;
        }
        return Object.keys(levels).length === 0 ? null : levels;
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
                    Model
                    <select
                        value={role}
                        disabled={busy || Object.keys(session?.models ?? {}).length === 0}
                        onChange={(event) => setRole(event.target.value)}
                    >
                        {Object.keys(session?.models ?? {}).length === 0 && (
                            <option value="">No models loaded</option>
                        )}
                        {Object.keys(session?.models ?? {}).map((name) => (
                            <option key={name} value={name}>{name}</option>
                        ))}
                    </select>
                </label>

                <label>
                    Model name
                    <input
                        value={modelName}
                        onChange={(event) => setModelName(event.target.value)}
                        placeholder="Name for the loaded model"
                    />
                </label>

                <label>
                    YAML Path
                    <input value={path} onChange={(event) => setPath(event.target.value)} />
                </label>

                <label>
                    Load YAML file
                    <input
                        type="file"
                        accept=".yaml,.yml"
                        disabled={busy || modelName.trim() === ""}
                        onChange={(event) => {
                            const file = event.target.files?.[0];
                            event.target.value = "";
                            if (!file) return;
                            void runAction(
                                async () => {
                                    const yaml = await file.text();
                                    const loaded = await loadYamlContent(modelName.trim(), yaml);
                                    setRole(loaded.model_name);
                                    setContent(loaded.raw_yaml ?? yaml);
                                },
                                `YAML loaded from ${file.name}.`,
                            );
                        }}
                    />
                </label>

                <div className="button-row">
                    <button
                        disabled={busy || modelName.trim() === "" || path.trim() === ""}
                        onClick={() =>
                            runAction(
                                async () => {
                                    const loaded = await loadYamlPath(modelName.trim(), path.trim());
                                    setRole(loaded.model_name);
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
                    canLoad={modelName.trim() !== ""}
                    loadContentAction={() =>
                        runAction(
                            async () => {
                                const loaded = await loadYamlContent(modelName.trim(), content);
                                setRole(loaded.model_name);
                            },
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
                    shockNames={shockNames}
                    registry={registry}
                />
                <OutputsView
                    hidden={view !== "outputs"}
                    busy={busy}
                    activeModel={activeModel}
                    simT={simT}
                    setSimT={setSimT}
                    shockScale={shockScale}
                    setShockScale={setShockScale}
                    x0Text={x0Text}
                    setX0Text={setX0Text}
                    includeObs={includeObs}
                    setIncludeObs={setIncludeObs}
                    runSimulationAction={() =>
                        runAction(async () => {
                            const sim = await runSimulation(role, {
                                T: simT,
                                x0: buildX0(),
                                shock_scale: shockScale,
                                observables: includeObs,
                                shocks: simShocks,
                            } satisfies SimSpecWire);
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
                    workspace={session?.workspace ?? null}
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
                        <MCPipelineView hidden={view !== "mc"} session={session} theme={theme} />
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
    canLoad,
    syncAction,
}: {
    hidden?: boolean;
    role: string;
    busy: boolean;
    theme: "light" | "dark";
    content: string;
    setContent: Dispatch<SetStateAction<string>>;
    loadContentAction: () => void;
    canLoad: boolean;
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
                    <button disabled={busy || !canLoad || content.trim() === ""} onClick={loadContentAction}>
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
    shockNames,
    registry,
}: {
    hidden?: boolean;
    role: string;
    theme: "light" | "dark";
    activeModel: ModelSummary;
    shockNames: string[];
    registry: ShockRegistryState;
}) {
    const overviewPanels: PanelDef[] = [
        {
            id: "summary",
            title: "Model",
            badge: activeModel.name ?? activeModel.model_name,
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
                    <ShockRegistryEditor
                        role={role}
                        shockNames={shockNames}
                        entries={registry.entries}
                        onChange={registry.setRegistry}
                    />
                    {registry.error !== "" && (
                        <span className="status error shock-registry-error">
                            {registry.error}
                        </span>
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
    shockScale,
    setShockScale,
    x0Text,
    setX0Text,
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
    shockScale: number;
    setShockScale: Dispatch<SetStateAction<number>>;
    x0Text: Record<string, string>;
    setX0Text: Dispatch<SetStateAction<Record<string, string>>>;
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
                <label>
                    Shock scale
                    <input
                        type="number"
                        step={0.1}
                        value={shockScale}
                        onChange={(event) => setShockScale(Number(event.target.value))}
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

            <X0Panel
                variables={activeModel.variables ?? []}
                x0Text={x0Text}
                setX0Text={setX0Text}
            />

            {(result?.transform_errors ?? []).length > 0 && (
                <section className="transform-errors">
                    {result?.transform_errors?.map((failure) => (
                        <span key={failure.name} className="status error">
                            {`${failure.name}: ${failure.error}`}
                        </span>
                    ))}
                </section>
            )}

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

function X0Panel({
    variables,
    x0Text,
    setX0Text,
}: {
    variables: string[];
    x0Text: Record<string, string>;
    setX0Text: Dispatch<SetStateAction<Record<string, string>>>;
}) {
    const set = variables.filter((name) => (x0Text[name] ?? "").trim() !== "").length;
    return (
        <details className="x0-panel">
            <summary>
                Initial state
                <span className="muted">
                    {set === 0 ? "steady state" : `${set} set`}
                </span>
            </summary>
            {variables.length === 0 ? (
                <span className="muted">Solve the model to set initial levels.</span>
            ) : (
                <>
                    <span className="muted">
                        Levels, not deviations. A variable left blank starts at its steady
                        state.
                    </span>
                    <div className="param-grid">
                        {variables.map((name) => (
                            <label key={name}>
                                {name}
                                <input
                                    type="number"
                                    value={x0Text[name] ?? ""}
                                    onChange={(event) =>
                                        setX0Text((current) => ({
                                            ...current,
                                            [name]: event.target.value,
                                        }))
                                    }
                                />
                            </label>
                        ))}
                    </div>
                </>
            )}
        </details>
    );
}

function ModelStatus({ model }: { model: ModelSummary }) {
    return (
        <div>
            <h2>{model.name ?? model.model_name}</h2>
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
