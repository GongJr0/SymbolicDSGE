import {
  Background,
  Controls,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from "@xyflow/react";
import type {
  Connection,
  Edge,
  EdgeChange,
  NodeChange,
  NodeMouseHandler,
  XYPosition,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Check,
  Play,
  Plus,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { DragEvent, PointerEvent } from "react";
import {
  fetchAvailableTraces,
  getMCCustomTemplate,
  putWorkspaceView,
  runMCPipeline,
  validateMCPipeline,
} from "../api";
import { PanelWorkspace } from "../PanelWorkspace";
import type { PanelDef } from "../PanelWorkspace";
import type {
  MCCatalog,
  MCEdgeSpec,
  MCPipelineResult,
  MCPipelineSpec,
  MCStepSpec,
  MCViewState,
  MCStepCatalogItem,
  MCStepCategory,
  Role,
  SessionSummary,
} from "../types";
import { StepInspector } from "./StepInspector";
import { StepNode } from "./StepNode";
import { MCResultPanel } from "./MCResultPanel";
import type { MCFlowNode, MCProducer } from "./types";

import type { MCStepDefinition } from "./catalog";
import { MC_CATALOG, stepDefinition } from "./catalog";
import { defaultStep, getField, setField } from "./fields";

const nodeTypes = { mcStep: StepNode };

// Starter source for a custom post-loop op. The pandas namespace applies, so the
// body may reference `pd` (and `np`); `import` stays banned, like `np`.
const POSTPROC_CUSTOM_TEMPLATE = `@pandas_operation
def postproc_op(*, traces):
    """Post-loop summary over the across-rep traces. Runs once after the loop.

    \`traces\` maps producer keys (e.g. "test.<name>.pval", "regression.<name>.coef",
    "payload.<name>") to length-R ndarrays. Return a scalar (-> Summary), an
    ndarray (-> Raw), a DataFrame (-> table), or a dict of several. \`pd\` and
    \`np\` are available (no imports).
    """
    pvals = traces["test.example.pval"]
    return pd.DataFrame({"rep": np.arange(pvals.size), "pval": pvals})
`;

export default function MCPipelineView({
  hidden,
  session,
  theme,
}: {
  hidden?: boolean;
  session: SessionSummary | null;
  theme: "light" | "dark";
}) {
  return (
    <div className="mc-layout" style={hidden ? { display: "none" } : undefined}>
      <ReactFlowProvider>
        <MCPipelineBuilder session={session} theme={theme} />
      </ReactFlowProvider>
    </div>
  );
}

function MCPipelineBuilder({
  session,
  theme,
}: {
  session: SessionSummary | null;
  theme: "light" | "dark";
}) {
  const customTemplateRef = useRef<string>("");
  const [catalog, setCatalog] = useState<MCCatalog | null>(null);
  const [nodes, setNodes, onNodesChangeBase] = useNodesState<MCFlowNode>([]);
  const [edges, setEdges, onEdgesChangeBase] = useEdgesState<Edge>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [nRep, setNRep] = useState(100);
  const [nJobs, setNJobs] = useState<number | null>(null);
  const [verbosity, setVerbosity] = useState(0);
  const [failFast, setFailFast] = useState(true);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [noticeError, setNoticeError] = useState(false);
  const [result, setResult] = useState<MCPipelineResult | null>(null);
  const [hydrated, setHydrated] = useState(false);
  // Read inside the one-shot hydration effect, so a later session refresh
  // updates the pills without re-seeding the canvas under the user.
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const [summaryShare, setSummaryShare] = useState(36);
  const [summaryFolded, setSummaryFolded] = useState(false);
  const { screenToFlowPosition, fitView, getViewport } =
    useReactFlow<MCFlowNode, Edge>();

  useEffect(() => {
    getMCCustomTemplate()
      .catch(() => ({ template: "" }))
      .then(async (tmpl) => {
        customTemplateRef.current = tmpl.template;
        const value: MCCatalog = { steps: MC_CATALOG };
        setCatalog(value);
        // Seeded from the session, which outlived the page. Read once: later
        // reads carry back only what was PUT from here.
        const mc = sessionRef.current?.workspace.mc ?? null;
        const view = mc?.view ?? null;
        // A bundle fills `spec`, not `view`: the pipeline is what it stores,
        // the canvas is not. Falling back to it is what makes a bundled run
        // visible when no result rode along with it.
        const pipeline = view?.pipeline ?? mc?.spec ?? null;
        const restored = restoreNodes(
          pipeline,
          { positions: view?.positions ?? {}, edges: view?.edges ?? [] },
          value,
        );
        if (restored !== null) {
          setNodes(restored.nodes);
          setEdges(restored.edges);
          setNRep(view?.n_rep ?? 100);
          setNJobs(view?.n_jobs ?? null);
          setVerbosity(view?.verbosity ?? 0);
          setFailFast(view?.fail_fast ?? true);
        } else {
          const simulation = stepDefinition("simulation");
          if (simulation !== undefined) {
            setNodes([makeNode(simulation, { x: 100, y: 140 }, [])]);
          }
        }
        setResult(mc?.result ?? null);
        setHydrated(true);
      })
      .catch((error: unknown) => {
        setNotice(error instanceof Error ? error.message : String(error));
        setNoticeError(true);
        setHydrated(true);
      });
  }, [setEdges, setNodes]);

  const selectedNode = nodes.find((node) => node.id === selectedId) ?? null;
  // A step may only read from its graph ancestors — producers reachable by
  // walking edges backward — so the flow arrows actually gate the source
  // dropdown. Without this, a step wired to one transform could still pick an
  // unrelated branch. Terminals (tests/regressions/postprocs) can't be sources.
  const ancestorIds = useMemo(
    () =>
      selectedId === null
        ? new Set<string>()
        : collectAncestorIds(selectedId, edges),
    [selectedId, edges],
  );
  const producers: MCProducer[] = nodes
    .filter((node) => ancestorIds.has(node.id))
    .map((node) => {
      const kind = sourceProducerKind(node);
      return kind === null ? null : { name: node.data.step.name, kind };
    })
    .filter((entry): entry is MCProducer => entry !== null);
  const pipeline = useMemo(() => toPipelineSpec(nodes), [nodes]);

  // The producible across-rep traces a POSTPROC op can consume. Refreshed from
  // the backend registry whenever the pipeline's producers change (debounced),
  // so the trace picker stays in sync without duplicating the key format here.
  const [availableTraces, setAvailableTraces] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    const handle = window.setTimeout(() => {
      fetchAvailableTraces(pipeline)
        .then((result) => {
          if (!cancelled) setAvailableTraces(result.traces);
        })
        .catch(() => {
          if (!cancelled) setAvailableTraces([]);
        });
    }, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [pipeline]);

  // Color edges by their producer's kind so a node's inputs are readable at a
  // glance (datagen / filter / transform) without opening the inspector.
  const styledEdges = useMemo(
    () =>
      edges.map((edge) => {
        const source = nodes.find((node) => node.id === edge.source);
        const color = producerColor(source?.data.step.step_type);
        return {
          ...edge,
          style: { ...edge.style, stroke: color, strokeWidth: 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color },
        };
      }),
    [edges, nodes],
  );
  const modelsReady =
    session?.models.reference?.solved === true && session.models.dgp?.solved === true;

  // Exogenous shock variables per model role, sourced from the loaded model
  // configs (independent of the pipeline), for the simulation shock checklist.
  const exogByRole: Record<Role, string[]> = useMemo(
    () => ({
      reference: (session?.models.reference?.shock_specs ?? []).map(
        (spec) => spec.shock,
      ),
      dgp: (session?.models.dgp?.shock_specs ?? []).map((spec) => spec.shock),
    }),
    [session],
  );

  const markDirty = useCallback(() => {
    setNotice("");
    setNoticeError(false);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    const timeout = window.setTimeout(() => {
      void putWorkspaceView("mc", {
        // The graph rides the view: a spec only reaches the tab's `spec` slot
        // once a run writes it, and an edited graph has not run yet.
        pipeline,
        positions: Object.fromEntries(
          // Keyed by name: the document carries no canvas id, so a name is what
          // a restored step can be looked up by.
          nodes.map((node) => [
            node.data.step.name,
            { x: node.position.x, y: node.position.y },
          ]),
        ),
        // Names, for the same reason, and only the pair: the rest of a React
        // Flow edge is styling this view reapplies on restore.
        edges: edges.map((edge) => ({ source: edge.source, target: edge.target })),
        n_rep: nRep,
        n_jobs: nJobs,
        verbosity,
        fail_fast: failFast,
      }).catch((error: unknown) => {
        setNotice(error instanceof Error ? error.message : String(error));
        setNoticeError(true);
      });
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [edges, failFast, hydrated, nJobs, nRep, nodes, pipeline, verbosity]);

  const onNodesChange = useCallback(
    (changes: NodeChange<MCFlowNode>[]) => {
      onNodesChangeBase(changes);
      if (changes.some((change) => change.type === "remove")) markDirty();
    },
    [markDirty, onNodesChangeBase],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => {
      onEdgesChangeBase(changes);
      markDirty();
    },
    [markDirty, onEdgesChangeBase],
  );

  const isValidConnection = useCallback(
    (connection: Connection | Edge) => {
      if (!connection.source || !connection.target) return false;
      if (connection.source === connection.target) return false;
      const source = nodes.find((node) => node.id === connection.source);
      const target = nodes.find((node) => node.id === connection.target);
      if (source === undefined || target === undefined) return false;
      // Postproc ops are a terminal phase referenced by trace key, not edges;
      // they are never wired into the DAG.
      if (isPostprocNode(source) || isPostprocNode(target)) return false;
      if (
        [
          "wald",
          "ljung_box",
          "jarque_bera",
          "breusch_pagan",
          "breusch_godfrey",
          "cusum",
          "cusumsq",
          "chow",
          "regression",
        ].includes(source.data.step.step_type)
      ) {
        return false;
      }
      if (target?.data.step.step_type === "simulation") return false;
      if (
        target.data.step.step_type === "filter" &&
        source.data.step.step_type !== "simulation"
      ) {
        return false;
      }
      // A node may now take several incoming edges — one per input leg (e.g. a
      // payload from a transform + a filter source). Only reject duplicate
      // edges between the same pair.
      if (
        edges.some(
          (edge) =>
            edge.source === connection.source && edge.target === connection.target,
        )
      ) {
        return false;
      }
      return true;
    },
    [edges, nodes],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!isValidConnection(connection)) return;
      setEdges((current) => addEdge({ ...connection, type: "smoothstep" }, current));
      markDirty();
    },
    [isValidConnection, markDirty, setEdges],
  );

  const onNodeClick: NodeMouseHandler<MCFlowNode> = useCallback((_, node) => {
    setSelectedId(node.id);
  }, []);

  function updateNode(updated: MCFlowNode) {
    setNodes((current) =>
      current.map((node) => (node.id === updated.id ? updated : node)),
    );
    markDirty();
  }

  function deleteNode(id: string) {
    setNodes((current) => current.filter((node) => node.id !== id));
    setEdges((current) =>
      current.filter((edge) => edge.source !== id && edge.target !== id),
    );
    setSelectedId((current) => (current === id ? null : current));
    markDirty();
  }

  // Spawn near the top-left of the *current* viewport (in flow coordinates),
  // cascading a little so repeated clicks don't stack — instead of marching off
  // to the right as if the canvas were a single uniform row.
  function viewportSpawnPosition(existingCount: number): { x: number; y: number } {
    const viewport = getViewport();
    const baseX = (80 - viewport.x) / viewport.zoom;
    const baseY = (80 - viewport.y) / viewport.zoom;
    const cascade = (existingCount % 6) * 28;
    return { x: baseX + cascade, y: baseY + cascade };
  }

  function addStep(item: MCStepCatalogItem, position?: { x: number; y: number }) {
    const definition = stepDefinition(item.step_type);
    if (definition === undefined) return;
    setNodes((current) => [
      ...current,
      makeNode(
        definition,
        position ?? viewportSpawnPosition(current.length),
        current,
        customTemplateRef.current,
      ),
    ]);
    markDirty();
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (catalog === null) return;
    const stepType = event.dataTransfer.getData("application/sdsge-mc-step");
    const item = catalog.steps.find((step) => step.step_type === stepType);
    if (item === undefined) return;
    addStep(item, screenToFlowPosition({ x: event.clientX, y: event.clientY }));
  }

  async function validate() {
    setBusy(true);
    try {
      const response = await validateMCPipeline(pipeline);
      const total = response.steps.length + response.postprocs.length;
      setNotice(`Valid dependency graph: ${total} executable steps.`);
      setNoticeError(false);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
      setNoticeError(true);
    } finally {
      setBusy(false);
    }
  }

  async function run() {
    setBusy(true);
    try {
      const output = await runMCPipeline(pipeline, nRep, nJobs, failFast, verbosity);
      setResult(output);
      setNotice(
        `MC run completed: ${output.n_successful}/${output.n_rep} replications at ${output.it_s.toFixed(2)} it/s.`,
      );
      setNoticeError(!output.succeeded);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
      setNoticeError(true);
    } finally {
      setBusy(false);
    }
  }

  async function clearWorkspace() {
    resetPipeline();
    setResult(null);
    setNRep(100);
    setNJobs(null);
    setVerbosity(0);
    setFailFast(true);
    try {
      await putWorkspaceView("mc", null);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
      setNoticeError(true);
    }
  }

  function resetPipeline() {
    const simulation = stepDefinition("simulation");
    setNodes(simulation ? [makeNode(simulation, { x: 100, y: 140 }, [])] : []);
    setEdges([]);
    setSelectedId(null);
    setNotice("");
    setNoticeError(false);
    window.requestAnimationFrame(() => void fitView());
  }

  const canvasPanels: PanelDef[] = [
    {
      id: "steps",
      title: "Steps",
      badge: catalog ? `${catalog.steps.length}` : undefined,
      scrollable: true,
      content: <StepPalette catalog={catalog} onAdd={addStep} />,
    },
    {
      id: "pipeline",
      title: "Pipeline",
      badge: `${nodes.length} steps`,
      noPadding: true,
      headerActions: (
        <button className="icon-button" onClick={resetPipeline} title="Clear pipeline">
          <Trash2 size={15} />
        </button>
      ),
      content: (
        <div
          className="mc-canvas-shell"
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          <ReactFlow
            nodes={nodes}
            edges={styledEdges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={() => setSelectedId(null)}
            isValidConnection={isValidConnection}
            deleteKeyCode={["Backspace", "Delete"]}
            fitView
            minZoom={0.25}
          >
            <Background gap={20} size={1} />
            <Controls />
          </ReactFlow>
          <EdgeLegend />
        </div>
      ),
    },
    {
      id: "inspector",
      title: "Step Inspector",
      badge: selectedNode?.data.step.name,
      scrollable: true,
      content: (
        <StepInspector
          node={selectedNode}
          onChange={updateNode}
          onDelete={deleteNode}
          theme={theme}
          producers={producers}
          availableTraces={availableTraces}
          exogByRole={exogByRole}
        />
      ),
    },
  ];

  const resultPanels: PanelDef[] = [
    {
      id: "mc-results",
      title: "Run Summary",
      badge: result ? `${result.n_successful}/${result.n_rep} successful` : undefined,
      scrollable: true,
      content: <MCResultPanel result={result} />,
    },
  ];

  function startSummaryResize(event: PointerEvent<HTMLDivElement>) {
    const stack = event.currentTarget.parentElement;
    if (stack === null) return;
    const rect = stack.getBoundingClientRect();
    const startY = event.clientY;
    const startShare = summaryShare;
    const move = (next: globalThis.PointerEvent) => {
      const delta = ((next.clientY - startY) / rect.height) * 100;
      setSummaryShare(Math.min(72, Math.max(20, startShare - delta)));
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }

  return (
    <>
      <section className="mc-runbar">
        <div className="mc-model-readiness">
          <ModelPill label="Reference" ready={session?.models.reference?.solved === true} />
          <ModelPill label="DGP" ready={session?.models.dgp?.solved === true} />
        </div>
        <label>
          Replications
          <input
            type="number"
            min={1}
            value={nRep}
            onChange={(event) => setNRep(Number(event.target.value))}
          />
        </label>
        <label>
          Workers
          <input
            type="number"
            min={1}
            placeholder="auto"
            value={nJobs ?? ""}
            onChange={(event) => {
              const value = event.target.value;
              setNJobs(value === "" ? null : Number(value));
            }}
          />
        </label>
        <label>
          Verbosity
          <select
            value={verbosity}
            onChange={(event) => setVerbosity(Number(event.target.value))}
          >
            <option value={0}>Quiet</option>
            <option value={1}>Run summary</option>
            <option value={2}>Step timings</option>
          </select>
        </label>
        <label className="switch-row">
          <span>Fail fast</span>
          <input
            type="checkbox"
            checked={failFast}
            onChange={(event) => setFailFast(event.target.checked)}
          />
        </label>
        <button className="secondary" disabled={busy} onClick={() => void validate()}>
          <Check size={15} />
          Validate
        </button>
        <button disabled={busy || !modelsReady} onClick={() => void run()}>
          <Play size={15} />
          Run pipeline
        </button>
        <button
          className="secondary"
          disabled={busy}
          onClick={() => void clearWorkspace()}
        >
          <Trash2 size={15} />
          Clear workspace
        </button>
        {notice !== "" && (
          <span className={noticeError ? "status error mc-notice" : "status mc-notice"}>
            {noticeError && <TriangleAlert size={14} />}
            {notice}
          </span>
        )}
      </section>
      <div className="mc-panel-stack">
        <div
          className="mc-main-row"
          style={{ flex: summaryFolded ? "1 1 auto" : `${100 - summaryShare} 1 0` }}
        >
          <PanelWorkspace
            panels={canvasPanels}
            defaultLayout="horizontal"
            defaultSizes={[18, 54, 28]}
            fillHeight
          />
        </div>
        {!summaryFolded && (
          <div
            className="mc-row-splitter"
            onPointerDown={startSummaryResize}
            title="Resize run summary"
          />
        )}
        <div
          className="mc-results-row"
          style={{ flex: summaryFolded ? "0 0 42px" : `${summaryShare} 1 0` }}
        >
          <PanelWorkspace
            panels={resultPanels}
            defaultLayout="vertical"
            fillHeight
            onFoldChange={(folded) => setSummaryFolded(Boolean(folded["mc-results"]))}
          />
        </div>
      </div>
    </>
  );
}

const STEP_CATEGORIES: { id: MCStepCategory; label: string }[] = [
  { id: "core", label: "Core" },
  { id: "transforms", label: "Transforms" },
  { id: "tests", label: "Tests" },
  { id: "regressions", label: "Regressions" },
  { id: "postproc", label: "Postproc" },
];

// Edge colors by producer kind, so each consumer's inputs are distinguishable.
const EDGE_KINDS: { kind: string; label: string; color: string }[] = [
  { kind: "datagen", label: "Datagen", color: "#2563eb" },
  { kind: "filter", label: "Filter", color: "#0d9488" },
  { kind: "transform", label: "Transform", color: "#7c3aed" },
];
const EDGE_COLOR_BY_KIND: Record<string, string> = Object.fromEntries(
  EDGE_KINDS.map((entry) => [entry.kind, entry.color]),
);

function producerKind(stepType: string | undefined): string {
  if (stepType === "simulation" || stepType === "raw_model_data") return "datagen";
  if (stepType === "filter") return "filter";
  return "transform";
}

// The source-leg producer kind of a node, or null when it can't be a source
// (tests/regressions/postprocs). Unlike `producerKind`, this distinguishes
// non-producers rather than defaulting them to "transform".
function sourceProducerKind(node: MCFlowNode): MCProducer["kind"] | null {
  const stepType: string = node.data.step.step_type;
  if (stepType === "simulation" || stepType === "raw_model_data") return "datagen";
  if (stepType === "filter") return "filter";
  if (stepDefinition(stepType)?.category === "transforms") return "transform";
  return null;
}

// Node ids reachable by walking edges backward from `nodeId` (its ancestors) —
// the only producers a step may legally read from. The filter's implicit read of
// the datagen still comes through as an edge on the canvas, so a test wired to
// the filter sees both the filter and the datagen behind it.
function collectAncestorIds(nodeId: string, edges: Edge[]): Set<string> {
  const parentsByTarget = new Map<string, string[]>();
  for (const edge of edges) {
    const parents = parentsByTarget.get(edge.target);
    if (parents) parents.push(edge.source);
    else parentsByTarget.set(edge.target, [edge.source]);
  }
  const ancestors = new Set<string>();
  const stack = [...(parentsByTarget.get(nodeId) ?? [])];
  while (stack.length > 0) {
    const current = stack.pop() as string;
    if (ancestors.has(current)) continue;
    ancestors.add(current);
    for (const parent of parentsByTarget.get(current) ?? []) stack.push(parent);
  }
  return ancestors;
}

function producerColor(stepType: string | undefined): string {
  return EDGE_COLOR_BY_KIND[producerKind(stepType)] ?? "#94a3b8";
}

function EdgeLegend() {
  return (
    <div className="mc-edge-legend">
      {EDGE_KINDS.map((entry) => (
        <span key={entry.kind} className="mc-edge-legend-item">
          <span
            className="mc-edge-legend-swatch"
            style={{ background: entry.color }}
          />
          {entry.label}
        </span>
      ))}
    </div>
  );
}

function StepPalette({
  catalog,
  onAdd,
}: {
  catalog: MCCatalog | null;
  onAdd: (item: MCStepCatalogItem) => void;
}) {
  const [activeTab, setActiveTab] = useState<MCStepCategory>("core");
  const steps = catalog?.steps ?? [];
  const visible = steps.filter((item) => item.category === activeTab);

  return (
    <div className="mc-palette">
      <div className="mc-palette-tabs" role="tablist">
        {STEP_CATEGORIES.map((category) => {
          const count = steps.filter((step) => step.category === category.id).length;
          return (
            <button
              key={category.id}
              role="tab"
              aria-selected={activeTab === category.id}
              className={`mc-palette-tab${activeTab === category.id ? " active" : ""}`}
              onClick={() => setActiveTab(category.id)}
            >
              {category.label}
              {count > 0 && <span className="mc-palette-tab-count">{count}</span>}
            </button>
          );
        })}
      </div>
      <div className="mc-palette-steps">
        {visible.length === 0 ? (
          <span className="muted mc-palette-empty">No steps in this group.</span>
        ) : (
          visible.map((item) => (
            <button
              key={item.step_type}
              className="mc-palette-step"
              draggable
              onDragStart={(event) => {
                event.dataTransfer.effectAllowed = "copy";
                event.dataTransfer.setData("application/sdsge-mc-step", item.step_type);
              }}
              onClick={() => onAdd(item)}
              title={item.description}
            >
              <Plus size={13} />
              {item.title}
            </button>
          ))
        )}
      </div>
    </div>
  );
}

function ModelPill({ label, ready }: { label: string; ready: boolean }) {
  return (
    <span className={`mc-model-pill ${ready ? "ready" : ""}`}>
      {ready ? <Check size={12} /> : <TriangleAlert size={12} />}
      {label}
    </span>
  );
}

function makeNode(
  definition: MCStepDefinition,
  position: { x: number; y: number },
  existing: MCFlowNode[],
  customTemplate = "",
): MCFlowNode {
  const count = existing.filter(
    (node) => node.data.step.step_type === definition.step_type,
  ).length;
  const name =
    count === 0
      ? definition.default_name
      : `${definition.default_name}_${count + 1}`;
  const code =
    definition.step_type === "transform:custom"
      ? customTemplate
      : definition.step_type === "postproc:custom"
        ? POSTPROC_CUSTOM_TEMPLATE
        : "";
  return {
    id: `${definition.step_type}-${crypto.randomUUID()}`,
    type: "mcStep",
    position,
    data: { step: defaultStep(definition, name, code) },
  };
}

function isPostprocNode(node: MCFlowNode): boolean {
  return stepDefinition(node.data.step.step_type)?.category === "postproc";
}

/** The canvas as the document the server takes: two step lists and nothing else.
 *
 * The phase split reads each step's own `op_type` rather than asking the
 * catalog what category the node was drawn from, which is the same
 * discrimination the library makes. Edges do not travel: a step names the
 * producers it reads in `source_args`, which is what the server orders the
 * graph from.
 */
function toPipelineSpec(nodes: MCFlowNode[]): MCPipelineSpec {
  const steps = nodes.map((node) => node.data.step);
  return {
    replication_steps: steps.filter((step) => step.op_type !== "postproc"),
    postproc_steps: steps.filter((step) => step.op_type === "postproc"),
  };
}

/** Somewhere to put a node the canvas has no remembered position for.
 *
 * A bundle stores the pipeline that ran, not the canvas it was drawn on, so a
 * restored run arrives with a graph and no coordinates. Layers by longest path
 * from a source, which reads left to right in dependency order.
 */
/** The dependency pairs a document carries, as `(producer, consumer)` names.
 *
 * A step names the producers it reads in its source legs, which is the record
 * the server orders the graph from. The canvas edge list is a separate thing:
 * it also holds connections drawn before their leg was bound, which no document
 * records. Those ride `MCViewState.edges`, and `restoreNodes` lays the two over
 * each other.
 */
function documentEdges(pipeline: MCPipelineSpec): MCEdgeSpec[] {
  const names = new Set(pipeline.replication_steps.map((step) => step.name));
  const seen = new Set<string>();
  const out: MCEdgeSpec[] = [];
  for (const step of pipeline.replication_steps) {
    for (const leg of step.source_args) {
      const key = `${leg.source_step}->${step.name}`;
      if (!names.has(leg.source_step) || seen.has(key)) continue;
      seen.add(key);
      out.push({ source: leg.source_step, target: step.name });
    }
  }
  return out;
}

/** Somewhere to put a step the canvas has no remembered position for.
 *
 * A bundle stores the pipeline that ran, not the canvas it was drawn on, so a
 * restored run arrives with steps and no coordinates. Layers by longest path
 * through the source legs, which reads left to right in dependency order.
 */
function autoLayout(pipeline: MCPipelineSpec): Record<string, XYPosition> {
  const incoming = new Map<string, string[]>();
  for (const step of pipeline.replication_steps) incoming.set(step.name, []);
  for (const edge of documentEdges(pipeline)) incoming.get(edge.target)?.push(edge.source);

  const depth = new Map<string, number>();
  const walking = new Set<string>();
  function depthOf(name: string): number {
    const cached = depth.get(name);
    if (cached !== undefined) return cached;
    // A spec is a DAG; the guard only keeps a malformed one from recursing.
    if (walking.has(name)) return 0;
    walking.add(name);
    const sources = incoming.get(name) ?? [];
    const value = sources.length === 0 ? 0 : Math.max(...sources.map(depthOf)) + 1;
    depth.set(name, value);
    return value;
  }

  const filled = new Map<number, number>();
  const positions: Record<string, XYPosition> = {};
  for (const step of pipeline.replication_steps) {
    const column = depthOf(step.name);
    const row = filled.get(column) ?? 0;
    filled.set(column, row + 1);
    positions[step.name] = { x: 100 + column * 260, y: 140 + row * 130 };
  }
  // Postprocs consume the finished run rather than a step, so they trail it.
  const trailing = Math.max(0, ...[...filled.keys()].map((column) => column + 1));
  pipeline.postproc_steps.forEach((step, index) => {
    positions[step.name] = { x: 100 + trailing * 260, y: 140 + index * 130 };
  });
  return positions;
}

/** Canvas nodes for a stored document.
 *
 * A step is keyed by its name, since the document carries no canvas id and a
 * name is unique across a pipeline. The form params are rebuilt from every
 * place the post scattered them: `kwargs` for the plain fields, `source_args`
 * for the legs, and the step's own slots for `n_retain` and `code`. Anything
 * the document does not name stays on its catalogue default.
 */
function restoreNodes(
  pipeline: MCPipelineSpec | null,
  canvas: { positions: Record<string, XYPosition>; edges: MCEdgeSpec[] },
  catalog: MCCatalog,
): { nodes: MCFlowNode[]; edges: Edge[] } | null {
  if (pipeline === null) return null;
  // Remembered positions win; anything without one is laid out, so a bundle's
  // graph does not land stacked on a single point.
  const placed = { ...autoLayout(pipeline), ...canvas.positions };
  const nodes: MCFlowNode[] = [];
  const steps = [...pipeline.replication_steps, ...pipeline.postproc_steps];
  for (const step of steps) {
    const definition = stepDefinition(step.step_type);
    if (definition === undefined) return null;
    // Defaults underneath, so a document written before a field existed still
    // fills the form. `kwargs` is merged rather than replaced, since a shallow
    // spread would drop every default the document does not mention.
    const defaults = defaultStep(definition, step.name, "");
    nodes.push({
      id: step.name,
      type: "mcStep",
      position: placed[step.name] ?? { x: 100, y: 140 },
      data: {
        step: {
          ...defaults,
          ...step,
          kwargs: { ...defaults.kwargs, ...step.kwargs },
        },
      },
    });
  }
  // The document's dependencies first, then whatever the canvas remembers on
  // top: the two overlap on every bound leg, and only the canvas holds an edge
  // drawn before its leg was bound. Both are filtered to steps that still
  // exist, since either can outlive a deleted one.
  const present = new Set(nodes.map((node) => node.id));
  const seen = new Set<string>();
  const edges: Edge[] = [];
  for (const edge of [...documentEdges(pipeline), ...canvas.edges]) {
    const key = `${edge.source}->${edge.target}`;
    if (seen.has(key)) continue;
    if (!present.has(edge.source) || !present.has(edge.target)) continue;
    seen.add(key);
    edges.push({
      id: `mc-edge-${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
    });
  }
  return { nodes, edges };
}
