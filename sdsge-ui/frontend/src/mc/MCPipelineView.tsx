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
  getMCCatalog,
  getMCCustomTemplate,
  runMCPipeline,
  validateMCPipeline,
} from "../api";
import { PanelWorkspace } from "../PanelWorkspace";
import type { PanelDef } from "../PanelWorkspace";
import type {
  MCCatalog,
  MCPipelineResult,
  MCPipelineSpec,
  MCStepCatalogItem,
  MCStepCategory,
  SessionSummary,
} from "../types";
import { StepInspector } from "./StepInspector";
import { StepNode } from "./StepNode";
import { MCResultPanel } from "./MCResultPanel";
import {
  clearMCWorkspace,
  loadMCResult,
  loadMCWorkspace,
  saveMCResult,
  saveMCWorkspace,
} from "./persistence";
import type { MCPersistedWorkspace } from "./persistence";
import type { MCFlowNode } from "./types";

const nodeTypes = { mcStep: StepNode };

// "transform:custom" has no backend StepDefinition (it isn't GUI-authored via
// fields), so it never appears in the catalogue payload. We inject a synthetic
// palette entry for it under the Transforms tab; its single "param" is the op
// source code.
const CUSTOM_CATALOG_ITEM: MCStepCatalogItem = {
  step_type: "transform:custom",
  title: "Custom Op",
  default_name: "custom_op",
  description: "User-defined Python transform, validated and run per replication.",
  category: "transforms",
  fields: [],
};

// Post-loop sibling of CUSTOM_CATALOG_ITEM: a user-authored summary op run once
// after the replication loop (pandas namespace; may return a DataFrame).
const POSTPROC_CUSTOM_CATALOG_ITEM: MCStepCatalogItem = {
  step_type: "postproc:custom",
  title: "Custom Postproc",
  default_name: "postproc_op",
  description: "User-defined post-loop summary op over the across-rep traces.",
  category: "postproc",
  fields: [],
};

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
  const [failFast, setFailFast] = useState(true);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [noticeError, setNoticeError] = useState(false);
  const [result, setResult] = useState<MCPipelineResult | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [summaryShare, setSummaryShare] = useState(36);
  const [summaryFolded, setSummaryFolded] = useState(false);
  const { screenToFlowPosition, fitView, getViewport } =
    useReactFlow<MCFlowNode, Edge>();

  useEffect(() => {
    Promise.all([
      getMCCatalog(),
      getMCCustomTemplate().catch(() => ({ template: "" })),
    ])
      .then(async ([rawCatalog, tmpl]) => {
        customTemplateRef.current = tmpl.template;
        const value: MCCatalog = {
          ...rawCatalog,
          steps: [
            ...rawCatalog.steps,
            CUSTOM_CATALOG_ITEM,
            POSTPROC_CUSTOM_CATALOG_ITEM,
          ],
        };
        setCatalog(value);
        const [workspace, persistedResult] = await Promise.all([
          loadMCWorkspace().catch(() => null),
          loadMCResult().catch(() => null),
        ]);
        const restored = restoreNodes(workspace, value);
        if (restored !== null) {
          setNodes(restored.nodes);
          setEdges(restored.edges);
          setNRep(workspace?.nRep ?? 100);
          setFailFast(workspace?.failFast ?? true);
        } else {
          const simulation = value.steps.find(
            (step) => step.step_type === "simulation",
          );
          if (simulation !== undefined) {
            setNodes([makeNode(simulation, { x: 100, y: 140 }, [])]);
          }
        }
        setResult(persistedResult);
        setHydrated(true);
      })
      .catch((error: unknown) => {
        setNotice(error instanceof Error ? error.message : String(error));
        setNoticeError(true);
        setHydrated(true);
      });
  }, [setEdges, setNodes]);

  const selectedNode = nodes.find((node) => node.id === selectedId) ?? null;
  // Every transform/custom step produces a payload addressable by its name — the
  // registry a consumer picks from when a leg reads "payload".
  const payloadProducers = nodes
    .filter((node) => node.data.catalog.category === "transforms")
    .map((node) => node.data.name);
  const pipeline = useMemo(() => toPipelineSpec(nodes, edges), [nodes, edges]);

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
        const color = producerColor(source?.data.stepType);
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

  const markDirty = useCallback(() => {
    setNotice("");
    setNoticeError(false);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    const timeout = window.setTimeout(() => {
      void saveMCWorkspace({
        version: 1,
        pipeline,
        positions: Object.fromEntries(
          nodes.map((node) => [
            // Postprocs carry no id in the spec, so key their positions by name.
            isPostprocNode(node) ? node.data.name : node.id,
            { x: node.position.x, y: node.position.y },
          ]),
        ),
        nRep,
        failFast,
      }).catch((error: unknown) => {
        setNotice(error instanceof Error ? error.message : String(error));
        setNoticeError(true);
      });
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [edges, failFast, hydrated, nRep, nodes, pipeline]);

  useEffect(() => {
    if (!hydrated || result === null) return;
    void saveMCResult(result).catch((error: unknown) => {
      setNotice(error instanceof Error ? error.message : String(error));
      setNoticeError(true);
    });
  }, [hydrated, result]);

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
        ].includes(source.data.stepType)
      ) {
        return false;
      }
      if (target?.data.stepType === "simulation") return false;
      if (target.data.stepType === "filter" && source.data.stepType !== "simulation") {
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
    setNodes((current) => [
      ...current,
      makeNode(
        item,
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
      const total = response.order.length + response.postprocs.length;
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
      const output = await runMCPipeline(pipeline, nRep, failFast);
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
    setFailFast(true);
    try {
      await clearMCWorkspace();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
      setNoticeError(true);
    }
  }

  function resetPipeline() {
    const simulation = catalog?.steps.find((step) => step.step_type === "simulation");
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
      badge: selectedNode?.data.name,
      scrollable: true,
      content: (
        <StepInspector
          node={selectedNode}
          onChange={updateNode}
          onDelete={deleteNode}
          theme={theme}
          payloadProducers={payloadProducers}
          availableTraces={availableTraces}
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
  if (stepType === "simulation" || stepType === "raw_data") return "datagen";
  if (stepType === "filter") return "filter";
  return "transform";
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
  item: MCStepCatalogItem,
  position: { x: number; y: number },
  existing: MCFlowNode[],
  customTemplate = "",
): MCFlowNode {
  const count = existing.filter((node) => node.data.stepType === item.step_type).length;
  const name = count === 0 ? item.default_name : `${item.default_name}_${count + 1}`;
  const params =
    item.step_type === "transform:custom"
      ? { code: customTemplate }
      : item.step_type === "postproc:custom"
        ? { code: POSTPROC_CUSTOM_TEMPLATE }
        : Object.fromEntries(item.fields.map((field) => [field.key, field.default]));
  return {
    id: `${item.step_type}-${crypto.randomUUID()}`,
    type: "mcStep",
    position,
    data: {
      stepType: item.step_type,
      name,
      params,
      catalog: item,
    },
  };
}

function isPostprocNode(node: MCFlowNode): boolean {
  return node.data.catalog.category === "postproc";
}

function toPipelineSpec(nodes: MCFlowNode[], edges: Edge[]): MCPipelineSpec {
  const perRep = nodes.filter((node) => !isPostprocNode(node));
  const postprocs = nodes.filter(isPostprocNode);
  const perRepIds = new Set(perRep.map((node) => node.id));
  return {
    // Per-replication DAG nodes only.
    nodes: perRep.map((node) => ({
      id: node.id,
      step_type: node.data.stepType,
      name: node.data.name,
      params: node.data.params,
    })),
    // Edges never touch a postproc node (they carry no edges), but filter
    // defensively so a stale edge can't leak into the spec.
    edges: edges
      .filter((edge) => perRepIds.has(edge.source) && perRepIds.has(edge.target))
      .map((edge) => ({ source: edge.source, target: edge.target })),
    // Post-loop ops: a separate terminal list, no id, no edges.
    postprocs: postprocs.map((node) => ({
      step_type: node.data.stepType,
      name: node.data.name,
      params: node.data.params,
    })),
  };
}

function restoreNodes(
  workspace: MCPersistedWorkspace | null,
  catalog: MCCatalog,
): { nodes: MCFlowNode[]; edges: Edge[] } | null {
  if (workspace === null || workspace.version !== 1) return null;
  const nodes: MCFlowNode[] = [];
  for (const spec of workspace.pipeline.nodes) {
    const item = catalog.steps.find((step) => step.step_type === spec.step_type);
    if (item === undefined) return null;
    nodes.push({
      id: spec.id,
      type: "mcStep",
      position: workspace.positions[spec.id] ?? { x: 100, y: 140 },
      data: {
        stepType: spec.step_type,
        name: spec.name,
        params: {
          ...Object.fromEntries(item.fields.map((field) => [field.key, field.default])),
          ...spec.params,
        },
        catalog: item,
      },
    });
  }
  // Postprocs are a separate spec list with no id; rebuild them as standalone
  // canvas nodes keyed by name. (`?? []` tolerates a pre-postprocs cached
  // workspace; any legacy postproc still in `nodes` is re-routed on next save.)
  for (const pp of workspace.pipeline.postprocs ?? []) {
    const item = catalog.steps.find((step) => step.step_type === pp.step_type);
    if (item === undefined) return null;
    nodes.push({
      id: crypto.randomUUID(),
      type: "mcStep",
      position: workspace.positions[pp.name] ?? { x: 100, y: 320 },
      data: {
        stepType: pp.step_type,
        name: pp.name,
        params: {
          ...Object.fromEntries(item.fields.map((field) => [field.key, field.default])),
          ...pp.params,
        },
        catalog: item,
      },
    });
  }
  return {
    nodes,
    edges: workspace.pipeline.edges.map((edge) => ({
      id: `mc-edge-${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
    })),
  };
}
