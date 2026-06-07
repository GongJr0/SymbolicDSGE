import {
  Background,
  Controls,
  MiniMap,
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
import { Check, Play, Plus, RefreshCw, TriangleAlert } from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { DragEvent, PointerEvent } from "react";
import { getMCCatalog, runMCPipeline, validateMCPipeline } from "../api";
import { PanelWorkspace } from "../PanelWorkspace";
import type { PanelDef } from "../PanelWorkspace";
import type {
  MCCatalog,
  MCPipelineResult,
  MCPipelineSpec,
  MCStepCatalogItem,
  SessionSummary,
} from "../types";
import { StepInspector } from "./StepInspector";
import { StepNode } from "./StepNode";
import { MCResultPanel } from "./MCResultPanel";
import type { MCFlowNode } from "./types";

const nodeTypes = { mcStep: StepNode };

export default function MCPipelineView({
  hidden,
  session,
}: {
  hidden?: boolean;
  session: SessionSummary | null;
}) {
  return (
    <div className="mc-layout" style={hidden ? { display: "none" } : undefined}>
      <ReactFlowProvider>
        <MCPipelineBuilder session={session} />
      </ReactFlowProvider>
    </div>
  );
}

function MCPipelineBuilder({ session }: { session: SessionSummary | null }) {
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
  const [summaryShare, setSummaryShare] = useState(36);
  const [summaryFolded, setSummaryFolded] = useState(false);
  const { screenToFlowPosition, fitView } = useReactFlow<MCFlowNode, Edge>();

  useEffect(() => {
    getMCCatalog()
      .then((value) => {
        setCatalog(value);
        const simulation = value.steps.find((step) => step.step_type === "simulation");
        if (simulation === undefined) return;
        setNodes((current) =>
          current.length > 0
            ? current
            : [makeNode(simulation, { x: 100, y: 140 }, current)],
        );
      })
      .catch((error: unknown) => {
        setNotice(error instanceof Error ? error.message : String(error));
        setNoticeError(true);
      });
  }, [setNodes]);

  const selectedNode = nodes.find((node) => node.id === selectedId) ?? null;
  const pipeline = useMemo(() => toPipelineSpec(nodes, edges), [nodes, edges]);
  const modelsReady =
    session?.models.reference?.solved === true && session.models.dgp?.solved === true;

  const markDirty = useCallback(() => {
    setNotice("");
    setNoticeError(false);
    setResult(null);
  }, []);

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
      if (["wald", "ljung_box", "regression"].includes(source.data.stepType)) {
        return false;
      }
      if (target?.data.stepType === "simulation") return false;
      if (target.data.stepType === "filter" && source.data.stepType !== "simulation") {
        return false;
      }
      if (edges.some((edge) => edge.target === connection.target)) return false;
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

  function addStep(item: MCStepCatalogItem, position?: { x: number; y: number }) {
    setNodes((current) => [
      ...current,
      makeNode(
        item,
        position ?? { x: 120 + current.length * 220, y: 140 },
        current,
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
      setNotice(`Valid dependency graph: ${response.order.length} executable steps.`);
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

  const canvasPanels: PanelDef[] = [
    {
      id: "pipeline",
      title: "Pipeline",
      badge: `${nodes.length} steps`,
      noPadding: true,
      headerActions: (
        <button className="icon-button" onClick={() => void fitView()} title="Fit pipeline">
          <RefreshCw size={15} />
        </button>
      ),
      content: (
        <div
          className="mc-canvas-shell"
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          <StepPalette catalog={catalog} onAdd={addStep} />
          <ReactFlow
            nodes={nodes}
            edges={edges}
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
            <MiniMap pannable zoomable />
          </ReactFlow>
        </div>
      ),
    },
    {
      id: "inspector",
      title: "Step Inspector",
      badge: selectedNode?.data.name,
      scrollable: true,
      content: (
        <StepInspector node={selectedNode} onChange={updateNode} onDelete={deleteNode} />
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
        <label className="checkbox">
          <input
            type="checkbox"
            checked={failFast}
            onChange={(event) => setFailFast(event.target.checked)}
          />
          fail fast
        </label>
        <button className="secondary" disabled={busy} onClick={() => void validate()}>
          <Check size={15} />
          Validate
        </button>
        <button disabled={busy || !modelsReady} onClick={() => void run()}>
          <Play size={15} />
          Run pipeline
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
            defaultSplit={72}
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

function StepPalette({
  catalog,
  onAdd,
}: {
  catalog: MCCatalog | null;
  onAdd: (item: MCStepCatalogItem) => void;
}) {
  return (
    <div className="mc-palette">
      <strong>Steps</strong>
      {catalog?.steps.map((item) => (
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
      ))}
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
): MCFlowNode {
  const count = existing.filter((node) => node.data.stepType === item.step_type).length;
  const name = count === 0 ? item.default_name : `${item.default_name}_${count + 1}`;
  return {
    id: `${item.step_type}-${crypto.randomUUID()}`,
    type: "mcStep",
    position,
    data: {
      stepType: item.step_type,
      name,
      params: Object.fromEntries(item.fields.map((field) => [field.key, field.default])),
      catalog: item,
    },
  };
}

function toPipelineSpec(nodes: MCFlowNode[], edges: Edge[]): MCPipelineSpec {
  return {
    nodes: nodes.map((node) => ({
      id: node.id,
      step_type: node.data.stepType,
      name: node.data.name,
      params: node.data.params,
    })),
    edges: edges.map((edge) => ({ source: edge.source, target: edge.target })),
  };
}
