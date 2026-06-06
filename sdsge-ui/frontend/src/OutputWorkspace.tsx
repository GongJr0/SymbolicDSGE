import { HotTable } from "@handsontable/react-wrapper";
import type { HotTableRef } from "@handsontable/react-wrapper";
import {
  ChevronDown,
  ChevronRight,
  Download,
  GripVertical,
} from "lucide-react";
import { registerAllModules } from "handsontable/registry";
import { memo, useEffect, useMemo, useRef, useState } from "react";
import type {
  Dispatch,
  DragEvent,
  PointerEvent,
  RefObject,
  SetStateAction,
} from "react";
import { Line } from "react-chartjs-2";
import { decodeArray } from "./api";
import type { SimResult } from "./types";

registerAllModules();

type PanelId = "graph" | "table";
type LayoutDirection = "horizontal" | "vertical";
type DropPlacement = "top" | "right" | "bottom" | "left";

interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
    pointRadius: number;
    borderWidth: number;
  }[];
}

export const OutputWorkspace = memo(function OutputWorkspace({
  result,
  graphSeries,
  selected,
  setSelected,
  chartData,
  theme,
}: {
  result: SimResult;
  graphSeries: SimResult["series"];
  selected: string[];
  setSelected: Dispatch<SetStateAction<string[]>>;
  chartData: ChartData;
  theme: "light" | "dark";
}) {
  const [order, setOrder] = useState<PanelId[]>(["graph", "table"]);
  const [folded, setFolded] = useState<Record<PanelId, boolean>>({
    graph: false,
    table: false,
  });
  const [panelHeights, setPanelHeights] = useState<Record<PanelId, number>>({
    graph: 470,
    table: 470,
  });
  const [split, setSplit] = useState(50);
  const [layout, setLayout] = useState<LayoutDirection>("horizontal");
  const [dragged, setDragged] = useState<PanelId | null>(null);
  const [dropTarget, setDropTarget] = useState<{
    panel: PanelId;
    placement: DropPlacement;
  } | null>(null);
  const tableRef = useRef<HotTableRef>(null);
  const tableContainerRef = useRef<HTMLDivElement>(null);
  const table = useMemo(() => createTableData(graphSeries), [graphSeries]);

  useEffect(() => {
    const container = tableContainerRef.current;
    if (container === null) return;
    let frame = 0;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        tableRef.current?.hotInstance?.refreshDimensions();
      });
    });
    observer.observe(container);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, []);

  function dragOverPanel(target: PanelId, event: DragEvent<HTMLElement>) {
    event.preventDefault();
    if (dragged === null || dragged === target) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const distances: Array<[DropPlacement, number]> = [
      ["top", y],
      ["right", rect.width - x],
      ["bottom", rect.height - y],
      ["left", x],
    ];
    const placement = distances.reduce((closest, candidate) =>
      candidate[1] < closest[1] ? candidate : closest,
    )[0];
    setDropTarget((current) =>
      current?.panel === target && current.placement === placement
        ? current
        : { panel: target, placement },
    );
  }

  function dropPanel(target: PanelId) {
    if (dragged === null || dragged === target || dropTarget === null) return;
    const draggedFirst =
      dropTarget.placement === "top" || dropTarget.placement === "left";
    setOrder(draggedFirst ? [dragged, target] : [target, dragged]);
    setLayout(
      dropTarget.placement === "top" || dropTarget.placement === "bottom"
        ? "vertical"
        : "horizontal",
    );
    setDragged(null);
    setDropTarget(null);
  }

  function startSplitResize(event: PointerEvent<HTMLDivElement>) {
    const workspace = event.currentTarget.parentElement;
    if (workspace === null) return;
    const startX = event.clientX;
    const startY = event.clientY;
    const startSplit = split;
    const first = order[0];
    const second = order[1];
    const firstHeight = panelHeights[first];
    const secondHeight = panelHeights[second];
    const width = workspace.getBoundingClientRect().width;

    function move(pointerEvent: globalThis.PointerEvent) {
      if (layout === "horizontal") {
        const delta = ((pointerEvent.clientX - startX) / width) * 100;
        setSplit(Math.min(75, Math.max(25, startSplit + delta)));
        return;
      }
      const delta = pointerEvent.clientY - startY;
      setPanelHeights((current) => ({
        ...current,
        [first]: Math.max(260, firstHeight + delta),
        [second]: Math.max(260, secondHeight - delta),
      }));
    }

    function stop() {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    }

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }

  function startHeightResize(panel: PanelId, event: PointerEvent<HTMLDivElement>) {
    const startY = event.clientY;
    const startHeight = panelHeights[panel];

    function move(pointerEvent: globalThis.PointerEvent) {
      setPanelHeights((current) => ({
        ...current,
        [panel]: Math.max(260, startHeight + pointerEvent.clientY - startY),
      }));
    }

    function stop() {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    }

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }

  function exportCsv() {
    tableRef.current?.hotInstance
      ?.getPlugin("exportFile")
      .downloadFile("csv", {
        bom: true,
        colHeaders: true,
        filename: `${result.role}-${result.run_id}-[YYYY]-[MM]-[DD]`,
      });
  }

  const panels: Record<PanelId, React.ReactNode> = {
    graph: (
      <GraphPanel
        graphSeries={graphSeries}
        selected={selected}
        setSelected={setSelected}
        chartData={chartData}
      />
    ),
    table: (
      <TablePanel
        tableRef={tableRef}
        tableContainerRef={tableContainerRef}
        table={table}
        theme={theme}
      />
    ),
  };

  return (
    <section
      className={`output-workspace ${layout}`}
      style={
        layout === "horizontal"
          ? { gridTemplateColumns: `${split}fr 6px ${100 - split}fr` }
          : undefined
      }
    >
      {order.map((panel, index) => (
        <div key={panel} className={index === 1 ? "output-slot second" : "output-slot"}>
          {index === 1 && !folded[order[0]] && !folded[order[1]] && (
            <div
              className="output-splitter"
              onPointerDown={startSplitResize}
              title="Resize output panels"
            />
          )}
          <section
            className={`output-panel ${folded[panel] ? "folded" : ""}`}
            style={{ height: folded[panel] ? undefined : panelHeights[panel] }}
            onDragOver={(event) => dragOverPanel(panel, event)}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node)) {
                setDropTarget(null);
              }
            }}
            onDrop={() => dropPanel(panel)}
          >
            {dropTarget?.panel === panel && (
              <div className={`panel-drop-indicator ${dropTarget.placement}`} />
            )}
            <header
              className="output-panel-header"
              draggable
              onDragStart={(event: DragEvent<HTMLElement>) => {
                event.dataTransfer.effectAllowed = "move";
                setDragged(panel);
              }}
              onDragEnd={() => {
                setDragged(null);
                setDropTarget(null);
              }}
            >
              <div>
                <GripVertical size={16} />
                <strong>{panel === "graph" ? "Graph" : "Table"}</strong>
                <span>{graphSeries.length} series</span>
              </div>
              <div className="output-panel-actions">
                {panel === "table" && (
                  <button
                    className="icon-button"
                    onClick={exportCsv}
                    title="Export table as CSV"
                  >
                    <Download size={16} />
                  </button>
                )}
                <button
                  className="icon-button"
                  onClick={() =>
                    setFolded((current) => ({ ...current, [panel]: !current[panel] }))
                  }
                  title={folded[panel] ? `Expand ${panel}` : `Fold ${panel}`}
                >
                  {folded[panel] ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
                </button>
              </div>
            </header>
            {!folded[panel] && (
              <>
                <div className="output-panel-body">{panels[panel]}</div>
                <div
                  className="output-height-handle"
                  onPointerDown={(event) => startHeightResize(panel, event)}
                  title={`Resize ${panel} height`}
                />
              </>
            )}
          </section>
        </div>
      ))}
    </section>
  );
});

const GraphPanel = memo(function GraphPanel({
  graphSeries,
  selected,
  setSelected,
  chartData,
}: {
  graphSeries: SimResult["series"];
  selected: string[];
  setSelected: Dispatch<SetStateAction<string[]>>;
  chartData: ChartData;
}) {
  return (
    <>
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
    </>
  );
});

const TablePanel = memo(function TablePanel({
  tableRef,
  tableContainerRef,
  table,
  theme,
}: {
  tableRef: RefObject<HotTableRef | null>;
  tableContainerRef: RefObject<HTMLDivElement | null>;
  table: ReturnType<typeof createTableData>;
  theme: "light" | "dark";
}) {
  return (
    <div ref={tableContainerRef} className="output-table">
      <HotTable
        ref={tableRef}
        data={table.rows}
        columns={table.columns}
        colHeaders={table.headers}
        readOnly
        width="100%"
        height="100%"
        stretchH="all"
        filters
        dropdownMenu={["filter_by_condition", "filter_by_value", "filter_action_bar"]}
        multiColumnSorting
        manualColumnResize
        navigableHeaders
        licenseKey="non-commercial-and-evaluation"
        className={theme === "dark" ? "ht-theme-main-dark" : "ht-theme-main"}
      />
    </div>
  );
});

function createTableData(series: SimResult["series"]) {
  const decoded = series.map((item) => ({
    name: item.name,
    values: Array.from(decodeArray(item.array)),
  }));
  const rowCount = Math.max(0, ...decoded.map((item) => item.values.length));
  return {
    headers: ["period", ...decoded.map((item) => item.name)],
    columns: [
      { type: "numeric", numericFormat: { pattern: "0" } },
      ...decoded.map(() => ({
        type: "numeric",
        numericFormat: { pattern: "0.000" },
      })),
    ],
    rows: Array.from({ length: rowCount }, (_, period) => [
      period,
      ...decoded.map((item) => item.values[period] ?? null),
    ]),
  };
}
