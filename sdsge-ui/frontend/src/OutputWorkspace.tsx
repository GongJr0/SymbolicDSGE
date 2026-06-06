import { HotTable } from "@handsontable/react-wrapper";
import type { HotTableRef } from "@handsontable/react-wrapper";
import { Download } from "lucide-react";
import { registerAllModules } from "handsontable/registry";
import {
  forwardRef,
  memo,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
} from "react";
import type { Dispatch, SetStateAction } from "react";
import { Line } from "react-chartjs-2";
import { decodeArray } from "./api";
import { FiguresPanel } from "./FiguresPanel";
import { PanelWorkspace } from "./PanelWorkspace";
import type { PanelDef } from "./PanelWorkspace";
import type { SimResult } from "./types";

registerAllModules();

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

interface TableHandle {
  exportCsv: (role: string, runId: string) => void;
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
  const tablePanelRef = useRef<TableHandle>(null);
  const table = useMemo(() => createTableData(graphSeries), [graphSeries]);
  const figures = result.figures ?? [];

  function exportCsv() {
    tablePanelRef.current?.exportCsv(result.role, result.run_id);
  }

  const topPanels: PanelDef[] = [
    {
      id: "graph",
      title: "Graph",
      badge: `${graphSeries.length} series`,
      content: (
        <GraphPanel
          graphSeries={graphSeries}
          selected={selected}
          setSelected={setSelected}
          chartData={chartData}
        />
      ),
    },
    {
      id: "table",
      title: "Table",
      badge: `${graphSeries.length} series`,
      content: <TablePanel ref={tablePanelRef} table={table} theme={theme} />,
      headerActions: (
        <button className="icon-button" onClick={exportCsv} title="Export table as CSV">
          <Download size={16} />
        </button>
      ),
    },
  ];

  const figurePanels: PanelDef[] = [
    {
      id: "figures",
      title: "Figures",
      badge: figures.length > 0 ? `${figures.length} plot${figures.length === 1 ? "" : "s"}` : undefined,
      content: <FiguresPanel figures={figures} />,
      scrollable: false,
    },
  ];

  return (
    <div className="output-layout">
      <div className="output-top-row">
        <PanelWorkspace panels={topPanels} defaultLayout="horizontal" defaultSplit={70} fillHeight />
      </div>
      <div className="output-figures-row">
        <PanelWorkspace panels={figurePanels} defaultLayout="vertical" />
      </div>
    </div>
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

const TablePanel = forwardRef<
  TableHandle,
  { table: ReturnType<typeof createTableData>; theme: "light" | "dark" }
>(function TablePanel({ table, theme }, ref) {
  const tableRef = useRef<HotTableRef>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useImperativeHandle(ref, () => ({
    exportCsv(role: string, runId: string) {
      tableRef.current?.hotInstance
        ?.getPlugin("exportFile")
        .downloadFile("csv", {
          bom: true,
          colHeaders: true,
          filename: `${role}-${runId}-[YYYY]-[MM]-[DD]`,
        });
    },
  }));

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
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

  return (
    <div ref={containerRef} className="output-table">
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
    values: decodeArray(item.array),
  }));
  const rowCount = Math.max(0, ...decoded.map((item) => item.values.length));
  return {
    headers: ["period", ...decoded.map((item) => item.name)],
    columns: [
      { type: "numeric", numericFormat: { pattern: "0" } },
      ...decoded.map(() => ({ type: "numeric", numericFormat: { pattern: "0.000" } })),
    ],
    rows: Array.from({ length: rowCount }, (_, period) => [
      period,
      ...decoded.map((item) => item.values[period] ?? null),
    ]),
  };
}
