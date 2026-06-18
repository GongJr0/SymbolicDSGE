import {
  Chart as ChartJS,
  BarController,
  BarElement,
  CategoryScale,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  ScatterController,
  Tooltip,
} from "chart.js";
import { Bar, Line, Scatter } from "react-chartjs-2";
import { useMemo, useState } from "react";
import type {
  MCPipelineResult,
  MCPostprocArtifact,
  MCRegressionSummary,
  MCTraceSummary,
} from "../types";

// Idempotent — App.tsx already registers the scales/plugins globally; we add the
// controllers/elements the postproc charts need (bar + line + scatter) so the
// combined density-over-histogram view renders regardless of load order.
ChartJS.register(
  BarController,
  BarElement,
  LineController,
  LineElement,
  PointElement,
  ScatterController,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
);

const POSTPROC_TAB_PREFIX = "pp:";

export function MCResultPanel({ result }: { result: MCPipelineResult | null }) {
  const [tab, setTab] = useState<string>("performance");
  if (result === null) {
    return (
      <div className="mc-empty">
        <span>Run a validated pipeline to inspect aggregate results.</span>
      </div>
    );
  }
  const postproc = Object.entries(result.postproc ?? {});
  const activePostproc =
    tab.startsWith(POSTPROC_TAB_PREFIX) &&
    (result.postproc ?? {})[tab.slice(POSTPROC_TAB_PREFIX.length)];
  return (
    <div className="mc-summary">
      <nav className="mc-summary-tabs">
        <SummaryTabButton active={tab === "performance"} onClick={() => setTab("performance")}>
          Performance
        </SummaryTabButton>
        <SummaryTabButton active={tab === "tests"} onClick={() => setTab("tests")}>
          Tests ({Object.keys(result.test_summaries).length})
        </SummaryTabButton>
        <SummaryTabButton active={tab === "regressions"} onClick={() => setTab("regressions")}>
          Regressions ({Object.keys(result.regression_summaries).length})
        </SummaryTabButton>
        {postproc.map(([key, artifact]) => (
          <SummaryTabButton
            key={key}
            active={tab === `${POSTPROC_TAB_PREFIX}${key}`}
            onClick={() => setTab(`${POSTPROC_TAB_PREFIX}${key}`)}
          >
            {artifact.title || key}
          </SummaryTabButton>
        ))}
      </nav>
      <div className="mc-summary-body">
        {tab === "performance" && <PerformanceSummary result={result} />}
        {tab === "tests" && <TestSummary result={result} />}
        {tab === "regressions" && <RegressionSummary result={result} />}
        {activePostproc && (
          <PostprocArtifactView
            name={tab.slice(POSTPROC_TAB_PREFIX.length)}
            artifact={activePostproc}
          />
        )}
      </div>
    </div>
  );
}

function SummaryTabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button className={active ? "active" : ""} onClick={onClick}>
      {children}
    </button>
  );
}

function PerformanceSummary({ result }: { result: MCPipelineResult }) {
  return (
    <div className="mc-summary-section">
      <div className="mc-metric-grid">
        <Metric label="Throughput" value={`${format(result.it_s)} it/s`} />
        <Metric label="Elapsed" value={`${format(result.elapsed_s)} s`} />
        <Metric label="Successful" value={`${result.n_successful}/${result.n_rep}`} />
        <Metric label="Failures" value={String(result.failures.length)} />
      </div>
      <h3>Steps</h3>
      <div className="mc-table-scroll">
        <table className="mc-summary-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Iterations</th>
              <th>Elapsed</th>
              <th>Throughput</th>
              <th>Failures</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(result.step_elapsed_s).map(([name, elapsed]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{result.step_counts[name] ?? 0}</td>
                <td>{format(elapsed)} s</td>
                <td>{format(result.step_it_s[name])} it/s</td>
                <td>{result.step_failures[name] ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>Generated Data</h3>
      {Object.keys(result.data_summaries).length === 0 ? (
        <span className="muted">No generated data summaries</span>
      ) : (
        <div className="mc-table-scroll">
          <table className="mc-summary-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Per-rep shape</th>
                <th>Finite values</th>
                <th>Mean</th>
                <th>Std.</th>
                <th>Min.</th>
                <th>Max.</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.data_summaries).map(([name, summary]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{summary.shape.join(" x ")}</td>
                  <td>{summary.n_finite}/{summary.n_values}</td>
                  <td>{format(summary.mean)}</td>
                  <td>{format(summary.std)}</td>
                  <td>{format(summary.min)}</td>
                  <td>{format(summary.max)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {result.failures.length > 0 && (
        <>
          <h3>Failures</h3>
          <div className="mc-table-scroll">
            <table className="mc-summary-table">
              <thead>
                <tr>
                  <th>Replication</th>
                  <th>Step</th>
                  <th>Type</th>
                  <th>Message</th>
                </tr>
              </thead>
              <tbody>
                {result.failures.map((failure) => (
                  <tr key={`${failure.rep_idx}:${failure.step_name}`}>
                    <td>{failure.rep_idx}</td>
                    <td>{failure.step_name}</td>
                    <td>{failure.error_type}</td>
                    <td>{failure.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function TestSummary({ result }: { result: MCPipelineResult }) {
  const tests = Object.entries(result.test_summaries);
  if (tests.length === 0) return <div className="mc-empty">No test summaries</div>;
  return (
    <div className="mc-summary-section">
      <div className="mc-table-scroll">
        <table className="mc-summary-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Reference</th>
              <th>N</th>
              <th>Status</th>
              <th>Mean statistic</th>
              <th>Statistic 95% CI</th>
              <th>Statistic SE</th>
              <th>Mean p-value</th>
              <th>Rejection rate</th>
              <th>Rejection 95% CI</th>
              <th>Rejection SE</th>
            </tr>
          </thead>
          <tbody>
            {tests.map(([name, summary]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{summary.distribution}({formatDf(summary.df)})</td>
                <td>{summary.n}</td>
                <td>{formatStatusCounts(summary.status_counts)}</td>
                <td>{format(summary.mean_statistic)}</td>
                <td>{formatInterval(summary.statistic_ci)}</td>
                <td>{format(summary.statistic_se)}</td>
                <td>{format(summary.mean_pval)}</td>
                <td>{format(summary.rejection_rate)}</td>
                <td>{formatInterval(summary.rejection_ci)}</td>
                <td>{format(summary.pval_se)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>Trace Distributions</h3>
      <div className="mc-table-scroll">
        <table className="mc-summary-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Trace</th>
              <th>Finite</th>
              <th>Mean</th>
              <th>Std.</th>
              <th>Min.</th>
              <th>2.5%</th>
              <th>97.5%</th>
              <th>Max.</th>
            </tr>
          </thead>
          <tbody>
            {tests.flatMap(([name, summary]) => [
              <TraceRow key={`${name}:stat`} name={name} trace="Statistic" summary={summary.statistic_summary} />,
              <TraceRow key={`${name}:pval`} name={name} trace="P-value" summary={summary.pval_summary} />,
            ])}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatStatusCounts(counts: Record<string, number>): string {
  return Object.entries(counts)
    .map(([status, count]) => `${status}: ${count}`)
    .join(", ");
}

function RegressionSummary({ result }: { result: MCPipelineResult }) {
  const regressions = Object.entries(result.regression_summaries);
  if (regressions.length === 0) {
    return <div className="mc-empty">No regression summaries</div>;
  }
  return (
    <div className="mc-regression-detail-list">
      {regressions.map(([name, summary]) => (
        <RegressionDetail key={name} name={name} summary={summary} />
      ))}
    </div>
  );
}

function RegressionDetail({
  name,
  summary,
}: {
  name: string;
  summary: MCRegressionSummary;
}) {
  return (
    <section className="mc-regression-detail">
      <header>
        <strong>{name}</strong>
        <span>{summary.n_rep} reps / n={summary.n} / k={summary.k}</span>
      </header>
      <div className="mc-status-list">
        {Object.entries(summary.status_counts).map(([status, count]) => (
          <span key={status}>{status}: {count}</span>
        ))}
      </div>
      <h3>Coefficients</h3>
      <div className="mc-table-scroll">
        <table className="mc-summary-table">
          <thead>
            <tr>
              <th>Variable</th>
              <th>Mean</th>
              <th>Std.</th>
              <th>Empirical 95% interval</th>
              {summary.ols !== null && <th>Mean SE</th>}
              {summary.ols !== null && <th>Mean t-stat</th>}
              {summary.ols !== null && <th>Mean p-value</th>}
              {summary.ols !== null && <th>Mean partial R2</th>}
            </tr>
          </thead>
          <tbody>
            {summary.coefficient_summaries.map((coefficient, index) => (
              <tr key={coefficient.variable}>
                <td>{coefficient.variable}</td>
                <td>{format(coefficient.mean)}</td>
                <td>{format(coefficient.std)}</td>
                <td>{formatInterval([coefficient.q025, coefficient.q975])}</td>
                {summary.ols !== null && <td>{format(summary.ols.mean_standard_errors[index])}</td>}
                {summary.ols !== null && <td>{format(summary.ols.mean_t_statistics[index])}</td>}
                {summary.ols !== null && <td>{format(summary.ols.mean_pvalues[index])}</td>}
                {summary.ols !== null && <td>{format(summary.ols.mean_partial_r2[index])}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>Fit Metrics</h3>
      <div className="mc-table-scroll">
        <table className="mc-summary-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Mean</th>
              <th>Std.</th>
              <th>Min.</th>
              <th>2.5%</th>
              <th>97.5%</th>
              <th>Max.</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(summary.metrics).map(([metric, trace]) => (
              <MetricTraceRow key={metric} name={metric} summary={trace} />
            ))}
            {summary.ols !== null && <MetricTraceRow name="F statistic" summary={summary.ols.f_statistic} />}
            {summary.ols !== null && <MetricTraceRow name="F p-value" summary={summary.ols.f_pvalue} />}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function TraceRow({
  name,
  trace,
  summary,
}: {
  name: string;
  trace: string;
  summary: MCTraceSummary;
}) {
  return (
    <tr>
      <td>{name}</td>
      <td>{trace}</td>
      <td>{summary.n_finite}/{summary.n}</td>
      <td>{format(summary.mean)}</td>
      <td>{format(summary.std)}</td>
      <td>{format(summary.min)}</td>
      <td>{format(summary.q025)}</td>
      <td>{format(summary.q975)}</td>
      <td>{format(summary.max)}</td>
    </tr>
  );
}

function MetricTraceRow({ name, summary }: { name: string; summary: MCTraceSummary }) {
  return (
    <tr>
      <td>{name}</td>
      <td>{format(summary.mean)}</td>
      <td>{format(summary.std)}</td>
      <td>{format(summary.min)}</td>
      <td>{format(summary.q025)}</td>
      <td>{format(summary.q975)}</td>
      <td>{format(summary.max)}</td>
    </tr>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="mc-metric">
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  );
}

function format(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "--" : value.toFixed(4);
}

function formatInterval(values: Array<number | null>): string {
  return `[${format(values[0])}, ${format(values[1])}]`;
}

function formatDf(value: number | Array<number | null> | null): string {
  // Parameter-free reference distributions (e.g. CUSUM) carry no df; the
  // backend sends null. Render "N/A" so it reads as "not applicable" rather
  // than a missing/failed value.
  if (value == null) return "N/A";
  if (Array.isArray(value)) {
    return value.length === 0 ? "N/A" : value.map(format).join(", ");
  }
  return format(value);
}

// ---- POSTPROC artifact surfaces (#184) ----------------------------------

function PostprocArtifactView({
  name,
  artifact,
}: {
  name: string;
  artifact: MCPostprocArtifact;
}) {
  if (artifact.artifact === "scalar") {
    return <ScalarArtifactView name={name} artifact={artifact} />;
  }
  if (artifact.artifact === "table") {
    return <TableArtifactView name={name} artifact={artifact} />;
  }
  return <ArrayArtifactView name={name} artifact={artifact} />;
}

function ScalarArtifactView({
  name,
  artifact,
}: {
  name: string;
  artifact: MCPostprocArtifact;
}) {
  return (
    <div className="mc-summary-section">
      <div className="mc-metric-grid">
        <Metric label={artifact.title || name} value={formatCell(artifact.value)} />
      </div>
    </div>
  );
}

function TableArtifactView({
  name,
  artifact,
}: {
  name: string;
  artifact: MCPostprocArtifact;
}) {
  const { headers, rows } = tableArtifactRows(artifact);
  return (
    <div className="mc-summary-section">
      <FoldableTable
        title={artifact.title || name}
        headers={headers}
        rows={rows}
        csvName={name}
        defaultOpen
      />
    </div>
  );
}

type ArrayView = "line" | "histogram" | "scatter";

function ArrayArtifactView({
  name,
  artifact,
}: {
  name: string;
  artifact: MCPostprocArtifact;
}) {
  const rows = useMemo(() => toNumericRows(artifact.value), [artifact.value]);
  const width = rows.length > 0 ? rows[0].length : 0;
  const isCurve = width === 2; // (x, y) curve, e.g. a KDE density
  // KDE / curves default to the density line; 1-D arrays to a histogram.
  const [view, setView] = useState<ArrayView>(isCurve ? "line" : "histogram");

  const points = useMemo<{ x: number; y: number }[]>(
    () =>
      isCurve
        ? rows.map((r) => ({ x: r[0], y: r[1] }))
        : rows.map((r, i) => ({ x: i, y: r[0] ?? Number.NaN })),
    [rows, isCurve],
  );
  const bins = useMemo(
    () =>
      isCurve
        ? binCurve(points, 20)
        : histogram(points.map((p) => p.y), 20),
    [points, isCurve],
  );

  const headers = width <= 1 ? ["i", "value"] : ["i", ...range(width).map((j) => `c${j}`)];
  const tableRows: unknown[][] = rows.map((r, i) =>
    width <= 1 ? [i, r[0]] : [i, ...r],
  );

  return (
    <div className="mc-summary-section">
      <div className="mc-postproc-views">
        {(["line", "histogram", "scatter"] as ArrayView[]).map((option) => (
          <button
            key={option}
            className={view === option ? "active" : ""}
            onClick={() => setView(option)}
          >
            {option}
          </button>
        ))}
      </div>
      <div className="mc-postproc-chart" style={{ height: 280 }}>
        {view === "line" && (
          <Line data={lineData(artifact.title || name, points)} options={CHART_OPTIONS} />
        )}
        {view === "scatter" && (
          <Scatter
            data={scatterData(artifact.title || name, points)}
            options={CHART_OPTIONS}
          />
        )}
        {view === "histogram" && (
          <Bar data={barData(name, bins)} options={BAR_OPTIONS} />
        )}
      </div>
      <FoldableTable
        title="Data"
        headers={headers}
        rows={tableRows}
        csvName={name}
        defaultOpen={false}
      />
    </div>
  );
}

function FoldableTable({
  title,
  headers,
  rows,
  csvName,
  defaultOpen,
}: {
  title: string;
  headers: (string | number)[];
  rows: unknown[][];
  csvName: string;
  defaultOpen: boolean;
}) {
  return (
    <details open={defaultOpen} className="mc-foldable-table">
      <summary>
        <span>{title}</span>
        <button
          className="secondary"
          onClick={(event) => {
            event.preventDefault();
            downloadCsv(csvName, headers, rows);
          }}
        >
          Export CSV
        </button>
      </summary>
      <div className="mc-table-scroll">
        <table className="mc-summary-table">
          <thead>
            <tr>
              {headers.map((header) => (
                <th key={String(header)}>{String(header)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{formatCell(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

// ---- artifact data helpers ----------------------------------------------

const CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  plugins: { legend: { display: true } },
  scales: { x: { type: "linear" as const } },
};

const BAR_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  plugins: { legend: { display: false } },
};

function lineData(label: string, points: { x: number; y: number }[]) {
  return {
    datasets: [
      {
        label,
        data: points,
        borderColor: "#4f46e5",
        backgroundColor: "rgb(79 70 229 / 14%)",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
      },
    ],
  };
}

function scatterData(label: string, points: { x: number; y: number }[]) {
  return {
    datasets: [
      { label, data: points, backgroundColor: "#4f46e5", pointRadius: 3 },
    ],
  };
}

function barData(label: string, bins: { center: number; value: number }[]) {
  return {
    labels: bins.map((b) => b.center.toPrecision(3)),
    datasets: [
      {
        label,
        data: bins.map((b) => b.value),
        backgroundColor: "rgb(79 70 229 / 55%)",
      },
    ],
  };
}

// Flatten an artifact `value` (1-D or 2-D nested arrays) into rows of numbers.
function toNumericRows(value: unknown): number[][] {
  if (!Array.isArray(value)) return [];
  if (value.length === 0) return [];
  if (Array.isArray(value[0])) {
    return (value as unknown[][]).map((row) => row.map(toNumber));
  }
  return (value as unknown[]).map((v) => [toNumber(v)]);
}

function toNumber(value: unknown): number {
  return typeof value === "number" ? value : Number(value);
}

// Equal-width histogram of scalar values -> 20 bin centers + counts.
function histogram(values: number[], binCount: number): { center: number; value: number }[] {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return [];
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (min === max) return [{ center: min, value: finite.length }];
  const width = (max - min) / binCount;
  const counts = new Array<number>(binCount).fill(0);
  for (const v of finite) {
    const idx = Math.min(binCount - 1, Math.floor((v - min) / width));
    counts[idx] += 1;
  }
  return counts.map((value, i) => ({ center: min + (i + 0.5) * width, value }));
}

// Bin an (x, y) curve into 20 x-bins, bar height = mean y in the bin.
function binCurve(
  points: { x: number; y: number }[],
  binCount: number,
): { center: number; value: number }[] {
  const xs = points.map((p) => p.x).filter(Number.isFinite);
  if (xs.length === 0) return [];
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  if (min === max) return [{ center: min, value: mean(points.map((p) => p.y)) }];
  const width = (max - min) / binCount;
  const sums = new Array<number>(binCount).fill(0);
  const counts = new Array<number>(binCount).fill(0);
  for (const p of points) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    const idx = Math.min(binCount - 1, Math.floor((p.x - min) / width));
    sums[idx] += p.y;
    counts[idx] += 1;
  }
  return sums.map((sum, i) => ({
    center: min + (i + 0.5) * width,
    value: counts[i] > 0 ? sum / counts[i] : 0,
  }));
}

function tableArtifactRows(artifact: MCPostprocArtifact): {
  headers: string[];
  rows: unknown[][];
} {
  const columns = artifact.columns ?? [];
  const data = artifact.data ?? {};
  const labeled = artifact.index?.kind === "labeled";
  const indexHeader = artifact.index?.name || "index";
  const headers = labeled ? [indexHeader, ...columns] : columns;
  const nRows = columns.length > 0 ? (data[columns[0]]?.length ?? 0) : 0;
  const rows: unknown[][] = [];
  for (let i = 0; i < nRows; i += 1) {
    const body = columns.map((column) => data[column]?.[i]);
    rows.push(labeled ? [data["__index__"]?.[i], ...body] : body);
  }
  return { headers, rows };
}

function downloadCsv(name: string, headers: (string | number)[], rows: unknown[][]): void {
  const escape = (value: unknown): string => {
    const text = value == null ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const lines = [
    headers.map(escape).join(","),
    ...rows.map((row) => row.map(escape).join(",")),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${name}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function formatCell(value: unknown): string {
  if (value == null) return "--";
  if (typeof value === "number") {
    return Number.isFinite(value) ? format(value) : "--";
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

function mean(values: number[]): number {
  const finite = values.filter(Number.isFinite);
  return finite.length === 0 ? 0 : finite.reduce((a, b) => a + b, 0) / finite.length;
}
