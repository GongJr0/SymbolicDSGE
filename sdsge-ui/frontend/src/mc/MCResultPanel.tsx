import { useState } from "react";
import type {
  MCPipelineResult,
  MCRegressionSummary,
  MCTraceSummary,
} from "../types";

type SummaryTab = "performance" | "tests" | "regressions";

export function MCResultPanel({ result }: { result: MCPipelineResult | null }) {
  const [tab, setTab] = useState<SummaryTab>("performance");
  if (result === null) {
    return (
      <div className="mc-empty">
        <span>Run a validated pipeline to inspect aggregate results.</span>
      </div>
    );
  }
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
      </nav>
      <div className="mc-summary-body">
        {tab === "performance" && <PerformanceSummary result={result} />}
        {tab === "tests" && <TestSummary result={result} />}
        {tab === "regressions" && <RegressionSummary result={result} />}
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
