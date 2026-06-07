import { Play, RefreshCw, Trash2, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Line } from "react-chartjs-2";
import { getEstimationCatalog, runEstimation } from "./api";
import {
  clearEstimationWorkspace,
  loadEstimationWorkspace,
  saveEstimationWorkspace,
} from "./estimationPersistence";
import { PanelWorkspace } from "./PanelWorkspace";
import type { PanelDef } from "./PanelWorkspace";
import type {
  EstimationCatalog,
  EstimationMethod,
  EstimationParameterSpec,
  EstimationRunResult,
  ModelSummary,
  Role,
} from "./types";

export function EstimationView({
  hidden,
  role,
  model,
  onSessionRefresh,
}: {
  hidden?: boolean;
  role: Role;
  model: ModelSummary;
  onSessionRefresh: () => Promise<void>;
}) {
  const [catalog, setCatalog] = useState<EstimationCatalog | null>(null);
  const [method, setMethod] = useState<EstimationMethod>("mle");
  const [parameters, setParameters] = useState<EstimationParameterSpec[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [observables, setObservables] = useState("");
  const [dataVectors, setDataVectors] = useState<Record<string, string>>({});
  const [maxIter, setMaxIter] = useState(1000);
  const [nDraws, setNDraws] = useState(1000);
  const [burnIn, setBurnIn] = useState(500);
  const [thin, setThin] = useState(1);
  const [seed, setSeed] = useState(0);
  const [adapt, setAdapt] = useState(true);
  const [proposalScale, setProposalScale] = useState(0.1);
  const [posteriorPoint, setPosteriorPoint] = useState("mean");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState(false);
  const [result, setResult] = useState<EstimationRunResult | null>(null);
  const [modeFolded, setModeFolded] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [workspaceRevision, setWorkspaceRevision] = useState(0);
  const [chartRevision, setChartRevision] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getEstimationCatalog()
      .then(setCatalog)
      .catch((reason: unknown) => {
        setNotice(reason instanceof Error ? reason.message : String(reason));
        setError(true);
      });
  }, []);

  useEffect(() => {
    if (hidden) return;
    const frame = window.requestAnimationFrame(() => {
      setChartRevision((current) => current + 1);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [hidden]);

  const parameterKey = Object.keys(model.parameter_values ?? {}).join(",");
  const observableKey = (model.observables ?? []).join(",");
  const modelKey = JSON.stringify({
    role,
    source: model.source ?? null,
    parameters: parameterKey,
    observables: observableKey,
  });

  useEffect(() => {
    if (catalog === null) return;
    let cancelled = false;
    const values = model.parameter_values ?? {};
    const names = model.observables ?? [];
    setHydrated(false);
    void loadEstimationWorkspace(role)
      .catch(() => null)
      .then((workspace) => {
        if (cancelled) return;
        if (workspace !== null && workspace.modelKey === modelKey) {
          setMethod(workspace.method);
          setParameters(workspace.parameters);
          setSelected(workspace.selected);
          setObservables(workspace.observables);
          setDataVectors(workspace.dataVectors);
          setMaxIter(workspace.maxIter);
          setNDraws(workspace.nDraws);
          setBurnIn(workspace.burnIn);
          setThin(workspace.thin);
          setSeed(workspace.seed);
          setAdapt(workspace.adapt);
          setProposalScale(workspace.proposalScale);
          setPosteriorPoint(workspace.posteriorPoint);
          setResult(workspace.result);
          setModeFolded(workspace.modeFolded);
        } else {
          setParameters(
            Object.entries(values).map(([name, value]) =>
              makeParameter(name, value, catalog),
            ),
          );
          setSelected(Object.keys(values)[0] ?? null);
          setObservables(names.join(", "));
          setDataVectors(Object.fromEntries(names.map((name) => [name, ""])));
          setResult(null);
        }
        setHydrated(true);
      });
    return () => {
      cancelled = true;
    };
  }, [catalog, modelKey, role]);

  useEffect(() => {
    if (!hydrated) return;
    const timeout = window.setTimeout(() => {
      void saveEstimationWorkspace({
        version: 1,
        role,
        modelKey,
        method,
        parameters,
        selected,
        observables,
        dataVectors,
        maxIter,
        nDraws,
        burnIn,
        thin,
        seed,
        adapt,
        proposalScale,
        posteriorPoint,
        result,
        modeFolded,
      }).catch((reason: unknown) => {
        setNotice(reason instanceof Error ? reason.message : String(reason));
        setError(true);
      });
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [
    adapt,
    burnIn,
    dataVectors,
    hydrated,
    maxIter,
    method,
    modeFolded,
    modelKey,
    nDraws,
    observables,
    parameters,
    posteriorPoint,
    proposalScale,
    result,
    role,
    seed,
    selected,
    thin,
  ]);

  const active = parameters.find((parameter) => parameter.name === selected) ?? null;
  const estimatedCount = parameters.filter((parameter) => parameter.estimate).length;
  const observableNames = parseNames(observables) ?? [];

  function updateParameter(name: string, update: Partial<EstimationParameterSpec>) {
    setParameters((current) =>
      current.map((parameter) =>
        parameter.name === name ? { ...parameter, ...update } : parameter,
      ),
    );
  }

  function updatePrior(
    name: string,
    update: Partial<NonNullable<EstimationParameterSpec["prior"]>>,
  ) {
    setParameters((current) =>
      current.map((parameter) =>
        parameter.name === name && parameter.prior !== null
          ? { ...parameter, prior: { ...parameter.prior, ...update } }
          : parameter,
      ),
    );
  }

  async function submit(estimateAndSolve: boolean) {
    setBusy(true);
    setNotice("");
    setError(false);
    try {
      const output = await runEstimation({
        role,
        method,
        y: matrixFromVectors(observableNames, dataVectors),
        observables: observableNames,
        parameters,
        method_kwargs:
          method === "mcmc"
            ? {
                n_draws: nDraws,
                burn_in: burnIn,
                thin,
                random_state: seed,
                adapt,
                proposal_scale: proposalScale,
              }
            : { options: { maxiter: maxIter } },
        compile_kwargs: {},
        steady_state: null,
        posterior_point: posteriorPoint,
        estimate_and_solve: estimateAndSolve,
      });
      setResult(output);
      if (estimateAndSolve) await onSessionRefresh();
      setNotice(
        estimateAndSolve
          ? "Estimation completed and the model was solved."
          : "Estimation completed.",
      );
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : String(reason));
      setError(true);
    } finally {
      setBusy(false);
    }
  }

  async function importCsv(file: File) {
    setNotice("");
    setError(false);
    try {
      const parsed = parseCsv(await file.text(), model.observables ?? []);
      setObservables(parsed.names.join(", "));
      setDataVectors(
        Object.fromEntries(
          parsed.names.map((name, index) => [
            name,
            parsed.columns[index].join("\n"),
          ]),
        ),
      );
      setNotice(`Loaded ${parsed.rowCount} observations from ${file.name}.`);
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : String(reason));
      setError(true);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function clearWorkspace() {
    const values = model.parameter_values ?? {};
    const names = model.observables ?? [];
    setMethod("mle");
    setParameters(
      Object.entries(values).map(([name, value]) => makeParameter(name, value, catalog)),
    );
    setSelected(Object.keys(values)[0] ?? null);
    setObservables(names.join(", "));
    setDataVectors(Object.fromEntries(names.map((name) => [name, ""])));
    setMaxIter(1000);
    setNDraws(1000);
    setBurnIn(500);
    setThin(1);
    setSeed(0);
    setAdapt(true);
    setProposalScale(0.1);
    setPosteriorPoint("mean");
    setResult(null);
    setModeFolded(false);
    setWorkspaceRevision((current) => current + 1);
    try {
      await clearEstimationWorkspace(role);
      setNotice("Estimation workspace cleared.");
      setError(false);
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : String(reason));
      setError(true);
    }
  }

  const modePanels: PanelDef[] = [
    {
      id: "estimation-mode",
      title: "Estimation Mode",
      badge: `${estimatedCount} selected`,
      scrollable: true,
      content: (
        <div className="estimation-mode-content">
          <div className="estimation-method-section">
            <div className="segmented-control">
              {(["mle", "map", "mcmc"] as EstimationMethod[]).map((item) => (
                <button
                  key={item}
                  className={method === item ? "active" : ""}
                  onClick={() => setMethod(item)}
                >
                  {item.toUpperCase()}
                </button>
              ))}
            </div>
            <div className="estimation-method-fields">
              {method === "mcmc" ? (
                <>
                  <NumberField label="Draws" value={nDraws} onChange={setNDraws} />
                  <NumberField label="Burn-in" value={burnIn} onChange={setBurnIn} />
                  <NumberField label="Thin" value={thin} onChange={setThin} />
                  <NumberField label="Seed" value={seed} onChange={setSeed} />
                  <NumberField
                    label="Proposal scale"
                    value={proposalScale}
                    onChange={setProposalScale}
                  />
                  <label className="switch-row">
                    <span>Adapt proposal</span>
                    <input
                      type="checkbox"
                      checked={adapt}
                      onChange={(event) => setAdapt(event.target.checked)}
                    />
                  </label>
                  <label>
                    Posterior point
                    <select
                      value={posteriorPoint}
                      onChange={(event) => setPosteriorPoint(event.target.value)}
                    >
                      {(catalog?.posterior_points ?? ["mean", "map", "last"]).map(
                        (point) => <option key={point}>{point}</option>,
                      )}
                    </select>
                  </label>
                </>
              ) : (
                <NumberField
                  label="Max iterations"
                  value={maxIter}
                  onChange={setMaxIter}
                />
              )}
            </div>
          </div>
          <div className="estimation-data-section">
            <header>
              <label>
                Observable columns
                <input
                  value={observables}
                  onChange={(event) => setObservables(event.target.value)}
                />
              </label>
              <input
                ref={fileInputRef}
                className="estimation-file-input"
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void importCsv(file);
                }}
              />
              <button
                className="secondary"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload size={15} />
                Import CSV
              </button>
            </header>
            <div className="estimation-vector-list">
              {observableNames.length === 0 ? (
                <span className="muted">Add observable column names to enter data.</span>
              ) : (
                observableNames.map((name) => (
                  <label key={name} className="estimation-vector-field">
                    <span>{name}</span>
                    <textarea
                      value={dataVectors[name] ?? ""}
                      onChange={(event) =>
                        setDataVectors((current) => ({
                          ...current,
                          [name]: event.target.value,
                        }))
                      }
                      placeholder={"1.0\n1.1\n1.2"}
                    />
                  </label>
                ))
              )}
            </div>
            <div className="estimation-actions">
              <button disabled={busy || !model.loaded} onClick={() => void submit(false)}>
                <Play size={15} />
                Run Estimation
              </button>
              <button
                className="secondary"
                disabled={busy || !model.loaded}
                onClick={() => void submit(true)}
              >
                <RefreshCw size={15} />
                Estimate & Solve
              </button>
              <button
                className="secondary"
                disabled={busy}
                onClick={() => void clearWorkspace()}
              >
                <Trash2 size={15} />
                Clear
              </button>
              {notice !== "" && (
                <span className={error ? "status error" : "status"}>{notice}</span>
              )}
            </div>
          </div>
        </div>
      ),
    },
  ];

  const detailPanels: PanelDef[] = [
    {
      id: "estimation-parameters",
      title: "Parameters",
      badge: `${parameters.length}`,
      scrollable: true,
      content: (
        <div className="estimation-parameter-list">
          {parameters.map((parameter) => (
            <button
              key={parameter.name}
              className={`estimation-parameter-card ${
                parameter.estimate ? "estimated" : ""
              } ${selected === parameter.name ? "selected" : ""}`}
              onClick={() => setSelected(parameter.name)}
            >
              <strong>{parameter.name}</strong>
              <span>{format(parameter.initial)}</span>
            </button>
          ))}
        </div>
      ),
    },
    {
      id: "estimation-details",
      title: "Estimation Details",
      badge: active?.name,
      scrollable: true,
      content: active ? (
        <ParameterDetails
          parameter={active}
          method={method}
          catalog={catalog}
          result={result}
          chartRevision={chartRevision}
          onChange={(update) => updateParameter(active.name, update)}
          onPriorChange={(update) => updatePrior(active.name, update)}
        />
      ) : (
        <div className="muted">Load a model to configure its parameters.</div>
      ),
    },
  ];

  return (
    <div className="estimation-layout" style={hidden ? { display: "none" } : undefined}>
      <div className={`estimation-mode-row${modeFolded ? " folded" : ""}`}>
        <PanelWorkspace
          key={`mode:${workspaceRevision}:${hydrated ? "ready" : "loading"}`}
          panels={modePanels}
          defaultLayout="vertical"
          initialFolded={{ "estimation-mode": modeFolded }}
          onFoldChange={(folded) => setModeFolded(folded["estimation-mode"] ?? false)}
        />
      </div>
      <div className="estimation-detail-row">
        <PanelWorkspace
          key={`details:${workspaceRevision}`}
          panels={detailPanels}
          defaultLayout="horizontal"
          defaultSplit={32}
          fillHeight
        />
      </div>
    </div>
  );
}

function ParameterDetails({
  parameter,
  method,
  catalog,
  result,
  chartRevision,
  onChange,
  onPriorChange,
}: {
  parameter: EstimationParameterSpec;
  method: EstimationMethod;
  catalog: EstimationCatalog | null;
  result: EstimationRunResult | null;
  chartRevision: number;
  onChange: (update: Partial<EstimationParameterSpec>) => void;
  onPriorChange: (
    update: Partial<NonNullable<EstimationParameterSpec["prior"]>>,
  ) => void;
}) {
  const prior = parameter.prior;
  const estimatedValue =
    result?.result.theta?.[parameter.name] ??
    result?.result.posterior_mean?.[parameter.name];
  return (
    <div className="estimation-details">
      <label className="switch-row estimation-estimate-switch">
        <span>Estimate</span>
        <input
          type="checkbox"
          checked={parameter.estimate}
          onChange={(event) => onChange({ estimate: event.target.checked })}
        />
      </label>
      <div className="estimation-form-grid">
        <NumberField
          label="Initial value"
          value={parameter.initial}
          onChange={(initial) => onChange({ initial })}
        />
        {method !== "mcmc" && (
          <>
            <OptionalNumberField
              label="Lower bound"
              value={parameter.lower}
              onChange={(lower) => onChange({ lower })}
            />
            <OptionalNumberField
              label="Upper bound"
              value={parameter.upper}
              onChange={(upper) => onChange({ upper })}
            />
          </>
        )}
      </div>
      {method !== "mle" && prior !== null && catalog !== null && (
        <>
          <h3>Prior</h3>
          <div className="estimation-form-grid">
            <label>
              Distribution
              <select
                value={prior.distribution}
                onChange={(event) => {
                  const distribution = event.target.value;
                  onPriorChange({
                    distribution,
                    parameters: numericDefaults(catalog.distributions[distribution]),
                  });
                }}
              >
                {Object.keys(catalog.distributions).map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </label>
            <label>
              Transform
              <select
                value={prior.transform}
                onChange={(event) => {
                  const transform = event.target.value;
                  onPriorChange({
                    transform,
                    transform_kwargs: transformDefaults(
                      catalog.transforms[transform],
                      parameter,
                    ),
                  });
                }}
              >
                {Object.keys(catalog.transforms).map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </label>
            {Object.entries(prior.parameters).map(([name, value]) => (
              <NumberField
                key={`dist:${name}`}
                label={name}
                value={value}
                onChange={(next) =>
                  onPriorChange({ parameters: { ...prior.parameters, [name]: next } })
                }
              />
            ))}
            {Object.entries(prior.transform_kwargs).map(([name, value]) => (
              <NumberField
                key={`transform:${name}`}
                label={name}
                value={value}
                onChange={(next) =>
                  onPriorChange({
                    transform_kwargs: { ...prior.transform_kwargs, [name]: next },
                  })
                }
              />
            ))}
          </div>
        </>
      )}
      {result !== null && (
        <section className="estimation-result">
          <h3>Latest Result</h3>
          <div className="estimation-result-grid">
            <ResultValue label="Method" value={result.method.toUpperCase()} />
            <ResultValue label="Estimate" value={format(estimatedValue)} />
            <ResultValue label="Solved" value={result.solved ? "yes" : "no"} />
            {result.result.accept_rate !== undefined && (
              <ResultValue
                label="Acceptance"
                value={format(result.result.accept_rate)}
              />
            )}
            {result.result.loglik !== undefined && (
              <ResultValue label="Log likelihood" value={format(result.result.loglik)} />
            )}
            {result.result.logpost_mean !== undefined && (
              <ResultValue
                label="Mean log posterior"
                value={format(result.result.logpost_mean)}
              />
            )}
          </div>
          {result.method === "mcmc" && (
            <MCMCCharts
              key={`${result.run_id}:${parameter.name}`}
              chartRevision={chartRevision}
              result={result}
              parameter={parameter.name}
            />
          )}
        </section>
      )}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function OptionalNumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number | null;
  onChange: (value: number | null) => void;
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) =>
          onChange(event.target.value === "" ? null : Number(event.target.value))
        }
      />
    </label>
  );
}

function ResultValue({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  );
}

function MCMCCharts({
  result,
  parameter,
  chartRevision,
}: {
  result: EstimationRunResult;
  parameter: string;
  chartRevision: number;
}) {
  const trace = result.result.logpost_trace ?? [];
  const samples = result.result.samples?.[parameter] ?? [];
  const histogram = useMemo(() => makeHistogram(samples), [samples]);
  const tracePlot = useMemo(() => downsampleTrace(trace), [trace]);
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#94a3b8", maxTicksLimit: 7 } },
      y: { ticks: { color: "#94a3b8", maxTicksLimit: 6 } },
    },
  };
  if (trace.length === 0 && samples.length === 0) return null;
  return (
    <div className="estimation-chart-grid">
      {tracePlot.values.length > 0 && (
        <section className="estimation-chart">
          <h4>Log-posterior trace</h4>
          <div>
            <Line
              key={`trace:${result.run_id}:${chartRevision}`}
              redraw
              data={{
                labels: tracePlot.indices.map((index) => String(index + 1)),
                datasets: [
                  {
                    label: "Log posterior",
                    data: tracePlot.values,
                    borderColor: "#4f46e5",
                    backgroundColor: "rgb(79 70 229 / 14%)",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                  },
                ],
              }}
              options={options}
            />
          </div>
        </section>
      )}
      {samples.length > 0 && (
        <section className="estimation-chart">
          <h4>{parameter} posterior distribution</h4>
          <div>
            <Line
              key={`posterior:${result.run_id}:${parameter}:${chartRevision}`}
              redraw
              data={{
                labels: histogram.labels,
                datasets: [
                  {
                    label: parameter,
                    data: histogram.counts,
                    borderColor: "#0891b2",
                    backgroundColor: "rgb(8 145 178 / 16%)",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.18,
                  },
                ],
              }}
              options={options}
            />
          </div>
        </section>
      )}
    </div>
  );
}

function makeParameter(
  name: string,
  value: number,
  catalog: EstimationCatalog | null,
): EstimationParameterSpec {
  return {
    name,
    estimate: false,
    initial: value,
    lower: null,
    upper: null,
    prior: {
      distribution: "normal",
      parameters: numericDefaults(catalog?.distributions.normal ?? { mean: 0, std: 1 }),
      transform: "identity",
      transform_kwargs: {},
    },
  };
}

function numericDefaults(
  defaults: Record<string, number | null> | undefined,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(defaults ?? {})
      .filter((entry): entry is [string, number] => entry[1] !== null)
      .map(([name, value]) => [name, Number(value)]),
  );
}

function transformDefaults(
  defaults: Record<string, number | null> | undefined,
  parameter: EstimationParameterSpec,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(defaults ?? {}).map(([name, value]) => [
      name,
      value ?? (name === "low" ? parameter.lower ?? 0 : parameter.upper ?? 1),
    ]),
  );
}

function matrixFromVectors(
  names: string[],
  vectors: Record<string, string>,
): number[][] {
  if (names.length === 0) throw new Error("At least one observable column is required.");
  const columns = names.map((name) => parseVector(vectors[name] ?? "", name));
  const length = columns[0].length;
  if (length === 0) throw new Error("Observed data is required.");
  if (columns.some((column) => column.length !== length)) {
    throw new Error("Observable vectors must contain the same number of observations.");
  }
  return Array.from({ length }, (_, row) => columns.map((column) => column[row]));
}

function parseNames(value: string): string[] | null {
  const names = value.split(/[\s,;]+/).filter(Boolean);
  return names.length > 0 ? names : null;
}

function parseVector(value: string, name: string): number[] {
  const values = value
    .trim()
    .split(/[\s,;]+/)
    .filter(Boolean)
    .map(Number);
  if (values.some((item) => !Number.isFinite(item))) {
    throw new Error(`Observable '${name}' contains a non-numeric value.`);
  }
  return values;
}

function parseCsv(
  content: string,
  preferredNames: string[],
): { names: string[]; columns: number[][]; rowCount: number } {
  const lines = content.split(/\r?\n/).filter((line) => line.trim() !== "");
  if (lines.length === 0) throw new Error("The selected CSV is empty.");
  const delimiter = detectDelimiter(lines[0]);
  const rows = lines.map((line) => parseDelimitedLine(line, delimiter));
  const width = rows[0].length;
  if (width === 0 || rows.some((row) => row.length !== width)) {
    throw new Error("CSV rows must have a consistent number of columns.");
  }

  const hasHeader = rows[0].some((value) => !isNumeric(value));
  const header = hasHeader ? rows.shift() ?? [] : [];
  let names: string[];
  let indices: number[];
  if (hasHeader && preferredNames.length > 0 && preferredNames.every((name) => header.includes(name))) {
    names = preferredNames;
    indices = names.map((name) => header.indexOf(name));
  } else if (hasHeader) {
    names = header.map((name, index) => name.trim() || `column_${index + 1}`);
    indices = names.map((_, index) => index);
  } else {
    names =
      preferredNames.length === width
        ? preferredNames
        : Array.from({ length: width }, (_, index) => `column_${index + 1}`);
    indices = names.map((_, index) => index);
  }

  const columns = indices.map(() => [] as number[]);
  for (const [rowIndex, row] of rows.entries()) {
    for (const [columnIndex, sourceIndex] of indices.entries()) {
      const raw = row[sourceIndex]?.trim() ?? "";
      if (!isNumeric(raw)) {
        throw new Error(
          `CSV value at row ${rowIndex + (hasHeader ? 2 : 1)}, column ${sourceIndex + 1} is not numeric.`,
        );
      }
      columns[columnIndex].push(Number(raw));
    }
  }
  return { names, columns, rowCount: rows.length };
}

function detectDelimiter(line: string): string {
  const candidates = [",", ";", "\t"];
  return candidates.reduce((best, candidate) =>
    line.split(candidate).length > line.split(best).length ? candidate : best,
  );
}

function parseDelimitedLine(line: string, delimiter: string): string[] {
  const values: string[] = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === delimiter && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  if (quoted) throw new Error("CSV contains an unterminated quoted value.");
  values.push(value);
  return values;
}

function isNumeric(value: string): boolean {
  return value.trim() !== "" && Number.isFinite(Number(value));
}

function makeHistogram(values: number[]): { labels: string[]; counts: number[] } {
  if (values.length === 0) return { labels: [], counts: [] };
  let min = values[0];
  let max = values[0];
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (min === max) return { labels: [format(min)], counts: [values.length] };
  const bins = Math.max(8, Math.min(36, Math.ceil(Math.sqrt(values.length))));
  const width = (max - min) / bins;
  const counts = Array.from({ length: bins }, () => 0);
  for (const value of values) {
    counts[Math.min(bins - 1, Math.floor((value - min) / width))] += 1;
  }
  return {
    labels: counts.map((_, index) => format(min + (index + 0.5) * width)),
    counts,
  };
}

function downsampleTrace(
  values: number[],
  maxPoints = 2000,
): { indices: number[]; values: number[] } {
  if (values.length <= maxPoints) {
    return { indices: values.map((_, index) => index), values };
  }
  const stride = (values.length - 1) / (maxPoints - 1);
  const indices = Array.from({ length: maxPoints }, (_, index) =>
    Math.round(index * stride),
  );
  return { indices, values: indices.map((index) => values[index]) };
}

function format(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value) ? "--" : value.toFixed(4);
}
