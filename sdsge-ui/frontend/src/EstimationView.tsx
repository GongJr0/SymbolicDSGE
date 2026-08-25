import { Play, RefreshCw, Trash2, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Line } from "react-chartjs-2";
import { getEstimationCatalog, putWorkspaceView, runEstimation } from "./api";
import { PanelWorkspace } from "./PanelWorkspace";
import type { PanelDef } from "./PanelWorkspace";
import type {
  EstimationCatalog,
  EstimationMethod,
  EstimationParameterSpec,
  EstimationResultWire,
  EstimationViewState,
  EstimationViewsByRole,
  MapOptions,
  ModelSummary,
  Role,
  SessionWorkspace,
} from "./types";

// Mirrors the estimator's own kwarg defaults, so an untouched form runs what
// `Estimator.mle`/`map`/`mcmc` would run. `nDraws` has no library default.
const DEFAULTS = {
  optimizer: "L-BFGS-B",
  maxIter: 15000,
  maxFun: 15000,
  m: 10,
  maxLs: 20,
  factr: 1e7,
  pgtol: 1e-5,
  fdStep: 0,
  xatol: 1e-4,
  fatol: 1e-4,
  nDraws: 1000,
  burnIn: 1000,
  thin: 1,
  seed: 0,
  proposalScale: 0.1,
  adapt: true,
  adaptStart: 100,
  adaptEpsilon: 1e-8,
  posteriorPoint: "mean",
  cov: true,
  jacobian: false,
  computeMap: true,
  covFdStepScale: 1.0,
  covFdAbsoluteFloor: 0.1,
  mapOptions: null as MapOptions | null,
  proposalCov: null as number[][] | null,
};

// What the sampler falls back to for the MAP presolve when `map_options` is
// absent. Shares the optimizer defaults above, under the estimator's own key
// names, since the dict reaches `run_mcmc` unmapped.
const MAP_OPTION_DEFAULTS: Required<Omit<MapOptions, "bounds">> = {
  method: DEFAULTS.optimizer,
  m: DEFAULTS.m,
  maxiter: DEFAULTS.maxIter,
  maxfun: DEFAULTS.maxFun,
  maxls: DEFAULTS.maxLs,
  factr: DEFAULTS.factr,
  pgtol: DEFAULTS.pgtol,
  fd_step: DEFAULTS.fdStep,
  xatol: DEFAULTS.xatol,
  fatol: DEFAULTS.fatol,
};

// `cov_status` codes from _ckernels/estimation/estimation.h, as reasons.
const COV_STATUS_REASONS: Record<number, string> = {
  [-1800]: "the covariance workspace could not be allocated",
  [-1801]: "the Hessian at the optimum could not be formed",
  [-1802]: "the Hessian at the optimum is not positive definite",
};

export function EstimationView({
  hidden,
  role,
  model,
  workspace,
  onSessionRefresh,
}: {
  hidden?: boolean;
  role: Role;
  model: ModelSummary;
  workspace: SessionWorkspace | null;
  onSessionRefresh: () => Promise<void>;
}) {
  const [catalog, setCatalog] = useState<EstimationCatalog | null>(null);
  const [method, setMethod] = useState<EstimationMethod>("mle");
  const [parameters, setParameters] = useState<EstimationParameterSpec[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [observables, setObservables] = useState("");
  const [dataVectors, setDataVectors] = useState<Record<string, string>>({});
  const [optimizer, setOptimizer] = useState(DEFAULTS.optimizer);
  const [maxIter, setMaxIter] = useState(DEFAULTS.maxIter);
  const [maxFun, setMaxFun] = useState(DEFAULTS.maxFun);
  const [m, setM] = useState(DEFAULTS.m);
  const [maxLs, setMaxLs] = useState(DEFAULTS.maxLs);
  const [factr, setFactr] = useState(DEFAULTS.factr);
  const [pgtol, setPgtol] = useState(DEFAULTS.pgtol);
  const [fdStep, setFdStep] = useState(DEFAULTS.fdStep);
  const [xatol, setXatol] = useState(DEFAULTS.xatol);
  const [fatol, setFatol] = useState(DEFAULTS.fatol);
  const [nDraws, setNDraws] = useState(DEFAULTS.nDraws);
  const [burnIn, setBurnIn] = useState(DEFAULTS.burnIn);
  const [thin, setThin] = useState(DEFAULTS.thin);
  const [seed, setSeed] = useState(DEFAULTS.seed);
  const [proposalScale, setProposalScale] = useState(DEFAULTS.proposalScale);
  const [adapt, setAdapt] = useState(DEFAULTS.adapt);
  const [adaptStart, setAdaptStart] = useState(DEFAULTS.adaptStart);
  const [adaptEpsilon, setAdaptEpsilon] = useState(DEFAULTS.adaptEpsilon);
  const [posteriorPoint, setPosteriorPoint] = useState(DEFAULTS.posteriorPoint);
  const [cov, setCov] = useState(DEFAULTS.cov);
  const [jacobian, setJacobian] = useState(DEFAULTS.jacobian);
  const [computeMap, setComputeMap] = useState(DEFAULTS.computeMap);
  const [covFdStepScale, setCovFdStepScale] = useState(DEFAULTS.covFdStepScale);
  const [covFdAbsoluteFloor, setCovFdAbsoluteFloor] = useState(
    DEFAULTS.covFdAbsoluteFloor,
  );
  const [mapOptions, setMapOptions] = useState(DEFAULTS.mapOptions);
  const [proposalCov, setProposalCov] = useState(DEFAULTS.proposalCov);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState(false);
  const [result, setResult] = useState<EstimationResultWire | null>(null);
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

  // Seeded once from the session, then owned by this component. Later session
  // reads carry back only what was PUT from here, so re-reading them would
  // fight the user's typing.
  const viewsRef = useRef<EstimationViewsByRole>({});

  useEffect(() => {
    if (catalog === null || workspace === null || hydrated) return;
    const values = model.parameter_values ?? {};
    const names = model.observables ?? [];
    viewsRef.current = workspace.estimation?.view ?? {};
    // Merge over the defaults rather than requiring every field: a bundle
    // fills only what it can speak to, and a control added later still opens
    // at its default instead of undefined.
    const stored = viewsRef.current[role];
    const base: EstimationViewState = {
      ...DEFAULTS,
      method: "mle",
      parameters: Object.entries(values).map(([name, value]) =>
        makeParameter(name, value, catalog),
      ),
      selected: Object.keys(values)[0] ?? null,
      observables: names.join(", "),
      dataVectors: Object.fromEntries(names.map((name) => [name, ""])),
      modeFolded: false,
    };
    applyView({ ...base, ...stored });
    setResult(workspace.estimation?.result ?? null);
    setHydrated(true);
  }, [catalog, hydrated, model, role, workspace]);

  // Switching role swaps in that role's form without touching the other's.
  useEffect(() => {
    if (!hydrated) return;
    applyView({ ...currentView(), ...viewsRef.current[role] });
    // Only on a role change: the deps are deliberately not the form fields.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [role]);

  function currentView(): EstimationViewState {
    return {
      method,
      parameters,
      selected,
      observables,
      dataVectors,
      optimizer,
      maxIter,
      maxFun,
      m,
      maxLs,
      factr,
      pgtol,
      fdStep,
      xatol,
      fatol,
      nDraws,
      burnIn,
      thin,
      seed,
      proposalScale,
      adapt,
      adaptStart,
      adaptEpsilon,
      posteriorPoint,
      cov,
      jacobian,
      computeMap,
      covFdStepScale,
      covFdAbsoluteFloor,
      mapOptions,
      proposalCov,
      modeFolded,
    };
  }

  function applyView(view: EstimationViewState) {
    setMethod(view.method);
    setParameters(view.parameters);
    setSelected(view.selected);
    setObservables(view.observables);
    setDataVectors(view.dataVectors);
    setOptimizer(view.optimizer);
    setMaxIter(view.maxIter);
    setMaxFun(view.maxFun);
    setM(view.m);
    setMaxLs(view.maxLs);
    setFactr(view.factr);
    setPgtol(view.pgtol);
    setFdStep(view.fdStep);
    setXatol(view.xatol);
    setFatol(view.fatol);
    setNDraws(view.nDraws);
    setBurnIn(view.burnIn);
    setThin(view.thin);
    setSeed(view.seed);
    setProposalScale(view.proposalScale);
    setAdapt(view.adapt);
    setAdaptStart(view.adaptStart);
    setAdaptEpsilon(view.adaptEpsilon);
    setPosteriorPoint(view.posteriorPoint);
    setCov(view.cov);
    setJacobian(view.jacobian);
    setComputeMap(view.computeMap);
    setCovFdStepScale(view.covFdStepScale);
    setCovFdAbsoluteFloor(view.covFdAbsoluteFloor);
    setMapOptions(view.mapOptions);
    setProposalCov(view.proposalCov);
    setModeFolded(view.modeFolded);
  }

  useEffect(() => {
    if (!hydrated) return;
    const timeout = window.setTimeout(() => {
      viewsRef.current = { ...viewsRef.current, [role]: currentView() };
      void putWorkspaceView("estimation", viewsRef.current).catch(
        (reason: unknown) => {
          setNotice(reason instanceof Error ? reason.message : String(reason));
          setError(true);
        },
      );
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [
    adapt,
    adaptEpsilon,
    adaptStart,
    burnIn,
    computeMap,
    cov,
    covFdAbsoluteFloor,
    covFdStepScale,
    dataVectors,
    factr,
    fatol,
    fdStep,
    hydrated,
    jacobian,
    m,
    mapOptions,
    maxFun,
    maxIter,
    maxLs,
    method,
    modeFolded,
    nDraws,
    observables,
    optimizer,
    parameters,
    pgtol,
    posteriorPoint,
    proposalCov,
    proposalScale,
    role,
    seed,
    selected,
    thin,
    xatol,
  ]);

  const active = parameters.find((parameter) => parameter.name === selected) ?? null;
  const estimatedCount = parameters.filter((parameter) => parameter.estimate).length;
  const observableNames = parseNames(observables) ?? [];
  const estimatedNames = parameters
    .filter((parameter) => parameter.estimate)
    .map((parameter) => parameter.name);

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
                proposal_scale: proposalScale,
                adapt,
                ...(adapt
                  ? { adapt_start: adaptStart, adapt_epsilon: adaptEpsilon }
                  : {}),
                compute_map: computeMap,
                cov_fd_step_scale: covFdStepScale,
                cov_fd_absolute_floor: covFdAbsoluteFloor,
                ...(mapOptions === null ? {} : { map_options: mapOptions }),
                ...(proposalCov === null ? {} : { proposal_cov: proposalCov }),
              }
            : {
                method: optimizer,
                maxiter: maxIter,
                maxfun: maxFun,
                ...(optimizer === "Nelder-Mead"
                  ? { xatol, fatol }
                  : { m, maxls: maxLs, factr, pgtol, fd_step: fdStep }),
                cov,
                cov_fd_step_scale: covFdStepScale,
                cov_fd_absolute_floor: covFdAbsoluteFloor,
                // Only MAP takes it; mle has no such parameter.
                ...(method === "map" ? { jacobian } : {}),
              },
        compile_kwargs: {},
        ss_seed: null,
        posterior_point: posteriorPoint,
        estimate_and_solve: estimateAndSolve,
      });
      setResult(output.result);
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
    setOptimizer(DEFAULTS.optimizer);
    setMaxIter(DEFAULTS.maxIter);
    setMaxFun(DEFAULTS.maxFun);
    setM(DEFAULTS.m);
    setMaxLs(DEFAULTS.maxLs);
    setFactr(DEFAULTS.factr);
    setPgtol(DEFAULTS.pgtol);
    setFdStep(DEFAULTS.fdStep);
    setXatol(DEFAULTS.xatol);
    setFatol(DEFAULTS.fatol);
    setNDraws(DEFAULTS.nDraws);
    setBurnIn(DEFAULTS.burnIn);
    setThin(DEFAULTS.thin);
    setSeed(DEFAULTS.seed);
    setProposalScale(DEFAULTS.proposalScale);
    setAdapt(DEFAULTS.adapt);
    setAdaptStart(DEFAULTS.adaptStart);
    setAdaptEpsilon(DEFAULTS.adaptEpsilon);
    setPosteriorPoint(DEFAULTS.posteriorPoint);
    setCov(DEFAULTS.cov);
    setJacobian(DEFAULTS.jacobian);
    setComputeMap(DEFAULTS.computeMap);
    setCovFdStepScale(DEFAULTS.covFdStepScale);
    setCovFdAbsoluteFloor(DEFAULTS.covFdAbsoluteFloor);
    setMapOptions(DEFAULTS.mapOptions);
    setProposalCov(DEFAULTS.proposalCov);
    setResult(null);
    setModeFolded(false);
    setWorkspaceRevision((current) => current + 1);
    try {
      const { [role]: _cleared, ...rest } = viewsRef.current;
      viewsRef.current = rest;
      await putWorkspaceView("estimation", viewsRef.current);
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
                  <label className="switch-row">
                    <span>Adapt proposal</span>
                    <input
                      type="checkbox"
                      checked={adapt}
                      onChange={(event) => setAdapt(event.target.checked)}
                    />
                  </label>
                  {adapt && (
                    <>
                      <NumberField
                        label="Adapt start"
                        value={adaptStart}
                        onChange={setAdaptStart}
                      />
                      <NumberField
                        label="Adapt epsilon"
                        value={adaptEpsilon}
                        onChange={setAdaptEpsilon}
                      />
                    </>
                  )}
                  {computeMap !== DEFAULTS.computeMap && (
                    <SwitchField
                      label="Start from MAP"
                      value={computeMap}
                      onChange={setComputeMap}
                    />
                  )}
                  <CovarianceFields
                    stepScale={covFdStepScale}
                    absoluteFloor={covFdAbsoluteFloor}
                    onStepScale={setCovFdStepScale}
                    onAbsoluteFloor={setCovFdAbsoluteFloor}
                  />
                </>
              ) : (
                <>
                  <label>
                    Optimizer
                    <select
                      value={optimizer}
                      onChange={(event) => setOptimizer(event.target.value)}
                    >
                      {(catalog?.optimizer_methods ?? [DEFAULTS.optimizer]).map(
                        (name) => <option key={name}>{name}</option>,
                      )}
                    </select>
                  </label>
                  <NumberField
                    label="Max iterations"
                    value={maxIter}
                    onChange={setMaxIter}
                  />
                  <NumberField
                    label="Max evaluations"
                    value={maxFun}
                    onChange={setMaxFun}
                  />
                  {optimizer === "Nelder-Mead" ? (
                    <>
                      <NumberField label="xatol" value={xatol} onChange={setXatol} />
                      <NumberField label="fatol" value={fatol} onChange={setFatol} />
                    </>
                  ) : (
                    <>
                      <NumberField label="History size" value={m} onChange={setM} />
                      <NumberField
                        label="Max line search"
                        value={maxLs}
                        onChange={setMaxLs}
                      />
                      <NumberField label="factr" value={factr} onChange={setFactr} />
                      <NumberField label="pgtol" value={pgtol} onChange={setPgtol} />
                      <NumberField label="FD step" value={fdStep} onChange={setFdStep} />
                    </>
                  )}
                  {cov !== DEFAULTS.cov && (
                    <SwitchField
                      label="Standard errors"
                      value={cov}
                      onChange={setCov}
                    />
                  )}
                  {method === "map" && jacobian !== DEFAULTS.jacobian && (
                    <SwitchField
                      label="Include log-jacobian"
                      value={jacobian}
                      onChange={setJacobian}
                    />
                  )}
                  <CovarianceFields
                    stepScale={covFdStepScale}
                    absoluteFloor={covFdAbsoluteFloor}
                    onStepScale={setCovFdStepScale}
                    onAbsoluteFloor={setCovFdAbsoluteFloor}
                  />
                </>
              )}
            </div>
            {method === "mcmc" && computeMap && (
              <MapOptionsPanel
                options={{ ...MAP_OPTION_DEFAULTS, ...(mapOptions ?? {}) }}
                optimizers={catalog?.optimizer_methods ?? [DEFAULTS.optimizer]}
                bounds={mapOptions?.bounds ?? null}
                estimatedNames={estimatedNames}
                onChange={(update) =>
                  setMapOptions({ ...(mapOptions ?? {}), ...update })
                }
              />
            )}
            {method === "mcmc" && (
              <ProposalCovariance
                value={proposalCov}
                names={estimatedNames}
                computeMap={computeMap}
              />
            )}
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
          solved={model.solved ?? false}
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
  solved,
  chartRevision,
  onChange,
  onPriorChange,
}: {
  parameter: EstimationParameterSpec;
  method: EstimationMethod;
  catalog: EstimationCatalog | null;
  result: EstimationResultWire | null;
  solved: boolean;
  chartRevision: number;
  onChange: (update: Partial<EstimationParameterSpec>) => void;
  onPriorChange: (
    update: Partial<NonNullable<EstimationParameterSpec["prior"]>>,
  ) => void;
}) {
  const prior = parameter.prior;
  const estimatedValue =
    result?.theta?.[parameter.name] ??
    result?.posterior_mean?.[parameter.name];
  // A null `se` means the run computed no covariance, which shows the same as
  // having no result field at all.
  const standardErrors = result?.se ?? undefined;
  const covStatus = result?.cov_status ?? 0;
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
            <ResultValue label="Method" value={methodOf(result).toUpperCase()} />
            <ResultValue label="Estimate" value={format(estimatedValue)} />
            {standardErrors !== undefined && (
              <ResultValue
                label="Std. error"
                value={format(standardErrors[parameter.name])}
              />
            )}
            <ResultValue label="Solved" value={solved ? "yes" : "no"} />
            {result.accept_rate !== undefined && (
              <ResultValue
                label="Acceptance"
                value={format(result.accept_rate)}
              />
            )}
            {result.loglik !== undefined && (
              <ResultValue label="Log likelihood" value={format(result.loglik)} />
            )}
            {result.logpost !== undefined && (
              <ResultValue label="Log posterior" value={format(result.logpost)} />
            )}
            {result.logprior !== undefined && (
              <ResultValue label="Log prior" value={format(result.logprior)} />
            )}
            {result.logpost_mean !== undefined && (
              <ResultValue
                label="Mean log posterior"
                value={format(result.logpost_mean)}
              />
            )}
          </div>
          {covStatus !== 0 && (
            <span className="muted">
              {`No standard errors: ${
                COV_STATUS_REASONS[covStatus] ?? `covariance status ${covStatus}`
              }.`}
            </span>
          )}
          {methodOf(result) === "mcmc" && (
            <MCMCCharts
              key={`${chartRevision}:${parameter.name}`}
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

/** The MAP presolve's optimizer options.
 *
 * A second full option set that only matters when the chain starts from a MAP
 * estimate, so it folds away rather than crowding the sampler's own fields.
 * Stays null until touched, which leaves the sampler on its own defaults.
 */
function MapOptionsPanel({
  options,
  optimizers,
  bounds,
  estimatedNames,
  onChange,
}: {
  options: Required<Omit<MapOptions, "bounds">>;
  optimizers: string[];
  bounds: Array<[number | null, number | null]> | null;
  estimatedNames: string[];
  onChange: (update: MapOptions) => void;
}) {
  return (
    <details className="estimation-suboptions">
      <summary>MAP start options</summary>
      <div className="estimation-method-fields">
        <label>
          Optimizer
          <select
            value={options.method}
            onChange={(event) => onChange({ method: event.target.value })}
          >
            {optimizers.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
        <NumberField
          label="Max iterations"
          value={options.maxiter}
          onChange={(maxiter) => onChange({ maxiter })}
        />
        <NumberField
          label="Max evaluations"
          value={options.maxfun}
          onChange={(maxfun) => onChange({ maxfun })}
        />
        {options.method === "Nelder-Mead" ? (
          <>
            <NumberField
              label="xatol"
              value={options.xatol}
              onChange={(xatol) => onChange({ xatol })}
            />
            <NumberField
              label="fatol"
              value={options.fatol}
              onChange={(fatol) => onChange({ fatol })}
            />
          </>
        ) : (
          <>
            <NumberField
              label="History size"
              value={options.m}
              onChange={(m) => onChange({ m })}
            />
            <NumberField
              label="Max line search"
              value={options.maxls}
              onChange={(maxls) => onChange({ maxls })}
            />
            <NumberField
              label="factr"
              value={options.factr}
              onChange={(factr) => onChange({ factr })}
            />
            <NumberField
              label="pgtol"
              value={options.pgtol}
              onChange={(pgtol) => onChange({ pgtol })}
            />
            <NumberField
              label="FD step"
              value={options.fd_step}
              onChange={(fd_step) => onChange({ fd_step })}
            />
          </>
        )}
      </div>
      {bounds !== null && (
        <>
          <h4>Bounds</h4>
          <MatrixTable
            rowLabels={estimatedNames}
            columnLabels={["lower", "upper"]}
            values={bounds.map(([low, high]) => [low, high])}
          />
        </>
      )}
    </details>
  );
}

/** A read-only numeric table with labelled axes.
 *
 * What it shows arrives already decided: a covariance the sampler was handed,
 * or the bounds a MAP presolve ran under. Making them legible is the point;
 * neither is something this form sets.
 */
function MatrixTable({
  rowLabels,
  columnLabels,
  values,
}: {
  rowLabels: string[];
  columnLabels: string[];
  values: Array<Array<number | null>>;
}) {
  return (
    <div className="estimation-matrix">
      <table>
        <thead>
          <tr>
            <th />
            {columnLabels.map((label) => (
              <th key={label}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {values.map((row, index) => (
            <tr key={rowLabels[index] ?? index}>
              <th>{rowLabels[index] ?? index}</th>
              {row.map((value, column) => (
                <td key={columnLabels[column] ?? column}>
                  {value === null || !Number.isFinite(value)
                    ? "--"
                    : value.toPrecision(3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Which of the sampler's two proposal sources this run used.
 *
 * A covariance that was handed over is shown; one the sampler derives has no
 * value to show until it has run, so the readout names where it comes from
 * instead. There is no third state: a Hessian that fails aborts the run
 * rather than degrading to anything.
 */
function ProposalCovariance({
  value,
  names,
  computeMap,
}: {
  value: number[][] | null;
  names: string[];
  computeMap: boolean;
}) {
  if (value === null) {
    return (
      <span className="muted">
        {`Proposal covariance: derived from the Hessian at ${
          computeMap ? "the MAP estimate" : "the starting values"
        }.`}
      </span>
    );
  }
  return (
    <details className="estimation-suboptions">
      <summary>Proposal covariance (pre-specified)</summary>
      <MatrixTable rowLabels={names} columnLabels={names} values={value} />
    </details>
  );
}

function SwitchField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="switch-row">
      <span>{label}</span>
      <input
        type="checkbox"
        checked={value}
        onChange={(event) => onChange(event.target.checked)}
      />
    </label>
  );
}

/** The finite-difference covariance steps, shown only once a run has moved
 * one off its default. Both routines take them, so both render this. */
function CovarianceFields({
  stepScale,
  absoluteFloor,
  onStepScale,
  onAbsoluteFloor,
}: {
  stepScale: number;
  absoluteFloor: number;
  onStepScale: (value: number) => void;
  onAbsoluteFloor: (value: number) => void;
}) {
  return (
    <>
      {stepScale !== DEFAULTS.covFdStepScale && (
        <NumberField label="Cov FD scale" value={stepScale} onChange={onStepScale} />
      )}
      {absoluteFloor !== DEFAULTS.covFdAbsoluteFloor && (
        <NumberField
          label="Cov FD floor"
          value={absoluteFloor}
          onChange={onAbsoluteFloor}
        />
      )}
    </>
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

/** The routine a result came from, read off the wire it produced.
 *
 * Only the MCMC wire names itself; a point estimate is told apart by which
 * objective it carries, exactly as the emitter decides which one to write.
 */
function methodOf(result: EstimationResultWire): EstimationMethod {
  if (result.kind === "mcmc") return "mcmc";
  return result.logpost === undefined ? "mle" : "map";
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
  result: EstimationResultWire;
  parameter: string;
  chartRevision: number;
}) {
  const trace = result.logpost_trace ?? [];
  const samples = result.samples?.[parameter] ?? [];
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
              key={`trace:${chartRevision}`}
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
              key={`posterior:${parameter}:${chartRevision}`}
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

// `null` is how the wire says "the run produced no finite value here", which
// reads the same as absent.
function format(value: number | null | undefined): string {
  return value === undefined || value === null || !Number.isFinite(value)
    ? "--"
    : value.toFixed(4);
}
