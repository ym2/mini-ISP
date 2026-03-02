const stageList = document.getElementById("stage-list");
const previewA = document.getElementById("preview-a");
const previewB = document.getElementById("preview-b");
const stageName = document.getElementById("stage-name");
const debugPanelA = document.getElementById("debug-a");
const debugPanelB = document.getElementById("debug-b");
const debugPanel = document.getElementById("debug-panel");
const debugLabelA = document.getElementById("debug-label-a");
const debugLabelB = document.getElementById("debug-label-b");
const timingA = document.getElementById("timing-a");
const timingB = document.getElementById("timing-b");
const timingDelta = document.getElementById("timing-delta");
const placeholderA = document.getElementById("placeholder-a");
const placeholderB = document.getElementById("placeholder-b");
const labelA = document.getElementById("label-a");
const labelB = document.getElementById("label-b");
const toggleRoi = document.getElementById("toggle-roi");
const toggleDebug = document.getElementById("toggle-debug");
const runTitle = document.getElementById("run-title");
const metricsLabelA = document.getElementById("metrics-label-a");
const metricsLabelB = document.getElementById("metrics-label-b");
const metricsPanelA = document.getElementById("metrics-a");
const metricsPanelB = document.getElementById("metrics-b");
const diffMetricsPanel = document.getElementById("diff-metrics");
const diffMetricsSection = document.getElementById("diff-metrics-section");
const diagnosticsPanel = document.getElementById("diagnostics-panel");
const diagnosticsControls = document.getElementById("diagnostics-controls");
const diagnosticsStatus = document.getElementById("diagnostics-status");
const metricsToggle = document.getElementById("metrics-toggle");
const signalsPanel = document.getElementById("signals-panel");
const signalsTable = document.getElementById("signals-table");

let manifest = null;
let manifestB = null;
let compareBundle = null;
let compareStages = null;
let compareRunDirs = null;
let activeIndex = 0;
let activeStageName = "";
let useRoi = false;
let diagMode = "preview";
let metricsRequestId = 0;
let diagRequestId = 0;
let debugRequestId = 0;
let diagAvailability = {};
let showAllMetrics = false;
let metricsCache = { a: null, b: null };
let diffMetricsCache = {};
let debugCache = { a: null, b: null };

const METRICS_DEFAULT_KEYS = ["luma_mean", "clip_pct", "p99", "max", "min", "psnr"];
const SIGNAL_SPECS = {
  __common: [
    { label: "luma_mean", source: "metrics", path: "luma_mean" },
    { label: "clip_pct", source: "metrics", path: "clip_pct" },
    { label: "p99", source: "metrics", path: "p99" },
  ],
  raw_norm: [
    { label: "raw_min", source: "debug", path: "metrics.min" },
    { label: "raw_max", source: "debug", path: "metrics.max" },
    { label: "dtype", source: "debug", path: "metrics.dtype" },
  ],
  dpc: [
    { label: "n_fixed", source: "debug", path: "metrics.n_fixed" },
    { label: "threshold", source: "debug", path: "metrics.threshold" },
    { label: "neighbor_policy", source: "debug", path: "metrics.neighbor_policy" },
  ],
  lsc: [
    { label: "gain_mean", source: "debug", path: "metrics.gain_mean" },
    { label: "gain_max", source: "debug", path: "metrics.gain_max" },
    { label: "k", source: "debug", path: "metrics.k" },
  ],
  wb_gains: [
    { label: "wb_mode", source: "debug", path: "params.wb_mode" },
    { label: "wb_source", source: "debug", path: "params.wb_source" },
    { label: "wb_gains", source: "debug", path: "params.wb_gains" },
  ],
  demosaic: [
    { label: "method", source: "debug", path: "params.method" },
    { label: "clip_applied", source: "debug", path: "metrics.clip_applied" },
  ],
  denoise: [
    { label: "method", source: "debug", path: "params.method" },
    { label: "sigma", source: "debug", path: "params.sigma" },
    { label: "sigma_y", source: "debug", path: "params.sigma_y" },
    { label: "sigma_c", source: "debug", path: "params.sigma_c" },
  ],
  ccm: [
    { label: "mode", source: "debug", path: "params.mode" },
    { label: "ccm_source", source: "debug", path: "params.ccm_source" },
    { label: "meta_rule", source: "debug", path: "params.non_dng_meta_rule" },
    { label: "meta_branch", source: "debug", path: "params.non_dng_meta_branch" },
    { label: "wp_err_d50", source: "debug", path: "params.non_dng_meta_wp_err_d50" },
    { label: "wp_err_d65", source: "debug", path: "params.non_dng_meta_wp_err_d65" },
    { label: "outlier_trigger", source: "debug", path: "params.non_dng_meta_outlier_confidence_trigger" },
    { label: "matrix_policy", source: "debug", path: "params.non_dng_matrix_source_policy" },
  ],
  stats_3a: [
    { label: "ae_mean", source: "debug", path: "metrics.stats_3a.ae.mean" },
    { label: "ae_clip_pct", source: "debug", path: "metrics.stats_3a.ae.clip_pct" },
    { label: "af_tenengrad", source: "debug", path: "metrics.stats_3a.af.tenengrad" },
  ],
  tone: [
    { label: "method", source: "debug", path: "params.method" },
    { label: "exposure", source: "debug", path: "params.exposure" },
    { label: "white_point", source: "debug", path: "params.white_point" },
    { label: "gamma", source: "debug", path: "params.gamma" },
  ],
  color_adjust: [
    { label: "method", source: "debug", path: "params.method" },
    { label: "sat_scale", source: "debug", path: "params.sat_scale" },
  ],
  sharpen: [
    { label: "method", source: "debug", path: "params.method" },
    { label: "sigma", source: "debug", path: "params.sigma" },
    { label: "amount", source: "debug", path: "params.amount" },
    { label: "threshold", source: "debug", path: "params.threshold" },
    { label: "luma_only", source: "debug", path: "params.luma_only" },
  ],
  oetf_encode: [
    { label: "oetf", source: "debug", path: "params.oetf" },
    { label: "bit_depth", source: "debug", path: "metrics.bit_depth" },
  ],
  __single_diff: [
    { label: "stage_l1", source: "diff", path: "l1" },
    { label: "stage_l2", source: "diff", path: "l2" },
    { label: "stage_psnr", source: "diff", path: "psnr" },
  ],
};

function getQueryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

function loadJson(path) {
  return fetch(path).then((resp) => resp.json());
}

function loadJsonSafe(path) {
  return fetch(path).then((resp) => {
    if (!resp.ok) {
      throw new Error("not found");
    }
    return resp.json();
  });
}

function toAbsolutePath(path) {
  if (!path) return path;
  if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("/")) {
    return path;
  }
  return "/" + path.replace(/^\//, "");
}

function loadManifests() {
  const comparePath = getQueryParam("compare");
  const manifestPath = getQueryParam("manifest");
  if (comparePath) {
    return loadJson(comparePath).then((bundle) => {
      compareBundle = bundle;
      compareRunDirs = {
        a: bundle.a.run_dir,
        b: bundle.b.run_dir,
      };
      const aPath = toAbsolutePath(`${bundle.a.run_dir}/manifest.json`);
      const bPath = toAbsolutePath(`${bundle.b.run_dir}/manifest.json`);
      return Promise.all([loadJson(aPath), loadJson(bPath)]).then(([a, b]) => {
        manifest = a;
        manifestB = b;
      });
    });
  }
  const path = manifestPath || "../manifest.json";
  return loadJson(path).then((data) => {
    manifest = data;
  });
}

function normalizeCompareLabel(label, side) {
  const raw = String(label || "").trim();
  if (!raw) return side;
  const withoutPrefix = raw.replace(new RegExp(`^${side}\\s*[:\\-]?\\s*`, "i"), "").trim();
  return withoutPrefix || side;
}

function buildCompareSubtitle(bundle) {
  const variantA = normalizeCompareLabel(bundle && bundle.a ? bundle.a.label : "", "A");
  const variantB = normalizeCompareLabel(bundle && bundle.b ? bundle.b.label : "", "B");
  if (variantA === "A" && variantB === "B") {
    return "Compare";
  }
  return `Compare: ${variantA} vs ${variantB}`;
}

function renderStageList() {
  stageList.innerHTML = "";
  const stages = compareStages || manifest.stages;
  stages.forEach((stage, idx) => {
    const li = document.createElement("li");
    const stageA = stage.a;
    const stageB = stage.b;
    li.textContent =
      (stageA && (stageA.display_name || stageA.name)) ||
      (stageB && (stageB.display_name || stageB.name)) ||
      stage.display_name ||
      stage.name ||
      stage.name_b ||
      stage.name_a ||
      `Stage ${idx}`;
    li.addEventListener("click", () => setActiveStage(idx));
    if (idx === activeIndex) {
      li.classList.add("active");
    }
    stageList.appendChild(li);
  });
  if (compareBundle) {
    runTitle.textContent = buildCompareSubtitle(compareBundle);
    labelA.textContent = "A";
    labelB.textContent = "B";
    debugLabelA.textContent = labelA.textContent;
    debugLabelB.textContent = labelB.textContent;
    metricsLabelA.textContent = labelA.textContent;
    metricsLabelB.textContent = labelB.textContent;
    document.body.classList.remove("single-run");
  } else {
    runTitle.textContent = "Single run";
    labelA.textContent = "Run";
    labelB.textContent = "";
    debugLabelA.textContent = labelA.textContent;
    debugLabelB.textContent = "";
    metricsLabelA.textContent = labelA.textContent;
    metricsLabelB.textContent = "";
    document.body.classList.add("single-run");
  }
}

function setActiveStage(index) {
  activeIndex = index;
  renderStageList();
  const stage = compareStages ? compareStages[index] : { a: manifest.stages[index] };
  const stageA = stage.a;
  const stageB = stage.b;
  activeStageName = (stageA && stageA.name) || (stageB && stageB.name) || "";
  debugCache = { a: null, b: null };
  diffMetricsCache = {};
  setSignalsLoadingState();
  stageName.textContent =
    (stageA && (stageA.display_name || stageA.name)) ||
    (stageB && (stageB.display_name || stageB.name)) ||
    "";
  updateTiming(stageA, stageB);
  updateRoiToggle(stageA, stageB);
  updateDiagnosticsAvailability(stageA, stageB);
  updatePreview(stageA, stageB);
  loadDebug(stageA ? stageA.artifacts.debug : null, stageB ? stageB.artifacts.debug : null);
  updateMetrics(stageA, stageB);
}

function setPreviewImage(imgEl, placeholderEl, path) {
  if (!path) {
    imgEl.src = "";
    placeholderEl.classList.remove("hidden");
    return;
  }
  imgEl.onload = () => {
    placeholderEl.classList.add("hidden");
  };
  imgEl.onerror = () => {
    imgEl.src = "";
    placeholderEl.classList.remove("hidden");
  };
  imgEl.src = path;
  placeholderEl.classList.add("hidden");
}

function updatePreview(stageA, stageB) {
  const stageDirA = stageA ? stageDirFromArtifacts(stageA.artifacts) : null;
  const stageDirB = stageB ? stageDirFromArtifacts(stageB.artifacts) : null;
  const diagKey = diagMode !== "preview" ? diagMode : null;
  if (!compareBundle) {
    const stage = stageA;
    let path = null;
    if (diagKey) {
      path = stageDirA ? resolveDiagPath(null, stageDirA, diagKey) : null;
    } else {
      path = useRoi && stage.artifacts.roi ? stage.artifacts.roi : stage.artifacts.preview;
      path = "../" + path;
    }
    setPreviewImage(previewA, placeholderA, path);
    setPreviewImage(previewB, placeholderB, "");
    return;
  }

  if (stageA) {
    let pathA = null;
    if (diagKey) {
      pathA = stageDirA ? resolveDiagPath(compareRunDirs.a, stageDirA, diagKey) : null;
    } else {
      const previewPath = useRoi && stageA.artifacts.roi ? stageA.artifacts.roi : stageA.artifacts.preview;
      pathA = toAbsolutePath(`${compareRunDirs.a}/${previewPath}`);
    }
    setPreviewImage(previewA, placeholderA, pathA);
  } else {
    setPreviewImage(previewA, placeholderA, "");
  }

  if (stageB) {
    let pathB = null;
    if (diagKey) {
      pathB = stageDirB ? resolveDiagPath(compareRunDirs.b, stageDirB, diagKey) : null;
    } else {
      const previewPath = useRoi && stageB.artifacts.roi ? stageB.artifacts.roi : stageB.artifacts.preview;
      pathB = toAbsolutePath(`${compareRunDirs.b}/${previewPath}`);
    }
    setPreviewImage(previewB, placeholderB, pathB);
  } else {
    setPreviewImage(previewB, placeholderB, "");
  }
}

function loadDebug(pathA, pathB) {
  const requestId = ++debugRequestId;
  if (!compareBundle) {
    const singlePath = pathA ? "../" + pathA : null;
    const loadA = singlePath ? fetch(singlePath).then((resp) => resp.json()).catch(() => null) : Promise.resolve(null);
    Promise.all([loadA]).then(([dataA]) => {
      if (requestId !== debugRequestId) return;
      debugCache = { a: dataA, b: null };
      debugPanelA.textContent = JSON.stringify(dataA || {}, null, 2);
      debugPanelB.textContent = "";
      renderStageSignals();
    });
    return;
  }
  const loadA = pathA
    ? fetch(toAbsolutePath(`${compareRunDirs.a}/${pathA}`))
        .then((resp) => resp.json())
        .catch(() => null)
    : Promise.resolve(null);
  const loadB = pathB
    ? fetch(toAbsolutePath(`${compareRunDirs.b}/${pathB}`))
        .then((resp) => resp.json())
        .catch(() => null)
    : Promise.resolve(null);
  Promise.all([loadA, loadB]).then(([dataA, dataB]) => {
    if (requestId !== debugRequestId) return;
    debugCache = { a: dataA, b: dataB };
    debugPanelA.textContent = JSON.stringify(dataA || {}, null, 2);
    debugPanelB.textContent = JSON.stringify(dataB || {}, null, 2);
    renderStageSignals();
  });
}

function toggleDebugPanel() {
  debugPanel.classList.toggle("hidden");
  toggleDebug.classList.toggle("active", !debugPanel.classList.contains("hidden"));
}

function toggleRoiView() {
  if (diagMode !== "preview") {
    return;
  }
  useRoi = !useRoi;
  toggleRoi.classList.toggle("active", useRoi);
  const stage = compareStages ? compareStages[activeIndex] : { a: manifest.stages[activeIndex] };
  updatePreview(stage.a, stage.b);
}

function handleKey(event) {
  const maxIndex = (compareStages || manifest.stages).length - 1;
  if (event.key === "ArrowRight") {
    setActiveStage(Math.min(activeIndex + 1, maxIndex));
  }
  if (event.key === "ArrowLeft") {
    setActiveStage(Math.max(activeIndex - 1, 0));
  }
}

function matchStages(primary, secondary) {
  const matches = [];
  const usedB = new Set();

  primary.forEach((stageA) => {
    let stageB = null;
    const idx = stageA.index;
    if (idx != null && idx < secondary.length && secondary[idx].index === idx) {
      if (secondary[idx].name === stageA.name) {
        stageB = secondary[idx];
      }
    }
    if (!stageB) {
      const byName = secondary.find((s) => s.name === stageA.name && !usedB.has(s));
      if (byName) {
        stageB = byName;
      }
    }
    if (stageB) {
      usedB.add(stageB);
    }
    matches.push({ a: stageA, b: stageB });
  });

  secondary.forEach((stageB) => {
    if (!usedB.has(stageB)) {
      matches.push({ a: null, b: stageB });
    }
  });
  return matches;
}

function updateTiming(stageA, stageB) {
  if (!compareBundle) {
    timingA.textContent = stageA && stageA.timing_ms ? `Timing: ${stageA.timing_ms.toFixed(2)} ms` : "";
    timingB.textContent = "";
    timingDelta.textContent = "";
    return;
  }
  const tA = stageA && typeof stageA.timing_ms === "number" ? stageA.timing_ms : null;
  const tB = stageB && typeof stageB.timing_ms === "number" ? stageB.timing_ms : null;
  timingA.textContent = tA != null ? `Timing A: ${tA.toFixed(2)} ms` : "Timing A: N/A";
  timingB.textContent = tB != null ? `Timing B: ${tB.toFixed(2)} ms` : "Timing B: N/A";
  if (tA != null && tB != null) {
    const delta = tB - tA;
    timingDelta.textContent = `Delta (B-A): ${delta.toFixed(2)} ms`;
  } else {
    timingDelta.textContent = "Delta (B-A): N/A";
  }
}

function stageDirFromArtifacts(artifacts) {
  if (!artifacts) return null;
  const path = artifacts.preview || artifacts.debug || artifacts.timing_ms;
  if (!path) return null;
  const parts = path.split("/");
  parts.pop();
  return parts.join("/");
}

function buildStagePath(runDir, stageDir, file) {
  if (!stageDir) return null;
  if (!runDir) {
    return `../${stageDir}/${file}`;
  }
  return toAbsolutePath(`${runDir}/${stageDir}/${file}`);
}

function resolveDiagPath(runDir, stageDir, key) {
  const info = diagAvailability[key];
  if (info) {
    const hasDiag = runDir ? info.b : info.a;
    if (hasDiag) {
      return buildStagePath(runDir, stageDir, `extra/diagnostics/${key}.png`);
    }
  }
  return buildStagePath(runDir, stageDir, `extra/${key}.png`);
}

function formatMetricValue(value) {
  if (typeof value !== "number") return String(value);
  if (!isFinite(value)) return String(value);
  return value.toFixed(4);
}

function getNestedValue(obj, path) {
  if (!obj || !path) return undefined;
  return path.split(".").reduce((acc, key) => {
    if (acc && Object.prototype.hasOwnProperty.call(acc, key)) {
      return acc[key];
    }
    return undefined;
  }, obj);
}

function hasDisplayValue(value) {
  return value !== undefined && value !== null;
}

function isFiniteNumber(value) {
  return typeof value === "number" && isFinite(value);
}

function formatSignalValue(value) {
  if (!hasDisplayValue(value)) return "N/A";
  if (isFiniteNumber(value)) return formatMetricValue(value);
  if (Array.isArray(value)) {
    return (
      "[" +
      value
        .map((item) => (isFiniteNumber(item) ? formatMetricValue(item) : String(item)))
        .join(", ") +
      "]"
    );
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function signalSourceObject(source, side) {
  if (source === "metrics") return side === "b" ? metricsCache.b : metricsCache.a;
  if (source === "debug") return side === "b" ? debugCache.b : debugCache.a;
  if (source === "diff") return diffMetricsCache;
  return null;
}

function renderMetricsTable(container, data, keys, emptyLabel) {
  if (!data || !keys || keys.length === 0) {
    container.textContent = `${emptyLabel}: N/A`;
    return;
  }
  const rows = keys
    .map((key) => {
      const value = Object.prototype.hasOwnProperty.call(data, key) ? formatMetricValue(data[key]) : "N/A";
      return `<tr><td class="metric-key">${key}</td><td class="metric-value">${value}</td></tr>`;
    })
    .join("");
  container.innerHTML = `<table><tbody>${rows}</tbody></table>`;
}

function selectMetricKeys(dataA, dataB) {
  const allKeys = Array.from(new Set([...Object.keys(dataA || {}), ...Object.keys(dataB || {})]));
  if (showAllMetrics) {
    return allKeys.sort();
  }
  const subset = METRICS_DEFAULT_KEYS.filter((key) => allKeys.includes(key));
  return subset;
}

function extractNumericMetrics(obj) {
  if (!obj) return {};
  const out = {};
  Object.entries(obj).forEach(([key, value]) => {
    if (typeof value === "number") {
      out[key] = value;
    }
  });
  return out;
}

function loadJsonFirst(paths) {
  return paths.reduce((promise, path) => {
    return promise.catch(() => loadJsonSafe(path));
  }, Promise.reject());
}

function updateMetrics(stageA, stageB) {
  const requestId = ++metricsRequestId;
  const stageDirA = stageA ? stageDirFromArtifacts(stageA.artifacts) : null;
  const stageDirB = stageB ? stageDirFromArtifacts(stageB.artifacts) : null;
  const metricsPathsA = stageDirA
    ? [
        buildStagePath(null, stageDirA, "extra/metrics.json"),
        buildStagePath(null, stageDirA, "metrics.json"),
      ]
    : [];
  const diffStageDir = stageDirA || stageDirB;
  const diffRunDir = stageDirA ? null : compareRunDirs ? compareRunDirs.b : null;
  const diffPathsA = diffStageDir
    ? [
        buildStagePath(diffRunDir, diffStageDir, "extra/diff_metrics.json"),
        buildStagePath(diffRunDir, diffStageDir, "diff_metrics.json"),
      ]
    : [];
  const metricsPathsB = stageDirB
    ? [
        buildStagePath(compareRunDirs ? compareRunDirs.b : null, stageDirB, "extra/metrics.json"),
        buildStagePath(compareRunDirs ? compareRunDirs.b : null, stageDirB, "metrics.json"),
      ]
    : [];

  const metricsA = metricsPathsA.length ? loadJsonFirst(metricsPathsA) : Promise.resolve(null);
  const metricsB = compareBundle && metricsPathsB.length ? loadJsonFirst(metricsPathsB) : Promise.resolve(null);
  const diffA = diffPathsA.length ? loadJsonFirst(diffPathsA) : Promise.resolve(null);

  Promise.all([metricsA.catch(() => null), metricsB.catch(() => null), diffA.catch(() => null)]).then(
    ([dataA, dataB, diffData]) => {
      if (requestId !== metricsRequestId) return;
      const metricsAData = extractNumericMetrics(dataA);
      const metricsBData = extractNumericMetrics(dataB);
      metricsCache = { a: metricsAData, b: metricsBData };
      const diffMetrics = extractNumericMetrics(diffData);
      diffMetricsCache = diffMetrics;
      const diffKeys = Object.keys(diffMetrics).sort();
      updateMetricsPanelVisibility(metricsAData, metricsBData, diffKeys);
      renderMetricsPanels();
      renderStageSignals();
      if (diffKeys.length === 0) {
        diffMetricsSection.classList.add("hidden");
      } else {
        diffMetricsSection.classList.remove("hidden");
        renderMetricsTable(diffMetricsPanel, diffMetrics, diffKeys, "Diff metrics");
      }
    }
  );
}

function updateMetricsPanelVisibility(metricsAData, metricsBData, diffKeys) {
  const hasA = Object.keys(metricsAData || {}).length > 0;
  const hasB = Object.keys(metricsBData || {}).length > 0;
  const hasDiff = (diffKeys || []).length > 0;
  const showPanel = compareBundle ? hasA || hasB || hasDiff : hasA;
  const metricsPanel = document.getElementById("metrics-panel");
  if (!metricsPanel) return;
  metricsPanel.classList.toggle("hidden", !showPanel);
}

function renderMetricsPanels() {
  const metricsAData = metricsCache.a || {};
  const metricsBData = metricsCache.b || {};
  const keys = compareBundle ? selectMetricKeys(metricsAData, metricsBData) : selectMetricKeys(metricsAData, {});
  renderMetricsTable(metricsPanelA, metricsAData, keys, "Metrics");
  if (compareBundle) {
    renderMetricsTable(metricsPanelB, metricsBData, keys, "Metrics");
  } else {
    metricsPanelB.textContent = "";
  }
  metricsToggle.textContent = showAllMetrics ? "Show subset" : "Show all";
  metricsToggle.classList.toggle("active", showAllMetrics);
}

function setSignalsLoadingState() {
  if (!signalsTable) return;
  signalsTable.textContent = "Signals: loading...";
}

function buildSignalRows() {
  const stageSpecs = SIGNAL_SPECS[activeStageName] || [];
  const specs = [...SIGNAL_SPECS.__common, ...stageSpecs];
  if (!compareBundle) {
    const withDiff =
      diffMetricsCache && Object.keys(diffMetricsCache).length > 0 ? [...specs, ...SIGNAL_SPECS.__single_diff] : specs;
    return withDiff
      .map((spec) => {
        const value = getNestedValue(signalSourceObject(spec.source, "a"), spec.path);
        return { label: spec.label, value };
      })
      .filter((row) => hasDisplayValue(row.value));
  }

  return specs
    .map((spec) => {
      const sourceA = signalSourceObject(spec.source, "a");
      const sourceB = signalSourceObject(spec.source, "b");
      const valueA = getNestedValue(sourceA, spec.path);
      const valueB = getNestedValue(sourceB, spec.path);
      let delta = null;
      if (isFiniteNumber(valueA) && isFiniteNumber(valueB)) {
        delta = valueB - valueA;
      }
      return { label: spec.label, valueA, valueB, delta };
    })
    .filter((row) => hasDisplayValue(row.valueA) || hasDisplayValue(row.valueB));
}

function renderSignalsSingle(rows) {
  if (rows.length === 0) {
    signalsTable.textContent = "Signals: N/A";
    return;
  }
  const body = rows
    .map(
      (row) =>
        `<tr><td class="signal-key">${row.label}</td><td class="signal-value">${formatSignalValue(row.value)}</td></tr>`
    )
    .join("");
  signalsTable.innerHTML = `<table class="signals-single"><tbody>${body}</tbody></table>`;
}

function renderSignalsCompare(rows) {
  if (rows.length === 0) {
    signalsTable.textContent = "Signals: N/A";
    return;
  }
  const header =
    "<thead><tr><th class=\"signal-key\">signal</th><th class=\"signal-value\">A</th><th class=\"signal-value\">B</th><th class=\"signal-value\">Δ(B-A)</th></tr></thead>";
  const body = rows
    .map((row) => {
      const deltaText = hasDisplayValue(row.delta)
        ? `${row.delta >= 0 ? "+" : ""}${formatMetricValue(row.delta)}`
        : "N/A";
      return `<tr><td class="signal-key">${row.label}</td><td class="signal-value">${formatSignalValue(row.valueA)}</td><td class="signal-value">${formatSignalValue(row.valueB)}</td><td class="signal-value">${deltaText}</td></tr>`;
    })
    .join("");
  signalsTable.innerHTML = `<table class="signals-compare">${header}<tbody>${body}</tbody></table>`;
}

function renderStageSignals() {
  if (!signalsPanel || !signalsTable) return;
  const rows = buildSignalRows();
  if (compareBundle) {
    renderSignalsCompare(rows);
  } else {
    renderSignalsSingle(rows);
  }
}

function checkImage(path) {
  return fetch(path, { method: "HEAD" })
    .then((resp) => resp.ok)
    .catch(() => false);
}

function updateDiagnosticsAvailability(stageA, stageB) {
  const requestId = ++diagRequestId;
  const diagKeys = ["false_color", "zipper", "halo"];
  const stageDirA = stageA ? stageDirFromArtifacts(stageA.artifacts) : null;
  const stageDirB = stageB ? stageDirFromArtifacts(stageB.artifacts) : null;

  const checks = diagKeys.map((key) => {
    const pathA = stageDirA ? buildStagePath(null, stageDirA, `extra/diagnostics/${key}.png`) : null;
    const pathAFallback = stageDirA ? buildStagePath(null, stageDirA, `extra/${key}.png`) : null;
    const pathB =
      compareBundle && stageDirB ? buildStagePath(compareRunDirs.b, stageDirB, `extra/diagnostics/${key}.png`) : null;
    const pathBFallback =
      compareBundle && stageDirB ? buildStagePath(compareRunDirs.b, stageDirB, `extra/${key}.png`) : null;
    const checkA = pathA
      ? checkImage(pathA).then((ok) => (ok ? true : pathAFallback ? checkImage(pathAFallback) : false))
      : Promise.resolve(false);
    const checkB = pathB
      ? checkImage(pathB).then((ok) => (ok ? true : pathBFallback ? checkImage(pathBFallback) : false))
      : Promise.resolve(false);
    return Promise.all([checkA, checkB]).then(([a, b]) => ({ key, a, b }));
  });

  Promise.all(checks).then((results) => {
    if (requestId !== diagRequestId) return;
    const availability = {};
    results.forEach((result) => {
      availability[result.key] = result;
    });
    diagAvailability = availability;
    updateDiagnosticsButtons(availability);
    updateDiagnosticsStatus(availability);
    updateRoiToggle(stageA, stageB);
    updatePreview(stageA, stageB);
  });
}

function updateDiagnosticsButtons(availability) {
  const buttons = diagnosticsControls.querySelectorAll("button[data-diag]");
  let anyAvailable = false;
  buttons.forEach((button) => {
    const mode = button.dataset.diag;
    if (mode === "preview") {
      button.disabled = false;
      return;
    }
    const info = availability[mode];
    const available = compareBundle ? info && (info.a || info.b) : info && info.a;
    button.disabled = !available;
    if (available) {
      anyAvailable = true;
    }
  });

  if (!anyAvailable) {
    diagMode = "preview";
  }
  const activeButton = diagnosticsControls.querySelector(`button[data-diag="${diagMode}"]`);
  if (activeButton && activeButton.disabled) {
    diagMode = "preview";
  }
  updateDiagnosticsButtonsState();
}

function updateDiagnosticsButtonsState() {
  diagnosticsControls.querySelectorAll("button[data-diag]").forEach((button) => {
    const mode = button.dataset.diag;
    if (mode === diagMode) {
      button.classList.add("active");
    } else {
      button.classList.remove("active");
    }
  });
}

function updateDiagnosticsStatus(availability) {
  const diagKeys = ["false_color", "zipper", "halo"];
  const anyAvailable = diagKeys.some((key) => {
    const info = availability[key];
    return compareBundle ? info && (info.a || info.b) : info && info.a;
  });
  diagnosticsPanel.classList.toggle("hidden", !anyAvailable);
  if (!anyAvailable) {
    diagnosticsStatus.textContent = "Diagnostics: N/A";
    return;
  }
  if (diagMode === "preview") {
    diagnosticsStatus.textContent = "Diagnostics: Preview";
    return;
  }
  const activeButton = diagnosticsControls.querySelector(`button[data-diag="${diagMode}"]`);
  const label = activeButton ? activeButton.textContent.trim() : diagMode.replace("_", " ");
  diagnosticsStatus.textContent = `Diagnostics: ${label}`;
}

function updateRoiToggle(stageA, stageB) {
  const roiPathA = stageA && stageA.artifacts && stageA.artifacts.roi;
  const roiPathB = stageB && stageB.artifacts && stageB.artifacts.roi;
  const hasRoi = !!(roiPathA || roiPathB);
  toggleRoi.disabled = !hasRoi || diagMode !== "preview";
  if (!hasRoi || diagMode !== "preview") {
    useRoi = false;
  }
  toggleRoi.classList.toggle("active", useRoi);
}

diagnosticsControls.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-diag]");
  if (!button || button.disabled) return;
  diagMode = button.dataset.diag;
  updateDiagnosticsButtonsState();
  updateDiagnosticsStatus(diagAvailability);
  const stage = compareStages ? compareStages[activeIndex] : { a: manifest.stages[activeIndex] };
  updateRoiToggle(stage.a, stage.b);
  updatePreview(stage.a, stage.b);
});

metricsToggle.addEventListener("click", () => {
  showAllMetrics = !showAllMetrics;
  renderMetricsPanels();
});

loadManifests().then(() => {
  if (compareBundle) {
    compareStages = matchStages(manifest.stages, manifestB.stages);
  }
  renderStageList();
  setActiveStage(0);
});

toggleRoi.addEventListener("click", toggleRoiView);
toggleDebug.addEventListener("click", toggleDebugPanel);
document.addEventListener("keydown", handleKey);
