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
const diagnosticsControls = document.getElementById("diagnostics-controls");
const diagnosticsStatus = document.getElementById("diagnostics-status");

let manifest = null;
let manifestB = null;
let compareBundle = null;
let compareStages = null;
let compareRunDirs = null;
let activeIndex = 0;
let useRoi = false;
let diagMode = "preview";
let metricsRequestId = 0;
let diagRequestId = 0;
let diagAvailability = {};

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
    runTitle.textContent = compareBundle.title || "Compare runs";
    labelA.textContent = compareBundle.a.label || "A";
    labelB.textContent = compareBundle.b.label || "B";
    debugLabelA.textContent = labelA.textContent;
    debugLabelB.textContent = labelB.textContent;
    metricsLabelA.textContent = labelA.textContent;
    metricsLabelB.textContent = labelB.textContent;
    document.body.classList.remove("single-run");
  } else {
    runTitle.textContent = manifest.title || manifest.run_id;
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
  if (!compareBundle) {
    fetch("../" + pathA)
      .then((resp) => resp.json())
      .then((data) => {
        debugPanelA.textContent = JSON.stringify(data, null, 2);
        debugPanelB.textContent = "";
      })
      .catch(() => {
        debugPanelA.textContent = "{}";
      });
    return;
  }
  if (pathA) {
    fetch(toAbsolutePath(`${compareRunDirs.a}/${pathA}`))
      .then((resp) => resp.json())
      .then((data) => {
        debugPanelA.textContent = JSON.stringify(data, null, 2);
      })
      .catch(() => {
        debugPanelA.textContent = "{}";
      });
  } else {
    debugPanelA.textContent = "{}";
  }
  if (pathB) {
    fetch(toAbsolutePath(`${compareRunDirs.b}/${pathB}`))
      .then((resp) => resp.json())
      .then((data) => {
        debugPanelB.textContent = JSON.stringify(data, null, 2);
      })
      .catch(() => {
        debugPanelB.textContent = "{}";
      });
  } else {
    debugPanelB.textContent = "{}";
  }
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
      const keys = compareBundle
        ? Array.from(new Set([...Object.keys(metricsAData), ...Object.keys(metricsBData)])).sort()
        : Object.keys(metricsAData).sort();
      renderMetricsTable(metricsPanelA, metricsAData, keys, "Metrics");
      if (compareBundle) {
        renderMetricsTable(metricsPanelB, metricsBData, keys, "Metrics");
      } else {
        metricsPanelB.textContent = "";
      }
      const diffMetrics = extractNumericMetrics(diffData);
      const diffKeys = Object.keys(diffMetrics).sort();
      renderMetricsTable(diffMetricsPanel, diffMetrics, diffKeys, "Diff metrics");
    }
  );
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
  if (!anyAvailable) {
    diagnosticsStatus.textContent = "Diagnostics: N/A";
    return;
  }
  if (diagMode === "preview") {
    diagnosticsStatus.textContent = "Diagnostics: Preview";
    return;
  }
  const label = diagMode.replace("_", " ");
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
