const stageList = document.getElementById("stage-list");
const previewA = document.getElementById("preview-a");
const previewB = document.getElementById("preview-b");
const stageName = document.getElementById("stage-name");
const debugPanelA = document.getElementById("debug-a");
const debugPanelB = document.getElementById("debug-b");
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

let manifest = null;
let manifestB = null;
let compareBundle = null;
let compareStages = null;
let compareRunDirs = null;
let activeIndex = 0;
let useRoi = false;

function getQueryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

function loadJson(path) {
  return fetch(path).then((resp) => resp.json());
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
    document.body.classList.remove("single-run");
  } else {
    runTitle.textContent = manifest.title || manifest.run_id;
    labelA.textContent = "Run";
    labelB.textContent = "";
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
  const roiPathA = stageA && stageA.artifacts && stageA.artifacts.roi;
  const roiPathB = stageB && stageB.artifacts && stageB.artifacts.roi;
  toggleRoi.disabled = !(roiPathA || roiPathB);
  useRoi = false;
  updatePreview(stageA, stageB);
  loadDebug(stageA ? stageA.artifacts.debug : null, stageB ? stageB.artifacts.debug : null);
}

function updatePreview(stageA, stageB) {
  if (!compareBundle) {
    const stage = stageA;
    const path = useRoi && stage.artifacts.roi ? stage.artifacts.roi : stage.artifacts.preview;
    previewA.src = "../" + path;
    previewB.src = "";
    placeholderA.classList.add("hidden");
    placeholderB.classList.add("hidden");
    return;
  }

  if (stageA) {
    const pathA = useRoi && stageA.artifacts.roi ? stageA.artifacts.roi : stageA.artifacts.preview;
    previewA.src = toAbsolutePath(`${compareRunDirs.a}/${pathA}`);
    placeholderA.classList.add("hidden");
  } else {
    previewA.src = "";
    placeholderA.classList.remove("hidden");
  }

  if (stageB) {
    const pathB = useRoi && stageB.artifacts.roi ? stageB.artifacts.roi : stageB.artifacts.preview;
    previewB.src = toAbsolutePath(`${compareRunDirs.b}/${pathB}`);
    placeholderB.classList.add("hidden");
  } else {
    previewB.src = "";
    placeholderB.classList.remove("hidden");
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
  debugPanelA.classList.toggle("hidden");
  debugPanelB.classList.toggle("hidden");
}

function toggleRoiView() {
  useRoi = !useRoi;
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
