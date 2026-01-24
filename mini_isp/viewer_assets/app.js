const stageList = document.getElementById("stage-list");
const preview = document.getElementById("preview");
const stageName = document.getElementById("stage-name");
const debugPanel = document.getElementById("debug");
const timing = document.getElementById("timing");
const toggleRoi = document.getElementById("toggle-roi");
const toggleDebug = document.getElementById("toggle-debug");
const runTitle = document.getElementById("run-title");

let manifest = null;
let activeIndex = 0;
let useRoi = false;

function loadManifest() {
  return fetch("../manifest.json")
    .then((resp) => resp.json())
    .then((data) => {
      manifest = data;
    });
}

function renderStageList() {
  stageList.innerHTML = "";
  manifest.stages.forEach((stage, idx) => {
    const li = document.createElement("li");
    li.textContent = stage.display_name || stage.name;
    li.addEventListener("click", () => setActiveStage(idx));
    if (idx === activeIndex) {
      li.classList.add("active");
    }
    stageList.appendChild(li);
  });
  runTitle.textContent = manifest.title || manifest.run_id;
}

function setActiveStage(index) {
  activeIndex = index;
  renderStageList();
  const stage = manifest.stages[index];
  stageName.textContent = stage.display_name || stage.name;
  timing.textContent = stage.timing_ms ? `Timing: ${stage.timing_ms.toFixed(2)} ms` : "";
  const roiPath = stage.artifacts && stage.artifacts.roi;
  toggleRoi.disabled = !roiPath;
  useRoi = false;
  updatePreview();
  loadDebug(stage.artifacts.debug);
}

function updatePreview() {
  const stage = manifest.stages[activeIndex];
  const path = useRoi && stage.artifacts.roi ? stage.artifacts.roi : stage.artifacts.preview;
  preview.src = "../" + path;
}

function loadDebug(path) {
  fetch("../" + path)
    .then((resp) => resp.json())
    .then((data) => {
      debugPanel.textContent = JSON.stringify(data, null, 2);
    })
    .catch(() => {
      debugPanel.textContent = "{}";
    });
}

function toggleDebugPanel() {
  debugPanel.classList.toggle("hidden");
}

function toggleRoiView() {
  useRoi = !useRoi;
  updatePreview();
}

function handleKey(event) {
  if (event.key === "ArrowRight") {
    setActiveStage(Math.min(activeIndex + 1, manifest.stages.length - 1));
  }
  if (event.key === "ArrowLeft") {
    setActiveStage(Math.max(activeIndex - 1, 0));
  }
}

loadManifest().then(() => {
  renderStageList();
  setActiveStage(0);
});

toggleRoi.addEventListener("click", toggleRoiView);
toggleDebug.addEventListener("click", toggleDebugPanel);
document.addEventListener("keydown", handleKey);
