# Config

mini-ISP is configured with YAML files. The config is intentionally simple in v0.1:
- select a `pipeline_mode`
- choose per-stage methods/parameters
- define dumping/metrics/viewer behavior

This doc specifies the **schema**, **defaults**, and **where values come from**.

---

## 1) Top-level schema (v0.1)

```yaml
# configs/default.yaml (example shape)

pipeline_mode: classic            # classic | jdd | drc_plus

input:
  path: data/sample.dng
  bayer_pattern: RGGB            # optional; overrides RAW metadata/autodetect for CFA pattern
  bit_depth: 12                  # optional
  black_level: 64                # optional
  white_level: 4095              # optional

output:
  dir: runs
  name: run_001                  # optional; else timestamp
  save_final: true
  format: png                    # png | jpg (v0.1)
  bit_depth: 8                   # 8 (v0.1 default; extend later)

dump:
  enable: true
  preview_max_side: 1024
  roi:
    enable: true
    xywh: [0.35, 0.35, 0.30, 0.30]   # normalized coords (x,y,w,h)
  stages: all                     # all | [list of stage names]
  # example:
  # stages: [raw_norm, demosaic, tone]

metrics:
  enable: true
  timing: true
  histograms: true
  deltas: false                   # enable stage-to-stage comparisons later

skin_mask:
  enable: false
  method: heuristic               # heuristic | seg_model (future)
  dump: true

stages:
  raw_norm: { ... }
  dpc: { ... }
  lsc: { ... }
  wb_gains: { ... }
  demosaic: { ... }
  denoise: { ... }
  ccm: { ... }
  stats_3a: { ... }
  tone: { ... }
  color_adjust: { ... }
  sharpen: { ... }
  oetf_encode: { ... }

viewer:
  enable: true
  title: mini-ISP run