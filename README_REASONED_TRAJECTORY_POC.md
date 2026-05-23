# Reasoned Trajectory Program POC for openpilot

This repository is a local-PC proof of concept for adding a VLM-conditioned
trajectory compiler upstream of openpilot controls. It is not a production
driving stack, not validated for road use, and not intended to bypass panda,
opendbc, car safety hooks, driver monitoring, or the existing openpilot vehicle
interface.

The POC implements the "pseudo-Alpamayo" architecture: instead of trying to run
a full trained action decoder, the system asks a small VLM to identify bounded
scene constraints from a driver-UI-style scene board. Those constraints are
compiled into a deterministic trajectory program, then into path/curvature and
speed constraints that remain inside the existing planner/control boundary.

The current working demo runs on this PC in MetaDrive with a Qwen2.5-VL-3B
backend on the local NVIDIA GPU. The original target direction remains an eGPU
path, but the checked-in code currently uses the local PC GPU backend for the
working demo.

## Current Status

Working:

- Openpilot master fork with a new `reasoned_plannerd` process.
- New `reasonedTrajectoryPlan` cereal event for audit/debug telemetry.
- Strict `RTPv1` parser with bounded fields and grammar validation.
- Deterministic PathSynth compiler for lateral bias, avoid zones, speed
  modifiers, stop/yield constraints, and candidate selection.
- Qwen2.5-VL-3B label worker that sees a full 384 px driver-UI-style overlay
  scene board.
- Optimized async loop that keeps the sim/control path nonblocking and tracks
  source-frame age.
- MetaDrive closed-loop demo comparing stock path following versus the VLM
  reasoned trajectory path.
- Video generation for stock, VLM, padded stock, and side-by-side comparisons.
- Unit tests for RTP parsing, PathSynth, relative speed caps, durable speed
  replacement, construction side detection, simulator sign conversion, and
  contradictory durable lateral plan replacement.

Known limitations:

- Qwen2.5-VL-3B via PyTorch/CUDA is too slow for synchronous 20 Hz model-loop
  use on this PC. The working path is bounded async.
- The current Qwen backend is CUDA/NVIDIA. It is not yet a tinygrad AMD eGPU
  backend.
- The measured Qwen VRAM footprint is about 7.4 GiB after model load and about
  7.7 GiB after one full384 score inference. A clean 8 GiB eGPU may be possible
  only with very tight memory discipline, quantization, or a smaller model. It
  is not a comfortable target for this exact fp16 Qwen path.
- Model weights and run artifacts are intentionally ignored by git. They must
  be downloaded or regenerated locally.
- The MetaDrive demo proves control influence and closed-loop behavior in sim.
  It does not prove real-world safety.

## Architecture

At a high level:

```text
camera frame + vehicle state
        |
        v
driver-UI-style scene board
        |
        v
Qwen VLM label scorer
        |
        v
strict RTPv1 program
        |
        v
PathSynth deterministic compiler
        |
        v
bounded lateral/speed plan
        |
        v
lateralManeuverPlan / controlsd path
```

The VLM does not command steering, torque, throttle, brake, CAN, or panda. It
only emits bounded semantic constraints. The compiler turns those constraints
into path candidates, speed caps, and avoid-zone costs.

The important boundary is:

- VLM output is text and can be invalid.
- Parser validates the text.
- Compiler clamps all geometry and speed changes.
- Controls consume only the compiled plan.

## Repository Changes

Core runtime:

- `selfdrive/controls/reasoned/rtp.py`
  - Strict RTP parser.
  - Bounded `scene`, `evidence`, `meta`, `branch`, `lat_bias_m`,
    `speed_cap_mps`, `stop_s`, `avoid`, `weights`, and `confidence`.
  - `speed_cap_mps` now accepts absolute values, `none`, percentage strings
    such as `25%`, and scale strings such as `0.25x`.

- `selfdrive/controls/reasoned/pathsynth.py`
  - Base plan abstraction.
  - Candidate path generation.
  - Deterministic trajectory compiler.
  - Relative speed scaling against desired/base speed.
  - Curvature clipping and bounded lateral offset handling.

- `selfdrive/controls/reasoned/planner.py`
  - Planner orchestration.
  - Scene-board rendering.
  - VLM backend call.
  - Same-frame synchronous mode and bounded async mode.
  - RTP validation and PathSynth timing.

- `selfdrive/controls/reasoned/vlm.py`
  - Static RTP backend for non-VLM tests.
  - External subprocess backend for GPU VLM workers.
  - Async worker mode with latest-frame behavior and source-frame age metadata.

- `selfdrive/controls/reasoned/ui_scene_board.py`
  - Full-frame, driver-UI-style scene board used by the VLM.
  - Includes camera frame, path overlay, HUD state, and visual affordances.

- `selfdrive/controls/reasoned_plannerd.py`
  - PC-only process that consumes `modelV2` and `carState`, runs the reasoned
    planner, publishes `lateralManeuverPlan`, and publishes audit telemetry.

Openpilot integration:

- `cereal/custom.capnp`
  - Defines `ReasonedTrajectoryPlan`.

- `cereal/log.capnp`
  - Wires the custom event at the reserved custom slot.

- `cereal/services.py`
  - Adds `reasonedTrajectoryPlan` service.

- `system/manager/process_config.py`
  - Adds PC-only `reasoned_plannerd`, gated by `ReasonedPlanner` param or
    `ENABLE_REASONED_PLANNER=1`.

- `selfdrive/controls/controlsd.py`
  - Uses `lateralManeuverPlan` mono time when a valid lateral maneuver plan is
    present, otherwise falls back to the model timing path.

POC tools:

- `tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py`
  - Current default VLM worker.
  - Uses full384 scene boards.
  - Uses batched yes/no label scoring instead of free-form RTP generation.
  - Rotates label groups to reduce latency.
  - Keeps durable labels through short occlusion/splash, with negative-score
    clearing.
  - Emits deterministic strict RTP from labels.

- `tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py`
  - Closed-loop MetaDrive demo runner.
  - Runs stock and VLM episodes.
  - Spawns random mixed construction and pedestrian scenes.
  - Applies durable lateral and speed plans.
  - Tracks collisions, latency, RTP age, path deltas, speed deltas, and saved
    input frames.

- `tools/reasoned_trajectory_poc/render_demo_videos.py`
  - Builds stock, padded-stock, VLM, and side-by-side MP4s from saved PNG
    frames.

- `tools/reasoned_trajectory_poc/run_local_demo.py`
  - Hardware-free parser/compiler demo using static RTP.

- `tools/reasoned_trajectory_poc/benchmark_vlm_backend.py`
  - Backend timing harness.

- `tools/reasoned_trajectory_poc/probe_qwen_novel_scenes.py`
  - One-frame Qwen scene probing in MetaDrive.

- `tools/reasoned_trajectory_poc/diagnose_qwen_scene_perception.py`
  - Diagnostic prompt/label experiments.

- `tools/reasoned_trajectory_poc/*smolvlm*`, `tools/reasoned_trajectory_poc/nanovlm_worker.py`
  - Earlier backend experiments retained for reference.

Tests:

- `selfdrive/controls/tests/test_reasoned_trajectory.py`
  - Main focused unit test suite for this POC.

## RTPv1 Program Format

The VLM path ultimately compiles to this fixed text shape:

```text
RTPv1
scene=construction_right
evidence=[cones_barrier_right_edge]
meta=BIAS_LEFT_AND_SLOW
branch=base
lat_bias_m=1.25
speed_cap_mps=25%
stop_s=none
avoid=[right_edge_s8_48_margin1.25]
weights=[obs2.5,lane1.4,comfort1.0,base0.7,vlm1.0]
confidence=0.72
```

Important fields:

- `scene`: compact scene class.
- `evidence`: visible evidence tokens.
- `meta`: maneuver prior.
- `branch`: candidate branch hint.
- `lat_bias_m`: openpilot/PathSynth lateral bias. Positive means left in the
  openpilot compiler convention.
- `speed_cap_mps`: legacy name, now supports:
  - `none`
  - absolute m/s, for compatibility
  - percent, for example `25%`
  - scale, for example `0.25x`
- `stop_s`: stop/yield distance, or `none`.
- `avoid`: bounded avoid-zone tokens such as `right_edge_s8_48_margin1.25`.
- `weights`: bounded optimizer weights.
- `confidence`: `0.0` to `1.0`.

The current Qwen label compiler emits percentages for normal slowdowns:

- Construction: `speed_cap_mps=25%`
- Mixed agent/construction yield: `speed_cap_mps=15%`
- Stop: `speed_cap_mps=0.0`

The parser stores percentage/scale values as `speed_scale`, and PathSynth
resolves them against the current desired speed. A `25%` cap means:

```text
desired 8 m/s  -> 2.0 m/s cap
desired 10 m/s -> 2.5 m/s cap
desired 12 m/s -> 3.0 m/s cap
desired 20 m/s -> 5.0 m/s cap
```

## Construction Side and Simulator Sign Convention

A previous bug made the sim appear to steer into cones. The root cause was two
separate issues:

1. Generic construction labels collapsed into a right-edge avoid program.
2. MetaDrive lane lateral sign in the harness is opposite the openpilot/PathSynth
   sign convention.

The current path fixes this by:

- Scoring `construction_left` and `construction_right` relative to the green
  planned path.
- Emitting side-specific RTP:
  - Right-side construction -> `BIAS_LEFT_AND_SLOW`, positive openpilot
    `lat_bias_m`, `right_edge...`
  - Left-side construction -> `BIAS_RIGHT_AND_SLOW`, negative openpilot
    `lat_bias_m`, `left_edge...`
- Converting openpilot lateral sign at the MetaDrive boundary.
- Clearing active durable lateral plans that pull the opposite direction when a
  new signed plan arrives with sufficient confidence.

This is not a "cones always mean left" shortcut. The VLM has to identify the
construction side relative to the path, and the compiler/sign conversion then
does the deterministic movement.

## VLM Backend

The current working backend is:

```text
tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py
```

Defaults:

```text
--image-mode full
--label-mode score
--score-rotate-groups
--score-cache-ttl-frames 60
--score-negative-clear-threshold 2.0
```

The worker loads model files from:

```text
models/vlm/qwen2_5_vl_3b_instruct
```

Those files are intentionally ignored by git. Download them locally before
running the Qwen backend. The expected model is:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Example download command:

```powershell
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models\vlm\qwen2_5_vl_3b_instruct
```

The current code uses PyTorch CUDA for Qwen. AMD/tinygrad eGPU support is a
future backend target, not the active working path.

## VRAM Requirement

Measured on this machine with Qwen2.5-VL-3B, full384 score mode:

```text
model load incremental VRAM:        ~7,364 MiB
after one full384 score inference:  ~7,666 MiB
PyTorch peak reserved:              ~7,522 MiB
model files on disk:                ~7.0 GB
```

Interpretation:

- 8 GB dedicated GPU: technically close, not reliable for this exact fp16 Qwen
  path.
- 8 GB desktop GPU with other applications sharing VRAM: not realistic.
- 12 GB GPU: realistic minimum for the current Qwen path.
- 16 GB GPU: comfortable, and matches the current working local setup.

For an 8 GB AMD eGPU target, the likely path is a smaller model, quantized model,
or a memory-frugal tinygrad/AMD implementation. The current CUDA worker does not
prove the 8 GB AMD path.

## Local Demo Commands

Run focused unit tests:

```powershell
py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
```

Run syntax checks for the main POC files:

```powershell
py -3.11 -m py_compile `
  selfdrive\controls\reasoned\rtp.py `
  selfdrive\controls\reasoned\pathsynth.py `
  selfdrive\controls\reasoned\planner.py `
  selfdrive\controls\reasoned\vlm.py `
  selfdrive\controls\reasoned_plannerd.py `
  tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py `
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py `
  tools\reasoned_trajectory_poc\render_demo_videos.py
```

Run a static local parser/compiler demo without the VLM:

```powershell
py -3.11 tools\reasoned_trajectory_poc\run_local_demo.py `
  --frames 3 `
  --scenario construction `
  --speed-mps 12 `
  --out artifacts\reasoned_trajectory_poc\local_static_demo
```

Run the current mixed MetaDrive VLM demo:

```powershell
$out = 'artifacts\reasoned_trajectory_poc\random_mixed_loop_vlm_relative_speed_600f'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:RTP_VLM_IMAGE_SIZE = '384'
$env:RTP_VLM_STDERR_PATH = "$out\qwen_worker_stderr.log"
$env:RTP_VLM_SERVER_COMMAND = 'py -3.11 tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py'

py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py `
  --frames 600 `
  --engine vlm `
  --async-vlm `
  --vlm-period-frames 1 `
  --vlm-max-age-frames 8 `
  --vlm-latest-only `
  --vlm-drop-stale-results `
  --vlm-max-result-age-frames 8 `
  --prewarm-seconds 30 `
  --deadline-ms 50 `
  --tick-sec 0.05 `
  --map O `
  --novel-scene random_mixed `
  --save-every 5 `
  --out $out
```

Render videos after a run:

```powershell
py -3.11 tools\reasoned_trajectory_poc\render_demo_videos.py `
  --run-dir artifacts\reasoned_trajectory_poc\random_mixed_loop_vlm_relative_speed_600f `
  --prefix random_mixed_relative_speed_600f
```

Expected videos:

```text
artifacts/.../videos/side_by_side_<prefix>.mp4
artifacts/.../videos/stock_<prefix>.mp4
artifacts/.../videos/stock_<prefix>_padded.mp4
artifacts/.../videos/vlm_<prefix>.mp4
```

## Latest Measured Demo

Latest relative-speed mixed demo:

```text
artifacts/reasoned_trajectory_poc/random_mixed_loop_vlm_relative_speed_600f
```

Summary:

```text
stock frames:              94
stock object-crash frames: 23
stock human-crash frames:  1

VLM frames:                600
VLM valid publishes:       381
VLM deadline misses:       0
VLM p99 planner overhead:  4.8246 ms
VLM mean path delta:       1.2336 m
VLM object-crash frames:   0
VLM human-crash frames:    0
VLM min object distance:   1.3529 m
```

RTP speed fields in that run:

```text
25%:  273 publishes
15%:  103 publishes
none: 5 publishes
```

Because this run uses a `10 m/s` desired speed, `25%` resolves to `2.5 m/s` and
`15%` resolves to `1.5 m/s`. That is close to the previous fixed-cap behavior
for this specific run, but it now scales correctly if the desired speed changes.

## Git Hygiene

Ignored local-only paths:

```text
artifacts/
models/
tools/nanoVLM/
tools/nanoVLM_v01/
__pycache__/
```

Do not push:

- Downloaded Hugging Face model weights.
- Generated PNG frame dumps.
- Generated MP4 videos.
- Benchmark artifacts.
- Nested third-party git clones used during experiments.

Push source and docs only:

- `GOAL.MD`
- `README_REASONED_TRAJECTORY_POC.md`
- `.gitignore`
- `cereal/custom.capnp`
- `cereal/log.capnp`
- `cereal/services.py`
- `selfdrive/controls/controlsd.py`
- `system/manager/process_config.py`
- `selfdrive/controls/reasoned/`
- `selfdrive/controls/reasoned_plannerd.py`
- `selfdrive/controls/tests/test_reasoned_trajectory.py`
- `tools/reasoned_trajectory_poc/`

## Suggested Push Workflow

Once the remote repository URL is known:

```powershell
git status --short
py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
py -3.11 -m py_compile selfdrive\controls\reasoned\*.py selfdrive\controls\reasoned_plannerd.py tools\reasoned_trajectory_poc\*.py

git add .gitignore GOAL.MD README_REASONED_TRAJECTORY_POC.md
git add cereal\custom.capnp cereal\log.capnp cereal\services.py
git add selfdrive\controls\controlsd.py system\manager\process_config.py
git add selfdrive\controls\reasoned selfdrive\controls\reasoned_plannerd.py selfdrive\controls\tests\test_reasoned_trajectory.py
git add tools\reasoned_trajectory_poc

git commit -m "Add reasoned trajectory VLM POC"
git remote add origin <repo-url>
git push -u origin HEAD
```

If a remote already exists, replace the `git remote add origin` line with:

```powershell
git remote set-url origin <repo-url>
```

Before pushing, verify that `git status --short` does not list `artifacts/`,
`models/`, `tools/nanoVLM/`, or `tools/nanoVLM_v01/`.
