# Qwen VLM 50 ms Progress

## Current Verified State

- Hardware: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, 16 GB VRAM.
- Driver: 596.36.
- CUDA toolkit installed for builds: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`.
- `nvcc` 13.2 verifies `compute_120`, `compute_121`, `sm_120`, and `sm_121`.
- User-level `CUDA_PATH` and `CUDA_HOME` now point to CUDA 13.2.
- User-level CUDA 12.6 `PATH` entries were removed; new shells should prefer CUDA 13.2. This already-running Codex process may still inherit stale process-level CUDA 12.6 variables, so benchmark commands keep an explicit CUDA 13.2 prefix.
- Runtime stack verified:
  - `torch 2.11.0.dev20260120+cu128`
  - `tensorrt 10.16.1.11`
  - TensorRT exposes `DataType.FP4` and `BuilderFlag.FP4`.
  - `nvidia-modelopt 0.44.0`
  - `onnx 1.20.1`
  - `onnx-graphsurgeon 0.6.1`
  - `onnxslim 0.1.94`
  - `polygraphy 0.49.26`

## Model Path Under Test

- Model: `Qwen2.5-VL-3B-Instruct`.
- Local model directory: `models/vlm/qwen2_5_vl_3b_instruct`.
- Image mode: `full`.
- Image size: `168`.
- Scoring mode: two label prompts, default proof set `construction_left,construction_right`.
- Shape used by the optimized path:
  - Vision input: `pixel_values=(96,1176)`, fixed Qwen image grid `[[1,8,12]]`.
  - Vision output: `image_features=(24,2048)`.
  - Text input: `inputs_embeds=(2,220,2048)`, `position_ids=(3,2,220)`.
  - Text output: selected yes/no logits for the two label prompts.

## Measurements

Latest measured hot path, using TensorRT FP16 static vision plus TensorRT NVFP4 text, after refactoring the benchmark around the reusable worker scorer:

```text
processor_ms  median 3.954   p90 4.066   max 4.125
trt_vision_ms median 5.637   p90 6.070   max 6.214
embed_ms      median 0.116   p90 0.133   max 0.178
scatter_ms    median 0.586   p90 0.613   max 0.651
rope_ms       median 2.208   p90 2.460   max 2.558
trt_text_ms   median 17.302  p90 17.419  max 17.574
total_ms      median 31.027  p90 31.601  max 32.165
```

This proves the fixed full168, two-label Qwen scoring path can run under the 50 ms model-loop budget on the RTX 5060 Ti.

Repeatable repo command added and verified:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --iters 30 --warmup 5 benchmark
```

Result from the repeatable command:

```text
processor_ms  median 4.012   p90 4.208   max 4.359
trt_vision_ms median 5.600   p90 5.924   max 6.057
embed_ms      median 0.113   p90 0.145   max 0.173
scatter_ms    median 0.591   p90 0.663   max 0.753
rope_ms       median 2.204   p90 2.474   max 3.085
trt_text_ms   median 17.185  p90 17.649  max 18.375
total_ms      median 29.803  p90 30.320  max 30.868
```

Persisted benchmark artifact from the same script:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --iters 20 --warmup 5 --out artifacts\reasoned_trajectory_poc\qwen_trt_label_benchmark.json benchmark
```

Persisted result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_label_benchmark.json
total_ms median 31.027  p90 31.601  max 32.165
```

Comparison points:

```text
PyTorch HF full168, two labels: about 97 to 116 ms hot, depending prompt shape.
PyTorch HF language portion only: about 72 to 73 ms for two labels.
TensorRT FP16 full 36-layer text engine: 35.7 ms median.
TensorRT NVFP4 full 36-layer text engine: 16.8 to 17.1 ms median.
PyTorch HF visual tower in end-to-end path: about 52 ms median.
TensorRT FP16 static visual engine: 5.6 ms median.
```

## CUDA 13.2 / compute_120 Update

Verified on 2026-05-23:

```text
Driver: 596.36
GPU: NVIDIA GeForce RTX 5060 Ti
CUDA toolkit selected: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
nvcc: release 13.2, V13.2.78
nvcc arch support: compute_120, compute_121
nvcc code support: sm_120, sm_121
torch: 2.11.0.dev20260120+cu128
torch GPU capability: (12, 0)
TensorRT: 10.16.1.11
TensorRT FP4 support: DataType.FP4=True, BuilderFlag.FP4=True
```

The Qwen TensorRT helper now force-selects CUDA 13.2 inside the process when that toolkit exists, instead of leaving stale inherited `CUDA_PATH=v12.6` in place.

Current CUDA 13.2 smoke benchmark:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --iters 6 --warmup 2 --out artifacts\reasoned_trajectory_poc\qwen_trt_cuda132_smoke.json benchmark-groups
```

Result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_cuda132_smoke.json
total_ms median 30.914  p90 31.528  max 31.608
trt_text_ms median 17.124  p90 17.461  max 17.669
trt_vision_ms median 5.615  p90 5.644  max 5.940
```

## Synchronous Sim Timing With TensorRT Worker

The optimized TensorRT worker now runs inside the MetaDrive overlay demo path, not only as a standalone benchmark. The worker supports an optional `--ready-jsonl` marker after loading and warmup; `PersistentRtpEngine` consumes it when `RTP_VLM_WAIT_READY=1`. This keeps one-time model load and engine warmup out of the recorded model-loop frames without burning a fake rotating-label inference.

Transport issue found and fixed before this run:

```text
PNG scene-board payloads made planner wall time about 60 ms even though Qwen inference stages summed to about 33 ms.
RTP_VLM_IMAGE_FORMAT=jpeg with quality 85 reduces scene-board serialization and pipe overhead enough for synchronous 20 Hz use.
```

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:RTP_VLM_IMAGE_FORMAT='jpeg'
$env:RTP_VLM_JPEG_QUALITY='85'
$env:RTP_VLM_WAIT_READY='1'
$env:RTP_VLM_STDERR_PATH='E:\ture_opamayo\openpilot\artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_30\vlm_stderr.log'
$env:RTP_VLM_SERVER_COMMAND='py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-groups --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --warmup 1 --ready-jsonl serve'
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --frames 30 --speed-mps 5 --engine vlm --novel-scene construction --deadline-ms 50 --tick-sec 0 --save-every 10 --out artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_30
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_30\comparison_vlm.json
artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_30\vlm\episode.json

frames 30
publish_count 30
valid_count 30
deadline_miss_count 0
reasoned_latency_ms median 36.349  p90 37.201  p99 38.363  max 38.695
same_frame_all True
max_rtp_age_frames 0
frame0_latency_ms 37.545
frame0_deadline_met True
```

Stage timing from the recorded sim frames:

```text
camera_to_scene_board_ms        median 2.750   p90 2.872   p99 3.550   max 3.633
scene_board_to_vlm_prefill_ms   median 12.686  p90 13.274  p99 13.590  max 13.711
vlm_decode_ms                   median 17.320  p90 17.687  p99 17.914  max 17.967
rtp_parse_ms                    median 0.053   p90 0.061   p99 0.073   max 0.077
path_synth_ms                   median 0.027   p90 0.035   p99 0.048   max 0.049
```

## Longer Mixed-Scene Synchronous Timing

Extended the proof from a 30-frame construction smoke to a 300-frame random mixed scene with construction and moving pedestrian/vehicle labels rotating through the same shared fixed-shape TensorRT language engine.

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:RTP_VLM_IMAGE_FORMAT='jpeg'
$env:RTP_VLM_JPEG_QUALITY='85'
$env:RTP_VLM_WAIT_READY='1'
$env:RTP_VLM_STDERR_PATH='E:\ture_opamayo\openpilot\artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_mixed_300\vlm_stderr.log'
$env:RTP_VLM_SERVER_COMMAND='py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-groups --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --warmup 1 --ready-jsonl serve'
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --frames 300 --speed-mps 5 --engine vlm --novel-scene random_mixed --deadline-ms 50 --tick-sec 0 --save-every 100 --out artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_mixed_300
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_mixed_300\comparison_vlm.json
artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_mixed_300\vlm\episode.json

frames 300
publish_count 300
valid_count 300
deadline_miss_count 0
reasoned_latency_ms median 35.719  p90 37.006  p99 37.780  p99.9 38.068  max 38.113
same_frame_all True
max_rtp_age_frames 0
selected_candidates C0=24 C1=276
path_delta_nonzero_frames 276
speed_delta_nonzero_frames 276
```

Stage timing from the recorded mixed-scene frames:

```text
camera_to_scene_board_ms        median 2.771   p90 2.810   p99 3.161   p99.9 3.338   max 3.348
scene_board_to_vlm_prefill_ms   median 12.138  p90 13.150  p99 13.507  p99.9 13.627  max 13.629
vlm_decode_ms                   median 17.288  p90 17.699  p99 17.970  p99.9 18.052  max 18.062
rtp_parse_ms                    median 0.051   p90 0.055   p99 0.100   p99.9 0.150   max 0.167
path_synth_ms                   median 0.026   p90 0.028   p99 0.036   p99.9 0.075   max 0.085
```

The demo summary now records `p999_latency_ms`, `max_latency_ms`, `same_frame_count`, `same_frame_all`, and `max_rtp_age_frames` directly in future episode JSON files.

## Crashout Video TensorRT Evaluation

Added `tools\reasoned_trajectory_poc\evaluate_qwen_trt_video.py` to run the same optimized TensorRT Qwen label scorer on a real video file. This is not a closed-loop control replay, so it cannot prove the car would physically avoid the obstacle. It does prove the scorer latency and whether the RTP compiler would request lateral trajectory modification from the video frames.

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\evaluate_qwen_trt_video.py --video E:\ture_opamayo\crashout.mp4 --save-first 8 --save-lateral 8 --out artifacts\reasoned_trajectory_poc\crashout_qwen_trt_video_eval_full
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\crashout_qwen_trt_video_eval_full\video_eval.json
artifacts\reasoned_trajectory_poc\crashout_qwen_trt_video_eval_full\input_samples
artifacts\reasoned_trajectory_poc\crashout_qwen_trt_video_eval_full\lateral_samples

video_frames 1237
video_fps 29.348591
sampled_frames 1237
deadline_miss_count 0
would_change_lateral_count 1201
first_lateral_frame 3
first_lateral_time_sec 0.102
first_lateral_lat_bias_m 1.25
first_lateral_scene mixed_agent_construction_right
max_abs_lat_bias_m 1.25
scored_fps_wall 30.614
```

Crashout scorer timing:

```text
total_ms      median 31.660  p90 32.642  p99 33.437  p99.9 34.096  max 34.111
processor_ms  median 4.208   p90 4.485   p99 4.723   p99.9 5.024   max 5.057
trt_vision_ms median 5.620   p90 6.065   p99 6.316   p99.9 6.464   max 6.602
trt_text_ms   median 17.425  p90 17.802  p99 18.361  p99.9 19.142  max 19.415
embed_ms      median 0.119   p90 0.150   p99 0.214   p99.9 0.305   max 0.407
scatter_ms    median 0.596   p90 0.668   p99 0.857   p99.9 0.942   max 0.944
rope_ms       median 2.250   p90 2.529   p99 2.814   p99.9 3.202   max 3.363
```

Crashout RTP distribution:

```text
lat_bias positive frames 1161
lat_bias negative frames 40
lat_bias zero frames 36
top scenes:
  mixed_agent_construction_right 775
  construction_right 386
  mixed_agent_construction_left 21
  path_conflict_agent 20
  construction_left 19
  nominal 16
```

## Artifact And Shape Validation

Added `check-artifacts` to `tools\reasoned_trajectory_poc\qwen_trt_label_engine.py`. It validates the external TensorRT artifacts and runtime environment before a run, without loading the full Qwen model through Transformers.

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --out artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check.json check-artifacts
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check.json
ok true
issues []
CUDA_PATH C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
nvcc compute_120 true
nvcc sm_120 true
torch_device_name NVIDIA GeForce RTX 5060 Ti
torch_device_capability [12,0]
TensorRT 10.16.1.11
TensorRT FP4 true
```

Validated engine shapes:

```text
vision pixel_values    [96,1176]  FLOAT input
vision image_features  [24,2048]  HALF output
text inputs_embeds     [2,220,2048]  HALF input
text position_ids      [3,2,220]     INT64 input
text selected_logits   [2,1,8]       HALF output
```

## Runtime Manifest Contract

Added a runtime manifest contract for the optimized TensorRT Qwen path. The manifest hashes the prompt contract, score-question text, label groups, image mode/size, text sequence length, vehicle-state text, scoring thresholds, rotating-cache behavior, selected model config/tokenizer files, and model weight filenames/sizes. Runtime commands can now use `--require-manifest` to reject mismatched fixed-shape engines before measuring latency.

Manifest path:

```text
F:\qwen_trt_export\qwen_trt_runtime_manifest.json
contract_sha256 db472aec5aad73627bdb475c8ff253bc6475822dae7ee0950b7a881979c98f97
```

Manifest write command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --write-manifest --out artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_with_behavior_manifest.json check-artifacts
```

Strict validation command:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --require-manifest --out artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_require_manifest.json check-artifacts
```

Strict validation result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_require_manifest.json
ok true
issues []
actual_contract_sha256 db472aec5aad73627bdb475c8ff253bc6475822dae7ee0950b7a881979c98f97
expected_contract_sha256 db472aec5aad73627bdb475c8ff253bc6475822dae7ee0950b7a881979c98f97
```

Fail-closed probes:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_manifest_mismatch.json
changed text_seq_len 220 -> 224
result: rejected
issues:
  text inputs_embeds shape (2,220,2048) != expected (2,224,2048)
  text position_ids shape (3,2,220) != expected (3,2,224)
  manifest contract sha mismatch

artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_manifest_label_mismatch.json
changed label groups while keeping tensor shapes valid
result: rejected
issues:
  manifest contract sha mismatch
```

## 50 ms Pass/Fail Gate

Added `gate` to `tools\reasoned_trajectory_poc\qwen_trt_label_engine.py`. The gate always requires the runtime manifest, validates TensorRT artifacts first, runs the rotating label benchmark, and exits nonzero if either p99 or max total latency exceeds `--deadline-ms`.

Passing 50 ms gate command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --deadline-ms 50 --iters 30 --warmup 3 --out artifacts\reasoned_trajectory_poc\qwen_trt_50ms_gate_behavior_manifest.json gate
```

Passing result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_50ms_gate_behavior_manifest.json
ok true
deadline_ms 50.0
p99_total_ms 33.133
max_total_ms 33.133
issues []
manifest ok true
contract_sha256 db472aec5aad73627bdb475c8ff253bc6475822dae7ee0950b7a881979c98f97
```

Gate stage timing:

```text
total_ms      median 30.832  p90 31.516  p99 33.133  p99.9 33.133  max 33.133
processor_ms  median 3.875   p90 4.132   p99 4.914   p99.9 4.914   max 4.914
trt_vision_ms median 5.623   p90 5.935   p99 6.129   p99.9 6.129   max 6.129
trt_text_ms   median 17.273  p90 17.562  p99 17.973  p99.9 17.973  max 17.973
```

Gate failure-path probe:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --deadline-ms 1 --iters 3 --warmup 1 --out artifacts\reasoned_trajectory_poc\qwen_trt_1ms_gate_expected_fail.json gate
```

Failure result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_1ms_gate_expected_fail.json
ok false
exit code 2
issues:
  p99 total latency 31.254 ms exceeds deadline 1.000 ms
  max total latency 31.254 ms exceeds deadline 1.000 ms
```

## Strict-Manifest MetaDrive Timing

Ran the synchronous MetaDrive VLM loop with the worker itself started under `--require-manifest`, using the behavior-aware manifest that includes score thresholds and rotating-cache settings. This verifies the production-style persistent server path, not just the standalone gate.

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:RTP_VLM_IMAGE_FORMAT='jpeg'
$env:RTP_VLM_JPEG_QUALITY='85'
$env:RTP_VLM_WAIT_READY='1'
$env:RTP_VLM_STDERR_PATH='E:\ture_opamayo\openpilot\artifacts\reasoned_trajectory_poc\metadrive_trt_require_manifest_mixed_120\vlm_stderr.log'
$env:RTP_VLM_SERVER_COMMAND='py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-groups --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --warmup 1 --require-manifest --ready-jsonl serve'
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --frames 120 --speed-mps 5 --engine vlm --novel-scene random_mixed --deadline-ms 50 --tick-sec 0 --save-every 60 --out artifacts\reasoned_trajectory_poc\metadrive_trt_require_manifest_mixed_120
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_require_manifest_mixed_120\comparison_vlm.json
artifacts\reasoned_trajectory_poc\metadrive_trt_require_manifest_mixed_120\vlm\episode.json

frames 120
publish_count 120
valid_count 120
deadline_miss_count 0
reasoned_latency_ms median 36.192  p90 37.082  p99 38.121  p99.9 38.924  max 39.024
same_frame_all True
same_frame_count 120
max_rtp_age_frames 0
selected_candidates C0=18 C1=102
path_delta_nonzero_frames 102
speed_delta_nonzero_frames 102
```

Stage timing:

```text
camera_to_scene_board_ms        median 2.778   p90 2.825   p99 3.157   p99.9 3.698   max 3.769
scene_board_to_vlm_prefill_ms   median 12.718  p90 13.208  p99 13.609  p99.9 13.826  max 13.855
vlm_decode_ms                   median 17.274  p90 17.587  p99 17.956  p99.9 18.098  max 18.115
rtp_parse_ms                    median 0.052   p90 0.057   p99 0.076   p99.9 0.084   max 0.085
path_synth_ms                   median 0.026   p90 0.029   p99 0.056   p99.9 0.064   max 0.065
```

## Rotating Six-Label Worker Verification

The fast path now supports rotating two-label groups while reusing one fixed-shape TensorRT language engine. This covers the proposed six-label cadence:

```text
group 0: construction_left,construction_right
group 1: pedestrian_in_path,pedestrian_entering_path
group 2: vehicle_in_path,vehicle_entering_path
```

The key implementation detail is `--text-seq-len 220`: current verbose prompts tokenize to 220, 216, and 208 tokens respectively, so a fixed 220-token text shape lets all three groups share the same `(2,220,2048)` TensorRT text engine without truncation.

Benchmark command:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  --artifact-dir F:\qwen_trt_export `
  --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine `
  --text-seq-len 220 `
  --score-rotate-shared-engine `
  --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" `
  --iters 9 --warmup 2 `
  --out artifacts\reasoned_trajectory_poc\qwen_trt_rotating_shared_benchmark.json `
  benchmark-groups
```

Persisted rotating benchmark:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_rotating_shared_benchmark.json
total_ms median 31.051  p90 31.907  max 32.216
trt_text_ms median 17.300  p90 17.783  max 17.909
```

Per-group total latency:

```text
construction side: median 30.639  max 31.907
pedestrian conflict: median 31.185  max 32.216
vehicle conflict: median 29.902  max 31.114
```

Persistent JSONL serve command:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  --artifact-dir F:\qwen_trt_export `
  --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine `
  --text-seq-len 220 `
  --score-rotate-groups `
  --score-rotate-shared-engine `
  --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" `
  --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" `
  serve
```

Persistent JSONL smoke on the same scene-board input:

```text
frame 0 group construction_left,construction_right: labels none, total_ms 30.930
frame 1 group pedestrian_in_path,pedestrian_entering_path: labels none, total_ms 30.232
frame 2 group vehicle_in_path,vehicle_entering_path: labels none, total_ms 30.246
```

The vehicle group raw score on that construction frame was `vehicle_in_path=0.25`, so per-label thresholds are now supported and the smoke used `vehicle_in_path:0.5` to avoid turning that weak score into a false yield.

## Prior Persistent Worker Verification

`tools\reasoned_trajectory_poc\qwen_trt_label_engine.py` now has a `serve` subcommand. It keeps the build and benchmark commands intact and adds a JSONL worker compatible with `selfdrive.controls.reasoned.vlm.PersistentRtpEngine`.

Worker command:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --warmup 2 --score-labels construction_left,construction_right serve
```

Direct JSONL smoke, using `artifacts\reasoned_trajectory_poc\qwen_construction_loop_proof\vlm\vlm_input_0020.png`:

```text
backend qwen2.5-vl-3b-trt-nvfp4-full168-score
frame_id 20
source_frame_id 20
prefill_ms 11.498
decode_ms 16.987
total_ms 29.549
labels ["none"]
scores construction_left=-0.5625 construction_right=-0.75
```

Planner-contract smoke through `PersistentRtpEngine`:

```text
text_first_line RTPv1
backend qwen2.5-vl-3b-trt-nvfp4-full168-score
source_frame_id 20
prefill_ms 11.736
decode_ms 17.379
```

## Generated Artifacts

Current working TensorRT artifacts were moved from `C:\Users\user\AppData\Local\Temp\qwen_trt_export` to `F:\qwen_trt_export` because C: ran out of space during grouped exports. They are not committed:

```text
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.onnx
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_construction_left__construction_right_trt.engine
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_construction_left__construction_right_trt.onnx
F:\qwen_trt_export\vision_static_fp16\qwen_vision_full168_static_fp16.engine
F:\qwen_trt_export\vision_static_fp16\qwen_vision_full168_static_fp16.onnx
```

The temp artifact directory is intentionally outside the repo because the text and vision engines are large.

## Implementation Notes

- Direct FP8 or FP4 fake quant in PyTorch is not useful for runtime. ModelOpt fake-quant configs were slower than FP16 PyTorch.
- TensorRT-LLM is not available as a Windows wheel in this environment. The successful path is direct ONNX plus TensorRT engines.
- The Qwen vision tower could not export as-is because:
  - SDPA export hit `scaled_dot_product_attention` with `enable_gqa=True`.
  - TensorRT rejected a half-typed rotary `Range`.
- The working vision engine uses an eager-attention wrapper and bakes fixed full168 constants:
  - `rotary_pos_emb`
  - `window_index`
  - `cu_window_seqlens`
  - `cu_seqlens`
  - reverse window index
- The working text engine removes the full vocabulary LM head and only computes logits for yes/no token IDs used by label scoring.
- The text engine is fixed-shape for the prompt/image/label set used at build time. Changing prompt text, label group, image mode, or image size can change token/image shapes and requires rebuilding or adding fixed profiles for those variants.
- The worker defaults to a fixed vehicle-state string for shape stability. `--use-payload-vehicle-state` exists, but it will reject requests if the resulting text shape differs from the built TensorRT engine.
- Separate label-keyed text engine builds are possible, but not the preferred runtime path. A grouped build hit disk-space first, then a subsequent non-construction keyed build went idle in TensorRT build. The better working path is a shared fixed sequence length matching all current two-label prompts.

## Next Work

- Decide whether to keep the engines in a configured external artifact directory or add an explicit build step for local users.
- Rebuild or add additional fixed-shape engines for the actual production label rotation sets if they differ from the two-label proof set.

## 2026-05-23 Fast TensorRT Mixed MetaDrive Video Demo

Ran a synchronous mixed pedestrian plus construction MetaDrive comparison using the TensorRT Qwen fast path, strict runtime manifest, CUDA 13.2, JPEG scene-board transport, 5.0 m/s target speed, and every-frame video capture.

Command shape:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:RTP_VLM_WAIT_READY='1'
$env:RTP_VLM_IMAGE_FORMAT='jpeg'
$env:RTP_VLM_JPEG_QUALITY='85'
$env:RTP_VLM_SERVER_COMMAND='py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-groups --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --require-manifest --ready-jsonl serve'
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --engine vlm --novel-scene random_mixed --frames 300 --speed-mps 5.0 --tick-sec 0.05 --deadline-ms 50 --save-every 1 --map 3 --seed 7 --random-scene-seed 42 --out artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043
```

Run artifacts:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043\videos\side_by_side_fast_mixed_300_sync_5mps.mp4
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043\videos\stock_fast_mixed_300_sync_5mps.mp4
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043\videos\vlm_fast_mixed_300_sync_5mps.mp4
```

Summary:

```text
stock frames 300, mean_speed_mps 4.611, target_speed_mps 5.0
vlm frames 300, publish_count 300, valid_count 300, deadline_miss_count 0
vlm same_frame_all true, max_rtp_age_frames 0
vlm latency median 36.430 ms, p90 37.291 ms, p99 37.881 ms, p99.9 38.602 ms, max 38.602 ms
vlm selected_candidates C1=276, C0=24
vlm active durable lateral frames 276
vlm active durable speed frames 276
vlm mean_path_delta_m 1.15
vlm lateral offset range -1.25..0.0 MetaDrive meters
vlm mean_speed_mps 0.920 because pedestrian/agent speed-plan logic was active
stock min_spawned_object_distance_m 0.805
vlm min_spawned_object_distance_m 1.133
```

Stage timing:

```text
camera_to_scene_board_ms median 2.761, p99 3.079, max 3.680
scene_board_to_vlm_prefill_ms median 12.677, p99 13.575, max 13.763
vlm_decode_ms median 17.496, p99 18.240, max 18.315
rtp_parse_ms median 0.051, p99 0.073, max 0.104
path_synth_ms median 0.026, p99 0.049, max 0.176
```

The first non-base RTP occurred at frame 24:

```text
scene=construction_right
meta=BIAS_LEFT_AND_SLOW
lat_bias_m=1.25
speed_cap_mps=25%
avoid=[right_edge_s8_48_margin1.25]
confidence=0.72
```

The speed controller then fired mixed pedestrian/agent constraints at frame 56:

```text
scene=mixed_agent_construction_right
meta=YIELD
lat_bias_m=1.25
speed_cap_mps=15%
avoid=[right_edge_s8_48_margin1.25,corridor_object_s18_28]
confidence=0.72
```

## 2026-05-23 Fast TensorRT Mixed MetaDrive Lateral-Only 2.5 m/s Demo

Reran the same mixed construction plus pedestrian MetaDrive setup with target speed capped at 2.5 m/s and VLM speed control disabled. This isolates lateral behavior while keeping stock and VLM longitudinal behavior effectively identical.

Command difference from the prior run:

```powershell
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --engine vlm --novel-scene random_mixed --frames 300 --speed-mps 2.5 --disable-vlm-speed-control --tick-sec 0.05 --deadline-ms 50 --save-every 1 --map 3 --seed 7 --random-scene-seed 42 --out artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214340_2p5mps_lateral_only
```

Video artifacts:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214340_2p5mps_lateral_only\videos\side_by_side_fast_mixed_300_sync_2p5mps_lateral_only.mp4
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214340_2p5mps_lateral_only\videos\stock_fast_mixed_300_sync_2p5mps_lateral_only.mp4
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214340_2p5mps_lateral_only\videos\vlm_fast_mixed_300_sync_2p5mps_lateral_only.mp4
```

Summary:

```text
stock frames 300, mean_speed_mps 2.322, target_speed_mps fixed 2.5
vlm frames 300, publish_count 300, valid_count 300, deadline_miss_count 0
vlm same_frame_all true, max_rtp_age_frames 0
vlm mean_speed_mps 2.318, target_speed_mps fixed 2.5
vlm speed-control enabled false on every frame
vlm latency median 37.298 ms, p90 38.523 ms, p99 39.876 ms, max 41.072 ms
vlm selected_candidates C1=289, C0=11
vlm active durable lateral frames 276
vlm active durable speed frames 0
vlm mean_path_delta_m 1.15
vlm active lateral offset range 0.0..1.25 MetaDrive meters
stock min_spawned_object_distance_m 1.073
vlm min_spawned_object_distance_m 0.111
```

Interpretation: the 2.5 m/s cap and `--disable-vlm-speed-control` worked. The VLM no longer slows the car relative to stock. However, this seed exposes a lateral-policy failure: the VLM path modification got closer to a spawned object than stock. The fast TensorRT path is meeting the realtime budget, but lateral target selection/sign/corridor handling still needs correction before this mixed-scene behavior is acceptable.

## 2026-05-23 Corridor And Side-Grounding Follow-up

The 2.5 m/s lateral-only run showed the car moving toward the cone cluster. Inspection found multiple issues:

```text
frame 24 exact VLM input showed foreground cones on image-right of the green planned path
runtime RTP nevertheless said construction_left
fixed text_seq_len=220 had previously truncated left/right questions so construction_left and construction_right prompts were token-identical
after moving the scored label/question to the front of the prompt, the left/right prompts are no longer identical
Qwen still needs stronger path-relevance semantics: only hazards overlapping, intruding into, narrowing, blocking, or imminently entering the green corridor should count
```

Changes made:

```text
tools\reasoned_trajectory_poc\qwen_trt_label_engine.py
  moved Scored label and Question before the long common prompt so text_seq_len=220 preserves the side-specific question

tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py
  tightened prompts/questions so Qwen should only consider hazards affecting the green planned path or imminent path entrants
  removed first-label-wins behavior for exclusive construction_left/construction_right ties

tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
  corrected the MetaDrive lateral conversion to match the observed route convention in this harness

selfdrive\controls\reasoned\ui_scene_board.py
  expanded the green planned corridor from 0.48 m half-width to 0.60 m half-width, exactly 25% wider
```

Verification:

```text
py -3.11 -m py_compile selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
Ran 20 tests OK
```

The strict TensorRT behavior manifest should be treated as stale after these prompt and scene-board changes. Do not use `--require-manifest` again until a new side-grounding probe and mixed lateral-only run pass, then rewrite `F:\qwen_trt_export\qwen_trt_runtime_manifest.json`.

## 2026-05-23 Wide-Corridor Sign-Fixed Demo And Manifest Rewrite

The first wide-corridor rerun still failed side behavior because the MetaDrive sign conversion had been changed in the wrong direction. It produced `construction_right` correctly, but converted that right-edge avoid into a positive MetaDrive target while the spawned right-side cones also had positive lateral coordinates. That run was rejected and was not used for the manifest.

Fixed the MetaDrive sign conversion back to the observed harness convention:

```text
PathSynth/openpilot positive lat_bias_m = left
MetaDrive positive lateral in this harness = visually right
right_edge_s8_48_margin1.25 -> openpilot +1.25 -> MetaDrive -1.25
left_edge_s8_48_margin1.25 -> openpilot -1.25 -> MetaDrive +1.25
```

Reran the mixed construction/pedestrian demo at 2.5 m/s with VLM speed control disabled:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_220505_2p5mps_lateral_only_widecorridor_signfix
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_220505_2p5mps_lateral_only_widecorridor_signfix\videos\side_by_side_fast_mixed_300_sync_2p5mps_lateral_only_widecorridor_signfix.mp4
```

Run result:

```text
stock frames 300, mean_speed_mps 2.3219, min_spawned_object_distance_m 1.0730
vlm frames 300, publish_count 300, valid_count 300, deadline_miss_count 0
vlm same_frame_all true, max_rtp_age_frames 0
vlm mean_speed_mps 2.3217, target_speed fixed 2.5, VLM speed control disabled
vlm latency p90 37.904 ms, p99 38.906 ms, max 39.366 ms
vlm min_spawned_object_distance_m 1.3115
vlm active lateral offset range -1.25..0.0
vlm source_counts right_edge_s8_48_margin1.25=294, left_edge_s8_48_margin1.25=6
```

Side audit on the same run:

```text
frame 24 visible construction color mass: image-right
frame 24 RTP source: right_edge_s8_48_margin1.25
frame 24 spawned construction laterals ahead: +1.069, +1.236, +1.157, +1.345, +1.337, +1.378
frame 24 active MetaDrive lateral target: -0.252

frame 98 spawned construction laterals ahead: +1.236, +1.157, +1.345, +1.337, +1.378, +1.253
frame 98 active MetaDrive lateral target: -1.25
```

Direct Qwen side score probe on the saved frame:

```text
image: artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_220505_2p5mps_lateral_only_widecorridor_signfix\vlm\vlm_input_0024.png
construction_left score: 4.30078125
construction_right score: 5.28125
total_ms p90 32.939 ms
```

Rewrote the TensorRT runtime manifest after this validation:

```text
F:\qwen_trt_export\qwen_trt_runtime_manifest.json
contract_sha256 cf6c028ed0580f03db61300884c0b777bad6c750741998d64fdf98a0d5319f29
```

The manifest contract now includes:

```text
score prompt hash
score question hash
label groups
score_rotate_groups
score_rotate_shared_engine
thresholds
image mode/size
text_seq_len
model config metadata
scene-board geometry, including planned_corridor_half_width_m=0.60
```

Manifest-gated 50 ms check passed:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_50ms_gate_widecorridor_signfix_manifest.json
ok true
manifest ok true
p99_total_ms 34.458
max_total_ms 34.458
contract_sha256 cf6c028ed0580f03db61300884c0b777bad6c750741998d64fdf98a0d5319f29
```

Remaining caveat: the lateral-only demo disables VLM speed control, so pedestrian/vehicle path-conflict labels cannot yield or stop the car. A pedestrian in the planned path during a lateral-only run is not avoidable by this configuration; production mixed scenes need speed control enabled or a separate explicitly validated lateral pedestrian avoidance policy.
