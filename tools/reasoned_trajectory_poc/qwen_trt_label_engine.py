#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def _prefer_cuda_13_2() -> None:
  cuda_13 = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
  if cuda_13.exists():
    os.environ["CUDA_PATH"] = str(cuda_13)
    os.environ["CUDA_HOME"] = str(cuda_13)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    cuda_bin = str(cuda_13 / "bin")
    cuda_libnvvp = str(cuda_13 / "libnvvp")
    cuda_cupti = str(cuda_13 / "extras" / "CUPTI" / "lib64")
    for part in (cuda_cupti, cuda_libnvvp, cuda_bin):
      if part not in path_parts:
        path_parts.insert(0, part)
    os.environ["PATH"] = os.pathsep.join(path_parts)
    os.environ.setdefault("CL", "/Zc:preprocessor")


_prefer_cuda_13_2()

import torch
from PIL import Image
import tensorrt as trt
from torch import nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from tools.reasoned_trajectory_poc.qwen_label_rtp_worker import (
  DEFAULT_DURABLE_SCORE_LABELS,
  DEFAULT_MODEL_DIR,
  DEFAULT_SCORE_LABEL_GROUPS,
  RotatingScoreState,
  SCORE_PROMPT,
  SCORE_QUESTIONS,
  _image_from_payload,
  _inference_images,
  _labels_to_rtp,
  _parse_score_label_groups,
  _score_label_ids,
  _validate_score_labels,
  _with_visual_fallbacks,
)

try:
  from selfdrive.controls.reasoned.ui_scene_board import OverlayGeometry
except Exception:
  OverlayGeometry = None


DEFAULT_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "qwen_trt_export"
DEFAULT_IMAGE = REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "qwen_construction_loop_proof" / "vlm" / "vlm_input_0020.png"
MANIFEST_VERSION = 1
MODEL_CONTRACT_FILES = (
  "config.json",
  "generation_config.json",
  "preprocessor_config.json",
  "tokenizer_config.json",
  "chat_template.json",
  "model.safetensors.index.json",
)


def percentile(values: Sequence[float], pct: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
  return float(ordered[idx])


def _load_qwen(model_dir: Path):
  processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa",
  )
  model.eval()
  return processor, model


def _score_labels(raw: str) -> tuple[str, ...]:
  labels = tuple(label.strip() for label in raw.split(",") if label.strip())
  unknown = [label for label in labels if label not in SCORE_QUESTIONS]
  if unknown:
    raise ValueError(f"unknown score labels: {unknown}")
  if not labels:
    raise ValueError("at least one score label is required")
  return labels


def _labels_key(labels: Sequence[str]) -> str:
  return "__".join(label.replace("-", "_").replace("/", "_") for label in labels)


def _text_seq_suffix(args) -> str:
  text_seq_len = int(getattr(args, "text_seq_len", 0))
  return "" if text_seq_len <= 0 else f"_seq{text_seq_len}"


def _generic_text_engine_path(args) -> Path:
  return args.artifact_dir / "nvfp4_trt" / f"qwen_text_36layer_nvfp4{_text_seq_suffix(args)}_trt.engine"


def _generic_text_onnx_path(args) -> Path:
  return args.artifact_dir / "nvfp4_trt" / f"qwen_text_36layer_nvfp4{_text_seq_suffix(args)}_trt.onnx"


def _keyed_text_engine_path(args, labels: Sequence[str]) -> Path:
  return args.artifact_dir / "nvfp4_trt" / f"qwen_text_36layer_nvfp4{_text_seq_suffix(args)}_{_labels_key(labels)}_trt.engine"


def _keyed_text_onnx_path(args, labels: Sequence[str]) -> Path:
  return args.artifact_dir / "nvfp4_trt" / f"qwen_text_36layer_nvfp4{_text_seq_suffix(args)}_{_labels_key(labels)}_trt.onnx"


def _resolve_text_engine_path(args, labels: Sequence[str], *, require_keyed: bool = False) -> Path:
  if args.text_engine is not None and not require_keyed:
    return args.text_engine
  keyed = _keyed_text_engine_path(args, labels)
  if keyed.exists():
    return keyed
  if require_keyed:
    raise FileNotFoundError(f"missing label-specific TensorRT text engine for {','.join(labels)}: {keyed}")
  return _generic_text_engine_path(args)


def _score_groups(raw: str) -> tuple[tuple[str, ...], ...]:
  groups = _parse_score_label_groups(raw)
  if not groups:
    raise ValueError("at least one score label group is required")
  for group in groups:
    unknown = [label for label in group if label not in SCORE_QUESTIONS]
    if unknown:
      raise ValueError(f"unknown score label group entries: {unknown}")
  return groups


def _parse_score_threshold_map(raw: str) -> dict[str, float]:
  thresholds: dict[str, float] = {}
  if not raw.strip():
    return thresholds
  for item in raw.split(","):
    if not item.strip():
      continue
    if ":" not in item:
      raise ValueError(f"invalid score threshold item: {item}")
    label, value_raw = item.split(":", 1)
    label = label.strip()
    if label not in SCORE_QUESTIONS:
      raise ValueError(f"unknown score threshold label: {label}")
    thresholds[label] = float(value_raw)
  return thresholds


def _sha256_text(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _manifest_path(args) -> Path:
  manifest = getattr(args, "manifest", None)
  return manifest if manifest is not None else args.artifact_dir / "qwen_trt_runtime_manifest.json"


def _model_revision(model_dir: Path) -> dict:
  files = {}
  for name in MODEL_CONTRACT_FILES:
    path = model_dir / name
    if path.exists():
      files[name] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
      }
    else:
      files[name] = {
        "missing": True,
      }
  weights = []
  for path in sorted(model_dir.glob("*.safetensors")):
    weights.append({
      "name": path.name,
      "bytes": path.stat().st_size,
      "mtime": path.stat().st_mtime,
    })
  return {
    "model_dir_name": model_dir.name,
    "files": files,
    "weights": weights,
  }


def _scene_board_contract() -> dict:
  if OverlayGeometry is None:
    return {"overlay_geometry_importable": False}
  geometry = OverlayGeometry()
  return {
    "overlay_geometry_importable": True,
    "planned_corridor_half_width_m": float(geometry.planned_corridor_half_width_m),
    "lane_width_m": float(geometry.lane_width_m),
    "camera_height_m": float(geometry.camera_height_m),
    "horizon_ratio": float(geometry.horizon_ratio),
    "focal_ratio": float(geometry.focal_ratio),
    "max_draw_distance_m": float(geometry.max_draw_distance_m),
  }


def _runtime_contract(args, groups: Sequence[Sequence[str]]) -> dict:
  normalized_groups = tuple(tuple(group) for group in groups)
  labels = tuple(dict.fromkeys(label for group in normalized_groups for label in group))
  questions = {label: SCORE_QUESTIONS[label] for label in labels}
  payload = {
    "manifest_version": MANIFEST_VERSION,
    "model": _model_revision(args.model_dir),
    "prompt": {
      "score_prompt_sha256": _sha256_text(SCORE_PROMPT),
      "score_questions_sha256": _sha256_text(json.dumps(questions, sort_keys=True, separators=(",", ":"))),
      "labels": labels,
    },
    "scene_board": _scene_board_contract(),
    "runtime": {
      "image_mode": args.image_mode,
      "image_size": int(args.image_size),
      "text_seq_len": int(args.text_seq_len),
      "score_label_groups": normalized_groups,
      "score_rotate_groups": bool(args.score_rotate_groups),
      "score_rotate_shared_engine": bool(args.score_rotate_shared_engine),
      "vehicle_state": args.vehicle_state,
      "score_threshold": float(args.score_threshold),
      "score_thresholds": dict(sorted(getattr(args, "score_thresholds_map", {}).items())),
      "score_cache_ttl_frames": int(args.score_cache_ttl_frames),
      "score_durable_labels": tuple(label.strip() for label in args.score_durable_labels.split(",") if label.strip()),
      "score_negative_clear_threshold": float(args.score_negative_clear_threshold),
    },
  }
  return {
    "contract_sha256": _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
    "contract": payload,
  }


def _read_manifest(path: Path) -> tuple[dict | None, str | None]:
  if not path.exists():
    return None, f"missing manifest: {path}"
  try:
    return json.loads(path.read_text(encoding="utf-8")), None
  except Exception as exc:
    return None, f"failed to read manifest {path}: {exc!r}"


def _validate_manifest(args, groups: Sequence[Sequence[str]]) -> dict:
  path = _manifest_path(args)
  expected = _runtime_contract(args, groups)
  actual, error = _read_manifest(path)
  issues = []
  if error is not None:
    issues.append(error)
    actual_sha = ""
  else:
    actual_sha = str(actual.get("contract_sha256", ""))
    if actual_sha != expected["contract_sha256"]:
      issues.append(
        f"manifest contract sha mismatch: actual {actual_sha} expected {expected['contract_sha256']}"
      )
  return {
    "path": str(path),
    "exists": path.exists(),
    "ok": not issues,
    "issues": issues,
    "actual_contract_sha256": actual_sha,
    "expected_contract_sha256": expected["contract_sha256"],
  }


def _write_manifest(args, groups: Sequence[Sequence[str]], result: dict) -> dict:
  path = _manifest_path(args)
  contract = _runtime_contract(args, groups)
  manifest = {
    "kind": "qwen_trt_runtime_manifest",
    "created_unix": time.time(),
    **contract,
    "artifact_dir": str(args.artifact_dir),
    "vision_engine": result.get("vision_engine", {}),
    "text_engine": result.get("text_engine", {}),
    "cuda": result.get("cuda", {}),
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  return {
    "path": str(path),
    "written": True,
    "contract_sha256": contract["contract_sha256"],
  }


def _groups_for_runtime(args) -> tuple[tuple[str, ...], ...]:
  if getattr(args, "score_rotate_groups", False) or getattr(args, "cmd", "") in ("benchmark-groups", "check-artifacts"):
    return _score_groups(args.score_label_groups)
  return (_score_labels(args.score_labels),)


def _enforce_manifest(args, groups: Sequence[Sequence[str]]) -> None:
  manifest = _validate_manifest(args, groups)
  if not manifest["ok"]:
    raise RuntimeError("; ".join(manifest["issues"]))


def _summarize_timing_rows(timing_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
  if not timing_rows:
    return {}
  summary = {}
  for key in timing_rows[0]:
    vals = [row[key] for row in timing_rows]
    summary[key] = {
      "median": statistics.median(vals),
      "p90": percentile(vals, 90),
      "p99": percentile(vals, 99),
      "p999": percentile(vals, 99.9),
      "max": max(vals),
      "min": min(vals),
    }
  return summary


def _build_inputs(
  processor,
  image: Image.Image,
  labels: Sequence[str],
  image_mode: str,
  image_size: int,
  vehicle_state: str,
  text_seq_len: int = 0,
):
  images, _ = _inference_images(image, image_mode, image_size)
  prompts: list[str] = []
  batch_images: list[Image.Image] = []
  for label in labels:
    content = [{"type": "image", "image": view} for view in images]
    content.append({
      "type": "text",
      "text": f"Scored label: {label}\nQuestion: {SCORE_QUESTIONS[label]}\n{SCORE_PROMPT}\nVehicle state: {vehicle_state}",
    })
    prompts.append(processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
    batch_images.extend(images)
  processor_kwargs = {
    "text": prompts,
    "images": batch_images,
    "padding": True,
    "return_tensors": "pt",
  }
  if text_seq_len > 0:
    processor_kwargs["padding"] = "max_length"
    processor_kwargs["max_length"] = text_seq_len
    processor_kwargs["truncation"] = True
  return processor(**processor_kwargs).to("cuda")


def _prepare_text_tensors(
  processor,
  model,
  image: Image.Image,
  labels: Sequence[str],
  image_mode: str,
  image_size: int,
  vehicle_state: str,
  text_seq_len: int = 0,
):
  inputs = _build_inputs(processor, image, labels, image_mode, image_size, vehicle_state, text_seq_len)
  with torch.no_grad():
    qwen = model.model
    inputs_embeds = qwen.get_input_embeddings()(inputs.input_ids)
    image_features = torch.cat(qwen.get_image_features(inputs.pixel_values, inputs.image_grid_thw), dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = qwen.get_placeholder_mask(inputs.input_ids, inputs_embeds=inputs_embeds, image_features=image_features)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features).contiguous().detach().clone()
    position_ids, _ = qwen.get_rope_index(
      inputs.input_ids,
      inputs.image_grid_thw,
      None,
      second_per_grid_ts=None,
      attention_mask=inputs.attention_mask,
    )
    position_ids = position_ids.contiguous().detach().clone()
    yes_ids, no_ids = _score_label_ids(processor)
    selected_ids = torch.tensor(list(yes_ids) + list(no_ids), device="cuda", dtype=torch.long)
  return inputs, inputs_embeds, position_ids, selected_ids


class TextScore(nn.Module):
  def __init__(self, language_model, lm_head, selected_ids: torch.Tensor):
    super().__init__()
    self.language_model = language_model
    self.lm_head = lm_head
    self.register_buffer("selected_ids", selected_ids.detach().clone())

  def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    out = self.language_model(
      input_ids=None,
      position_ids=position_ids,
      attention_mask=None,
      past_key_values=None,
      inputs_embeds=inputs_embeds,
      use_cache=False,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=False,
      cache_position=None,
    )
    last = out[0][:, -1:, :]
    selected_weight = self.lm_head.weight.index_select(0, self.selected_ids)
    return torch.matmul(last, selected_weight.t())


class VisionStatic(nn.Module):
  def __init__(
    self,
    visual,
    rotary_pos_emb: torch.Tensor,
    window_index: torch.Tensor,
    cu_window: torch.Tensor,
    cu_seqlens: torch.Tensor,
    reverse: torch.Tensor,
  ):
    super().__init__()
    self.visual = visual
    self.register_buffer("rotary_const", rotary_pos_emb.detach().clone())
    self.register_buffer("window_index", window_index.detach().clone())
    self.register_buffer("cu_window", cu_window.detach().clone())
    self.register_buffer("cu_seqlens", cu_seqlens.detach().clone())
    self.register_buffer("reverse", reverse.detach().clone())

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    hidden_states = self.visual.patch_embed(pixel_values)
    seq_len = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
    hidden_states = hidden_states[self.window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    rotary_pos_emb = self.rotary_const.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[self.window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    for layer_num, block in enumerate(self.visual.blocks):
      cu_now = self.cu_seqlens if layer_num in self.visual.fullatt_block_indexes else self.cu_window
      hidden_states = block(hidden_states, cu_seqlens=cu_now, position_embeddings=position_embeddings)

    hidden_states = self.visual.merger(hidden_states)
    return hidden_states[self.reverse, :]


def _build_trt_engine(
  onnx_path: Path,
  engine_path: Path,
  *,
  fp16: bool = True,
  fp4: bool = False,
  workspace_gb: int = 6,
) -> dict:
  logger = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(logger)
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, logger)
  parse_start = time.perf_counter()
  ok = parser.parse_from_file(str(onnx_path))
  parse_ms = (time.perf_counter() - parse_start) * 1000.0
  errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
  if not ok:
    raise RuntimeError(f"TensorRT failed to parse {onnx_path}: {errors}")

  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
  if fp16:
    config.set_flag(trt.BuilderFlag.FP16)
  if fp4:
    config.set_flag(trt.BuilderFlag.FP4)

  build_start = time.perf_counter()
  serialized = builder.build_serialized_network(network, config)
  build_ms = (time.perf_counter() - build_start) * 1000.0
  if serialized is None:
    raise RuntimeError(f"TensorRT failed to build {onnx_path}")

  engine_path.parent.mkdir(parents=True, exist_ok=True)
  serialized_bytes = bytes(serialized)
  engine_path.write_bytes(serialized_bytes)
  return {
    "onnx": str(onnx_path),
    "engine": str(engine_path),
    "parse_ms": parse_ms,
    "build_ms": build_ms,
    "engine_bytes": len(serialized_bytes),
    "parse_errors": errors,
  }


def build_text_engine(args) -> dict:
  import modelopt.torch.quantization as mtq
  from modelopt.onnx.export.nvfp4_exporter import NVFP4QuantExporter
  from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers
  import onnx

  labels = _score_labels(args.score_labels)
  out_dir = args.artifact_dir / "nvfp4_trt"
  raw_dir = args.artifact_dir / "nvfp4_raw"
  shutil.rmtree(raw_dir, ignore_errors=True)
  out_dir.mkdir(parents=True, exist_ok=True)
  raw_dir.mkdir(parents=True, exist_ok=True)

  image = Image.open(args.image).convert("RGB")
  processor, model = _load_qwen(args.model_dir)
  model.config._attn_implementation = "eager"
  model.model.language_model.config._attn_implementation = "eager"
  _, inputs_embeds, position_ids, selected_ids = _prepare_text_tensors(
    processor,
    model,
    image,
    labels,
    args.image_mode,
    args.image_size,
    args.vehicle_state,
    args.text_seq_len,
  )

  wrapper = TextScore(model.model.language_model, model.lm_head, selected_ids).cuda().half().eval()
  with torch.no_grad():
    ref = wrapper(inputs_embeds, position_ids)
    torch.cuda.synchronize()

  quant_start = time.perf_counter()
  qwrapper = mtq.quantize(wrapper, mtq.NVFP4_DEFAULT_CFG, forward_loop=lambda mdl: mdl(inputs_embeds, position_ids))
  torch.cuda.synchronize()
  quant_ms = (time.perf_counter() - quant_start) * 1000.0
  qwrapper.eval()

  raw_onnx = raw_dir / "qwen_text_36layer_nvfp4_raw.onnx"
  if getattr(args, "label_keyed_text_engine", False):
    final_onnx = _keyed_text_onnx_path(args, labels)
    engine_path = _keyed_text_engine_path(args, labels)
  else:
    final_onnx = _generic_text_onnx_path(args)
    engine_path = _generic_text_engine_path(args)
  export_start = time.perf_counter()
  with torch.no_grad(), configure_linear_module_onnx_quantizers(qwrapper):
    torch.onnx.export(
      qwrapper,
      (inputs_embeds, position_ids),
      str(raw_onnx),
      input_names=["inputs_embeds", "position_ids"],
      output_names=["selected_logits"],
      opset_version=21,
      dynamo=False,
      do_constant_folding=True,
      external_data=True,
    )
  raw_export_ms = (time.perf_counter() - export_start) * 1000.0

  post_start = time.perf_counter()
  raw_model = onnx.load(str(raw_onnx))
  processed = NVFP4QuantExporter.process_model(raw_model)
  postprocess_ms = (time.perf_counter() - post_start) * 1000.0

  save_start = time.perf_counter()
  onnx.save(processed, str(final_onnx))
  save_ms = (time.perf_counter() - save_start) * 1000.0
  shutil.rmtree(raw_dir, ignore_errors=True)

  build = _build_trt_engine(final_onnx, engine_path, fp16=True, fp4=True, workspace_gb=args.workspace_gb)
  return {
    "kind": "text_nvfp4",
    "labels": labels,
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "ref_mean": float(ref.mean()),
    "quant_ms": quant_ms,
    "raw_export_ms": raw_export_ms,
    "postprocess_ms": postprocess_ms,
    "save_ms": save_ms,
    **build,
  }


def build_vision_engine(args) -> dict:
  out_dir = args.artifact_dir / "vision_static_fp16"
  out_dir.mkdir(parents=True, exist_ok=True)
  image = Image.open(args.image).convert("RGB")
  processor, model = _load_qwen(args.model_dir)
  visual = model.model.visual
  visual.config._attn_implementation = "eager"

  images, _ = _inference_images(image, args.image_mode, args.image_size)
  prompt = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image", "image": images[0]}, {"type": "text", "text": "x"}]}],
    tokenize=False,
    add_generation_prompt=True,
  )
  inputs = processor(text=[prompt], images=images[:1], padding=True, return_tensors="pt").to("cuda")
  pixel_values = inputs.pixel_values.contiguous()
  grid = inputs.image_grid_thw[:1].contiguous()

  with torch.no_grad():
    rotary = visual.rot_pos_emb(grid)
    window_index, cu_window_list = visual.get_window_index(grid)
    cu_window = torch.tensor(cu_window_list, device=pixel_values.device, dtype=torch.int32)
    cu_window = torch.unique_consecutive(cu_window)
    cu_seqlens = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    reverse = torch.argsort(window_index)

  wrapper = VisionStatic(visual, rotary, window_index, cu_window, cu_seqlens, reverse).cuda().half().eval()
  with torch.no_grad():
    ref_hf = visual(pixel_values, grid)
    ref = wrapper(pixel_values)
    torch.cuda.synchronize()
    mse = float((ref.float() - ref_hf.float()).square().mean())

  onnx_path = out_dir / "qwen_vision_full168_static_fp16.onnx"
  engine_path = out_dir / "qwen_vision_full168_static_fp16.engine"
  export_start = time.perf_counter()
  with torch.no_grad():
    torch.onnx.export(
      wrapper,
      (pixel_values,),
      str(onnx_path),
      input_names=["pixel_values"],
      output_names=["image_features"],
      opset_version=20,
      dynamo=False,
      do_constant_folding=True,
      external_data=True,
    )
  export_ms = (time.perf_counter() - export_start) * 1000.0

  build = _build_trt_engine(onnx_path, engine_path, fp16=True, fp4=False, workspace_gb=args.workspace_gb)
  return {
    "kind": "vision_static_fp16",
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "pixel_shape": tuple(pixel_values.shape),
    "image_feature_shape": tuple(ref.shape),
    "static_wrapper_mse_vs_hf": mse,
    "export_ms": export_ms,
    **build,
  }


def _load_engine(runtime, engine_path: Path):
  if not engine_path.exists():
    raise FileNotFoundError(engine_path)
  engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
  if engine is None:
    raise RuntimeError(f"failed to deserialize {engine_path}")
  return engine, engine.create_execution_context()


def _trt_shape(engine, tensor_name: str) -> tuple[int, ...]:
  return tuple(int(dim) for dim in engine.get_tensor_shape(tensor_name))


def _file_info(path: Path) -> dict:
  exists = path.exists()
  return {
    "path": str(path),
    "exists": exists,
    "bytes": path.stat().st_size if exists else 0,
    "mtime": path.stat().st_mtime if exists else 0.0,
  }


def _engine_info(runtime: trt.Runtime, path: Path) -> tuple[dict, list[str]]:
  issues: list[str] = []
  info = _file_info(path)
  if not path.exists():
    issues.append(f"missing engine: {path}")
    info["deserialized"] = False
    info["tensors"] = {}
    return info, issues
  engine = runtime.deserialize_cuda_engine(path.read_bytes())
  if engine is None:
    issues.append(f"failed to deserialize engine: {path}")
    info["deserialized"] = False
    info["tensors"] = {}
    return info, issues
  tensors = {}
  for idx in range(engine.num_io_tensors):
    name = engine.get_tensor_name(idx)
    tensors[name] = {
      "shape": _trt_shape(engine, name),
      "dtype": str(engine.get_tensor_dtype(name)),
      "mode": str(engine.get_tensor_mode(name)),
    }
  info["deserialized"] = True
  info["tensors"] = tensors
  return info, issues


def _check_shape(info: dict, tensor_name: str, expected: tuple[int, ...], issues: list[str], label: str) -> None:
  tensors = info.get("tensors", {})
  actual = tuple(tensors.get(tensor_name, {}).get("shape", ()))
  if actual != expected:
    issues.append(f"{label} {tensor_name} shape {actual} != expected {expected}")


def _nvcc_info() -> dict:
  cuda_path = Path(os.environ.get("CUDA_PATH", ""))
  nvcc = cuda_path / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
  if not nvcc.exists():
    found = shutil.which("nvcc")
    nvcc = Path(found) if found else nvcc
  info = {
    "path": str(nvcc),
    "exists": nvcc.exists(),
    "version": "",
    "has_compute_120": False,
    "has_sm_120": False,
  }
  if not nvcc.exists():
    return info
  try:
    version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=15, check=False)
    arch = subprocess.run([str(nvcc), "--list-gpu-arch"], capture_output=True, text=True, timeout=15, check=False)
    code = subprocess.run([str(nvcc), "--list-gpu-code"], capture_output=True, text=True, timeout=15, check=False)
  except Exception as exc:
    info["version"] = repr(exc)
    return info
  info["version"] = version.stdout.strip()
  info["has_compute_120"] = "compute_120" in arch.stdout
  info["has_sm_120"] = "sm_120" in code.stdout
  return info


class TrtVisionRunner:
  def __init__(self, args, runtime: trt.Runtime):
    self.args = args
    self.vision_engine_path = args.vision_engine or (args.artifact_dir / "vision_static_fp16" / "qwen_vision_full168_static_fp16.engine")
    self.vision_engine, self.vision_ctx = _load_engine(runtime, self.vision_engine_path)
    self.vision_stream = torch.cuda.Stream()
    self.vision_in_shape = _trt_shape(self.vision_engine, "pixel_values")
    self.vision_out_shape = _trt_shape(self.vision_engine, "image_features")
    self.vision_out = torch.empty(self.vision_out_shape, device="cuda", dtype=torch.float16)
    self.vision_ctx.set_tensor_address("image_features", self.vision_out.data_ptr())

  def run(self, inputs, label_count: int) -> tuple[torch.Tensor, float]:
    rows_per_image = inputs.pixel_values.shape[0] // label_count
    pixel_one = inputs.pixel_values[:rows_per_image].contiguous()
    if tuple(pixel_one.shape) != self.vision_in_shape:
      raise RuntimeError(f"vision input shape {tuple(pixel_one.shape)} does not match engine {self.vision_in_shape}")
    self.vision_ctx.set_tensor_address("pixel_values", pixel_one.data_ptr())
    start = time.perf_counter()
    with torch.cuda.stream(self.vision_stream):
      if not self.vision_ctx.execute_async_v3(self.vision_stream.cuda_stream):
        raise RuntimeError("vision TensorRT execute_async_v3 failed")
    self.vision_stream.synchronize()
    return self.vision_out, (time.perf_counter() - start) * 1000.0


class TrtTextRunner:
  def __init__(self, args, labels: Sequence[str], runtime: trt.Runtime, yes_count: int, no_count: int, *, require_keyed: bool = False):
    self.args = args
    self.labels = tuple(labels)
    self.text_engine_path = _resolve_text_engine_path(args, self.labels, require_keyed=require_keyed)
    self.text_engine, self.text_ctx = _load_engine(runtime, self.text_engine_path)
    self.text_stream = torch.cuda.Stream()
    self.text_embed_shape = _trt_shape(self.text_engine, "inputs_embeds")
    self.text_position_shape = _trt_shape(self.text_engine, "position_ids")
    self.text_out_shape = _trt_shape(self.text_engine, "selected_logits")
    self.text_out = torch.empty(self.text_out_shape, device="cuda", dtype=torch.float16)
    self.text_ctx.set_tensor_address("selected_logits", self.text_out.data_ptr())
    if self.text_out_shape[0] != len(self.labels):
      raise RuntimeError(f"text engine label batch {self.text_out_shape[0]} does not match labels {len(self.labels)}")
    if self.text_out_shape[-1] != yes_count + no_count:
      raise RuntimeError(
        f"text engine selected logit width {self.text_out_shape[-1]} does not match yes/no ids "
        f"{yes_count + no_count}"
      )

  def run(
    self,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    yes_count: int,
    labels: Sequence[str] | None = None,
  ) -> tuple[dict[str, float], float]:
    output_labels = tuple(labels) if labels is not None else self.labels
    if len(output_labels) != self.text_out_shape[0]:
      raise RuntimeError(f"text runner output batch {self.text_out_shape[0]} does not match labels {len(output_labels)}")
    if tuple(inputs_embeds.shape) != self.text_embed_shape:
      raise RuntimeError(f"text inputs_embeds shape {tuple(inputs_embeds.shape)} does not match engine {self.text_embed_shape}")
    if tuple(position_ids.shape) != self.text_position_shape:
      raise RuntimeError(f"text position_ids shape {tuple(position_ids.shape)} does not match engine {self.text_position_shape}")
    self.text_ctx.set_tensor_address("inputs_embeds", inputs_embeds.data_ptr())
    self.text_ctx.set_tensor_address("position_ids", position_ids.data_ptr())
    start = time.perf_counter()
    with torch.cuda.stream(self.text_stream):
      if not self.text_ctx.execute_async_v3(self.text_stream.cuda_stream):
        raise RuntimeError("text TensorRT execute_async_v3 failed")
    self.text_stream.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logits = self.text_out[:, 0, :]
    yes = logits[:, :yes_count].max(dim=1).values
    no = logits[:, yes_count:].max(dim=1).values
    raw_scores = (yes - no).detach().cpu().tolist()
    return {label: float(score) for label, score in zip(output_labels, raw_scores, strict=True)}, elapsed_ms


class QwenTrtLabelScorer:
  def __init__(
    self,
    args,
    labels: Sequence[str] | None = None,
    *,
    processor=None,
    model=None,
    runtime: trt.Runtime | None = None,
    vision_runner: TrtVisionRunner | None = None,
    text_runner: TrtTextRunner | None = None,
    require_keyed_text_engine: bool = False,
  ):
    self.args = args
    self.labels = tuple(labels) if labels is not None else _score_labels(args.score_labels)
    self.processor = processor
    self.model = model
    if self.processor is None or self.model is None:
      self.processor, self.model = _load_qwen(args.model_dir)
    self.yes_ids, self.no_ids = _score_label_ids(self.processor)
    self._runtime_owner = runtime is None
    if runtime is None:
      logger = trt.Logger(trt.Logger.WARNING)
      runtime = trt.Runtime(logger)
    self.runtime = runtime
    self.vision_runner = vision_runner or TrtVisionRunner(args, runtime)
    self.text_runner = text_runner or TrtTextRunner(
        args,
        self.labels,
        runtime,
        len(self.yes_ids),
        len(self.no_ids),
        require_keyed=require_keyed_text_engine,
      )
    self.text_engine_path = self.text_runner.text_engine_path
    self.vision_engine_path = self.vision_runner.vision_engine_path

  def warmup(self, count: int) -> None:
    warm = Image.open(self.args.image).convert("RGB") if self.args.image.exists() else Image.new("RGB", (384, 216), (20, 20, 20))
    for _ in range(max(0, count)):
      self.score(warm, self.args.vehicle_state)

  def score(self, image: Image.Image, vehicle_state: str) -> dict:
    parts: dict[str, float] = {}
    wall_start = time.perf_counter()

    start = time.perf_counter()
    inputs = _build_inputs(
      self.processor,
      image,
      self.labels,
      self.args.image_mode,
      self.args.image_size,
      vehicle_state,
      self.args.text_seq_len,
    )
    torch.cuda.synchronize()
    parts["processor_ms"] = (time.perf_counter() - start) * 1000.0

    vision_out, parts["trt_vision_ms"] = self.vision_runner.run(inputs, len(self.labels))

    with torch.no_grad():
      qwen = self.model.model
      start = time.perf_counter()
      inputs_embeds = qwen.get_input_embeddings()(inputs.input_ids)
      torch.cuda.synchronize()
      parts["embed_ms"] = (time.perf_counter() - start) * 1000.0

      start = time.perf_counter()
      image_features = vision_out.repeat((len(self.labels), 1)).contiguous()
      image_mask, _ = qwen.get_placeholder_mask(inputs.input_ids, inputs_embeds=inputs_embeds, image_features=image_features)
      inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features).contiguous()
      torch.cuda.synchronize()
      parts["scatter_ms"] = (time.perf_counter() - start) * 1000.0

      start = time.perf_counter()
      position_ids, _ = qwen.get_rope_index(
        inputs.input_ids,
        inputs.image_grid_thw,
        None,
        second_per_grid_ts=None,
        attention_mask=inputs.attention_mask,
      )
      position_ids = position_ids.contiguous()
      torch.cuda.synchronize()
      parts["rope_ms"] = (time.perf_counter() - start) * 1000.0

    scores, parts["trt_text_ms"] = self.text_runner.run(inputs_embeds, position_ids, len(self.yes_ids), self.labels)
    selected = [
      label for label in self.labels
      if scores[label] >= self.args.score_thresholds_map.get(label, self.args.score_threshold)
    ]
    labels = tuple(selected) if selected else ("none",)
    labels = _with_visual_fallbacks(image, labels)
    rtp_text = _labels_to_rtp(labels)
    total_ms = (time.perf_counter() - wall_start) * 1000.0
    parts["total_ms"] = total_ms
    return {
      "text": rtp_text,
      "rtp_text": rtp_text,
      "labels_text": ",".join(labels),
      "labels": list(labels),
      "label_mode": "score",
      "image_mode": self.args.image_mode,
      "label_scores": scores,
      "generated_token_count": 0,
      "prefill_ms": parts["processor_ms"] + parts["trt_vision_ms"] + parts["embed_ms"] + parts["scatter_ms"] + parts["rope_ms"],
      "decode_ms": parts["trt_text_ms"],
      "total_ms": total_ms,
      "timings_ms": parts,
      "backend": f"qwen2.5-vl-3b-trt-nvfp4-{self.args.image_mode}{self.args.image_size}-score",
    }


class QwenTrtRotatingLabelScorer:
  def __init__(self, args, groups: Sequence[Sequence[str]]):
    self.args = args
    self.groups = tuple(tuple(group) for group in groups)
    if getattr(args, "require_manifest", False):
      _enforce_manifest(args, self.groups)
    self.processor, self.model = _load_qwen(args.model_dir)
    logger = trt.Logger(trt.Logger.WARNING)
    self.runtime = trt.Runtime(logger)
    self.vision_runner = TrtVisionRunner(args, self.runtime)
    shared_text_runner = None
    if args.score_rotate_shared_engine:
      shared_text_runner = TrtTextRunner(
        args,
        self.groups[0],
        self.runtime,
        *[len(ids) for ids in _score_label_ids(self.processor)],
        require_keyed=False,
      )
    scorer_cache: dict[tuple[str, ...], QwenTrtLabelScorer] = {}
    self.scorers = []
    for group in self.groups:
      if group not in scorer_cache:
        scorer_cache[group] = QwenTrtLabelScorer(
          args,
          group,
          processor=self.processor,
          model=self.model,
          runtime=self.runtime,
          vision_runner=self.vision_runner,
          text_runner=shared_text_runner,
          require_keyed_text_engine=not args.score_rotate_shared_engine,
        )
      self.scorers.append(scorer_cache[group])
    durable_labels = tuple(label.strip() for label in args.score_durable_labels.split(",") if label.strip())
    self.rotating_state = RotatingScoreState(
      self.groups,
      args.score_cache_ttl_frames,
      durable_labels or DEFAULT_DURABLE_SCORE_LABELS,
      args.score_negative_clear_threshold,
    )

  def warmup(self, count: int) -> None:
    warm = Image.open(self.args.image).convert("RGB") if self.args.image.exists() else Image.new("RGB", (384, 216), (20, 20, 20))
    for idx, scorer in enumerate(self.scorers):
      for _ in range(max(0, count)):
        scorer.score(warm, self.args.vehicle_state)

  def score(self, image: Image.Image, vehicle_state: str, frame_id: int) -> dict:
    group_idx, request_labels = self.rotating_state.next_group()
    scorer = self.scorers[group_idx]
    response = scorer.score(image, vehicle_state)
    cached_labels = self.rotating_state.update(request_labels, response["labels"], response["label_scores"], frame_id)
    rtp_text = _labels_to_rtp(cached_labels)
    response["labels_scored_this_request"] = list(request_labels)
    response["score_group_index"] = group_idx
    response["labels_current_group"] = response["labels"]
    response["labels"] = list(cached_labels)
    response["label_scores_cached"] = self.rotating_state.active_scores(frame_id)
    response["rtp_text"] = rtp_text
    response["text"] = rtp_text
    response["backend"] = f"{response['backend']}-rotating"
    return response


def build_text_group_engines(args) -> dict:
  groups = _score_groups(args.score_label_groups)
  unique_groups = tuple(dict.fromkeys(groups))
  results = []
  for group in unique_groups:
    group_args = argparse.Namespace(**vars(args))
    group_args.score_labels = ",".join(group)
    group_args.label_keyed_text_engine = True
    engine_path = _keyed_text_engine_path(group_args, group)
    onnx_path = _keyed_text_onnx_path(group_args, group)
    if engine_path.exists():
      results.append({
        "kind": "text_nvfp4",
        "labels": group,
        "image_mode": args.image_mode,
        "image_size": args.image_size,
        "engine": str(engine_path),
        "onnx": str(onnx_path),
        "skipped_existing": True,
      })
    else:
      results.append(build_text_engine(group_args))
  return {
    "kind": "text_nvfp4_groups",
    "groups": groups,
    "unique_groups": unique_groups,
    "results": results,
  }


def benchmark(args) -> dict:
  if args.require_manifest:
    _enforce_manifest(args, (_score_labels(args.score_labels),))
  image = Image.open(args.image).convert("RGB")
  scorer = QwenTrtLabelScorer(args)

  for _ in range(args.warmup):
    scorer.score(image, args.vehicle_state)

  rows: list[dict[str, float]] = []
  response: dict | None = None
  for _ in range(args.iters):
    response = scorer.score(image, args.vehicle_state)
    rows.append(response["timings_ms"])

  return {
    "kind": "benchmark",
    "labels": scorer.labels,
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "iters": args.iters,
    "text_engine": str(scorer.text_engine_path),
    "vision_engine": str(scorer.vision_engine_path),
    "stage_summary": _summarize_timing_rows(rows),
    "scores": {} if response is None else response["label_scores"],
  }


def benchmark_groups(args) -> dict:
  image = Image.open(args.image).convert("RGB")
  groups = _score_groups(args.score_label_groups)
  scorer = QwenTrtRotatingLabelScorer(args, groups)
  scorer.warmup(args.warmup)

  rows: list[dict[str, float]] = []
  group_rows: dict[int, list[dict[str, float]]] = {idx: [] for idx in range(len(groups))}
  responses: list[dict] = []
  for idx in range(args.iters):
    response = scorer.score(image, args.vehicle_state, idx)
    responses.append(response)
    timings = response["timings_ms"]
    rows.append(timings)
    group_rows[int(response["score_group_index"])].append(timings)

  return {
    "kind": "benchmark_groups",
    "groups": groups,
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "iters": args.iters,
    "vision_engine": str(scorer.vision_runner.vision_engine_path),
    "text_engines": [str(group_scorer.text_engine_path) for group_scorer in scorer.scorers],
    "stage_summary": _summarize_timing_rows(rows),
    "group_stage_summary": {str(idx): _summarize_timing_rows(group_rows[idx]) for idx in group_rows},
    "last_response": responses[-1] if responses else {},
  }


def check_artifacts(args) -> dict:
  issues: list[str] = []
  groups = _score_groups(args.score_label_groups)
  first_group = groups[0]
  expected_label_count = len(first_group)
  if args.score_rotate_shared_engine:
    for group in groups:
      if len(group) != expected_label_count:
        issues.append(f"shared text engine requires equal group sizes: {groups}")

  model_dir_info = _file_info(args.model_dir)
  if not args.model_dir.exists():
    issues.append(f"missing model dir: {args.model_dir}")

  logger = trt.Logger(trt.Logger.WARNING)
  runtime = trt.Runtime(logger)
  vision_path = args.vision_engine or (args.artifact_dir / "vision_static_fp16" / "qwen_vision_full168_static_fp16.engine")
  text_path = _resolve_text_engine_path(args, first_group, require_keyed=not args.score_rotate_shared_engine)
  vision_info, vision_issues = _engine_info(runtime, vision_path)
  text_info, text_issues = _engine_info(runtime, text_path)
  issues.extend(vision_issues)
  issues.extend(text_issues)

  if vision_info.get("deserialized"):
    if args.image_mode == "full" and args.image_size == 168:
      _check_shape(vision_info, "pixel_values", (96, 1176), issues, "vision")
      _check_shape(vision_info, "image_features", (24, 2048), issues, "vision")
  if text_info.get("deserialized"):
    if args.text_seq_len > 0:
      _check_shape(text_info, "inputs_embeds", (expected_label_count, args.text_seq_len, 2048), issues, "text")
      _check_shape(text_info, "position_ids", (3, expected_label_count, args.text_seq_len), issues, "text")
    selected_shape = tuple(text_info.get("tensors", {}).get("selected_logits", {}).get("shape", ()))
    if selected_shape and selected_shape[0] != expected_label_count:
      issues.append(f"text selected_logits batch {selected_shape[0]} != expected label count {expected_label_count}")

  nvcc = _nvcc_info()
  if not nvcc["exists"]:
    issues.append("nvcc not found")
  if not nvcc["has_compute_120"]:
    issues.append("nvcc does not report compute_120")
  if not nvcc["has_sm_120"]:
    issues.append("nvcc does not report sm_120")

  cuda = {
    "cuda_path": os.environ.get("CUDA_PATH", ""),
    "cuda_home": os.environ.get("CUDA_HOME", ""),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    "torch_device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (),
    "tensorrt_version": trt.__version__,
    "tensorrt_fp4": hasattr(trt.DataType, "FP4") and hasattr(trt.BuilderFlag, "FP4"),
    "nvcc": nvcc,
  }
  if cuda["torch_device_capability"] and tuple(cuda["torch_device_capability"]) != (12, 0):
    issues.append(f"unexpected CUDA device capability: {cuda['torch_device_capability']}")
  if not cuda["tensorrt_fp4"]:
    issues.append("TensorRT FP4 support is unavailable")

  contract = _runtime_contract(args, groups)
  manifest = _validate_manifest(args, groups)
  if args.require_manifest:
    issues.extend(manifest["issues"])

  result = {
    "kind": "check_artifacts",
    "ok": not issues,
    "issues": issues,
    "model_dir": model_dir_info,
    "artifact_dir": str(args.artifact_dir),
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "text_seq_len": args.text_seq_len,
    "score_label_groups": groups,
    "score_rotate_shared_engine": args.score_rotate_shared_engine,
    "expected_label_count": expected_label_count,
    "vision_engine": vision_info,
    "text_engine": text_info,
    "cuda": cuda,
    "runtime_contract": {
      "contract_sha256": contract["contract_sha256"],
      "manifest_version": MANIFEST_VERSION,
    },
    "manifest": manifest,
  }
  if args.write_manifest and not issues:
    result["manifest"] = _write_manifest(args, groups, result)
  return result


def gate(args) -> dict:
  if args.iters <= 0:
    raise ValueError("gate requires --iters > 0")
  gate_args = argparse.Namespace(**vars(args))
  gate_args.require_manifest = True
  artifact_check = check_artifacts(gate_args)
  if not artifact_check["ok"]:
    return {
      "kind": "gate",
      "ok": False,
      "deadline_ms": args.deadline_ms,
      "artifact_check": artifact_check,
      "benchmark": None,
      "issues": ["artifact or manifest validation failed"],
    }

  benchmark_result = benchmark_groups(gate_args)
  total = benchmark_result["stage_summary"].get("total_ms", {})
  p99 = float(total.get("p99", 0.0))
  max_latency = float(total.get("max", 0.0))
  issues = []
  if p99 > args.deadline_ms:
    issues.append(f"p99 total latency {p99:.3f} ms exceeds deadline {args.deadline_ms:.3f} ms")
  if max_latency > args.deadline_ms:
    issues.append(f"max total latency {max_latency:.3f} ms exceeds deadline {args.deadline_ms:.3f} ms")
  return {
    "kind": "gate",
    "ok": not issues,
    "deadline_ms": args.deadline_ms,
    "p99_total_ms": p99,
    "max_total_ms": max_latency,
    "issues": issues,
    "artifact_check": {
      "ok": artifact_check["ok"],
      "issues": artifact_check["issues"],
      "manifest": artifact_check["manifest"],
      "runtime_contract": artifact_check["runtime_contract"],
    },
    "benchmark": benchmark_result,
  }


def serve(args) -> None:
  if args.require_manifest:
    _enforce_manifest(args, _groups_for_runtime(args))
  if args.score_rotate_groups:
    scorer = QwenTrtRotatingLabelScorer(args, _score_groups(args.score_label_groups))
  else:
    scorer = QwenTrtLabelScorer(args)
  scorer.warmup(args.warmup)
  if args.ready_jsonl:
    print(json.dumps({"ready": True}, separators=(",", ":")), flush=True)
  for line in sys.stdin:
    try:
      payload = json.loads(line)
      image = _image_from_payload(payload)
      vehicle_state = args.vehicle_state
      if args.use_payload_vehicle_state:
        vehicle_state = str(payload.get("scene_board_state_text", args.vehicle_state))
      frame_id = int(payload.get("frame_id", 0))
      if args.score_rotate_groups:
        response = scorer.score(image, vehicle_state, frame_id)
      else:
        response = scorer.score(image, vehicle_state)
      response["frame_id"] = frame_id
      response["source_frame_id"] = response["frame_id"]
    except Exception as exc:
      response = {"error": repr(exc)}
    print(json.dumps(response, separators=(",", ":")), flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(description="Build and benchmark fixed-shape Qwen2.5-VL TensorRT label scoring engines.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--artifact-dir", type=Path, default=Path(os.environ.get("QWEN_TRT_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)))
  parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
  parser.add_argument("--image-mode", choices=("full",), default="full")
  parser.add_argument("--image-size", type=int, default=168)
  parser.add_argument("--text-seq-len", type=int, default=0, help="Fixed processor max_length for shared-shape text engines. 0 keeps dynamic padding.")
  parser.add_argument("--score-labels", default="construction_left,construction_right")
  parser.add_argument("--score-label-groups", default=";".join(",".join(group) for group in DEFAULT_SCORE_LABEL_GROUPS))
  parser.add_argument("--score-rotate-groups", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--score-rotate-shared-engine", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--score-cache-ttl-frames", type=int, default=60)
  parser.add_argument("--score-durable-labels", default=",".join(DEFAULT_DURABLE_SCORE_LABELS))
  parser.add_argument("--score-negative-clear-threshold", type=float, default=2.0)
  parser.add_argument("--vehicle-state", default="speed=5.0 mps")
  parser.add_argument("--workspace-gb", type=int, default=6)
  parser.add_argument("--text-engine", type=Path, default=None)
  parser.add_argument("--vision-engine", type=Path, default=None)
  parser.add_argument("--score-threshold", type=float, default=0.0)
  parser.add_argument("--score-thresholds", default="")
  parser.add_argument(
    "--use-payload-vehicle-state",
    action="store_true",
    help="Use scene_board_state_text from each JSONL request. This requires text engine shapes to match that state text.",
  )
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--ready-jsonl", action="store_true", help="Emit a one-line JSON ready marker on stdout after loading and warmup.")
  parser.add_argument("--iters", type=int, default=60)
  parser.add_argument("--deadline-ms", type=float, default=50.0, help="Latency deadline for the gate subcommand.")
  parser.add_argument("--out", type=Path, default=None, help="Optional JSON path for the command result.")
  parser.add_argument("--manifest", type=Path, default=None, help="Runtime manifest path. Defaults to artifact-dir/qwen_trt_runtime_manifest.json.")
  parser.add_argument("--write-manifest", action="store_true", help="Write the runtime contract manifest from check-artifacts when validation succeeds.")
  parser.add_argument("--require-manifest", action="store_true", help="Reject runtime commands if the manifest contract does not match current args/code/model metadata.")
  parser.set_defaults(label_keyed_text_engine=False)
  sub = parser.add_subparsers(dest="cmd", required=True)
  sub.add_parser("build-text")
  sub.add_parser("build-text-groups")
  sub.add_parser("build-vision")
  sub.add_parser("benchmark")
  sub.add_parser("benchmark-groups")
  sub.add_parser("check-artifacts")
  sub.add_parser("gate")
  sub.add_parser("serve")
  sub.add_parser("all")
  args = parser.parse_args()
  args.score_thresholds_map = _parse_score_threshold_map(args.score_thresholds)

  args.artifact_dir.mkdir(parents=True, exist_ok=True)
  if args.cmd == "build-text":
    result = build_text_engine(args)
  elif args.cmd == "build-text-groups":
    result = build_text_group_engines(args)
  elif args.cmd == "build-vision":
    result = build_vision_engine(args)
  elif args.cmd == "benchmark":
    result = benchmark(args)
  elif args.cmd == "benchmark-groups":
    result = benchmark_groups(args)
  elif args.cmd == "check-artifacts":
    result = check_artifacts(args)
  elif args.cmd == "gate":
    result = gate(args)
  elif args.cmd == "serve":
    serve(args)
    return
  elif args.cmd == "all":
    result = {
      "text": build_text_engine(args),
      "vision": build_vision_engine(args),
      "benchmark": benchmark(args),
    }
  else:
    raise AssertionError(args.cmd)

  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps(result, indent=2))
  if args.cmd in ("check-artifacts", "gate") and not result.get("ok", False):
    sys.exit(2)


if __name__ == "__main__":
  main()
