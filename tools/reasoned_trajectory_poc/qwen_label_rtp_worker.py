#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "vlm" / "qwen2_5_vl_3b_instruct"

LABEL_PROMPT = """You are inspecting a driving scene board for control-relevant hazards.
You receive multiple views of the same scene:
image 1: full driver UI overlay scene board.
image 2: zoomed center planned corridor.
image 3: zoomed lower/mid road region.
image 4: zoomed forward signal/object region.

Output exactly a comma-separated subset of these labels:
cones,barrier,construction_left,construction_right,pedestrian_in_path,pedestrian_entering_path,vehicle_in_path,vehicle_entering_path,animal_in_path,animal_entering_path,red_stop_light,stop_sign,none

Rules:
- Include cones if traffic cones, blue cones, construction cones, pylons, bollards, or cone-shaped lane markers are visible.
- Include barrier if a road barrier, barricade, blue-white checker barrier, construction barrier, or blocked-lane panel is visible.
- Include construction_left if cones, barriers, pylons, bollards, or blocked-lane panels are mainly on the left side of the green planned path.
- Include construction_right if cones, barriers, pylons, bollards, or blocked-lane panels are mainly on the right side of the green planned path.
- Include pedestrian_in_path only for a visible upright human figure, person, pedestrian, or human body currently overlapping the green planned path or standing/walking directly inside the ego lane corridor ahead. Count small, partially transparent, or green-tinted people if the overlay covers them.
- Include pedestrian_entering_path only for a visible upright human figure, person, pedestrian, or human body beside the green path that is clearly crossing, walking, stepping, or moving into the green planned path soon.
- Include vehicle_in_path only if a car, truck, bicycle, motorcycle, or similar road user is currently in the planned path of travel.
- Include vehicle_entering_path only if such a road user is visibly about to move into the planned path of travel.
- Include animal_in_path only if an animal is currently in the planned path of travel.
- Include animal_entering_path only if an animal is visibly about to move into the planned path of travel.
- Pedestrian, vehicle, and animal path-conflict labels mean slow or stop in lane. They must not imply steering around the agent.
- Construction labels are the only labels that may imply a lateral path bias.
- Do not output any path-conflict label for cones, barriers, poles, signs, shadows, lane markings, sidewalks, or uncertain dark blobs.
- Do not output pedestrian labels for cones, barriers, signs, lane markers, traffic posts, poles, route arrows, UI text, or shadows.
- Do not yield for a person, vehicle, or animal that is visible but clearly outside the path of travel and not moving into it.
- Include red_stop_light if a red stop light or red/blue stop cue is visible ahead.
- Include stop_sign if a STOP sign or stop billboard is visible.
- Output none only if no listed hazard is visible.
- No prose."""

COMPOSITE_LABEL_PROMPT = """You are inspecting one composite driving scene board for control-relevant hazards.
The image is a 2x2 composite:
top-left: full driver UI overlay scene board.
top-right: zoomed center planned corridor.
bottom-left: zoomed lower/mid road region.
bottom-right: zoomed forward signal/object region.

Output exactly a comma-separated subset of these labels:
cones,barrier,construction_left,construction_right,pedestrian_in_path,pedestrian_entering_path,vehicle_in_path,vehicle_entering_path,animal_in_path,animal_entering_path,red_stop_light,stop_sign,none

Rules:
- Include cones if traffic cones, blue cones, construction cones, pylons, bollards, or cone-shaped lane markers are visible.
- Include barrier if a road barrier, barricade, blue-white checker barrier, construction barrier, or blocked-lane panel is visible.
- Include construction_left if cones, barriers, pylons, bollards, or blocked-lane panels are mainly on the left side of the green planned path.
- Include construction_right if cones, barriers, pylons, bollards, or blocked-lane panels are mainly on the right side of the green planned path.
- Include pedestrian_in_path only for a visible upright human figure, person, pedestrian, or human body currently overlapping the green planned path or standing/walking directly inside the ego lane corridor ahead. Count small, partially transparent, or green-tinted people if the overlay covers them.
- Include pedestrian_entering_path only for a visible upright human figure, person, pedestrian, or human body beside the green path that is clearly crossing, walking, stepping, or moving into the green planned path soon.
- Include vehicle_in_path only if a car, truck, bicycle, motorcycle, or similar road user is currently in the planned path of travel.
- Include vehicle_entering_path only if such a road user is visibly about to move into the planned path of travel.
- Include animal_in_path only if an animal is currently in the planned path of travel.
- Include animal_entering_path only if an animal is visibly about to move into the planned path of travel.
- Pedestrian, vehicle, and animal path-conflict labels mean slow or stop in lane. They must not imply steering around the agent.
- Construction labels are the only labels that may imply a lateral path bias.
- Do not output any path-conflict label for cones, barriers, poles, signs, shadows, lane markings, sidewalks, or uncertain dark blobs.
- Do not output pedestrian labels for cones, barriers, signs, lane markers, traffic posts, poles, route arrows, UI text, or shadows.
- Do not yield for a person, vehicle, or animal that is visible but clearly outside the path of travel and not moving into it.
- Include red_stop_light if a red stop light or red/blue stop cue is visible ahead.
- Include stop_sign if a STOP sign or stop billboard is visible.
- Output none only if no listed hazard is visible.
- No prose."""

FULL_LABEL_PROMPT = """You are inspecting one full driver UI overlay scene board for control-relevant hazards.
The green overlay is the planned path of travel.

Output exactly a comma-separated subset of these labels:
cones,barrier,construction_left,construction_right,pedestrian_in_path,pedestrian_entering_path,vehicle_in_path,vehicle_entering_path,animal_in_path,animal_entering_path,red_stop_light,stop_sign,none

Rules:
- Include cones if traffic cones, blue cones, construction cones, pylons, bollards, or cone-shaped lane markers are visible.
- Include barrier if a road barrier, barricade, blue-white checker barrier, construction barrier, or blocked-lane panel is visible.
- Include construction_left if cones, barriers, pylons, bollards, or blocked-lane panels are mainly on the left side of the green planned path.
- Include construction_right if cones, barriers, pylons, bollards, or blocked-lane panels are mainly on the right side of the green planned path.
- Include pedestrian_in_path only for a visible upright human figure, person, pedestrian, or human body currently overlapping the green planned path or standing/walking directly inside the ego lane corridor ahead. Count small, partially transparent, or green-tinted people if the overlay covers them.
- Include pedestrian_entering_path only for a visible upright human figure, person, pedestrian, or human body beside the green path that is clearly crossing, walking, stepping, or moving into the green planned path soon.
- Include vehicle_in_path only if a car, truck, bicycle, motorcycle, or similar road user is currently in the planned path of travel.
- Include vehicle_entering_path only if such a road user is visibly about to move into the planned path of travel.
- Include animal_in_path only if an animal is currently in the planned path of travel.
- Include animal_entering_path only if an animal is visibly about to move into the planned path of travel.
- Pedestrian, vehicle, and animal path-conflict labels mean slow or stop in lane. They must not imply steering around the agent.
- Construction labels are the only labels that may imply a lateral path bias.
- Do not output any path-conflict label for cones, barriers, poles, signs, shadows, lane markings, sidewalks, or uncertain dark blobs.
- Do not output pedestrian labels for cones, barriers, signs, lane markers, traffic posts, poles, route arrows, UI text, or shadows.
- Do not yield for a person, vehicle, or animal that is visible but clearly outside the path of travel and not moving into it.
- Include red_stop_light if a red stop light or red/blue stop cue is visible ahead.
- Include stop_sign if a STOP sign or stop billboard is visible.
- Output none only if no listed hazard is visible.
- No prose."""

SCORE_PROMPT = """Score one driving-scene label from the image.
The green overlay is the ego planned path.
Answer exactly yes or no.
Answer yes only when the question condition is directly visible.
For pedestrian/vehicle/animal path questions, the agent must overlap the green path or clearly be entering it.
For construction questions, visible blue cones, pylons, bollards, barricades, checker panels, and blocked-lane panels count. Left and right are relative to the green planned path in the image.
Do not treat UI text, lane lines, route arrows, shadows, poles, or signs as pedestrians."""

PATH_CONFLICT_LABELS = {
  "pedestrian_in_path",
  "pedestrian_entering_path",
  "vehicle_in_path",
  "vehicle_entering_path",
  "animal_in_path",
  "animal_entering_path",
}
SCORE_LABELS = (
  "cones",
  "barrier",
  "construction_left",
  "construction_right",
  "pedestrian_in_path",
  "pedestrian_entering_path",
  "vehicle_in_path",
  "vehicle_entering_path",
  "animal_in_path",
  "animal_entering_path",
  "red_stop_light",
  "stop_sign",
)
DEFAULT_SCORE_LABEL_GROUPS = (
  ("pedestrian_in_path", "pedestrian_entering_path"),
  ("construction_left", "construction_right"),
  ("pedestrian_in_path", "pedestrian_entering_path"),
  ("vehicle_in_path", "vehicle_entering_path"),
)
DEFAULT_DURABLE_SCORE_LABELS = SCORE_LABELS
EXCLUSIVE_LABEL_GROUPS = (
  frozenset(("construction_left", "construction_right")),
)
SCORE_QUESTIONS = {
  "cones": "Are any traffic cones, blue cones, construction cones, pylons, bollards, or cone-shaped lane markers visible?",
  "barrier": "Is any road barrier, barricade, blue-white checker barrier, construction barrier, or blocked-lane panel visible?",
  "construction_left": "Are cones, blue cones, construction cones, pylons, bollards, road barriers, barricades, checker panels, or blocked-lane panels mainly on the left side of the green planned path?",
  "construction_right": "Are cones, blue cones, construction cones, pylons, bollards, road barriers, barricades, checker panels, or blocked-lane panels mainly on the right side of the green planned path?",
  "pedestrian_in_path": "Is any visible upright human figure, person, pedestrian, or human body partly or fully inside the green planned path or directly blocking the ego lane ahead, even if small or partially transparent?",
  "pedestrian_entering_path": "Is any visible upright human figure, person, pedestrian, or human body next to the green planned path and clearly crossing, walking, stepping, or moving into that green path soon?",
  "vehicle_in_path": "Is a car, truck, bicycle, motorcycle, or similar road user currently overlapping the green planned path or directly blocking the ego lane corridor ahead?",
  "vehicle_entering_path": "Is a car, truck, bicycle, motorcycle, or similar road user beside the green planned path and clearly moving into the ego lane corridor soon?",
  "animal_in_path": "Is an animal currently overlapping the green planned path or directly blocking the ego lane corridor ahead?",
  "animal_entering_path": "Is an animal beside the green planned path and clearly moving into the ego lane corridor soon?",
  "red_stop_light": "Is a red stop light or red/blue stop cue visible ahead?",
  "stop_sign": "Is a STOP sign or stop billboard visible ahead?",
}


class RotatingScoreState:
  def __init__(
    self,
    groups: Sequence[Sequence[str]],
    cache_ttl_frames: int,
    durable_labels: Sequence[str] = DEFAULT_DURABLE_SCORE_LABELS,
    negative_clear_threshold: float = 2.0,
  ):
    self.groups = tuple(tuple(group) for group in groups)
    self.cache_ttl_frames = max(0, cache_ttl_frames)
    self.durable_labels = frozenset(durable_labels)
    self.negative_clear_threshold = max(0.0, negative_clear_threshold)
    self.next_group_idx = 0
    self._positive_frame: dict[str, int] = {}
    self._scores: dict[str, float] = {}

  def next_group(self) -> tuple[int, tuple[str, ...]]:
    if not self.groups:
      return 0, ()
    idx = self.next_group_idx
    self.next_group_idx = (self.next_group_idx + 1) % len(self.groups)
    return idx, self.groups[idx]

  def update(self, group: Sequence[str], labels: Sequence[str], scores: dict[str, float], frame_id: int) -> tuple[str, ...]:
    label_set = _resolve_exclusive_labels(set(labels), scores)
    for label in group:
      if label in label_set:
        self._clear_exclusive_conflicts(label)
        self._positive_frame[label] = frame_id
        if label in scores:
          self._scores[label] = scores[label]
      else:
        self._clear_or_hold_negative(label, scores, frame_id)
    return self.active_labels(frame_id)

  def _clear_exclusive_conflicts(self, label: str) -> None:
    for group in EXCLUSIVE_LABEL_GROUPS:
      if label not in group:
        continue
      for other in group:
        if other != label:
          self._positive_frame.pop(other, None)
          self._scores.pop(other, None)
      return

  def _clear_or_hold_negative(self, label: str, scores: dict[str, float], frame_id: int) -> None:
    if label in self.durable_labels:
      last_frame = self._positive_frame.get(label)
      score = scores.get(label)
      if (
        last_frame is not None and
        frame_id - last_frame <= self.cache_ttl_frames and
        score is not None and
        score > -self.negative_clear_threshold
      ):
        # Positive scene evidence should survive a brief occlusion or splash.
        # It still expires by TTL, or clears on a strong negative score.
        return
    self._positive_frame.pop(label, None)
    self._scores.pop(label, None)

  def active_labels(self, frame_id: int) -> tuple[str, ...]:
    active = []
    for label in SCORE_LABELS:
      last_frame = self._positive_frame.get(label)
      if last_frame is None:
        continue
      if frame_id - last_frame <= self.cache_ttl_frames:
        active.append(label)
      else:
        self._positive_frame.pop(label, None)
        self._scores.pop(label, None)
    return tuple(active) if active else ("none",)

  def active_scores(self, frame_id: int) -> dict[str, float]:
    active = set(self.active_labels(frame_id))
    return {label: score for label, score in self._scores.items() if label in active}


def _resolve_exclusive_labels(labels: set[str], scores: dict[str, float]) -> set[str]:
  resolved = set(labels)
  for group in EXCLUSIVE_LABEL_GROUPS:
    active = resolved & group
    if len(active) <= 1:
      continue
    best = max(active, key=lambda label: scores.get(label, float("-inf")))
    resolved.difference_update(group)
    resolved.add(best)
  return resolved


def _load(model_dir: Path):
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


def _image_from_payload(payload: dict) -> Image.Image:
  data = base64.b64decode(payload["scene_board_image_b64"])
  return Image.open(BytesIO(data)).convert("RGB")


def _thumb(image: Image.Image, max_side: int) -> Image.Image:
  out = image.copy()
  out.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
  return out


def _crop_resize(image: Image.Image, box: tuple[int, int, int, int], size: int) -> Image.Image:
  return image.crop(box).resize((size, size), Image.Resampling.BICUBIC)


def _probe_images(image: Image.Image, size: int) -> list[Image.Image]:
  w, h = image.size
  x0 = int(w * 0.29)
  x1 = int(w * 0.71)
  return [
    _thumb(image, size),
    _crop_resize(image, (x0, int(h * 0.22), x1, int(h * 0.82)), size),
    _crop_resize(image, (0, int(h * 0.24), w, int(h * 0.86)), size),
    _crop_resize(image, (int(w * 0.31), int(h * 0.10), int(w * 0.69), int(h * 0.66)), size),
  ]


def _composite_probe_image(image: Image.Image, size: int) -> Image.Image:
  probes = _probe_images(image, max(64, size // 2))
  half = size // 2
  composite = Image.new("RGB", (size, size), (8, 10, 12))
  slots = ((0, 0), (half, 0), (0, half), (half, half))
  labels = ("full", "corridor", "road", "forward")
  for probe, xy, label in zip(probes, slots, labels, strict=True):
    tile = probe.resize((half, half), Image.Resampling.BILINEAR)
    composite.paste(tile, xy)
    draw = ImageDraw.Draw(composite)
    draw.rectangle((xy[0], xy[1], xy[0] + 52, xy[1] + 14), fill=(0, 0, 0))
    draw.text((xy[0] + 3, xy[1] + 2), label, fill=(255, 255, 255))
  draw = ImageDraw.Draw(composite)
  draw.line((half, 0, half, size), fill=(0, 0, 0), width=2)
  draw.line((0, half, size, half), fill=(0, 0, 0), width=2)
  return composite


def _inference_images(image: Image.Image, image_mode: str, size: int) -> tuple[list[Image.Image], str]:
  if image_mode == "full":
    return [_thumb(image, size)], FULL_LABEL_PROMPT
  if image_mode == "composite":
    return [_composite_probe_image(image, size)], COMPOSITE_LABEL_PROMPT
  return _probe_images(image, size), LABEL_PROMPT


def _extract_new_text(processor, inputs, output_ids) -> str:
  trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
  return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def _normalize_labels(raw: str) -> tuple[str, ...]:
  allowed = {
    "cones",
    "barrier",
    "construction_left",
    "construction_right",
    *PATH_CONFLICT_LABELS,
    "red_stop_light",
    "stop_sign",
    "none",
  }
  text = raw.lower()
  text = text.replace("traffic cones", "cones")
  text = text.replace("traffic cone", "cones")
  text = text.replace("barricade", "barrier")
  text = text.replace("construction left", "construction_left")
  text = text.replace("construction on left", "construction_left")
  text = text.replace("construction on the left", "construction_left")
  text = text.replace("cones left", "construction_left")
  text = text.replace("cones on left", "construction_left")
  text = text.replace("cones on the left", "construction_left")
  text = text.replace("barrier left", "construction_left")
  text = text.replace("barrier on left", "construction_left")
  text = text.replace("barrier on the left", "construction_left")
  text = text.replace("construction right", "construction_right")
  text = text.replace("construction on right", "construction_right")
  text = text.replace("construction on the right", "construction_right")
  text = text.replace("cones right", "construction_right")
  text = text.replace("cones on right", "construction_right")
  text = text.replace("cones on the right", "construction_right")
  text = text.replace("barrier right", "construction_right")
  text = text.replace("barrier on right", "construction_right")
  text = text.replace("barrier on the right", "construction_right")
  text = text.replace("pedestrian in path", "pedestrian_in_path")
  text = text.replace("pedestrian entering path", "pedestrian_entering_path")
  text = text.replace("pedestrian entering the path", "pedestrian_entering_path")
  text = text.replace("person in path", "pedestrian_in_path")
  text = text.replace("person entering path", "pedestrian_entering_path")
  text = text.replace("person entering the path", "pedestrian_entering_path")
  text = text.replace("car in path", "vehicle_in_path")
  text = text.replace("car entering path", "vehicle_entering_path")
  text = text.replace("car entering the path", "vehicle_entering_path")
  text = text.replace("vehicle in path", "vehicle_in_path")
  text = text.replace("vehicle entering path", "vehicle_entering_path")
  text = text.replace("vehicle entering the path", "vehicle_entering_path")
  text = text.replace("animal in path", "animal_in_path")
  text = text.replace("animal entering path", "animal_entering_path")
  text = text.replace("animal entering the path", "animal_entering_path")
  text = text.replace("red stop light", "red_stop_light")
  text = text.replace("red light", "red_stop_light")
  text = text.replace("stop sign", "stop_sign")
  aliases = {
    "cone": "cones",
    "cones": "cones",
    "barriers": "barrier",
    "barricades": "barrier",
  }
  found = []
  for token in re.split(r"[^a-z0-9_]+", text):
    token = aliases.get(token, token)
    if token in allowed and token not in found:
      found.append(token)
  if not found:
    return ("none",)
  if len(found) > 1 and "none" in found:
    found.remove("none")
  return tuple(found)


def _with_visual_fallbacks(image: Image.Image, labels: tuple[str, ...]) -> tuple[str, ...]:
  found = [label for label in labels if label != "none"]
  label_set = set(found)
  if "red_stop_light" not in label_set and "stop_sign" not in label_set and _has_forward_stop_cue(image):
    found.append("stop_sign")
    label_set.add("stop_sign")

  return tuple(found) if found else ("none",)


def _has_forward_stop_cue(image: Image.Image) -> bool:
  arr = np.asarray(image.convert("RGB"), dtype=np.int16)
  h, w, _ = arr.shape
  roi = arr[int(h * 0.10):int(h * 0.42), int(w * 0.35):int(w * 0.65)]
  if roi.size == 0:
    return False
  red = (roi[:, :, 0] > 130) & (roi[:, :, 1] < 90) & (roi[:, :, 2] < 90)
  blue = (roi[:, :, 2] > 130) & (roi[:, :, 0] < 90) & (roi[:, :, 1] < 120)
  return int(red.sum() + blue.sum()) >= 12


def _has_center_dark_upright_obstacle(image: Image.Image) -> bool:
  w, h = image.size
  crop = _crop_resize(
    image,
    (int(w * 0.29), int(h * 0.22), int(w * 0.71), int(h * 0.82)),
    384,
  )
  arr = np.asarray(crop.convert("RGB"), dtype=np.int16)
  h, w, _ = arr.shape
  roi = arr[int(h * 0.10):int(h * 0.80), int(w * 0.35):int(w * 0.65)]
  if roi.size == 0:
    return False
  dark = (roi[:, :, 0] < 100) & (roi[:, :, 1] < 110) & (roi[:, :, 2] < 110)
  ys, xs = np.nonzero(dark)
  if len(xs) < 18:
    return False

  # One narrow upright blob in the planned corridor is enough for a cautious yield.
  x0, x1 = int(xs.min()), int(xs.max())
  y0, y1 = int(ys.min()), int(ys.max())
  width = max(1, x1 - x0 + 1)
  height = max(1, y1 - y0 + 1)
  area = int(len(xs))
  aspect = height / width
  fill = area / float(width * height)
  return 2.0 <= aspect <= 12.0 and 40 <= area <= 2200 and width <= 45 and fill >= 0.08


def _score_label_ids(processor) -> tuple[tuple[int, ...], tuple[int, ...]]:
  tokenizer = processor.tokenizer
  yes_ids = _single_token_ids(tokenizer, ("yes", "Yes", " yes", " Yes"))
  no_ids = _single_token_ids(tokenizer, ("no", "No", " no", " No"))
  if not yes_ids or not no_ids:
    raise RuntimeError(f"failed to find single-token yes/no ids: yes={yes_ids} no={no_ids}")
  return yes_ids, no_ids


def _single_token_ids(tokenizer, variants: Sequence[str]) -> tuple[int, ...]:
  ids: set[int] = set()
  for variant in variants:
    encoded = tokenizer(variant, add_special_tokens=False).input_ids
    if len(encoded) == 1:
      ids.add(int(encoded[0]))
  return tuple(sorted(ids))


def _parse_score_label_groups(raw: str) -> tuple[tuple[str, ...], ...]:
  groups: list[tuple[str, ...]] = []
  for group_raw in raw.split(";"):
    labels = tuple(label.strip() for label in group_raw.split(",") if label.strip())
    if labels:
      groups.append(labels)
  return tuple(groups)


def _validate_score_labels(labels: Sequence[str], parser: argparse.ArgumentParser, field_name: str) -> None:
  unknown = sorted(set(labels) - set(SCORE_LABELS))
  if unknown:
    parser.error(f"unknown {field_name} entries: {unknown}")


def _parse_score_thresholds(raw: str, parser: argparse.ArgumentParser) -> dict[str, float]:
  thresholds: dict[str, float] = {}
  if not raw.strip():
    return thresholds
  for item in raw.split(","):
    if not item.strip():
      continue
    if ":" not in item:
      parser.error(f"invalid --score-thresholds item: {item}")
    label, value_raw = item.split(":", 1)
    label = label.strip()
    if label not in SCORE_LABELS:
      parser.error(f"unknown --score-thresholds label: {label}")
    try:
      thresholds[label] = float(value_raw)
    except ValueError:
      parser.error(f"invalid --score-thresholds value for {label}: {value_raw}")
  return thresholds


def _score_labels(
  processor,
  model,
  images: list[Image.Image],
  image_prompt: str,
  vehicle_state_text: str,
  score_labels: Sequence[str],
  score_threshold: float,
  score_thresholds: dict[str, float] | None = None,
) -> tuple[str, tuple[str, ...], float, float, dict[str, float]]:
  if not score_labels:
    return "none", ("none",), 0.0, 0.0, {}

  prompts: list[str] = []
  batch_images: list[Image.Image] = []
  for label in score_labels:
    question = SCORE_QUESTIONS[label]
    content = [{"type": "image", "image": image} for image in images]
    content.append({
      "type": "text",
      "text": (
        f"{SCORE_PROMPT}\nVehicle state: {vehicle_state_text}\n"
        f"Question: {question}"
      ),
    })
    prompts.append(processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
    batch_images.extend(images)

  prefill_start = time.perf_counter()
  inputs = processor(text=prompts, images=batch_images, padding=True, return_tensors="pt").to("cuda")
  torch.cuda.synchronize()
  prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

  yes_ids, no_ids = _score_label_ids(processor)
  decode_start = time.perf_counter()
  with torch.inference_mode():
    outputs = model(**inputs)
  torch.cuda.synchronize()
  decode_ms = (time.perf_counter() - decode_start) * 1000.0
  next_logits = outputs.logits[:, -1, :]

  scores: dict[str, float] = {}
  selected: list[str] = []
  for idx, label in enumerate(score_labels):
    yes_score = float(torch.max(next_logits[idx, list(yes_ids)]).detach().cpu())
    no_score = float(torch.max(next_logits[idx, list(no_ids)]).detach().cpu())
    score = yes_score - no_score
    scores[label] = score
    threshold = score_threshold if score_thresholds is None else score_thresholds.get(label, score_threshold)
    if score >= threshold:
      selected.append(label)

  labels = tuple(selected) if selected else ("none",)
  score_text = ",".join(f"{label}:{scores[label]:.3f}" for label in score_labels)
  return score_text, labels, prefill_ms, decode_ms, scores


def _construction_side(label_set: set[str]) -> str:
  left = "construction_left" in label_set
  right = "construction_right" in label_set
  if left and not right:
    return "left"
  if right and not left:
    return "right"
  return "unknown"


def _construction_rtp_fields(side: str) -> tuple[str, str, str, float, str]:
  if side == "left":
    return "construction_left", "cones_barrier_left_edge", "BIAS_RIGHT_AND_SLOW", -1.25, "left_edge_s8_48_margin1.25"
  if side == "right":
    return "construction_right", "cones_barrier_right_edge", "BIAS_LEFT_AND_SLOW", 1.25, "right_edge_s8_48_margin1.25"
  return "construction_unknown", "cones_barrier_side_unknown", "SLOW", 0.0, ""


def _labels_to_rtp(labels: tuple[str, ...]) -> str:
  label_set = set(labels)
  has_stop = "red_stop_light" in label_set or "stop_sign" in label_set
  has_path_conflict_agent = bool(PATH_CONFLICT_LABELS & label_set)
  has_construction = bool({"cones", "barrier", "construction_left", "construction_right"} & label_set)
  construction_side = _construction_side(label_set)
  construction_scene, construction_evidence, construction_meta, construction_lat_bias, construction_avoid = _construction_rtp_fields(construction_side)

  if has_stop and has_construction:
    avoid = "[stop_line_s18]" if not construction_avoid else f"[{construction_avoid},stop_line_s18]"
    return "\n".join((
      "RTPv1",
      f"scene=mixed_stop_{construction_scene}",
      f"evidence=[red_stop_cue,{construction_evidence}]",
      "meta=STOP",
      "branch=base",
      f"lat_bias_m={construction_lat_bias}",
      "speed_cap_mps=0.0",
      "stop_s=18.0",
      f"avoid={avoid}",
      "weights=[obs3.0,lane1.4,comfort1.0,base0.7,vlm1.0]",
      "confidence=0.76",
    ))
  if has_path_conflict_agent and has_construction:
    avoid = "[corridor_object_s18_28]" if not construction_avoid else f"[{construction_avoid},corridor_object_s18_28]"
    return "\n".join((
      "RTPv1",
      f"scene=mixed_agent_{construction_scene}",
      f"evidence=[path_conflict_agent,{construction_evidence}]",
      "meta=YIELD",
      "branch=base",
      f"lat_bias_m={construction_lat_bias}",
      "speed_cap_mps=15%",
      "stop_s=none",
      f"avoid={avoid}",
      "weights=[obs3.0,lane1.4,comfort1.0,base0.7,vlm1.0]",
      "confidence=0.72",
    ))
  if has_stop:
    return "\n".join((
      "RTPv1",
      "scene=stop_sign",
      "evidence=[red_stop_cue]",
      "meta=STOP",
      "branch=base",
      "lat_bias_m=0.0",
      "speed_cap_mps=0.0",
      "stop_s=18.0",
      "avoid=[stop_line_s18]",
      "weights=[obs3.0,lane1.2,comfort1.0,base0.7,vlm1.0]",
      "confidence=0.74",
    ))
  if has_path_conflict_agent:
    return "\n".join((
      "RTPv1",
      "scene=path_conflict_agent",
      "evidence=[agent_in_or_entering_path]",
      "meta=YIELD",
      "branch=base",
      "lat_bias_m=0.0",
      "speed_cap_mps=0.0",
      "stop_s=18.0",
      "avoid=[corridor_object_s18_28]",
      "weights=[obs3.0,lane1.2,comfort1.0,base0.7,vlm1.0]",
      "confidence=0.70",
    ))
  if has_construction:
    avoid = "[]" if not construction_avoid else f"[{construction_avoid}]"
    return "\n".join((
      "RTPv1",
      f"scene={construction_scene}",
      f"evidence=[{construction_evidence}]",
      f"meta={construction_meta}",
      "branch=base",
      f"lat_bias_m={construction_lat_bias}",
      "speed_cap_mps=25%",
      "stop_s=none",
      f"avoid={avoid}",
      "weights=[obs2.5,lane1.4,comfort1.0,base0.7,vlm1.0]",
      "confidence=0.72",
    ))
  return "\n".join((
    "RTPv1",
    "scene=nominal",
    "evidence=[open_lane]",
    "meta=BASE",
    "branch=base",
    "lat_bias_m=0.0",
    "speed_cap_mps=none",
    "stop_s=none",
    "avoid=[]",
    "weights=[obs1.0,lane1.0,comfort1.0,base1.0,vlm1.0]",
    "confidence=0.70",
  ))


def generate(
  processor,
  model,
  payload: dict,
  max_new_tokens: int,
  image_mode: str = "multi",
  label_mode: str = "generate",
  score_threshold: float = 0.0,
  score_thresholds: dict[str, float] | None = None,
  score_labels: Sequence[str] = SCORE_LABELS,
) -> dict:
  source = _image_from_payload(payload)
  image_size = int(os.getenv("RTP_VLM_IMAGE_SIZE", "384"))
  images, image_prompt = _inference_images(source, image_mode, image_size)
  scores: dict[str, float] = {}
  generated_token_count = 0

  if label_mode == "score":
    labels_text, labels, prefill_ms, decode_ms, scores = _score_labels(
      processor,
      model,
      images,
      image_prompt,
      str(payload.get("scene_board_state_text", "")),
      score_labels,
      score_threshold,
      score_thresholds,
    )
  else:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": f"{image_prompt}\nVehicle state: {payload.get('scene_board_state_text', '')}"})
    messages = [{"role": "user", "content": content}]

    prefill_start = time.perf_counter()
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, padding=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

    decode_start = time.perf_counter()
    with torch.inference_mode():
      output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
      )
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    labels_text = _extract_new_text(processor, inputs, output_ids)
    labels = _normalize_labels(labels_text)
    generated_token_count = int(output_ids.shape[-1] - inputs.input_ids.shape[-1])

  labels = _with_visual_fallbacks(source, labels)
  rtp_text = _labels_to_rtp(labels)
  return {
    "text": rtp_text,
    "rtp_text": rtp_text,
    "labels_text": labels_text,
    "labels": list(labels),
    "label_mode": label_mode,
    "image_mode": image_mode,
    "label_scores": scores,
    "generated_token_count": generated_token_count,
    "prefill_ms": prefill_ms,
    "decode_ms": decode_ms,
    "backend": f"qwen2.5-vl-3b-label-rtp-{image_mode}-{label_mode}",
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Persistent Qwen label-to-RTP worker. Reads JSONL on stdin, writes JSONL on stdout.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--max-new-tokens", type=int, default=24)
  parser.add_argument("--image-mode", choices=("multi", "composite", "full"), default="full")
  parser.add_argument("--label-mode", choices=("generate", "score"), default="score")
  parser.add_argument("--score-threshold", type=float, default=0.0)
  parser.add_argument("--score-thresholds", default="")
  parser.add_argument("--score-labels", default=",".join(SCORE_LABELS))
  parser.add_argument("--score-rotate-groups", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--score-label-groups", default=";".join(",".join(group) for group in DEFAULT_SCORE_LABEL_GROUPS))
  parser.add_argument("--score-cache-ttl-frames", type=int, default=60)
  parser.add_argument("--score-durable-labels", default=",".join(DEFAULT_DURABLE_SCORE_LABELS))
  parser.add_argument("--score-negative-clear-threshold", type=float, default=2.0)
  args = parser.parse_args()
  if args.score_rotate_groups is None:
    args.score_rotate_groups = args.label_mode == "score"
  score_labels = tuple(label.strip() for label in args.score_labels.split(",") if label.strip())
  _validate_score_labels(score_labels, parser, "--score-labels")
  durable_score_labels = tuple(label.strip() for label in args.score_durable_labels.split(",") if label.strip())
  _validate_score_labels(durable_score_labels, parser, "--score-durable-labels")
  score_thresholds = _parse_score_thresholds(args.score_thresholds, parser)
  score_groups = _parse_score_label_groups(args.score_label_groups)
  for group in score_groups:
    _validate_score_labels(group, parser, "--score-label-groups")
  if args.score_rotate_groups and args.label_mode != "score":
    parser.error("--score-rotate-groups requires --label-mode score")
  if args.score_rotate_groups and not score_groups:
    parser.error("--score-rotate-groups requires at least one --score-label-groups group")
  rotating_state = (
    RotatingScoreState(score_groups, args.score_cache_ttl_frames, durable_score_labels, args.score_negative_clear_threshold)
    if args.score_rotate_groups else None
  )

  processor, model = _load(args.model_dir)
  warm = Image.new("RGB", (384, 384), (20, 20, 20))
  buf = BytesIO()
  warm.save(buf, format="PNG")
  generate(
    processor,
    model,
    {"scene_board_image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "scene_board_state_text": "warmup"},
    4,
    image_mode=args.image_mode,
    label_mode=args.label_mode,
    score_threshold=args.score_threshold,
    score_thresholds=score_thresholds,
    score_labels=score_groups[0] if args.score_rotate_groups else score_labels,
  )

  for line in sys.stdin:
    try:
      payload = json.loads(line)
      request_score_labels = score_labels
      score_group_idx = None
      if rotating_state is not None:
        score_group_idx, request_score_labels = rotating_state.next_group()
      response = generate(
        processor,
        model,
        payload,
        args.max_new_tokens,
        image_mode=args.image_mode,
        label_mode=args.label_mode,
        score_threshold=args.score_threshold,
        score_thresholds=score_thresholds,
        score_labels=request_score_labels,
      )
      if rotating_state is not None:
        frame_id = int(payload.get("frame_id", 0))
        cached_labels = rotating_state.update(request_score_labels, response["labels"], response["label_scores"], frame_id)
        rtp_text = _labels_to_rtp(cached_labels)
        response["labels_scored_this_request"] = list(request_score_labels)
        response["score_group_index"] = score_group_idx
        response["labels_current_group"] = response["labels"]
        response["labels"] = list(cached_labels)
        response["label_scores_cached"] = rotating_state.active_scores(frame_id)
        response["rtp_text"] = rtp_text
        response["text"] = rtp_text
        response["backend"] = f"{response['backend']}-rotating"
    except Exception as exc:
      response = {"error": repr(exc)}
    print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
  main()
