from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
  from openpilot.selfdrive.controls.reasoned.pathsynth import BasePlan
  from openpilot.selfdrive.controls.reasoned.scene_board import SceneBoard
except ModuleNotFoundError:
  from selfdrive.controls.reasoned.pathsynth import BasePlan
  from selfdrive.controls.reasoned.scene_board import SceneBoard


@dataclass(frozen=True)
class OverlayGeometry:
  lane_width_m: float = 4.5
  camera_height_m: float = 1.22
  horizon_ratio: float = 0.44
  focal_ratio: float = 1.35
  max_draw_distance_m: float = 90.0


class UiSceneBoardRenderer:
  """Render a VLM input that visually matches the onroad UI model overlay.

  This is intentionally PIL-based so it can run in the local PC POC, sim harness,
  and tests without starting the raylib UI process.
  """

  def __init__(self, width: int = 512, height: int = 384, geometry: OverlayGeometry | None = None):
    self.width = width
    self.height = height
    self.geometry = geometry or OverlayGeometry()

  def render(self, base_plan: BasePlan, vehicle_state: dict[str, Any] | None = None) -> SceneBoard:
    try:
      from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
      raise RuntimeError("UiSceneBoardRenderer requires Pillow") from exc

    state = vehicle_state or {}
    frame = state.get("road_frame")
    image = self._image_from_frame(frame, Image)
    draw = ImageDraw.Draw(image, "RGBA")

    self._draw_model_overlay(draw, base_plan)
    self._draw_metric_ticks(draw)
    self._draw_hud(draw, ImageFont, base_plan, state)

    state_text = (
      f"frame={base_plan.frame_id} "
      f"v_ego={base_plan.current_speed:.1f}mps "
      f"curv={base_plan.desired_curvature:.5f} "
      f"blinkers={int(bool(state.get('left_blinker', 0)))}/{int(bool(state.get('right_blinker', 0)))}"
    )
    return SceneBoard(self.width, self.height, bytearray(image.convert("RGB").tobytes()), state_text)

  def _image_from_frame(self, frame: Any, Image):
    if frame is None:
      image = Image.new("RGB", (self.width, self.height), (16, 20, 22))
    elif isinstance(frame, Image.Image):
      image = frame.convert("RGB")
    else:
      arr = np.asarray(frame)
      if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"road_frame must be HxWx3, got shape {arr.shape}")
      image = Image.fromarray(arr[:, :, :3].astype(np.uint8), "RGB")

    # Match UI behavior: preserve road aspect and center-crop rather than letterboxing.
    src_w, src_h = image.size
    scale = max(self.width / src_w, self.height / src_h)
    resized = image.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.BILINEAR)
    left = max(0, (resized.width - self.width) // 2)
    top = max(0, (resized.height - self.height) // 2)
    return resized.crop((left, top, left + self.width, top + self.height))

  def _draw_model_overlay(self, draw, base_plan: BasePlan) -> None:
    lane_width = self.geometry.lane_width_m
    base_points = self._plan_points(base_plan)
    lane_offsets = (-1.5 * lane_width, -0.5 * lane_width, 0.5 * lane_width, 1.5 * lane_width)
    lane_alpha = (75, 120, 165, 75)

    for offset, alpha in zip(lane_offsets, lane_alpha, strict=True):
      self._draw_strip(draw, base_points, lateral_offset=offset, half_width=0.035, color=(255, 255, 255, alpha))

    for offset in (-2.5 * lane_width, 2.5 * lane_width):
      self._draw_strip(draw, base_points, lateral_offset=offset, half_width=0.055, color=(245, 60, 50, 135))

    self._draw_strip(draw, base_points, lateral_offset=0.0, half_width=0.48, color=(35, 210, 105, 105))
    self._draw_polyline(draw, base_points, lateral_offset=0.0, color=(255, 255, 255, 225), width=2)

  def _draw_metric_ticks(self, draw) -> None:
    for s_m in (10.0, 20.0, 40.0, 60.0):
      left = self._project(s_m, 3.5)
      right = self._project(s_m, -3.5)
      if left is not None and right is not None:
        draw.line([left, right], fill=(255, 255, 255, 46), width=1)

  def _draw_hud(self, draw, ImageFont, base_plan: BasePlan, state: dict[str, Any]) -> None:
    font = ImageFont.load_default()
    speed_mph = base_plan.current_speed * 2.23694
    status = str(state.get("status", "SIM"))
    text = f"{speed_mph:4.0f} mph  {status}"
    draw.rounded_rectangle((8, 8, 146, 34), radius=6, fill=(0, 0, 0, 120))
    draw.text((16, 17), text, font=font, fill=(245, 245, 245, 255))

  def _plan_points(self, base_plan: BasePlan) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in zip(base_plan.x, base_plan.y) if x >= 0.0 and x <= self.geometry.max_draw_distance_m]
    if len(pts) >= 2:
      return pts
    return [(float(s), 0.0) for s in np.linspace(0.5, self.geometry.max_draw_distance_m, 24)]

  def _draw_strip(self, draw, points: list[tuple[float, float]], lateral_offset: float, half_width: float, color: tuple[int, int, int, int]) -> None:
    left = []
    right = []
    for s_m, y_m in points:
      lp = self._project(s_m, y_m + lateral_offset + half_width)
      rp = self._project(s_m, y_m + lateral_offset - half_width)
      if lp is not None and rp is not None:
        left.append(lp)
        right.append(rp)
    if len(left) > 1 and len(right) > 1:
      draw.polygon(left + right[::-1], fill=color)

  def _draw_polyline(self, draw, points: list[tuple[float, float]], lateral_offset: float, color: tuple[int, int, int, int], width: int) -> None:
    projected = []
    for s_m, y_m in points:
      p = self._project(s_m, y_m + lateral_offset)
      if p is not None:
        projected.append(p)
    if len(projected) > 1:
      draw.line(projected, fill=color, width=width, joint="curve")

  def _project(self, s_m: float, y_left_m: float) -> tuple[float, float] | None:
    if s_m <= 0.5:
      return None
    f = self.width * self.geometry.focal_ratio
    cx = self.width * 0.5
    horizon_y = self.height * self.geometry.horizon_ratio
    u = cx - f * y_left_m / s_m
    v = horizon_y + f * self.geometry.camera_height_m / s_m
    if u < -self.width or u > self.width * 2 or v < -self.height or v > self.height * 2:
      return None
    return (float(u), float(v))
