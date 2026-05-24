#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from selfdrive.controls.reasoned.pathsynth import BasePlan
from selfdrive.controls.reasoned.planner import ReasonedPlanner, ReasonedPlannerConfig
from selfdrive.controls.reasoned.ui_scene_board import UiSceneBoardRenderer
from selfdrive.controls.reasoned.vlm import StaticRtpEngine, build_rtp_engine
from tools.reasoned_trajectory_poc.run_local_demo import SCENARIOS


AVOID_ZONE_RE = re.compile(
  r"^(?P<side>left_edge|right_edge|corridor_object)_s"
  r"(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)"
  r"(?:_margin(?P<margin>\d+(?:\.\d+)?))?$"
)
STOP_LINE_RE = re.compile(r"^stop_line_s(?P<distance>\d+(?:\.\d+)?)$")


def wrap_angle(angle: float) -> float:
  return (angle + math.pi) % (2.0 * math.pi) - math.pi


def percentile(values: list[float], pct: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
  return ordered[idx]


def make_base_plan(frame_id: int, speed_mps: float, curvature: float = 0.0, xy: tuple[tuple[float, ...], tuple[float, ...]] | None = None, desired_speed_mps: float | None = None) -> BasePlan:
  if xy is None:
    xs = np.linspace(0.5, 80.0, 33)
    ys = 0.5 * curvature * xs * xs
  else:
    xs = np.asarray(xy[0], dtype=np.float32)
    ys = np.asarray(xy[1], dtype=np.float32)
  desired_speed = float(speed_mps if desired_speed_mps is None else desired_speed_mps)
  return BasePlan(
    frame_id=frame_id,
    model_log_mono_time_ns=frame_id * 50_000_000,
    t=tuple(float(i) * 0.2 for i in range(len(xs))),
    x=tuple(float(x) for x in xs),
    y=tuple(float(y) for y in ys),
    speeds=tuple(desired_speed for _ in xs),
    desired_curvature=float(curvature),
    v_ego=float(speed_mps),
  )


def make_base_plan_from_route(env, frame_id: int, speed_mps: float, desired_speed_mps: float | None = None) -> BasePlan:
  distances = np.linspace(0.5, 80.0, 33)
  xs: list[float] = []
  ys: list[float] = []
  for dist in distances:
    point = route_world_point(env, float(dist), 0.0)
    forward, left = world_to_ego(env, point)
    xs.append(max(0.1, forward))
    ys.append(left)

  curvature = 0.0
  for x, y in zip(xs, ys):
    if x >= 12.0:
      curvature = float(np.clip(2.0 * y / max(x * x, 1.0), -0.2, 0.2))
      break
  return make_base_plan(frame_id, speed_mps, curvature, (tuple(xs), tuple(ys)), desired_speed_mps)


def make_env(args: argparse.Namespace):
  from panda3d.core import Vec3
  from metadrive.component.sensors.rgb_camera import RGBCamera
  from metadrive.envs.metadrive_env import MetaDriveEnv
  map_arg: int | str
  map_arg = int(args.map) if str(args.map).isdigit() else args.map

  config = {
    "use_render": False,
    "image_observation": True,
    "sensors": {"rgb_road": (RGBCamera, args.camera_width, args.camera_height)},
    "vehicle_config": {"image_source": "rgb_road"},
    "interface_panel": [],
    "traffic_density": 0.0,
    "manual_control": False,
    "show_logo": False,
    "show_fps": False,
    "start_seed": args.seed,
    "num_scenarios": 1,
    "map": map_arg,
    "crash_vehicle_done": False,
    "crash_object_done": False,
    "out_of_route_done": False,
    "on_continuous_line_done": False,
  }
  env = MetaDriveEnv(config)
  env.reset(seed=args.seed)
  cam = env.engine.sensors["rgb_road"]
  cam.get_cam().reparentTo(env.vehicle.origin)
  cam.get_cam().setPos(Vec3(0.0, 0.0, 1.22))
  cam.get_cam().setHpr(Vec3(0.0, 0.0, 0.0))
  return env


def spawn_novel_scene(env, args: argparse.Namespace) -> list[dict[str, float | str]]:
  scene_name = args.novel_scene
  if scene_name == "none":
    return []

  spawned: list[dict[str, float | str]] = []
  env._rtp_moving_pedestrians = []
  lane = route_lane_for_vehicle(env)
  lane_heading = float(lane.heading_theta_at(max(0.0, lane.local_coordinates(env.vehicle.position)[0])))

  def spawn_at(kind: str, ahead_m: float, lateral_m: float, cls, **kwargs) -> None:
    point = route_world_point(env, ahead_m, lateral_m)
    obj = env.engine.spawn_object(
      cls,
      position=[float(point[0]), float(point[1])],
      heading_theta=lane_heading,
      force_spawn=True,
      **kwargs,
    )
    spawned.append({
      "kind": kind,
      "ahead_m": float(ahead_m),
      "lateral_m": float(lateral_m),
      "x": float(point[0]),
      "y": float(point[1]),
      "id": str(getattr(obj, "id", "")),
    })

  if scene_name == "construction":
    from metadrive.component.static_object.traffic_object import TrafficBarrier, TrafficCone
    for ahead_m in (14.0, 20.0, 26.0, 32.0):
      spawn_at("traffic_cone_right_edge", ahead_m, -1.35, TrafficCone, static=True)
    spawn_at("traffic_barrier_right_edge", 24.0, -1.20, TrafficBarrier, static=True)
  elif scene_name == "pedestrian":
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    spawn_at("pedestrian_center_lane", 24.0, 0.0, Pedestrian, random_seed=3)
  elif scene_name == "stop_sign":
    point = route_world_point(env, 18.0, -1.15)
    _spawn_stop_sign_billboard(env, float(point[0]), float(point[1]), lane_heading)
    from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
    light_point = route_world_point(env, 18.0, 0.0)
    light = env.engine.spawn_object(
      BaseTrafficLight,
      lane=lane,
      position=[float(light_point[0]), float(light_point[1])],
      force_spawn=True,
      show_model=True,
    )
    light.set_red()
    spawned.append({
      "kind": "stop_sign_billboard",
      "ahead_m": 18.0,
      "lateral_m": -1.15,
      "x": float(point[0]),
      "y": float(point[1]),
      "id": "billboard",
    })
    spawned.append({
      "kind": "red_stop_light",
      "ahead_m": 18.0,
      "lateral_m": 0.0,
      "x": float(light_point[0]),
      "y": float(light_point[1]),
      "id": str(getattr(light, "id", "")),
    })
  elif scene_name == "random_mixed":
    _spawn_random_mixed_scene(env, args, spawned)
  else:
    raise ValueError(f"unknown novel scene: {scene_name}")

  # Advance one render step so newly spawned objects are visible to the camera sensor.
  env.step([0.0, 0.0])
  return spawned


def _route_lanes(env):
  nav = env.vehicle.navigation
  lane_id = getattr(getattr(env.vehicle, "lane", None), "index", (None, None, 0))[2]
  lane_id = lane_id if isinstance(lane_id, int) else 0
  lanes = []
  net = nav.map.road_network
  for start, end in zip(nav.checkpoints[:-1], nav.checkpoints[1:]):
    road_lanes = net.graph.get(start, {}).get(end, [])
    if not road_lanes:
      continue
    lanes.append(road_lanes[min(max(lane_id, 0), len(road_lanes) - 1)])
  return lanes


def route_point_at_s(env, route_s_m: float, lateral_offset_m: float) -> tuple[np.ndarray, float]:
  remaining = max(0.0, float(route_s_m))
  lanes = _route_lanes(env)
  if not lanes:
    return route_world_point(env, route_s_m, lateral_offset_m), 0.0
  lane = lanes[-1]
  local_s = min(remaining, lane.length)
  for candidate in lanes:
    lane = candidate
    if remaining <= candidate.length:
      local_s = remaining
      break
    remaining -= candidate.length
  local_s = min(max(0.0, local_s), lane.length)
  half_width = max(0.1, float(lane.width_at(local_s)) * 0.5 - 0.45)
  point = np.asarray(lane.position(local_s, float(np.clip(lateral_offset_m, -half_width, half_width))), dtype=np.float32)
  heading = float(lane.heading_theta_at(local_s))
  return point, heading


def route_total_length_m(env) -> float:
  return float(sum(lane.length for lane in _route_lanes(env)))


def _spawn_random_mixed_scene(env, args: argparse.Namespace, spawned: list[dict[str, float | str]]) -> None:
  from metadrive.component.static_object.traffic_object import TrafficBarrier, TrafficCone
  from metadrive.component.traffic_participants.pedestrian import Pedestrian

  rng = np.random.default_rng(args.random_scene_seed)
  route_len = route_total_length_m(env)
  if route_len <= 1.0:
    route_len = args.random_scene_route_m
  end_s = min(route_len - 8.0, args.random_scene_route_m)
  start_s = 14.0

  construction_s = start_s
  cluster_id = 0
  while construction_s < end_s:
    side = -1.0 if rng.random() < args.random_construction_right_probability else 1.0
    base_lateral = side * float(rng.uniform(1.05, 1.45))
    cluster_len = int(rng.integers(2, args.random_construction_max_objects + 1))
    for idx in range(cluster_len):
      route_s = min(end_s, construction_s + idx * float(rng.uniform(4.0, 7.0)))
      lateral = base_lateral + float(rng.normal(0.0, 0.08))
      point, heading = route_point_at_s(env, route_s, lateral)
      cls = TrafficBarrier if idx == cluster_len // 2 and rng.random() < 0.45 else TrafficCone
      obj = env.engine.spawn_object(
        cls,
        position=[float(point[0]), float(point[1])],
        heading_theta=heading,
        force_spawn=True,
        static=True,
      )
      spawned.append({
        "kind": "random_traffic_barrier" if cls is TrafficBarrier else "random_traffic_cone",
        "route_s_m": float(route_s),
        "ahead_m": float(route_s),
        "lateral_m": float(lateral),
        "x": float(point[0]),
        "y": float(point[1]),
        "id": str(getattr(obj, "id", "")),
        "cluster": str(cluster_id),
      })
    construction_s += float(rng.uniform(args.random_construction_spacing_min_m, args.random_construction_spacing_max_m))
    cluster_id += 1

  pedestrians = []
  pedestrian_s = start_s + 8.0
  pedestrian_id = 0
  while pedestrian_s < end_s:
    start_side = -1.0 if rng.random() < 0.5 else 1.0
    start_lateral = start_side * float(rng.uniform(1.35, 1.75))
    target_lateral = -start_side * float(rng.uniform(1.20, 1.65))
    start_point, heading = route_point_at_s(env, pedestrian_s, start_lateral)
    end_point, _ = route_point_at_s(env, pedestrian_s + float(rng.uniform(-1.5, 2.0)), target_lateral)
    direction = np.asarray(end_point - start_point, dtype=np.float32)
    if float(np.linalg.norm(direction)) < 1e-3:
      direction = np.asarray([0.0, -start_side], dtype=np.float32)
    speed = float(rng.uniform(args.random_pedestrian_speed_min_mps, args.random_pedestrian_speed_max_mps))
    ped = env.engine.spawn_object(
      Pedestrian,
      position=[float(start_point[0]), float(start_point[1])],
      heading_theta=heading + (math.pi / 2.0) * -start_side,
      force_spawn=True,
      random_seed=int(rng.integers(0, 2**31 - 1)),
    )
    ped.set_velocity(direction.tolist(), speed)
    record = {
      "kind": "moving_pedestrian_crossing",
      "route_s_m": float(pedestrian_s),
      "ahead_m": float(pedestrian_s),
      "lateral_m": float(start_lateral),
      "target_lateral_m": float(target_lateral),
      "x": float(start_point[0]),
      "y": float(start_point[1]),
      "id": str(getattr(ped, "id", "")),
      "speed_mps": speed,
      "pedestrian": str(pedestrian_id),
    }
    spawned.append(record)
    pedestrians.append({
      "obj": ped,
      "direction": (float(direction[0]), float(direction[1])),
      "speed_mps": speed,
      "target_lateral_m": float(target_lateral),
      "record": record,
    })
    pedestrian_s += float(rng.uniform(args.random_pedestrian_spacing_min_m, args.random_pedestrian_spacing_max_m))
    pedestrian_id += 1

  env._rtp_moving_pedestrians = pedestrians
  env._rtp_random_scene_route_len_m = route_len


def update_moving_pedestrians(env) -> None:
  movers = getattr(env, "_rtp_moving_pedestrians", [])
  if not movers:
    return
  for mover in movers:
    try:
      ped = mover["obj"]
      ped.set_velocity(list(mover["direction"]), float(mover["speed_mps"]))
      record = mover["record"]
      record["x"] = float(ped.position[0])
      record["y"] = float(ped.position[1])
    except Exception:
      continue


def _spawn_stop_sign_billboard(env, x: float, y: float, lane_heading: float) -> None:
  from panda3d.core import CardMaker, TextNode

  root = env.engine.render.attachNewNode("rtp_stop_sign")
  root.setPos(x, y, 1.70)
  root.setHpr(math.degrees(lane_heading) + 90.0, 0.0, 0.0)
  root.setScale(1.10)

  card_maker = CardMaker("rtp_stop_sign_card")
  card_maker.setFrame(-1.0, 1.0, -1.0, 1.0)
  card = root.attachNewNode(card_maker.generate())
  card.setTwoSided(True)
  card.setLightOff(1)
  card.setTexture(_stop_sign_texture(env), 1)

  text_node = TextNode("rtp_stop_sign_billboard_text")
  text_node.setText("STOP")
  text_node.setAlign(TextNode.ACenter)
  text_node.setTextColor(1.0, 1.0, 1.0, 1.0)
  text = env.engine.render.attachNewNode(text_node)
  text.setPos(x, y - 0.05, 1.62)
  text.setScale(0.58)
  text.setLightOff(1)
  text.setBillboardPointEye()


def _stop_sign_texture(env):
  from panda3d.core import Filename, Texture
  from PIL import Image, ImageDraw, ImageFont

  texture_path = REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "stop_sign_texture_opaque.png"
  if not texture_path.exists():
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), (160, 0, 0))
    draw = ImageDraw.Draw(image)
    points = [
      (96, 10), (160, 10), (246, 96), (246, 160),
      (160, 246), (96, 246), (10, 160), (10, 96),
    ]
    draw.polygon(points, fill=(205, 0, 0), outline=(255, 255, 255))
    try:
      font = ImageFont.truetype("arial.ttf", 64)
    except Exception:
      font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), "STOP", font=font)
    draw.text(((256 - (bbox[2] - bbox[0])) / 2, (256 - (bbox[3] - bbox[1])) / 2 - 8), "STOP", font=font, fill=(255, 255, 255))
    image.save(texture_path)
  texture = env.engine.loader.loadTexture(Filename.fromOsSpecific(str(texture_path)))
  texture.setMinfilter(Texture.FTLinear)
  texture.setMagfilter(Texture.FTLinear)
  return texture


def camera_frame(env) -> np.ndarray:
  cam = env.engine.sensors["rgb_road"]
  frame = cam.perceive(to_float=False)
  if not isinstance(frame, np.ndarray):
    frame = frame.get()
  return frame.astype(np.uint8, copy=False)


def speed_mps(env) -> float:
  return float(np.linalg.norm(np.asarray(env.vehicle.velocity, dtype=np.float32)))


def route_lane_for_vehicle(env):
  vehicle = env.vehicle
  nav = getattr(vehicle, "navigation", None)
  ref_lanes = getattr(nav, "current_ref_lanes", None)
  if ref_lanes:
    if getattr(vehicle, "lane", None) in ref_lanes:
      return vehicle.lane
    lane_id = getattr(getattr(vehicle, "lane", None), "index", (None, None, 0))[2]
    if isinstance(lane_id, int) and 0 <= lane_id < len(ref_lanes):
      return ref_lanes[lane_id]
    return ref_lanes[0]
  return vehicle.lane


def next_route_lane(env, current_lane):
  nav = getattr(env.vehicle, "navigation", None)
  next_lanes = getattr(nav, "next_ref_lanes", None)
  if not next_lanes:
    return None
  lane_id = getattr(current_lane, "index", (None, None, 0))[2]
  if isinstance(lane_id, int) and 0 <= lane_id < len(next_lanes):
    return next_lanes[lane_id]
  return next_lanes[0]


def route_world_point(env, ahead_m: float, lateral_offset_m: float) -> np.ndarray:
  lane = route_lane_for_vehicle(env)
  long_m, _ = lane.local_coordinates(env.vehicle.position)
  target_long = long_m + ahead_m
  target_lane = lane
  if target_long > lane.length:
    overflow = target_long - lane.length
    next_lane = next_route_lane(env, lane)
    if next_lane is not None:
      target_lane = next_lane
      target_long = min(max(0.0, overflow), next_lane.length)
    else:
      target_long = lane.length
  else:
    target_long = min(max(0.0, target_long), lane.length)

  half_width = max(0.1, float(target_lane.width_at(target_long)) * 0.5 - 0.45)
  return np.asarray(target_lane.position(target_long, float(np.clip(lateral_offset_m, -half_width, half_width))), dtype=np.float32)


def world_to_ego(env, point: np.ndarray) -> tuple[float, float]:
  vehicle = env.vehicle
  dx = float(point[0] - vehicle.position[0])
  dy = float(point[1] - vehicle.position[1])
  heading = float(vehicle.heading_theta)
  cos_h = math.cos(heading)
  sin_h = math.sin(heading)
  forward = dx * cos_h + dy * sin_h
  left = -dx * sin_h + dy * cos_h
  return forward, left


def selected_lateral_offset_m(synth) -> float:
  if synth is None:
    return 0.0
  for candidate in synth.candidates:
    if candidate.name == synth.selected_candidate:
      return float(candidate.lateral_offset_m)
  return 0.0


def openpilot_to_metadrive_lateral_m(lateral_m: float) -> float:
  # PathSynth/openpilot convention: positive is left.
  # MetaDrive lane.position/local_coordinates convention in this harness:
  # positive lateral is visually/right-of-path for the ego route.
  return -float(lateral_m)


@dataclass
class DurableAvoidance:
  start_long_m: float
  end_long_m: float
  ramp_in_start_long_m: float
  ramp_out_end_long_m: float
  offset_m: float
  source_token: str
  source_meta: str
  confidence: float

  def active(self, current_long_m: float) -> bool:
    return current_long_m <= self.ramp_out_end_long_m

  def target_offset(self, current_long_m: float) -> float:
    if current_long_m < self.ramp_in_start_long_m or current_long_m > self.ramp_out_end_long_m:
      return 0.0
    if current_long_m < self.start_long_m:
      span = max(0.1, self.start_long_m - self.ramp_in_start_long_m)
      return self.offset_m * _smoothstep((current_long_m - self.ramp_in_start_long_m) / span)
    if current_long_m <= self.end_long_m:
      return self.offset_m
    span = max(0.1, self.ramp_out_end_long_m - self.end_long_m)
    return self.offset_m * (1.0 - _smoothstep((current_long_m - self.end_long_m) / span))


@dataclass
class DurableSpeedPlan:
  start_long_m: float
  end_long_m: float
  ramp_out_end_long_m: float
  speed_cap_mps: float
  stop_s: float | None
  source_token: str
  source_meta: str
  confidence: float

  def active(self, current_long_m: float) -> bool:
    return current_long_m <= self.ramp_out_end_long_m

  def target_speed_cap(self, current_long_m: float, nominal_speed_mps: float) -> float:
    if current_long_m > self.ramp_out_end_long_m:
      return nominal_speed_mps
    if current_long_m <= self.end_long_m:
      return self.speed_cap_mps
    span = max(0.1, self.ramp_out_end_long_m - self.end_long_m)
    blend = _smoothstep((current_long_m - self.end_long_m) / span)
    return self.speed_cap_mps + (nominal_speed_mps - self.speed_cap_mps) * blend


def _smoothstep(raw: float) -> float:
  x = float(np.clip(raw, 0.0, 1.0))
  return x * x * (3.0 - 2.0 * x)


def _slew(current: float, target: float, max_delta: float) -> float:
  return float(current + np.clip(target - current, -max_delta, max_delta))


def current_route_longitudinal_m(env) -> float:
  lane = route_lane_for_vehicle(env)
  long_m, _ = lane.local_coordinates(env.vehicle.position)
  return float(long_m)


def durable_avoidance_from_program(program, current_long_m: float, selected_offset_m: float, args: argparse.Namespace) -> DurableAvoidance | None:
  if program is None or not program.avoid:
    return None

  for token in program.avoid:
    match = AVOID_ZONE_RE.match(token)
    if match is None:
      continue
    side = match.group("side")
    start_s = float(match.group("start"))
    end_s = float(match.group("end"))
    if end_s <= start_s:
      continue

    if side == "corridor_object" and abs(selected_offset_m) <= 1e-3:
      continue

    margin_raw = match.group("margin")
    requested_margin = float(margin_raw) if margin_raw is not None else abs(selected_offset_m)
    offset_mag = max(abs(selected_offset_m), requested_margin, args.min_construction_offset_m)
    offset_mag = min(offset_mag, args.max_durable_offset_m)

    if side == "right_edge":
      offset_m = openpilot_to_metadrive_lateral_m(offset_mag)
    elif side == "left_edge":
      offset_m = openpilot_to_metadrive_lateral_m(-offset_mag)
    else:
      offset_m = openpilot_to_metadrive_lateral_m(selected_offset_m) if abs(selected_offset_m) > 1e-3 else 0.0

    return DurableAvoidance(
      start_long_m=current_long_m + start_s,
      end_long_m=current_long_m + end_s,
      ramp_in_start_long_m=current_long_m + max(0.0, start_s - args.avoid_lead_m),
      ramp_out_end_long_m=current_long_m + end_s + args.avoid_recover_m,
      offset_m=offset_m,
      source_token=token,
      source_meta=str(getattr(program, "meta", "")),
      confidence=float(getattr(program, "confidence", 0.0)),
    )
  return None


def durable_speed_plan_from_program(program, current_long_m: float, args: argparse.Namespace) -> DurableSpeedPlan | None:
  if program is None or not _program_requests_durable_speed(program):
    return None

  cap = args.speed_mps
  has_explicit_speed_cap = getattr(program, "speed_scale", None) is not None or program.speed_cap_mps is not None
  if getattr(program, "speed_scale", None) is not None:
    cap = min(cap, args.speed_mps * float(np.clip(program.speed_scale, 0.0, 1.0)))
  if program.speed_cap_mps is not None:
    cap = min(cap, float(program.speed_cap_mps))
  elif not has_explicit_speed_cap and getattr(program, "meta", "") in {"STOP", "YIELD"}:
    cap = 0.0
  elif not has_explicit_speed_cap and getattr(program, "meta", "") in {"SLOW", "BIAS_LEFT_AND_SLOW", "BIAS_RIGHT_AND_SLOW", "OCCLUSION_CAUTION", "EMERGENCY_CAUTION"}:
    cap = min(cap, args.speed_mps * float(np.clip(args.durable_slow_speed_scale, 0.0, 1.0)))

  if program.stop_s is not None:
    cap = min(cap, _stop_speed_cap_for_demo(float(program.stop_s), args.speed_mps))

  if cap >= args.speed_mps - 1e-3 and program.stop_s is None:
    return None

  source_token, start_s, end_s = _speed_plan_interval(program, args)
  if program.stop_s is not None:
    end_s = max(end_s, float(program.stop_s))
  end_s = max(end_s, args.durable_speed_min_horizon_m)

  return DurableSpeedPlan(
    start_long_m=current_long_m,
    end_long_m=current_long_m + end_s,
    ramp_out_end_long_m=current_long_m + end_s + args.durable_speed_recover_m,
    speed_cap_mps=float(np.clip(cap, 0.0, args.speed_mps)),
    stop_s=None if program.stop_s is None else float(program.stop_s),
    source_token=source_token,
    source_meta=str(getattr(program, "meta", "")),
    confidence=float(getattr(program, "confidence", 0.0)),
  )


def _merge_durable_avoidance(existing: DurableAvoidance, new: DurableAvoidance) -> DurableAvoidance:
  offset_m = existing.offset_m
  if _lateral_sign(existing.offset_m) != _lateral_sign(new.offset_m):
    offset_m = new.offset_m
  elif abs(new.offset_m) > abs(existing.offset_m):
    offset_m = new.offset_m
  return DurableAvoidance(
    start_long_m=min(existing.start_long_m, new.start_long_m),
    end_long_m=max(existing.end_long_m, new.end_long_m),
    ramp_in_start_long_m=min(existing.ramp_in_start_long_m, new.ramp_in_start_long_m),
    ramp_out_end_long_m=max(existing.ramp_out_end_long_m, new.ramp_out_end_long_m),
    offset_m=offset_m,
    source_token=existing.source_token,
    source_meta=new.source_meta,
    confidence=max(existing.confidence, new.confidence),
  )


def _merge_durable_speed_plan(existing: DurableSpeedPlan, new: DurableSpeedPlan) -> DurableSpeedPlan:
  # Same-source speed plans represent updated judgement about the same hazard.
  # Keep the broader spatial interval, but let the newest cap/stop decision replace
  # a stale stricter one so a transient YIELD cannot pin the car at zero forever.
  return DurableSpeedPlan(
    start_long_m=min(existing.start_long_m, new.start_long_m),
    end_long_m=max(existing.end_long_m, new.end_long_m),
    ramp_out_end_long_m=max(existing.ramp_out_end_long_m, new.ramp_out_end_long_m),
    speed_cap_mps=new.speed_cap_mps,
    stop_s=new.stop_s,
    source_token=existing.source_token,
    source_meta=new.source_meta,
    confidence=max(existing.confidence, new.confidence),
  )


def update_durable_lateral_plans(plans: dict[str, DurableAvoidance], new: DurableAvoidance | None, program, current_long_m: float, args: argparse.Namespace) -> dict[str, DurableAvoidance]:
  updated = {key: plan for key, plan in plans.items() if plan.active(current_long_m)}
  if _program_clears_lateral(program) and _program_confidence(program) >= args.durable_override_confidence:
    updated.clear()

  if new is None:
    return updated

  if _program_confidence(program) >= args.durable_override_confidence:
    if new.offset_m > 0.0:
      updated = {key: plan for key, plan in updated.items() if plan.offset_m >= 0.0}
    elif new.offset_m < 0.0:
      updated = {key: plan for key, plan in updated.items() if plan.offset_m <= 0.0}
  elif new.confidence >= args.durable_conflict_override_confidence:
    updated = {
      key: plan for key, plan in updated.items()
      if not _lateral_plans_conflict(plan, new)
    }

  existing = updated.get(new.source_token)
  updated[new.source_token] = new if existing is None else _merge_durable_avoidance(existing, new)
  return updated


def update_durable_speed_plans(plans: dict[str, DurableSpeedPlan], new: DurableSpeedPlan | None, program, current_long_m: float, args: argparse.Namespace) -> dict[str, DurableSpeedPlan]:
  updated = {key: plan for key, plan in plans.items() if plan.active(current_long_m)}
  if _program_clears_speed(program) and _program_confidence(program) >= args.durable_override_confidence:
    updated.clear()

  if new is None:
    return updated

  existing = updated.get(new.source_token)
  updated[new.source_token] = new if existing is None else _merge_durable_speed_plan(existing, new)
  return updated


def active_lateral_plans(plans: dict[str, DurableAvoidance], current_long_m: float) -> list[DurableAvoidance]:
  return [plan for plan in plans.values() if plan.active(current_long_m)]


def active_speed_plans(plans: dict[str, DurableSpeedPlan], current_long_m: float) -> list[DurableSpeedPlan]:
  return [plan for plan in plans.values() if plan.active(current_long_m)]


def _lateral_sign(offset_m: float) -> int:
  if offset_m > 1e-3:
    return 1
  if offset_m < -1e-3:
    return -1
  return 0


def _lateral_plans_conflict(existing: DurableAvoidance, new: DurableAvoidance) -> bool:
  existing_sign = _lateral_sign(existing.offset_m)
  new_sign = _lateral_sign(new.offset_m)
  return existing_sign != 0 and new_sign != 0 and existing_sign != new_sign


def compose_lateral_offset(plans: dict[str, DurableAvoidance], current_long_m: float, max_offset_m: float) -> float:
  offsets = [plan.target_offset(current_long_m) for plan in active_lateral_plans(plans, current_long_m)]
  offsets = [offset for offset in offsets if abs(offset) > 1e-3]
  if not offsets:
    return 0.0
  strongest = max(offsets, key=abs)
  return float(np.clip(strongest, -max_offset_m, max_offset_m))


def compose_speed_cap(plans: dict[str, DurableSpeedPlan], current_long_m: float, nominal_speed_mps: float) -> float | None:
  caps = [plan.target_speed_cap(current_long_m, nominal_speed_mps) for plan in active_speed_plans(plans, current_long_m)]
  caps = [float(np.clip(cap, 0.0, nominal_speed_mps)) for cap in caps if cap < nominal_speed_mps - 1e-3]
  return min(caps) if caps else None


def _program_requests_durable_speed(program) -> bool:
  if program is None:
    return False
  return (
    getattr(program, "speed_cap_mps", None) is not None
    or getattr(program, "speed_scale", None) is not None
    or getattr(program, "stop_s", None) is not None
    or getattr(program, "meta", "") in {
      "BIAS_LEFT_AND_SLOW",
      "BIAS_RIGHT_AND_SLOW",
      "SLOW",
      "YIELD",
      "STOP",
      "OCCLUSION_CAUTION",
      "EMERGENCY_CAUTION",
    }
  )


def _program_clears_lateral(program) -> bool:
  if program is None:
    return False
  return (
    getattr(program, "meta", "") == "BASE"
    and not getattr(program, "avoid", ())
    and abs(float(getattr(program, "lat_bias_m", 0.0))) <= 1e-3
  )


def _program_clears_speed(program) -> bool:
  if program is None:
    return False
  return (
    getattr(program, "meta", "") == "BASE"
    and not getattr(program, "avoid", ())
    and getattr(program, "speed_cap_mps", None) is None
    and getattr(program, "speed_scale", None) is None
    and getattr(program, "stop_s", None) is None
  )


def _speed_plan_interval(program, args: argparse.Namespace) -> tuple[str, float, float]:
  avoid = tuple(getattr(program, "avoid", ()))
  preferred_sides: tuple[str, ...]
  if getattr(program, "meta", "") in {"STOP", "YIELD"} or getattr(program, "stop_s", None) is not None:
    preferred_sides = ("stop_line", "corridor_object", "right_edge", "left_edge")
  else:
    preferred_sides = ("right_edge", "left_edge", "corridor_object", "stop_line")

  parsed: list[tuple[str, str, float, float]] = []
  for token in avoid:
    interval = _parse_plan_interval_token(token)
    if interval is not None:
      side, start_s, end_s = interval
      parsed.append((token, side, start_s, end_s))

  for side in preferred_sides:
    for token, parsed_side, start_s, end_s in parsed:
      if parsed_side == side:
        return token, start_s, end_s

  if getattr(program, "stop_s", None) is not None:
    stop_s = float(program.stop_s)
    return f"stop_s{stop_s:.1f}", max(0.0, stop_s - args.avoid_lead_m), stop_s + args.avoid_recover_m

  return str(getattr(program, "meta", "speed")), 0.0, args.durable_speed_min_horizon_m


def _parse_plan_interval_token(token: str) -> tuple[str, float, float] | None:
  match = AVOID_ZONE_RE.match(token)
  if match is not None:
    start_s = float(match.group("start"))
    end_s = float(match.group("end"))
    if end_s > start_s:
      return match.group("side"), start_s, end_s

  stop_match = STOP_LINE_RE.match(token)
  if stop_match is not None:
    distance = float(stop_match.group("distance"))
    return "stop_line", max(0.0, distance - 4.0), distance + 6.0

  return None


def _stop_speed_cap_for_demo(stop_s: float, desired_speed_mps: float) -> float:
  if stop_s <= 20.0:
    return 0.0
  if stop_s <= 40.0:
    return desired_speed_mps * 0.2
  return desired_speed_mps * 0.4


def _program_confidence(program) -> float:
  return 0.0 if program is None else float(getattr(program, "confidence", 0.0))


def spawned_min_distance_m(env, spawned_scene: list[dict[str, float | str]]) -> float | None:
  vehicle_pos = np.asarray(env.vehicle.position, dtype=np.float32)
  distances = []
  for item in spawned_scene:
    if "x" in item and "y" in item:
      obj_pos = np.asarray([float(item["x"]), float(item["y"])], dtype=np.float32)
      distances.append(float(np.linalg.norm(vehicle_pos - obj_pos)))
  return min(distances) if distances else None


def _jsonable_info_flags(info) -> dict[str, bool | float | int | str]:
  if not isinstance(info, dict):
    return {}
  out: dict[str, bool | float | int | str] = {}
  for key, value in info.items():
    if isinstance(value, (bool, str)):
      out[str(key)] = value
    elif isinstance(value, (int, float, np.integer, np.floating)):
      out[str(key)] = float(value)
  return out


class MetaDriveRouteFollower:
  def __init__(self, max_steer: float = 0.75, max_steer_rate_per_s: float = 0.9, steer_smoothing_alpha: float = 0.35):
    self.max_steer = max_steer
    self.max_steer_rate_per_s = max_steer_rate_per_s
    self.steer_smoothing_alpha = steer_smoothing_alpha
    self.last_steer = 0.0

  def action(self, env, target_speed_mps: float, lateral_offset_m: float, dt: float) -> tuple[float, float, dict[str, float]]:
    vehicle = env.vehicle
    speed = speed_mps(env)
    lane = route_lane_for_vehicle(env)
    long_m, current_lateral_m = lane.local_coordinates(vehicle.position)
    lookahead_m = float(np.clip(6.0 + speed * 1.65, 8.0, 28.0))
    target = route_world_point(env, lookahead_m, lateral_offset_m)
    dx = float(target[0] - vehicle.position[0])
    dy = float(target[1] - vehicle.position[1])
    desired_heading = math.atan2(dy, dx)
    heading_error = wrap_angle(desired_heading - float(vehicle.heading_theta))
    lateral_error = float(current_lateral_m - lateral_offset_m)

    raw_steer = float(np.clip(1.45 * heading_error + 0.04 * lateral_error, -self.max_steer, self.max_steer))
    filtered_steer = self.last_steer + self.steer_smoothing_alpha * (raw_steer - self.last_steer)
    steer = _slew(self.last_steer, filtered_steer, self.max_steer_rate_per_s * max(dt, 1e-3))
    steer = float(np.clip(steer, -self.max_steer, self.max_steer))
    self.last_steer = steer

    speed_error = target_speed_mps - speed
    gas = float(np.clip(0.16 * speed_error, -0.65, 0.55))
    return steer, gas, {
      "lane_longitudinal_m": float(long_m),
      "lane_lateral_m": float(current_lateral_m),
      "target_lateral_m": float(lateral_offset_m),
      "lookahead_m": lookahead_m,
      "heading_error_rad": float(heading_error),
      "raw_steer": raw_steer,
    }


def make_planner(args: argparse.Namespace, engine_name: str) -> ReasonedPlanner | None:
  if engine_name == "stock":
    return None
  if engine_name == "static":
    engine = StaticRtpEngine(SCENARIOS[args.scenario])
  else:
    if args.async_vlm:
      os.environ["RTP_VLM_ASYNC"] = "1"
      os.environ["RTP_VLM_ASYNC_PERIOD_FRAMES"] = str(args.vlm_period_frames)
      os.environ["RTP_VLM_ASYNC_MAX_AGE_FRAMES"] = str(args.vlm_max_age_frames)
      if args.vlm_latest_only:
        os.environ["RTP_VLM_ASYNC_LATEST_ONLY"] = "1"
      if args.vlm_drop_stale_results:
        os.environ["RTP_VLM_ASYNC_DROP_STALE_RESULTS"] = "1"
        os.environ["RTP_VLM_ASYNC_MAX_RESULT_AGE_FRAMES"] = str(args.vlm_max_result_age_frames)
    engine = build_rtp_engine()
  return ReasonedPlanner(
    config=ReasonedPlannerConfig(
      deadline_ms=args.deadline_ms,
      allow_async_rtp=args.async_vlm,
      max_async_age_frames=args.vlm_max_age_frames,
    ),
    renderer=UiSceneBoardRenderer(args.board_width, args.board_height),
    engine=engine,
  )


def run_episode(args: argparse.Namespace, mode: str) -> dict:
  env = make_env(args)
  spawned_scene = spawn_novel_scene(env, args)
  planner = make_planner(args, mode)
  out_dir = args.out / mode
  out_dir.mkdir(parents=True, exist_ok=True)
  records = []
  controller = MetaDriveRouteFollower(max_steer_rate_per_s=args.max_steer_rate_per_s, steer_smoothing_alpha=args.steer_smoothing_alpha)
  active_lateral_offset_m = 0.0
  desired_lateral_offset_m = 0.0
  durable_lateral_plans: dict[str, DurableAvoidance] = {}
  durable_speed_plans: dict[str, DurableSpeedPlan] = {}

  try:
    if planner is not None and mode == "vlm" and args.async_vlm and args.prewarm_seconds > 0:
      warm_until = time.perf_counter() + args.prewarm_seconds
      warm_frame_id = 0
      warm_road = camera_frame(env)
      while time.perf_counter() < warm_until:
        warm_plan = make_base_plan(warm_frame_id, speed_mps(env), curvature=0.0, desired_speed_mps=args.speed_mps)
        warm_result = planner.step(warm_plan, {"v_ego": warm_plan.current_speed, "road_frame": warm_road, "status": "WARM"})
        if warm_result.rtp_text or not warm_result.invalid_reason.startswith("async VLM has no RTP yet"):
          break
        time.sleep(0.05)

    for frame_id in range(args.frames):
      frame_start = time.perf_counter()
      ctrl_dt = args.tick_sec if args.tick_sec > 0 else 0.05
      current_long_m = current_route_longitudinal_m(env)
      durable_lateral_plans = {key: plan for key, plan in durable_lateral_plans.items() if plan.active(current_long_m)}
      durable_speed_plans = {key: plan for key, plan in durable_speed_plans.items() if plan.active(current_long_m)}
      desired_lateral_offset_m = compose_lateral_offset(durable_lateral_plans, current_long_m, args.max_durable_offset_m)
      active_lateral_offset_m = _slew(active_lateral_offset_m, desired_lateral_offset_m, args.max_lateral_offset_rate_mps * ctrl_dt)
      current_speed = speed_mps(env)
      active_speed_cap_mps = None if args.disable_vlm_speed_control else compose_speed_cap(durable_speed_plans, current_long_m, args.speed_mps)
      target_speed = min(args.speed_mps, active_speed_cap_mps) if active_speed_cap_mps is not None else args.speed_mps
      steer_cmd, gas, control_debug = controller.action(env, target_speed, active_lateral_offset_m, ctrl_dt)
      _, reward, terminated, truncated, info = env.step([steer_cmd, gas])
      update_moving_pedestrians(env)
      road = camera_frame(env)
      current_speed = speed_mps(env)
      current_long_m = current_route_longitudinal_m(env)
      base_plan = make_base_plan_from_route(env, frame_id, current_speed, args.speed_mps)

      result = None
      if planner is not None:
        result = planner.step(base_plan, {
          "v_ego": current_speed,
          "road_frame": road,
          "status": "VLM" if mode == "vlm" else "RTP",
        })
        if frame_id % args.save_every == 0 and result.board is not None:
          result.board.save(out_dir / f"vlm_input_{frame_id:04d}.png")
        if result.should_publish and result.synth is not None:
          compiled_lateral_offset_m = float(np.clip(selected_lateral_offset_m(result.synth), -args.max_durable_offset_m, args.max_durable_offset_m))
          new_durable_avoidance = durable_avoidance_from_program(result.program, current_long_m, compiled_lateral_offset_m, args)
          new_speed_plan = None if args.disable_vlm_speed_control else durable_speed_plan_from_program(result.program, current_long_m, args)
          durable_lateral_plans = update_durable_lateral_plans(durable_lateral_plans, new_durable_avoidance, result.program, current_long_m, args)
          durable_speed_plans = update_durable_speed_plans(durable_speed_plans, new_speed_plan, result.program, current_long_m, args)

          desired_lateral_offset_m = compose_lateral_offset(durable_lateral_plans, current_long_m, args.max_durable_offset_m)
          if not durable_lateral_plans:
            desired_lateral_offset_m = openpilot_to_metadrive_lateral_m(compiled_lateral_offset_m)
          active_speed_cap_mps = None if args.disable_vlm_speed_control else compose_speed_cap(durable_speed_plans, current_long_m, args.speed_mps)
      elif frame_id % args.save_every == 0:
        board = UiSceneBoardRenderer(args.board_width, args.board_height).render(base_plan, {"v_ego": current_speed, "road_frame": road, "status": "STOCK"})
        board.save(out_dir / f"stock_overlay_{frame_id:04d}.png")

      min_spawned_distance_m = spawned_min_distance_m(env, spawned_scene)
      active_lateral_plan_records = active_lateral_plans(durable_lateral_plans, current_long_m)
      active_speed_plan_records = active_speed_plans(durable_speed_plans, current_long_m)
      strongest_lateral_plan = max(active_lateral_plan_records, key=lambda plan: abs(plan.target_offset(current_long_m)), default=None)
      record_speed_cap_mps = None if args.disable_vlm_speed_control else compose_speed_cap(durable_speed_plans, current_long_m, args.speed_mps)
      records.append({
        "frame_id": frame_id,
        "mode": mode,
        "speed_mps": current_speed,
        "gas": gas,
        "steer_cmd": steer_cmd,
        "active_lateral_offset_m": active_lateral_offset_m,
        "desired_lateral_offset_m": desired_lateral_offset_m,
        "target_speed_mps": target_speed,
        "vlm_speed_control_enabled": not args.disable_vlm_speed_control,
        "control_debug": control_debug,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info_flags": _jsonable_info_flags(info),
        "min_spawned_object_distance_m": min_spawned_distance_m,
        "durable_avoidance_active": bool(active_lateral_plan_records),
        "durable_avoidance_offset_m": 0.0 if strongest_lateral_plan is None else strongest_lateral_plan.target_offset(current_long_m),
        "durable_avoidance_start_long_m": None if strongest_lateral_plan is None else strongest_lateral_plan.start_long_m,
        "durable_avoidance_end_long_m": None if strongest_lateral_plan is None else strongest_lateral_plan.end_long_m,
        "durable_avoidance_source": "" if strongest_lateral_plan is None else strongest_lateral_plan.source_token,
        "durable_avoidance_source_meta": "" if strongest_lateral_plan is None else strongest_lateral_plan.source_meta,
        "durable_avoidance_confidence": 0.0 if strongest_lateral_plan is None else strongest_lateral_plan.confidence,
        "durable_lateral_plan_count": len(active_lateral_plan_records),
        "durable_lateral_plan_sources": [plan.source_token for plan in active_lateral_plan_records],
        "durable_speed_plan_count": len(active_speed_plan_records),
        "durable_speed_plan_sources": [plan.source_token for plan in active_speed_plan_records],
        "durable_speed_cap_mps": record_speed_cap_mps,
        "route_completion": float(info.get("route_completion", 0.0)) if isinstance(info, dict) else 0.0,
        "reasoned_should_publish": False if result is None else result.should_publish,
        "reasoned_valid": False if result is None else result.valid,
        "reasoned_deadline_met": True if result is None else result.deadline_met,
        "reasoned_latency_ms": 0.0 if result is None else result.timings.publish_age_ms,
        "camera_to_scene_board_ms": 0.0 if result is None else result.timings.camera_to_scene_board_ms,
        "scene_board_to_vlm_prefill_ms": 0.0 if result is None else result.timings.scene_board_to_vlm_prefill_ms,
        "vlm_decode_ms": 0.0 if result is None else result.timings.vlm_decode_ms,
        "rtp_parse_ms": 0.0 if result is None else result.timings.rtp_parse_ms,
        "path_synth_ms": 0.0 if result is None else result.timings.path_synth_ms,
        "rtp_source_frame_id": None if result is None else result.rtp_source_frame_id,
        "rtp_age_frames": None if result is None else result.rtp_age_frames,
        "vlm_backend": "" if result is None else result.vlm_backend,
        "selected_candidate": None if result is None or result.synth is None else result.synth.selected_candidate,
        "path_delta_m": 0.0 if result is None or result.synth is None else result.synth.vlm_changed_path_meters,
        "speed_delta_mps": 0.0 if result is None or result.synth is None else result.synth.vlm_changed_speed_mps,
        "invalid_reason": "" if result is None else result.invalid_reason,
        "rtp_text": "" if result is None else result.rtp_text,
        "spawned_scene": spawned_scene,
      })
      if terminated or truncated:
        break
      if args.tick_sec > 0:
        time.sleep(max(0.0, args.tick_sec - (time.perf_counter() - frame_start)))
  finally:
    if planner is not None:
      planner.engine.close()
    env.close()

  latencies = [r["reasoned_latency_ms"] for r in records if mode != "stock"]
  published = [r for r in records if r["reasoned_should_publish"]]
  rtp_ages = [int(r["rtp_age_frames"]) for r in records if r["rtp_age_frames"] is not None]
  same_frame_records = [
    r for r in records
    if r["rtp_source_frame_id"] is not None and int(r["rtp_source_frame_id"]) == int(r["frame_id"])
  ]
  summary = {
    "mode": mode,
    "frames": len(records),
    "publish_count": len(published),
    "valid_count": sum(1 for r in records if r["reasoned_valid"]),
    "deadline_miss_count": sum(1 for r in records if not r["reasoned_deadline_met"]),
    "mean_speed_mps": statistics.fmean([r["speed_mps"] for r in records]) if records else 0.0,
    "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
    "p90_latency_ms": percentile(latencies, 90),
    "p99_latency_ms": percentile(latencies, 99),
    "p999_latency_ms": percentile(latencies, 99.9),
    "max_latency_ms": max(latencies) if latencies else 0.0,
    "same_frame_count": len(same_frame_records),
    "same_frame_all": len(same_frame_records) == len(records) if mode != "stock" else True,
    "max_rtp_age_frames": max(rtp_ages) if rtp_ages else 0,
    "mean_path_delta_m": statistics.fmean([r["path_delta_m"] for r in published]) if published else 0.0,
    "mean_speed_delta_mps": statistics.fmean([r["speed_delta_mps"] for r in published]) if published else 0.0,
    "records": records,
  }
  (out_dir / "episode.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description="Run an actual MetaDrive camera-frame demo with UI-style VLM overlays.")
  parser.add_argument("--frames", type=int, default=60)
  parser.add_argument("--speed-mps", type=float, default=10.0)
  parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="construction")
  parser.add_argument("--engine", choices=("static", "vlm"), default="static")
  parser.add_argument("--async-vlm", action="store_true")
  parser.add_argument("--vlm-period-frames", type=int, default=2)
  parser.add_argument("--vlm-max-age-frames", type=int, default=8)
  parser.add_argument("--vlm-latest-only", action="store_true")
  parser.add_argument("--vlm-drop-stale-results", action="store_true")
  parser.add_argument("--vlm-max-result-age-frames", type=int, default=8)
  parser.add_argument("--prewarm-seconds", type=float, default=90.0)
  parser.add_argument("--deadline-ms", type=float, default=50.0)
  parser.add_argument("--tick-sec", type=float, default=0.05)
  parser.add_argument("--map", default="3")
  parser.add_argument("--novel-scene", choices=("none", "construction", "pedestrian", "stop_sign", "random_mixed"), default="none")
  parser.add_argument("--camera-width", type=int, default=512)
  parser.add_argument("--camera-height", type=int, default=320)
  parser.add_argument("--board-width", type=int, default=512)
  parser.add_argument("--board-height", type=int, default=384)
  parser.add_argument("--save-every", type=int, default=5)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--avoid-lead-m", type=float, default=10.0)
  parser.add_argument("--avoid-recover-m", type=float, default=10.0)
  parser.add_argument("--min-construction-offset-m", type=float, default=1.25)
  parser.add_argument("--max-durable-offset-m", type=float, default=1.3)
  parser.add_argument("--max-lateral-offset-rate-mps", type=float, default=0.55)
  parser.add_argument("--max-steer-rate-per-s", type=float, default=0.9)
  parser.add_argument("--steer-smoothing-alpha", type=float, default=0.35)
  parser.add_argument("--durable-override-confidence", type=float, default=0.74)
  parser.add_argument("--durable-conflict-override-confidence", type=float, default=0.70)
  parser.add_argument("--disable-vlm-speed-control", action="store_true")
  parser.add_argument("--durable-slow-speed-scale", type=float, default=0.25)
  parser.add_argument("--durable-speed-min-horizon-m", type=float, default=30.0)
  parser.add_argument("--durable-speed-recover-m", type=float, default=10.0)
  parser.add_argument("--random-scene-seed", type=int, default=42)
  parser.add_argument("--random-scene-route-m", type=float, default=180.0)
  parser.add_argument("--random-construction-start-s", type=float, default=14.0)
  parser.add_argument("--random-construction-spacing-min-m", type=float, default=22.0)
  parser.add_argument("--random-construction-spacing-max-m", type=float, default=34.0)
  parser.add_argument("--random-construction-max-objects", type=int, default=4)
  parser.add_argument("--random-construction-right-probability", type=float, default=0.7)
  parser.add_argument("--random-pedestrian-start-s", type=float, default=24.0)
  parser.add_argument("--random-pedestrian-spacing-min-m", type=float, default=32.0)
  parser.add_argument("--random-pedestrian-spacing-max-m", type=float, default=48.0)
  parser.add_argument("--random-pedestrian-speed-min-mps", type=float, default=0.8)
  parser.add_argument("--random-pedestrian-speed-max-mps", type=float, default=1.6)
  parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "metadrive_overlay_demo")
  args = parser.parse_args()

  args.out.mkdir(parents=True, exist_ok=True)
  started = time.perf_counter()
  stock = run_episode(args, "stock")
  reasoned = run_episode(args, args.engine)
  comparison = {
    "engine": args.engine,
    "async_vlm": args.async_vlm,
    "elapsed_sec": time.perf_counter() - started,
    "stock": {k: v for k, v in stock.items() if k != "records"},
    "reasoned": {k: v for k, v in reasoned.items() if k != "records"},
    "delta_mean_speed_mps": reasoned["mean_speed_mps"] - stock["mean_speed_mps"],
    "delta_publish_count": reasoned["publish_count"] - stock["publish_count"],
  }
  out_path = args.out / f"comparison_{args.engine}.json"
  out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
  print(json.dumps(comparison, indent=2))
  print(f"artifacts={args.out}")


if __name__ == "__main__":
  main()
