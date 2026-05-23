#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import cv2
import numpy as np


FRAME_RE = re.compile(r"_(\d+)\.png$")


def frame_id(path: Path) -> int:
  match = FRAME_RE.search(path.name)
  return int(match.group(1)) if match is not None else 0


def read_frame(path: Path) -> np.ndarray:
  image = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if image is None:
    raise RuntimeError(f"failed to read frame: {path}")
  return image


def draw_label(image: np.ndarray, text: str) -> np.ndarray:
  out = image.copy()
  width = max(130, 12 * len(text))
  cv2.rectangle(out, (8, 8), (8 + width, 38), (0, 0, 0), -1)
  cv2.putText(out, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
  return out


def write_video(frames: list[Path], out_path: Path, fps: float) -> None:
  if not frames:
    raise RuntimeError(f"no frames for {out_path}")

  first = read_frame(frames[0])
  height, width = first.shape[:2]
  writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
  if not writer.isOpened():
    raise RuntimeError(f"failed to open video writer: {out_path}")

  try:
    for frame in frames:
      image = read_frame(frame)
      if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
      writer.write(image)
  finally:
    writer.release()


def nearest_stock_frame(stock_by_id: dict[int, Path], stock_ids: list[int], vlm_id: int) -> Path:
  candidates = [stock_id for stock_id in stock_ids if stock_id <= vlm_id]
  return stock_by_id[candidates[-1]] if candidates else stock_by_id[stock_ids[0]]


def validate_video(path: Path) -> dict[str, float | int]:
  capture = cv2.VideoCapture(str(path))
  try:
    return {
      "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
      "fps": float(capture.get(cv2.CAP_PROP_FPS)),
      "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
      "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
      "bytes": path.stat().st_size,
    }
  finally:
    capture.release()


def render_videos(run_dir: Path, prefix: str, fps: float) -> list[Path]:
  stock_dir = run_dir / "stock"
  vlm_dir = run_dir / "vlm"
  stock_frames = sorted(stock_dir.glob("stock_overlay_*.png"))
  vlm_frames = sorted(vlm_dir.glob("vlm_input_*.png"))
  if not stock_frames:
    raise RuntimeError(f"no stock frames found under {stock_dir}")
  if not vlm_frames:
    raise RuntimeError(f"no VLM frames found under {vlm_dir}")

  video_dir = run_dir / "videos"
  video_dir.mkdir(parents=True, exist_ok=True)

  stock_video = video_dir / f"stock_{prefix}.mp4"
  stock_padded_video = video_dir / f"stock_{prefix}_padded.mp4"
  vlm_video = video_dir / f"vlm_{prefix}.mp4"
  side_by_side_video = video_dir / f"side_by_side_{prefix}.mp4"

  write_video(stock_frames, stock_video, fps)
  write_video(vlm_frames, vlm_video, fps)

  first_vlm = read_frame(vlm_frames[0])
  height, width = first_vlm.shape[:2]
  stock_by_id = {frame_id(path): path for path in stock_frames}
  stock_ids = sorted(stock_by_id)
  last_stock = stock_frames[-1]
  last_stock_id = frame_id(last_stock)

  writer = cv2.VideoWriter(str(stock_padded_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
  if not writer.isOpened():
    raise RuntimeError(f"failed to open video writer: {stock_padded_video}")
  try:
    for vlm_frame in vlm_frames:
      stock_frame = nearest_stock_frame(stock_by_id, stock_ids, frame_id(vlm_frame))
      image = read_frame(stock_frame)
      if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
      if stock_frame == last_stock and last_stock_id < frame_id(vlm_frame):
        image = draw_label(image, f"STOCK ended at frame {last_stock_id}")
      writer.write(image)
  finally:
    writer.release()

  writer = cv2.VideoWriter(str(side_by_side_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
  if not writer.isOpened():
    raise RuntimeError(f"failed to open video writer: {side_by_side_video}")
  try:
    for vlm_frame in vlm_frames:
      vlm_id = frame_id(vlm_frame)
      stock_frame = nearest_stock_frame(stock_by_id, stock_ids, vlm_id)
      left = read_frame(stock_frame)
      right = read_frame(vlm_frame)
      if left.shape[:2] != (height, width):
        left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
      if right.shape[:2] != (height, width):
        right = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
      if stock_frame == last_stock and last_stock_id < vlm_id:
        left = draw_label(left, f"STOCK ended at frame {last_stock_id}")
      else:
        left = draw_label(left, "STOCK")
      right = draw_label(right, "VLM")
      writer.write(np.hstack([left, right]))
  finally:
    writer.release()

  return [side_by_side_video, stock_video, stock_padded_video, vlm_video]


def main() -> None:
  parser = argparse.ArgumentParser(description="Render stock/VLM/side-by-side MP4s from a MetaDrive RTP POC run directory.")
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--prefix", default=None, help="Filename suffix without stock_/vlm_/side_by_side_ prefix.")
  parser.add_argument("--fps", type=float, default=4.0, help="Default 4 FPS matches --save-every 5 at a 20 Hz sim tick.")
  args = parser.parse_args()

  run_dir = args.run_dir
  prefix = args.prefix or run_dir.name
  videos = render_videos(run_dir, prefix, args.fps)
  for video in videos:
    stats = validate_video(video)
    print(
      f"{video} {stats['bytes']} bytes "
      f"frames={stats['frames']} fps={stats['fps']} size={stats['width']}x{stats['height']}"
    )


if __name__ == "__main__":
  main()
