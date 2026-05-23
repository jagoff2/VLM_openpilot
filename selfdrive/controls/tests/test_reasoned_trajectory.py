import unittest

from selfdrive.controls.reasoned.pathsynth import BasePlan, PathSynth
from selfdrive.controls.reasoned.rtp import RtpValidationError, parse_rtp
from argparse import Namespace

from tools.reasoned_trajectory_poc.qwen_label_rtp_worker import RotatingScoreState, _labels_to_rtp
from tools.reasoned_trajectory_poc.run_metadrive_overlay_demo import (
  DurableAvoidance,
  DurableSpeedPlan,
  compose_lateral_offset,
  _merge_durable_speed_plan,
  durable_avoidance_from_program,
  durable_speed_plan_from_program,
  openpilot_to_metadrive_lateral_m,
  update_durable_lateral_plans,
)


SAMPLE_RTP = """RTPv1
scene=construction_merge
evidence=[cones_right_s22_45,lead_s18_braking,lane_left_open]
meta=BIAS_LEFT_AND_SLOW
branch=base
lat_bias_m=1.25
speed_cap_mps=2.5
stop_s=none
avoid=[right_edge_s8_48_margin1.25]
weights=[obs2.2,lane1.4,comfort1.0,base0.7,vlm1.0]
confidence=0.72"""


class TestRtpParser(unittest.TestCase):
  def test_accepts_bounded_rtp(self):
    program = parse_rtp(SAMPLE_RTP)
    self.assertEqual(program.scene, "construction_merge")
    self.assertEqual(program.meta, "BIAS_LEFT_AND_SLOW")
    self.assertEqual(program.branch, "base")
    self.assertAlmostEqual(program.lat_bias_m, 1.25)
    self.assertAlmostEqual(program.speed_cap_mps, 2.5)
    self.assertIn("cones_right_s22_45", program.evidence)

  def test_rejects_prose(self):
    with self.assertRaises(RtpValidationError):
      parse_rtp("I think the car should move left because cones are visible.")

  def test_rejects_out_of_bounds_bias(self):
    with self.assertRaises(RtpValidationError):
      parse_rtp(SAMPLE_RTP.replace("lat_bias_m=1.25", "lat_bias_m=4.0"))

  def test_accepts_percent_speed_cap(self):
    program = parse_rtp(SAMPLE_RTP.replace("speed_cap_mps=2.5", "speed_cap_mps=25%"))
    self.assertIsNone(program.speed_cap_mps)
    self.assertAlmostEqual(program.speed_scale, 0.25)
    self.assertIn("speed_cap_mps=25%", program.to_wire_text())


class TestPathSynth(unittest.TestCase):
  def test_bias_changes_path_and_speed_only_shrinks(self):
    program = parse_rtp(SAMPLE_RTP)
    base = BasePlan(
      frame_id=42,
      model_log_mono_time_ns=1_000_000,
      t=tuple(i * 0.2 for i in range(17)),
      x=tuple(i * 5.0 for i in range(17)),
      y=tuple(0.0 for _ in range(17)),
      speeds=tuple(15.0 for _ in range(17)),
      desired_curvature=0.0,
      v_ego=15.0,
    )
    result = PathSynth().compile(base, program)
    self.assertTrue(result.valid)
    self.assertEqual(result.frame_id, 42)
    self.assertEqual(result.selected_candidate, "C1")
    selected = next(candidate for candidate in result.candidates if candidate.name == result.selected_candidate)
    self.assertAlmostEqual(selected.lateral_offset_m, 1.25)
    self.assertGreater(result.vlm_changed_path_meters, 0.0)
    self.assertGreater(result.vlm_changed_speed_mps, 0.0)
    self.assertLessEqual(result.speed_cap_mps, base.current_speed)

  def test_high_speed_cap_does_not_expand_speed(self):
    program = parse_rtp(SAMPLE_RTP.replace("speed_cap_mps=11.0", "speed_cap_mps=40.0"))
    base = BasePlan(
      frame_id=1,
      model_log_mono_time_ns=1_000_000,
      t=(0.0, 0.2),
      x=(0.0, 5.0),
      y=(0.0, 0.0),
      speeds=(12.0, 12.0),
      desired_curvature=0.0,
      v_ego=12.0,
    )
    result = PathSynth().compile(base, program)
    self.assertTrue(result.valid)
    self.assertLessEqual(result.speed_cap_mps, 12.0)

  def test_percent_speed_cap_scales_desired_speed(self):
    program = parse_rtp(SAMPLE_RTP.replace("speed_cap_mps=2.5", "speed_cap_mps=25%"))
    base = BasePlan(
      frame_id=2,
      model_log_mono_time_ns=1_000_000,
      t=(0.0, 0.2),
      x=(0.0, 5.0),
      y=(0.0, 0.0),
      speeds=(20.0, 20.0),
      desired_curvature=0.0,
      v_ego=6.0,
    )
    result = PathSynth().compile(base, program)
    self.assertTrue(result.valid)
    self.assertAlmostEqual(result.speed_cap_mps, 5.0)


class TestConstructionSideCompiler(unittest.TestCase):
  def test_left_construction_compiles_to_right_bias(self):
    program = parse_rtp(_labels_to_rtp(("construction_left",)))
    self.assertEqual(program.meta, "BIAS_RIGHT_AND_SLOW")
    self.assertLess(program.lat_bias_m, 0.0)
    self.assertAlmostEqual(program.speed_scale, 0.25)
    self.assertIn("left_edge_s8_48_margin1.25", program.avoid)

  def test_right_construction_compiles_to_left_bias(self):
    program = parse_rtp(_labels_to_rtp(("construction_right",)))
    self.assertEqual(program.meta, "BIAS_LEFT_AND_SLOW")
    self.assertGreater(program.lat_bias_m, 0.0)
    self.assertAlmostEqual(program.speed_scale, 0.25)
    self.assertIn("right_edge_s8_48_margin1.25", program.avoid)

  def test_generic_construction_without_side_does_not_laterally_guess(self):
    program = parse_rtp(_labels_to_rtp(("cones",)))
    self.assertEqual(program.meta, "SLOW")
    self.assertAlmostEqual(program.lat_bias_m, 0.0)
    self.assertEqual(program.avoid, ())


class TestRotatingScoreState(unittest.TestCase):
  def test_durable_construction_label_survives_moderate_negative(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"),),
      cache_ttl_frames=60,
      durable_labels=("cones", "barrier"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(state.update(("cones", "barrier"), ("cones",), {"cones": 0.5, "barrier": -0.5}, 204), ("cones",))
    self.assertEqual(state.update(("cones", "barrier"), ("none",), {"cones": -1.4, "barrier": -3.0}, 252), ("cones",))
    self.assertEqual(state.update(("cones", "barrier"), ("none",), {"cones": -2.2, "barrier": -3.0}, 264), ("none",))

  def test_path_conflict_label_survives_moderate_negative_when_durable(self):
    state = RotatingScoreState(
      groups=(("pedestrian_in_path", "pedestrian_entering_path"),),
      cache_ttl_frames=60,
      durable_labels=("cones", "barrier", "pedestrian_in_path", "pedestrian_entering_path"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("pedestrian_in_path",), {"pedestrian_in_path": 0.5}, 10),
      ("pedestrian_in_path",),
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("none",), {"pedestrian_in_path": -0.1}, 11),
      ("pedestrian_in_path",),
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("none",), {"pedestrian_in_path": -2.1}, 12),
      ("none",),
    )

  def test_non_durable_label_can_still_clear_immediately(self):
    state = RotatingScoreState(
      groups=(("pedestrian_in_path", "pedestrian_entering_path"),),
      cache_ttl_frames=60,
      durable_labels=(),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("pedestrian_in_path",), {"pedestrian_in_path": 0.5}, 10),
      ("pedestrian_in_path",),
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("none",), {"pedestrian_in_path": -0.1}, 11),
      ("none",),
    )

  def test_mutually_exclusive_construction_side_keeps_higher_score(self):
    state = RotatingScoreState(
      groups=(("construction_left", "construction_right"),),
      cache_ttl_frames=60,
      durable_labels=("construction_left", "construction_right"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_left", "construction_right"),
        {"construction_left": 0.2, "construction_right": 1.0},
        10,
      ),
      ("construction_right",),
    )


class TestDurableSpeedPlans(unittest.TestCase):
  def test_percent_speed_cap_uses_desired_speed(self):
    program = parse_rtp(_labels_to_rtp(("construction_right",)))
    args = Namespace(
      speed_mps=12.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=30.0,
      durable_speed_recover_m=10.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
    )
    plan = durable_speed_plan_from_program(program, current_long_m=0.0, args=args)
    self.assertIsNotNone(plan)
    self.assertAlmostEqual(plan.speed_cap_mps, 3.0)

  def test_same_source_new_speed_cap_overrides_stale_stop(self):
    old = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=0.0,
      stop_s=18.0,
      source_token="corridor_object_s18_28",
      source_meta="YIELD",
      confidence=0.70,
    )
    new = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=1.5,
      stop_s=None,
      source_token="corridor_object_s18_28",
      source_meta="mixed_agent_construction",
      confidence=0.72,
    )
    merged = _merge_durable_speed_plan(old, new)
    self.assertAlmostEqual(merged.speed_cap_mps, 1.5)
    self.assertIsNone(merged.stop_s)
    self.assertEqual(merged.source_meta, "mixed_agent_construction")


class TestMetaDriveLateralConvention(unittest.TestCase):
  def test_openpilot_left_is_metadrive_negative(self):
    self.assertLess(openpilot_to_metadrive_lateral_m(1.25), 0.0)
    self.assertGreater(openpilot_to_metadrive_lateral_m(-1.25), 0.0)

  def test_right_edge_avoidance_moves_left_in_metadrive_coordinates(self):
    program = parse_rtp(_labels_to_rtp(("construction_right",)))
    args = Namespace(min_construction_offset_m=1.25, max_durable_offset_m=1.3, avoid_lead_m=10.0, avoid_recover_m=10.0)
    plan = durable_avoidance_from_program(program, current_long_m=0.0, selected_offset_m=program.lat_bias_m, args=args)
    self.assertIsNotNone(plan)
    self.assertLess(plan.offset_m, 0.0)

  def test_left_edge_avoidance_moves_right_in_metadrive_coordinates(self):
    program = parse_rtp(_labels_to_rtp(("construction_left",)))
    args = Namespace(min_construction_offset_m=1.25, max_durable_offset_m=1.3, avoid_lead_m=10.0, avoid_recover_m=10.0)
    plan = durable_avoidance_from_program(program, current_long_m=0.0, selected_offset_m=program.lat_bias_m, args=args)
    self.assertIsNotNone(plan)
    self.assertGreater(plan.offset_m, 0.0)

  def test_confident_contradictory_lateral_plan_replaces_old_side(self):
    args = Namespace(
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.70,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT_AND_SLOW",
      confidence=0.72,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT_AND_SLOW",
      confidence=0.72,
    )
    program = parse_rtp(_labels_to_rtp(("construction_left",)))
    updated = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": existing},
      new,
      program,
      current_long_m=10.0,
      args=args,
    )
    self.assertNotIn("right_edge_s8_48_margin1.25", updated)
    self.assertIn("left_edge_s8_48_margin1.25", updated)
    self.assertGreater(compose_lateral_offset(updated, 10.0, args.max_durable_offset_m), 0.0)


if __name__ == "__main__":
  unittest.main()
