"""Episodes recorded in the simulator, in the igris_pipeline episode format.

    raw/*   every tracking packet, exactly as 02_record.py stores it, so the offline stages
            (scripts/run_pipeline.py) can re-process the session at full quality
    sim/*   one row per control tick:
              sim/t                  (T,)        s since the episode start
              sim/robot/qpos         (T, 38)     simulated robot state (MJCF order, root wxyz)
              sim/robot/target       (T, 38)     streamed retarget target
              sim/human/pos          (T, 24, 3)  streamed corrected skeleton, world frame
              sim/human/contact      (T, 2)
              sim/objects/pos, quat  (T, K, 3/4) dynamic objects (names in attrs sim.objects)
              sim/grasp              (T, 2)      index into sim.objects held by the left / right hand, -1 none
                                                 (grasp.mode physics: held by contact, see igris_sim/world.py)
              sim/locomotion         (T, 3)      thumbstick offset x, y, yaw of the world transform
              sim/hand/cmd           (T, 12)     Robros hand command from the grips (igris_pipeline/hands.py):
                                                 normalized 0..1, motor order (sim.hand_motor_ids)
              sim/hand/joint_pos     (T, 22)     finger angles (sim.hand_joint_names), rad: with physical
                                                 hands the simulated ones (they stop on contact), else the
                                                 command's
            physical hands only (grasp.mode physics):
              sim/hand/joint_target  (T, 22)     finger angles the command maps to, rad
              sim/hand/pose          (T, 2, 7)   left / right hand root pose [pos, quat wxyz], world
              sim/hand/force         (T, 2)      contact normal force of each hand on objects, N
    group attrs "sim": scene name and JSON, physics mode, tracker profile, calibration, joint names,
                       hand_motor_ids / hand_motor_labels / hand_joint_names / hand_grip_full_command,
                       grasp_mode / grasp_cfg (/ hand_physics)
    The controller grips themselves are in raw/buttons/{left,right}_grip.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from igris_pipeline.pico.recorder import buffer_to_episode, next_episode_index
from igris_pipeline.pico.xrt_source import FrameBuffer, XRFrame


class SimRecorder:
    def __init__(self, out_dir: str | Path, min_frames: int = 10):
        self.out_dir = Path(out_dir)
        self.min_frames = int(min_frames)
        self.active = False
        self.buf: FrameBuffer | None = None
        self.rows: list[dict] = []
        self.t0 = 0.0

    def start(self, t: float):
        self.active = True
        self.buf = FrameBuffer()
        self.rows = []
        self.t0 = t

    def add_frame(self, fr: XRFrame):
        if self.active:
            self.buf.append(fr)

    def add_tick(self, t: float, row: dict):
        if self.active:
            self.rows.append({"t": t - self.t0, **row})

    def marker(self, t: float, label: str = "marker"):
        if self.active and self.buf is not None:
            self.buf.events.append((t - self.t0, label))

    def stop(self, meta: dict, sim_attrs: dict) -> Path | None:
        """Write the episode; returns its path, or None if it was too short."""
        self.active = False
        if self.buf is None or len(self.buf) <= self.min_frames or len(self.rows) < 2:
            return None
        # hand tracking (docs/sim_protocol.md "hands") is recorded when the app sent any
        self.buf.with_hands = any(f.left_hand is not None or f.right_hand is not None for f in self.buf.frames)
        ep = buffer_to_episode(self.buf, meta)
        cols = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        ep["sim/t"] = cols["t"].astype(np.float64)
        ep["sim/robot/qpos"] = cols["qpos"]
        ep["sim/robot/target"] = cols["target"]
        ep["sim/human/pos"] = cols["human_pos"].astype(np.float32)
        ep["sim/human/contact"] = cols["contact"]
        ep["sim/objects/pos"] = cols["obj_pos"].astype(np.float32)
        ep["sim/objects/quat"] = cols["obj_quat"].astype(np.float32)
        ep["sim/grasp"] = cols["grasp"].astype(np.int16)
        ep["sim/locomotion"] = cols["loco"]
        if "hand_cmd" in cols:
            ep["sim/hand/cmd"] = cols["hand_cmd"].astype(np.float32)
            ep["sim/hand/joint_pos"] = cols["hand_q"].astype(np.float32)
        if "hand_pose" in cols:
            ep["sim/hand/joint_target"] = cols["hand_target"].astype(np.float32)
            ep["sim/hand/pose"] = cols["hand_pose"].astype(np.float32)
            ep["sim/hand/force"] = cols["hand_force"].astype(np.float32)
        ep.group_attrs["sim"] = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                                 for k, v in sim_attrs.items()}
        path = self.out_dir / f"ep_{next_episode_index(self.out_dir):04d}.h5"
        ep.save(path)
        self.buf, self.rows = None, []
        return path
