"""One teleoperation session: tracking frames in, robot + scene state out, episodes recorded.

Life cycle
  calibrating  frames are collected for calibration.duration_s; the tracker profile is chosen
               from what the headset streams (igris_pipeline.pico.device_probe.select_profile)
               unless given; the operator stands still (T-pose or relaxed). A failed calibration
               (moving, missing sensors) is reported in `message` and retried.
  running      every tick: correct the newest frame, apply thumbstick locomotion, retarget,
               update grasps from the grip values, step physics, record.

Controller mapping (right / left as on the PICO controllers)
  A          start / stop an episode          B          event marker
  grips      grasp / release with that hand   hold Y     recalibrate
  left stick move (relative to the head)      right stick x  turn about the robot

World frame: MuJoCo / pipeline convention (Z up, +X forward). The calibrated human frame is placed
in the world by the spawn pose of the scene and the accumulated thumbstick locomotion.
"""
from __future__ import annotations

import dataclasses
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from igris_pipeline.pico.device_probe import ProbeReport, select_profile
from igris_pipeline.pico.xrt_source import XRFrame
from igris_pipeline.retarget.ik_config import load as load_ik_config
from igris_pipeline.retarget.robot import ROOT
from igris_pipeline.utils.rotations import yaw_of, yaw_quat

from .protocol import ROBOT_ASSETS
from .recording import SimRecorder
from .scene import SCENES_DIR, list_scenes, load_scene
from .streaming import CalibrationError, StreamingCorrector, StreamingRetargeter, _rot_z
from .world import SimWorld

CFG = ROOT / "configs"
PROTOCOL_VERSION = 1
HEAD_CAM_BODY = "Link_Neck_Pitch"
HEAD_CAM_OFFSET = np.array([0.08, 0.0, 0.10])  # robot "eyes" in the neck-pitch link frame


def load_sim_config(path: str | Path = CFG / "sim.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _yaml(name):
    with open(CFG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _r(a, n=5):
    return np.round(np.asarray(a, float), n).tolist()


class SimSession:
    def __init__(self, sim_cfg: dict, scene: str = "tabletop", profile: str = "auto", mode: str | None = None,
                 subject_height: float | None = None, body_trackers: int | None = None,
                 record_dir: str | Path | None = None, subject: str = "anonymous",
                 tracker_cfg: dict | None = None, retarget_cfg: dict | None = None):
        self.cfg = sim_cfg
        self.fps = float(sim_cfg.get("rate_hz", 50))
        self.tracker_cfg = tracker_cfg or _yaml("tracker_profiles.yaml")
        self.retarget_cfg = retarget_cfg or load_ik_config(CFG / "retarget_igris_c.yaml")
        self.profile_req = profile
        self.subject_height = subject_height
        self.body_trackers = body_trackers
        self.subject = subject
        self.world = SimWorld(load_scene(scene), sim_cfg, mode)
        self.retargeter = StreamingRetargeter(self.retarget_cfg, self.fps)
        rc = sim_cfg.get("recording", {})
        self.recorder = SimRecorder(record_dir, rc.get("min_frames", 10)) if record_dir else None
        self.saved: list[str] = []
        self.events = deque(maxlen=20)
        self.timing = {"correct_ms": 0.0, "retarget_ms": 0.0, "physics_ms": 0.0}
        self.latest: XRFrame | None = None
        self.n_frames = 0
        self._new_frame = False
        self._last_rx = None  # PC time of the newest frame (the headset clock is not synchronised)
        self.recalibrate()

    # ---- calibration -----------------------------------------------------------------
    def recalibrate(self):
        if self.recorder is not None and self.recorder.active:
            self._stop_recording()
        self.status = "calibrating"
        self.message = "stand still for calibration"
        self.progress = 0.0
        self.corrector: StreamingCorrector | None = None
        self.profile = None
        self.calibration = None
        self._calib: list[XRFrame] = []
        self._buttons_prev = {}
        self._pressed: set[str] = set()  # A / B seen pressed in any frame since the last tick
        self._body_frame: XRFrame | None = None  # newest body-carrying frame since the last tick
        self._y_since = None
        self.pose = None
        self._reset_motion()

    def _reset_motion(self):
        xy, yaw = self.world.scene.spawn
        self.loco = np.array([xy[0], xy[1], yaw], float)  # world transform of the calibrated frame
        self.retargeter.reset()
        self.world.reset()
        self.q_target = self.world.robot_qpos()

    def _choose_profile(self, frames) -> str:
        if self.profile_req and self.profile_req != "auto":
            return self.profile_req
        rep = ProbeReport(connected=True, headset_active=True, declared_body_trackers=self.body_trackers)
        rep.body_tracking = sum(f.body is not None for f in frames) >= 2
        rep.object_trackers = max((len(f.tracker_serials) for f in frames if f.trackers is not None), default=0)
        return select_profile(rep)

    def _calibration_tick(self):
        c = self.cfg.get("calibration", {})
        dur = float(c.get("duration_s", 3.0))
        if len(self._calib) < 2:
            return
        span = (self._calib[-1].t_ns - self._calib[0].t_ns) * 1e-9
        self.progress = min(1.0, span / dur)
        if span < dur:
            return
        profile = self._choose_profile(self._calib)
        corrector = StreamingCorrector(self.tracker_cfg, profile, self.subject_height,
                                       float(c.get("max_still_speed", 0.4)), rate_hz=self.fps)
        try:
            res = corrector.calibrate(self._calib)
        except CalibrationError as e:
            self.message = f"calibration failed: {e}"
            self._calib = self._calib[-int(0.5 * self.fps):]  # keep collecting
            return
        self.corrector, self.profile, self.calibration = corrector, profile, res["calibration"]
        self.retargeter.calibrate(res["arrays"]["proc/body/pos"], res["calibration"].get("stature"))
        self._reset_motion()
        self.status, self.message, self._calib = "running", "", []
        self._event("calibrated", profile)

    # ---- input -------------------------------------------------------------------------
    def push_frame(self, fr: XRFrame, rx_time: float | None = None):
        self.latest = fr
        self._new_frame = True
        self.n_frames += 1
        self._last_rx = time.time() if rx_time is None else rx_time
        # only the newest frame is processed per tick: keep what intermediate frames carried
        if fr.body is not None:
            self._body_frame = fr
        self._pressed.update(k for k in ("A", "B") if fr.buttons.get(k))
        if self.status == "calibrating":
            if self._calib and fr.t_ns < self._calib[-1].t_ns:  # headset clock restarted (app restart)
                self._calib = []
            self._calib.append(fr)
        if self.recorder is not None:
            self.recorder.add_frame(fr)

    def tick(self, now: float | None = None):
        """One control step (call at rate_hz)."""
        now = time.time() if now is None else now
        dt = 1.0 / self.fps
        if self.status == "calibrating":
            self._pressed.clear()  # buttons are ignored while calibrating
            self._body_frame = None
            self._calibration_tick()
            t0 = time.perf_counter()
            self.world.step(self.q_target, dt)  # objects keep simulating while the robot stands
            self.timing["physics_ms"] = (time.perf_counter() - t0) * 1e3
            return
        fr = self.latest
        if self._new_frame and fr is not None:
            self._new_frame = False
            if fr.body is None and self._body_frame is not None:
                # PICO streams body data slower than the headset: use the newest body sample of the
                # frames received since the last tick instead of treating this tick as stale
                fr = dataclasses.replace(fr, body=self._body_frame.body, body_joint_ts=self._body_frame.body_joint_ts)
            self._body_frame = None
            self._buttons(fr, now)
            if self.status != "running":  # recalibration requested by a button
                return
            t0 = time.perf_counter()
            self.pose = self.corrector.process(fr)
            self.timing["correct_ms"] = (time.perf_counter() - t0) * 1e3
            self._locomotion(fr, dt)
            self.q_target = self.retargeter.step(self.pose, self.loco[2], np.array([self.loco[0], self.loco[1], 0.0]))
            self.timing["retarget_ms"] = self.retargeter.timing_ms
            for side, action, obj in self.world.update_grasp({s: float(fr.buttons.get(f"{s}_grip", 0.0))
                                                              for s in ("left", "right")}):
                self._event(action, f"{side}:{obj}")
        t0 = time.perf_counter()
        self.world.step(self.q_target, dt)
        self.timing["physics_ms"] = (time.perf_counter() - t0) * 1e3
        if self.recorder is not None and self.recorder.active and self.pose is not None:
            self.recorder.add_tick(now, self._tick_row())

    def _buttons(self, fr, now):
        b = fr.buttons
        rising = {k: (bool(b.get(k)) or k in self._pressed) and not self._buttons_prev.get(k, False)
                  for k in ("A", "B", "Y")}
        self._buttons_prev = {k: bool(b.get(k)) for k in ("A", "B", "Y")}
        self._pressed.clear()
        if rising["A"]:
            self.set_recording(not (self.recorder is not None and self.recorder.active), now)
        if rising["B"] and self.recorder is not None and self.recorder.active:
            self.recorder.marker(now)
            self._event("marker", "")
        hold = float(self.cfg.get("calibration", {}).get("recalibrate_hold_s", 1.0))
        if b.get("Y"):
            self._y_since = now if self._y_since is None else self._y_since
            if now - self._y_since >= hold:
                self.recalibrate()
        else:
            self._y_since = None

    def _locomotion(self, fr, dt):
        lc = self.cfg.get("locomotion", {})
        dz = float(lc.get("deadzone", 0.15))

        def dead(v):
            v = float(v)
            return 0.0 if abs(v) < dz else np.sign(v) * (abs(v) - dz) / (1.0 - dz)

        left = fr.axes.get("left", np.zeros(2)) if fr.axes else np.zeros(2)
        right = fr.axes.get("right", np.zeros(2)) if fr.axes else np.zeros(2)
        lx, ly, rx = dead(left[0]), dead(left[1]), dead(right[0])
        if lx or ly:
            heading = self.loco[2] + float(yaw_of(self.pose.head_quat))
            fwd = np.array([np.cos(heading), np.sin(heading)])
            lft = np.array([-np.sin(heading), np.cos(heading)])
            self.loco[:2] += (ly * fwd - lx * lft) * float(lc.get("max_speed", 1.0)) * dt
        if rx:
            d = -rx * np.radians(float(lc.get("turn_rate_deg", 90.0))) * dt  # stick right -> turn right
            c = self.q_target[:2]  # turn about the robot's current position
            R2 = _rot_z(d)[:2, :2]
            self.loco[:2] = c + R2 @ (self.loco[:2] - c)
            self.loco[2] += d

    # ---- recording -----------------------------------------------------------------------
    def set_recording(self, on: bool, now: float | None = None):
        now = time.time() if now is None else now
        if self.recorder is None:
            self.message = "recording disabled (no --record-dir)"
            return
        if on and not self.recorder.active and self.status == "running":
            self.recorder.start(now)
            self._event("record", "start")
        elif not on and self.recorder.active:
            self._stop_recording()

    def _stop_recording(self):
        meta = {"subject": self.subject, "tracker_profile": self.profile or "", "source": "igris_sim",
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if self.subject_height:
            meta["subject_height"] = float(self.subject_height)
        sim_attrs = {"scene": self.world.scene.name, "scene_json": self.world.scene.spec, "mode": self.world.mode,
                     "profile": self.profile or "", "calibration": self.calibration or {}, "fps": self.fps,
                     "objects": self.world.dynamic, "joint_names": list(self.world.robot.joint_names)}
        path = self.recorder.stop(meta, sim_attrs)
        if path is None:
            self._event("record", "discarded (too short)")
        else:
            self.saved.append(str(path))
            self._event("record", f"saved {path.name}")

    def _tick_row(self) -> dict:
        w = self.world
        pos, quat = w.object_poses()
        names = w.dynamic
        Rw = _rot_z(self.loco[2])
        return {"qpos": w.robot_qpos(), "target": self.q_target.copy(),
                "human_pos": self.pose.pos @ Rw.T + np.array([self.loco[0], self.loco[1], 0.0]),
                "contact": self.pose.contact.copy(), "obj_pos": pos, "obj_quat": quat,
                "grasp": np.array([names.index(w.grasped[s]) if w.grasped[s] in names else -1
                                   for s in ("left", "right")]),
                "loco": self.loco.copy()}

    # ---- commands from the app / CLI -------------------------------------------------------
    def command(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        if cmd == "hello":
            return self.scene_message()
        if cmd == "list_scenes":
            return {"v": PROTOCOL_VERSION, "type": "scenes", "names": list_scenes(), "current": self.world.scene.name}
        if cmd == "load_scene":
            return self.load_scene(str(msg.get("scene", "")))
        if cmd == "set_mode":
            return self.set_mode(str(msg.get("mode", "")))
        if cmd == "reset":
            self._reset_motion()
            return self._ack(cmd)
        if cmd == "recalibrate":
            self.recalibrate()
            return self._ack(cmd)
        if cmd == "record":
            on = msg.get("on", True)
            if isinstance(on, str):  # tolerate "false" / "0" from loosely typed clients
                on = on.strip().lower() in ("1", "true", "yes", "on")
            self.set_recording(bool(on))
            return self._ack(cmd, recording=bool(self.recorder and self.recorder.active))
        if cmd == "marker":
            if self.recorder is not None and self.recorder.active:
                self.recorder.marker(time.time())
            return self._ack(cmd)
        return {"v": PROTOCOL_VERSION, "type": "error", "error": f"unknown command {cmd!r}"}

    def load_scene(self, name: str) -> dict:
        try:  # build the new world first: a scene that does not compile must not end the session
            world = SimWorld(load_scene(name), self.cfg, self.world.mode)
        except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
            return {"v": PROTOCOL_VERSION, "type": "error", "error": f"{type(e).__name__}: {e}"}
        if self.recorder is not None and self.recorder.active:
            self._stop_recording()
        self.world = world
        self._reset_motion()
        self._event("scene", name)
        return self.scene_message()

    def set_mode(self, mode: str) -> dict:
        if mode not in ("kinematic", "harness"):
            return {"v": PROTOCOL_VERSION, "type": "error", "error": f"unknown mode {mode!r}"}
        if self.recorder is not None and self.recorder.active:
            self._stop_recording()
        self.world = SimWorld(self.world.scene, self.cfg, mode)
        self._reset_motion()
        return self._ack("set_mode", mode=mode)

    def _ack(self, cmd, **kw):
        return {"v": PROTOCOL_VERSION, "type": "ack", "cmd": cmd, **kw}

    def _event(self, kind, detail):
        self.events.append({"t": round(time.time(), 3), "kind": kind, "detail": detail})

    # ---- messages -------------------------------------------------------------------------
    def rig_pose(self):
        """Pose of the PICO tracking space (pipeline axes) in the world: p_world = R p_tracking + t."""
        if self.corrector is None or self.corrector.calibration is None:
            xy, yaw = self.world.scene.spawn
            return np.array([xy[0], xy[1], 0.0]), yaw_quat(np.array(yaw))
        yaw = self.loco[2] - self.corrector.yaw0
        R = _rot_z(yaw)
        t = np.array([self.loco[0], self.loco[1], 0.0]) - R @ self.corrector.origin
        return t, yaw_quat(np.array(yaw))

    def head_cam(self):
        d, b = self.world.data, self.world.robot.bid(HEAD_CAM_BODY)
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ HEAD_CAM_OFFSET, d.xquat[b].copy()

    def state_message(self) -> dict:
        w = self.world
        pos, quat = w.object_poses()
        rig_p, rig_q = self.rig_pose()
        cam_p, cam_q = self.head_cam()
        age = None if self._last_rx is None else round(time.time() - self._last_rx, 3)
        return {
            "v": PROTOCOL_VERSION, "type": "state", "t": round(time.time(), 4),
            "status": self.status, "message": self.message, "progress": round(self.progress, 3),
            "profile": self.profile, "mode": w.mode, "scene": w.scene.name,
            "robot": {"qpos": _r(w.robot_qpos())},
            "objects": {"pos": _r(pos), "quat": _r(quat)},
            "grasp": {s: w.grasped[s] for s in ("left", "right")},
            "rig": {"pos": _r(rig_p), "quat": _r(rig_q)},
            "head_cam": {"pos": _r(cam_p), "quat": _r(cam_q)},
            "recording": bool(self.recorder is not None and self.recorder.active),
            "episodes": len(self.saved), "events": list(self.events),
            "harness_pct_mg": round(w.harness_force, 1) if w.mode == "harness" else None,
            "timing_ms": {k: round(v, 2) for k, v in self.timing.items()},
            "frames": self.n_frames, "data_age_s": age,
        }

    def scene_message(self, asset_base: str | None = None) -> dict:
        w = self.world
        try:  # scene files (meshes) are served from scenes/: <assets><folder>/<file>
            folder = w.scene.folder.resolve().relative_to(SCENES_DIR.resolve()).as_posix()
        except ValueError:
            folder = None
        return {"v": PROTOCOL_VERSION, "type": "scene", "name": w.scene.name, "folder": folder, "spec": w.scene.spec,
                "dynamic": w.dynamic, "joint_names": list(w.robot.joint_names),
                "robot_mjcf": str(Path(w.robot.path).relative_to(ROOT)).replace("\\", "/"),
                "robot": ROBOT_ASSETS + "robot.json", "assets": asset_base, "rate_hz": self.fps}

    def close(self):
        if self.recorder is not None and self.recorder.active:
            self._stop_recording()


def summary_line(s: SimSession) -> str:
    st = s.state_message()
    tm = st["timing_ms"]
    return (f"[{st['status']}] profile={st['profile']} mode={st['mode']} scene={st['scene']} frames={st['frames']} "
            f"correct={tm['correct_ms']:.1f}ms ik={tm['retarget_ms']:.1f}ms phys={tm['physics_ms']:.1f}ms "
            f"grasp={json.dumps(st['grasp'])} rec={st['recording']} saved={st['episodes']}"
            + (f" | {st['message']}" if st["message"] else ""))
