"""Live PICO -> IGRIS-C: XRoboToolkit body tracking retargeted frame by frame and shown in MuJoCo.

The batch pipeline (02_record -> 03_correct -> 04_label -> 05_retarget) stays the reference for
building datasets. This module runs the same chain causally, so the robot follows the person
while they move (and can record the session at the same time, see scripts/live_view.py):

  calibration  the first `calib_s` seconds of body data (stand still in a T-pose, facing the
               direction you will move in): floor, heading, origin, kinematic scaling ratios
  filtering    one-pole low-pass per joint group with the profile's cutoffs (batch: zero-phase)
  contacts     the profile's ankle height / speed thresholds, flat-foot pinning as in the batch path
  retargeting  Retargeter.solve_frame with the calibrated ratios (incl. the balance stage), then
               joint-range clip, joint rate limit and floor contact like Retargeter.postprocess

Only body-tracking profiles (PICO body solver, >= 2 calibrated Motion Trackers) are supported:
the sensor-only solver (hmd_3pt / sparse_trackers) needs the whole clip for its calibration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np

from .pico.correction import ANKLE_HEIGHT, _circular_median, _heading_from_body, _rot_z, profile_config
from .pico.frames import pose7_to_pipe
from .retarget.retargeter import Retargeter
from .skeleton import BONES, J, JOINT_GROUP, NUM_JOINTS
from .utils.rotations import quat_mul, quat_normalize, slerp, yaw_of, yaw_quat


def _alpha(cutoff_hz, dt):
    """One-pole low-pass gain for a sample interval dt."""
    return 1.0 - np.exp(-2.0 * np.pi * np.asarray(cutoff_hz, float) * dt)


@dataclass
class Calibration:
    floor_z: float
    yaw0: float
    origin: np.ndarray
    stature: float
    ratios: dict | None
    still_speed: float  # median pelvis speed in the window, m/s (should be ~0)


class LiveRetargeter:
    """Feed body samples with push(); returns the robot qpos once the calibration window is done."""

    def __init__(self, retarget_cfg: dict, profiles_cfg: dict, profile: str = "body_2trk", calib_s: float = 3.0):
        if not profile.startswith("body"):
            raise ValueError(f"live retargeting needs a body-tracking profile, not {profile!r}")
        self.rt = Retargeter(retarget_cfg)
        self.robot = self.rt.robot
        self.pcfg = profile_config(profiles_cfg, profile)
        self.calib_s = float(calib_s)
        self.calib: Calibration | None = None
        self._window: list[tuple[float, np.ndarray, np.ndarray]] = []
        self.cut = np.array([self.pcfg["cutoff_hz"][JOINT_GROUP[j]] for j in range(NUM_JOINTS)], float)
        self.conf = np.array([self.pcfg["confidence"][JOINT_GROUP[j]] for j in range(NUM_JOINTS)], float)
        fc = self.pcfg["foot_contact"]
        self.contact_height, self.contact_speed = float(fc["height"]), float(fc["speed"])
        self.foot_entries = {e["joint"] for st in self.rt.stages for e in st if e["body"].startswith("Link_Ankle_Roll")}
        pp = retarget_cfg.get("postprocess", {})
        self.q_lo, self.q_hi = self.rt.reference_range()
        ratio = pp.get("max_speed_ratio")
        if ratio:
            from .sim.mujoco_eval import load_joint_params
            self.vmax = load_joint_params(self.robot).velocity * float(ratio)
        else:
            self.vmax = None
        self.ground = pp.get("ground", "contact")
        self.speed_scale = float((self.rt.balance or {}).get("speed_scale", 0.25))
        self.human_pos = None  # last filtered, aligned human joints (for drawing)
        self.reset_state()

    def reset_state(self):
        self._t = None
        self._p = None
        self._q = None
        self._ankle_v = np.zeros(2)
        self._contact_w = np.zeros(2)
        self._pelvis_v = 0.0
        self._dz = 0.0
        self._out = None
        self._cfg = None

    # -- calibration ------------------------------------------------------------------------
    def _calibrate(self):
        t = np.array([w[0] for w in self._window])
        P = np.stack([w[1] for w in self._window])
        ank_z = np.minimum(P[:, J["left_ankle"], 2], P[:, J["right_ankle"], 2])
        floor = float(np.percentile(ank_z, 5) - ANKLE_HEIGHT)
        yaw0 = _circular_median(_heading_from_body(P))
        pel = P[:, J["pelvis"]]
        origin = np.array([pel[:, 0].mean(), pel[:, 1].mean(), floor])
        aligned = (P - origin) @ _rot_z(-yaw0).T
        stature = float(np.median(aligned[:, J["head"], 2]) + 0.12)
        mode = self.rt.cfg.get("scaling", {}).get("mode", "kinematic")
        ratios = self.rt._kinematic_ratios(aligned) if mode == "kinematic" else None
        # net displacement over ~0.5 s (frame-to-frame differences are dominated by solver jitter)
        k = max(1, int(np.searchsorted(t, t[0] + 0.5)))
        still = float(np.median(np.linalg.norm(pel[k:, :2] - pel[:-k, :2], axis=-1) / np.clip(t[k:] - t[:-k], 1e-3, None))) \
            if len(t) > k else 0.0
        self.calib = Calibration(floor, yaw0, origin, stature, ratios, still)
        self._window = []

    # -- per frame ----------------------------------------------------------------------------
    def push(self, t_s: float, pos: np.ndarray, quat: np.ndarray) -> np.ndarray | None:
        """One body sample in the pipeline frame (frames.pose7_to_pipe, not aligned yet)."""
        if self.calib is None:
            self._window.append((t_s, pos.copy(), quat.copy()))
            if t_s - self._window[0][0] >= self.calib_s and len(self._window) >= 10:
                self._calibrate()
            return None
        c = self.calib
        p = (pos - c.origin) @ _rot_z(-c.yaw0).T
        q = quat_mul(np.broadcast_to(yaw_quat(np.array(-c.yaw0)), quat.shape), quat)
        first = self._t is None
        dt = 1.0 / 60.0 if first else float(np.clip(t_s - self._t, 1.0 / 400.0, 0.1))
        self._t = t_s

        # 1) causal low-pass per joint group
        if first:
            self._p, self._q = p.copy(), quat_normalize(q)
            prev_ankles = p[[J["left_ankle"], J["right_ankle"]], :2]
            prev_pelvis = p[J["pelvis"], :2]
        else:
            prev_ankles = self._p[[J["left_ankle"], J["right_ankle"]], :2].copy()
            prev_pelvis = self._p[J["pelvis"], :2].copy()
            a = _alpha(self.cut, dt)[:, None]
            self._p = self._p + a * (p - self._p)
            q = np.where(np.sum(q * self._q, -1, keepdims=True) < 0, -q, q)
            self._q = quat_normalize(self._q + a * (q - self._q))
        p, q = self._p.copy(), self._q.copy()
        self.human_pos = p

        # 2) foot contacts (height + smoothed horizontal speed), smoothed pinning weight
        ankles = p[[J["left_ankle"], J["right_ankle"]]]
        v = np.linalg.norm(ankles[:, :2] - prev_ankles, axis=-1) / dt
        self._ankle_v += _alpha(3.0, dt) * (v - self._ankle_v)
        contact = (ankles[:, 2] - ANKLE_HEIGHT < self.contact_height) & (self._ankle_v < self.contact_speed)
        self._contact_w += _alpha(4.0, dt) * (contact.astype(float) - self._contact_w)
        vp = np.linalg.norm(p[J["pelvis"], :2] - prev_pelvis) / dt
        self._pelvis_v += _alpha(2.0, dt) * (vp - self._pelvis_v)
        staticness = float(np.exp(-(self._pelvis_v / self.speed_scale) ** 2))

        # 3) scale to robot proportions, pin feet in contact
        spos = self.rt.scale_human(p[None], q[None], c.stature, ratios=c.ratios)[0]
        for k, side in enumerate(("left", "right")):
            j = J[f"{side}_ankle"]
            if j in self.foot_entries:
                w = float(np.clip(self._contact_w[k], 0.0, 1.0))
                spos[j, 2] = (1 - w) * spos[j, 2] + w * self.robot.ankle_flat_height
                q[j] = slerp(q[j], yaw_quat(yaw_of(q[j])), w)

        # 4) IK (warm-started from the previous frame)
        if self._cfg is None:
            self._cfg = mink.Configuration(self.rt.model)
            q0 = self.robot.default_qpos.copy()
            q0[:2] = spos[J["pelvis"], :2]
            self._cfg.update(q0)
            self.rt._com_shift = np.zeros(2)
        self.rt.dt_frame = dt
        qik = self.rt.solve_frame(self._cfg, spos, q, self.conf, first=first, contact_t=contact,
                                  staticness=staticness)

        # 5) post: joint range, rate limit, floor (causal versions of Retargeter.postprocess)
        out = qik.copy()
        out[7:] = np.clip(out[7:], self.q_lo, self.q_hi)
        if self._out is not None and self.vmax is not None:
            step = self.vmax * dt
            out[7:] = self._out[7:] + np.clip(out[7:] - self._out[7:], -step, step)
        if self.ground != "none":
            sole = self.robot.sole_min_z(self.robot.forward(out))
            want = -sole[contact].min() if (self.ground == "contact" and contact.any()) else max(0.0, -sole.min())
            self._dz += _alpha(3.0, dt) * (want - self._dz)
            out[2] += max(self._dz, -sole.min())
        self._out = out
        return out


@dataclass
class LiveStats:
    packets: int = 0
    body_packets: int = 0
    solved: int = 0
    ik_ms: list = field(default_factory=list)
    t_first: float | None = None
    t_last: float | None = None

    def summary(self) -> dict:
        span = (self.t_last - self.t_first) if self.t_first is not None and self.t_last is not None else 0.0
        ik = np.array(self.ik_ms) if self.ik_ms else np.zeros(1)
        return {"packets": self.packets, "body_packets": self.body_packets, "solved": self.solved,
                "packet_hz": round(self.packets / span, 1) if span > 0 else 0.0,
                "ik_ms_mean": round(float(ik.mean()), 2), "ik_ms_p95": round(float(np.percentile(ik, 95)), 2)}


def _draw_skeleton(scn, pos, offset, rgba=(0.2, 0.6, 1.0, 1.0)):
    p = pos + offset
    for pa, c in BONES:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.015, p[pa], p[c])
        scn.ngeom += 1


def run_live(source, live: LiveRetargeter, viewer: bool = True, human_offset: float = 1.0,
             max_seconds: float | None = None, recorder=None, on_status=print,
             on_frame=None) -> LiveStats:
    """Poll `source` (XRTSource / SimulatedXRTSource), retarget every new body packet, show it.

    recorder: optional pico.recorder.RecordingSession fed with every packet (A/B buttons).
    on_frame: optional callback(t_s, qpos) for each solved frame (tests, streaming elsewhere).
    """
    stats = LiveStats()
    m = live.robot.model
    d = mujoco.MjData(m)
    d.qpos[:] = live.robot.default_qpos
    mujoco.mj_forward(m, d)
    handle = None
    if viewer:
        import mujoco.viewer as mj_viewer  # (a bare `import mujoco.viewer` here would shadow the module name)
        handle = mj_viewer.launch_passive(m, d)
    offset = np.array([0.0, human_offset, 0.0])
    t_wall0 = time.time()
    warned_no_body = False
    announced = False
    try:
        while handle is None or handle.is_running():
            if max_seconds is not None and time.time() - t_wall0 > max_seconds:
                break
            try:
                fr = source.poll()
            except StopIteration:
                break
            if fr is None:
                time.sleep(0.001)
                continue
            stats.packets += 1
            t_s = fr.t_ns * 1e-9
            stats.t_first = t_s if stats.t_first is None else stats.t_first
            stats.t_last = t_s
            if recorder is not None:
                recorder.feed(fr)
            if fr.body is None:
                if not warned_no_body and live.calib is None and stats.packets > 300 and stats.body_packets == 0:
                    on_status("[live] no body-tracking data: set Motion Tracker to 'Full body' in the headset app "
                              "(>= 2 calibrated trackers) and switch 'Send' on")
                    warned_no_body = True
                continue
            pos, quat, valid = pose7_to_pipe(fr.body)
            if not valid.all():
                continue
            stats.body_packets += 1
            was_calibrated = live.calib is not None
            t0 = time.perf_counter()
            q = live.push(t_s, pos, quat)
            if q is None:
                if stats.body_packets == 1:
                    on_status(f"[live] calibrating: stand still in a T-pose for {live.calib_s:.0f} s ...")
                continue
            if not was_calibrated:
                stats.ik_ms.clear()  # the calibration frame is not representative
            stats.ik_ms.append((time.perf_counter() - t0) * 1e3)
            stats.solved += 1
            if not announced:
                c = live.calib
                on_status(f"[live] calibrated: floor {c.floor_z:.2f} m, heading {np.degrees(c.yaw0):.0f} deg, "
                          f"stature {c.stature:.2f} m, pelvis speed {c.still_speed:.2f} m/s"
                          + ("  (moving during calibration: restart and stand still)" if c.still_speed > 0.15 else ""))
                announced = True
            if on_frame is not None:
                on_frame(t_s, q)
            if handle is not None:
                with handle.lock():
                    d.qpos[:] = q
                    mujoco.mj_forward(m, d)
                    handle.user_scn.ngeom = 0
                    if human_offset is not None and live.human_pos is not None:
                        _draw_skeleton(handle.user_scn, live.human_pos, offset)
                handle.sync()
    except KeyboardInterrupt:
        on_status("[live] interrupted")
    finally:
        if handle is not None:
            handle.close()
    return stats
