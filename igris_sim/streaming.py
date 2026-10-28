"""Causal (real-time) versions of pipeline stages 1-2 (correction) and 3 (retargeting).

Calibration reuses the offline code unchanged: the first seconds of the stream (a still pose)
are packed into an Episode and run through igris_pipeline.pico.correction.correct_episode, which
returns the world alignment, bone lengths, controller / tracker offsets (`res["stream"]`) and the
corrected window from which the retargeter's body-part scale factors are computed.

Afterwards every frame is processed causally with the same steps and parameters (per-profile
confidence / cutoffs from configs/tracker_profiles.yaml):

  body_2trk / body_3trk   align -> outlier gate -> 2nd-order low-pass per joint group ->
                          anchor wrists to the controllers and re-solve limbs with the calibrated
                          bone lengths -> foot contact -> foot lock while in contact
  sparse_trackers / hmd_3pt
                          align -> SparseBodySolver (trunk heading / pitch filtered causally)
                          -> low-pass per joint group -> foot contact -> foot lock

The zero-phase filters of the offline stages become causal Butterworth filters with the same
cutoffs, so the streamed skeleton lags by roughly one filter delay (~20-45 ms). Recorded raw data
can always be re-processed offline with the full-quality stages.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import mink
import numpy as np
from scipy.signal import butter

from igris_pipeline.pico.correction import (ANKLE_HEIGHT, SparseBodySolver, apply_tracker_offsets, correct_episode,
                                            profile_config, solve_limb, trunk_angles_from_head)
from igris_pipeline.pico.frames import pose7_to_pipe
from igris_pipeline.pico.recorder import buffer_to_episode
from igris_pipeline.pico.xrt_source import FrameBuffer, XRFrame
from igris_pipeline.retarget.retargeter import Retargeter
from igris_pipeline.skeleton import BONES, J, JOINT_GROUP, NUM_JOINTS
from igris_pipeline.utils.rotations import (quat_apply, quat_conj, quat_mul, quat_normalize, slerp, yaw_of,
                                            yaw_quat)

GROUPS = sorted(set(JOINT_GROUP))
GROUP_IDX = {g: [j for j in range(NUM_JOINTS) if JOINT_GROUP[j] == g] for g in GROUPS}


class CalibrationError(RuntimeError):
    """The calibration window cannot be used (subject moving, sensors missing)."""


# ----------------------------------------------------------------------------------------
# causal filters
# ----------------------------------------------------------------------------------------
class Biquad:
    """2nd-order Butterworth low-pass, causal, element-wise over arrays of any fixed shape.

    The state is initialised to steady state on the first sample (no start-up transient).
    A cutoff of None / >= Nyquist passes the signal through.
    """

    def __init__(self, fps: float, cutoff: float | None):
        self.passthrough = cutoff is None or cutoff <= 0 or cutoff >= 0.5 * fps
        if not self.passthrough:
            b, a = butter(2, cutoff / (0.5 * fps))
            self.b, self.a = b / a[0], a / a[0]
        self.z1 = self.z2 = None

    def reset(self):
        self.z1 = self.z2 = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.passthrough:
            return x.copy()
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        if self.z1 is None:
            self.z1 = (1.0 - b0) * x
            self.z2 = (b2 - a2) * x
        y = b0 * x + self.z1
        self.z1 = b1 * x - a1 * y + self.z2
        self.z2 = b2 * x - a2 * y
        return y


class QuatFilter:
    """Biquad on quaternion components with sign continuity, re-normalised. (...,4) wxyz."""

    def __init__(self, fps, cutoff):
        self.f = Biquad(fps, cutoff)
        self.prev = None

    def reset(self):
        self.f.reset()
        self.prev = None

    def __call__(self, q):
        q = np.asarray(q, dtype=np.float64).copy()
        if self.prev is not None:
            flip = np.sum(q * self.prev, axis=-1) < 0
            q[flip] *= -1.0
        out = quat_normalize(self.f(q))
        self.prev = out
        return out


class Debounce:
    """Boolean state that switches only after n consecutive samples of the other value."""

    def __init__(self, n: int, shape=()):
        self.n = max(1, int(n))
        self.state = None
        self.count = np.zeros(shape, int)

    def __call__(self, x):
        x = np.asarray(x, bool)
        if self.state is None:
            self.state = x.copy()
            return self.state.copy()
        other = x != self.state
        self.count = np.where(other, self.count + 1, 0)
        flip = self.count >= self.n
        self.state = np.where(flip, x, self.state)
        self.count[flip] = 0
        return self.state.copy()


def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------------------------
# streaming correction
# ----------------------------------------------------------------------------------------
@dataclass
class HumanPose:
    """Corrected human skeleton in the calibrated frame (floor z=0, start pose at the origin facing +X)."""
    t: float
    pos: np.ndarray                  # (24,3)
    quat: np.ndarray                 # (24,4) wxyz
    conf: np.ndarray                 # (24,)
    contact: np.ndarray              # (2,) bool, left / right foot
    valid: bool
    head_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    head_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    ctrl: dict = field(default_factory=dict)   # side -> (pos, quat, valid)


class StreamingCorrector:
    """rate_hz: rate at which process() is called (the filters' time base); default = the profile fps."""

    def __init__(self, cfg_all: dict, profile: str, subject_height: float | None = None,
                 max_still_speed: float = 0.4, rate_hz: float | None = None):
        self.cfg_all = cfg_all
        self.profile = profile
        self.cfg = profile_config(cfg_all, profile)
        self.fps = float(rate_hz or self.cfg["fps"])
        self.dt = 1.0 / self.fps
        self.is_body = profile.startswith("body")
        self.subject_height = subject_height
        self.max_still_speed = float(max_still_speed)
        self.calibration: dict | None = None

    # -- calibration ------------------------------------------------------------------
    def calibrate(self, frames: list[XRFrame]) -> dict:
        """Calibrate on a still window of frames; returns the correct_episode() result."""
        if len(frames) < 10:
            raise CalibrationError("not enough frames for calibration")
        buf = FrameBuffer()
        for f in frames:
            buf.append(f)
        meta = {"tracker_profile": self.profile}
        if self.subject_height:
            meta["subject_height"] = float(self.subject_height)
        ep = buffer_to_episode(buf, meta)
        try:
            res = correct_episode(ep, self.cfg_all, self.profile, self.subject_height)
        except ValueError as e:  # e.g. body profile without body data
            raise CalibrationError(str(e)) from e
        cal = res["calibration"]
        a, b = cal["still_window"]
        # headset + controllers (3-point / sparse) or body tracking (body profiles) must be tracked
        tracked = float(np.mean(res["arrays"]["proc/valid"][a:b])) if b > a else 0.0
        if tracked < 0.8:
            raise CalibrationError(f"sensors not tracked during calibration ({tracked:.0%} of the window; "
                                   "headset / controllers / body tracking)")
        # stillness of the corrected (low-passed) skeleton in the calibration window: sensor jitter
        # alone already gives ~0.1-0.5 m/s raw speeds, real motion shows up here
        pos = res["arrays"]["proc/body/pos"][a:b]
        speed = float(np.linalg.norm(np.gradient(pos, 1.0 / res["fps"], axis=0), axis=-1).mean(axis=1).max()) \
            if len(pos) > 2 else np.inf
        cal["still_speed_filtered"] = speed
        if speed > self.max_still_speed:
            raise CalibrationError(f"moving during calibration (peak {speed:.2f} m/s > {self.max_still_speed:.2f}); "
                                   "stand still")
        st = res["stream"]
        self.origin = np.asarray(st["origin"], float)
        self.yaw0 = float(st["yaw0"])
        self.Rz = _rot_z(-self.yaw0)
        self.q_align = yaw_quat(np.array(-self.yaw0))
        self.lengths = np.asarray(st["lengths"], float)
        self.ctrl_off = dict(st["ctrl_offsets"])
        self.hmd_to_head = st.get("hmd_to_head")
        serials = json.loads(ep.attrs.get("tracker_serials", "[]"))
        # tracker k of the calibration episode = k-th serial seen; map offsets to serial numbers
        self.trk_off = {(serials[k] if k < len(serials) else str(k)): v for k, v in st["tracker_offsets"].items()}
        self.calibration = cal
        self._reset_state()
        if self.is_body:  # start from the last calibration body sample, not an empty skeleton
            last = next((f for f in reversed(frames) if f.body is not None), None)
            if last is not None:
                bp, bq, bv = self._align(last.body)
                if bv.all():
                    self._last_body = (last.t_ns * 1e-9, bp, bq)
                    self._raw_prev = bp.copy()
        return res

    def _reset_state(self):
        cut, fps = self.cfg["cutoff_hz"], self.fps
        self.f_pos = {g: Biquad(fps, cut[g]) for g in GROUPS}
        self.f_quat = {g: QuatFilter(fps, cut[g]) for g in GROUPS}
        self.f_conf = Biquad(fps, 2.0)
        self.f_ankle = Biquad(fps, 3.0)
        self.f_yaw = Biquad(fps, 0.8)
        self.f_pitch = Biquad(fps, 1.0)
        fc = self.cfg["foot_contact"]
        self.contact = Debounce(max(1, int(fc["min_duration_s"] * fps)), (2,))
        self._prev_ankle_xy = None
        self._raw_prev = None
        self._out_count = np.zeros(NUM_JOINTS, int)
        self._last_body = None          # (t, pos, quat)
        self._yaw_prev = None
        self._lock = [None, None]       # foot lock xy while in contact
        self._lock_w = np.zeros(2)
        self._held = {}                 # sensor -> last valid aligned (pos, quat)
        self.solver = SparseBodySolver(self.lengths) if not self.is_body else None

    def _align(self, pose7):
        p, q, v = pose7_to_pipe(np.asarray(pose7, float))
        return (p - self.origin) @ self.Rz.T, quat_mul(np.broadcast_to(self.q_align, q.shape), q), v

    def _hold_valid(self, key, p, q, v):
        """Last valid pose while a sensor is not tracked (an untracked pose is all zeros), as the
        offline fill_invalid() does; validity still lowers the confidence."""
        if bool(v):
            self._held[key] = (p, q)
        elif key in self._held:
            p, q = self._held[key]
        return p, q

    # -- per frame ----------------------------------------------------------------------
    def process(self, fr: XRFrame) -> HumanPose:
        if self.calibration is None:
            raise RuntimeError("calibrate() first")
        t = fr.t_ns * 1e-9
        hp, hq, hv = self._align(fr.headset)
        hp, hq = self._hold_valid("head", hp, hq, hv)
        ctrl = {}
        for s in ("left", "right"):
            p, q, v = self._align(getattr(fr, f"{s}_ctrl"))
            p, q = self._hold_valid(s, p, q, v)
            ctrl[s] = (p, q, bool(v))
        if self.is_body:
            pos, quat, valid, outl, stale = self._body(fr, t, ctrl)
        else:
            pos, quat, valid, outl, stale = self._sparse(fr, hp, hq, ctrl)
            valid = valid and bool(hv)
        contact = self._contact(pos)
        if self.cfg.get("foot_skate_cleanup", False):
            self._foot_lock(pos, quat, contact)
        base = np.array([self.cfg["confidence"][JOINT_GROUP[j]] for j in range(NUM_JOINTS)])
        conf = base * float(valid) * (0.6 if stale else 1.0)
        conf = np.where(outl, conf * 0.4, conf)
        for s in ("left", "right"):
            if not ctrl[s][2]:
                conf[[J[f"{s}_wrist"], J[f"{s}_hand"]]] *= 0.5
        conf = np.clip(self.f_conf(conf), 0.0, 1.0)
        return HumanPose(t=t, pos=pos, quat=quat_normalize(quat), conf=conf, contact=contact, valid=bool(valid),
                         head_pos=hp, head_quat=hq, ctrl=ctrl)

    def _filter_groups(self, pos, quat):
        pos, quat = pos.copy(), quat.copy()
        for g in GROUPS:
            idx = GROUP_IDX[g]
            pos[idx] = self.f_pos[g](pos[idx])
            quat[idx] = self.f_quat[g](quat[idx])
        return pos, quat

    def _body(self, fr, t, ctrl):
        outl = np.zeros(NUM_JOINTS, bool)
        fresh = None
        if fr.body is not None:
            bp, bq, bv = self._align(fr.body)
            if bv.all():
                fresh = (bp, bq)
        if fresh is not None:
            bp, bq = fresh
            stale, valid = False, True
            if self._raw_prev is not None:
                dt = max(t - self._last_body[0], self.dt)
                d = np.linalg.norm(bp - self._raw_prev, axis=-1)
                jump = d > max(0.15, float(self.cfg["max_joint_speed"]) * dt)
                self._out_count = np.where(jump, self._out_count + 1, 0)
                outl = jump & (self._out_count <= 3)  # a jump that persists is real motion
                bp[outl] = self._raw_prev[outl]
            self._raw_prev = bp.copy()
            self._last_body = (t, bp, bq)
        else:
            # no fresh body sample (PICO streams body data slower than the headset): hold the last one
            if self._last_body is None:
                quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (NUM_JOINTS, 1))
                return np.zeros((NUM_JOINTS, 3)), quat, False, outl, True
            t0, bp, bq = self._last_body
            stale = True
            valid = (t - t0) <= float(self.cfg["max_gap_s"])
        pos, quat = self._filter_groups(bp, bq)
        if self.cfg.get("bone_length_normalize", True):
            P, Q = pos[None].copy(), quat[None].copy()
            for s in ("left", "right"):
                target = P[:, J[f"{s}_wrist"]].copy()
                if self.cfg.get("anchor", {}).get("hands") == "controllers" and s in self.ctrl_off and ctrl[s][2]:
                    off_p, off_q = self.ctrl_off[s]
                    wq = quat_mul(ctrl[s][1], quat_conj(off_q))
                    target = (ctrl[s][0] - quat_apply(wq, off_p))[None]
                solve_limb(P, Q, self.lengths, f"{s}_arm", target)
                solve_limb(P, Q, self.lengths, f"{s}_leg", P[:, J[f"{s}_ankle"]].copy())
            for end, child in (("left_wrist", "left_hand"), ("right_wrist", "right_hand"),
                               ("left_ankle", "left_foot"), ("right_ankle", "right_foot")):
                k = BONES.index((J[end], J[child]))
                d = P[0, J[child]] - P[0, J[end]]
                P[0, J[child]] = P[0, J[end]] + self.lengths[k] * d / max(np.linalg.norm(d), 1e-9)
            pos, quat = P[0], Q[0]
        return pos, quat, valid, outl, stale

    def _sparse(self, fr, hp, hq, ctrl):
        wrist, wrist_q = {}, {}
        for s in ("left", "right"):
            o_p, o_q = self.ctrl_off[s]
            wrist_q[s] = quat_mul(ctrl[s][1], quat_conj(o_q))[None]
            wrist[s] = (ctrl[s][0] - quat_apply(wrist_q[s][0], o_p))[None]
        head_q = hq[None]
        head_p = (hp - quat_apply(hq, self.hmd_to_head))[None]
        yaw, pitch = trunk_angles_from_head(head_q)
        yaw = float(yaw[0])
        if self._yaw_prev is not None:  # unwrap against the previous heading
            yaw = self._yaw_prev + float(np.angle(np.exp(1j * (yaw - self._yaw_prev))))
        self._yaw_prev = yaw
        trunk_yaw = self.f_yaw(np.array([yaw]))
        trunk_pitch = self.f_pitch(pitch)
        pelvis_in = ankles = None
        if self.trk_off and fr.trackers is not None:
            poses = {}
            for row, sn in enumerate(fr.tracker_serials):
                if sn in self.trk_off and row < len(fr.trackers):
                    p, q, v = self._align(fr.trackers[row])
                    if v:
                        poses[sn] = (p[None], q[None])
            pelvis_in, ankles = apply_tracker_offsets(self.trk_off, poses)
        P, Q = self.solver.solve(self.fps, head_p, head_q, wrist, wrist_q, 0.0, pelvis=pelvis_in, ankles=ankles,
                                 trunk_yaw=trunk_yaw, trunk_pitch=trunk_pitch)
        pos, quat = self._filter_groups(P[0], Q[0])
        valid = bool(ctrl["left"][2] and ctrl["right"][2])
        return pos, quat, valid, np.zeros(NUM_JOINTS, bool), False

    def _contact(self, pos):
        fc = self.cfg["foot_contact"]
        ank = pos[[J["left_ankle"], J["right_ankle"]]]
        xy = self.f_ankle(ank[:, :2])
        v = np.zeros(2) if self._prev_ankle_xy is None else np.linalg.norm(xy - self._prev_ankle_xy, axis=-1) / self.dt
        self._prev_ankle_xy = xy
        raw = (ank[:, 2] - ANKLE_HEIGHT < fc["height"]) & (v < fc["speed"])
        return self.contact(raw)

    def _foot_lock(self, pos, quat, contact):
        """Causal foot-skate cleanup: hold the ankle xy where the contact started (ramped in / out)."""
        P, Q = pos[None], quat[None]
        for k, s in enumerate(("left", "right")):
            aj = J[f"{s}_ankle"]
            if contact[k] and self._lock[k] is None:
                self._lock[k] = pos[aj, :2].copy()
            self._lock_w[k] = min(1.0, self._lock_w[k] + 1 / 3) if contact[k] else max(0.0, self._lock_w[k] - 1 / 3)
            if self._lock_w[k] <= 0.0:
                self._lock[k] = None
                continue
            target = pos[aj].copy()
            target[2] = max(target[2], ANKLE_HEIGHT * 0.8)
            w = 0.3 + 0.7 * self._lock_w[k]
            target[:2] = (1 - w) * target[:2] + w * self._lock[k]
            solve_limb(P, Q, self.lengths, f"{s}_leg", target[None])


# ----------------------------------------------------------------------------------------
# streaming retargeting
# ----------------------------------------------------------------------------------------
class StreamingRetargeter:
    """Per-frame mink IK with the offline Retargeter's tables, warm-started from the last frame.

    The world transform (spawn pose + joystick locomotion) is applied after body-part scaling,
    so joystick motion is not scaled. Post-processing is causal: joint-range clip, joint speed
    limit and ground alignment from the feet in contact.
    """

    def __init__(self, cfg: dict, fps: float, robot=None):
        self.rt = Retargeter(cfg, robot)
        self.robot = self.rt.robot
        self.cfg = cfg
        self.fps = float(fps)
        pp = cfg.get("postprocess", {})
        self.margin = float(pp.get("joint_margin", 0.0))
        self.flat_feet = bool(pp.get("flat_feet_in_contact", True))
        self.ground = pp.get("ground", "contact")
        ratio = pp.get("max_speed_ratio")
        self.vmax = None
        if ratio:
            from igris_pipeline.sim.mujoco_eval import load_joint_params
            self.vmax = load_joint_params(self.robot).velocity * float(ratio) / self.fps
        self.speed_scale = float((self.rt.balance or {}).get("speed_scale", 0.25))
        self.scale = None
        self.timing_ms = 0.0
        self.reset()

    def reset(self):
        self.config = None
        self.prev = None
        self._dz_hold = 0.0
        self.f_contact = Biquad(self.fps, 4.0)
        self.f_pelvis = Biquad(self.fps, 2.0)
        self.f_dz = Biquad(self.fps, 3.0)
        self._pelvis_prev = None

    def calibrate(self, window_pos: np.ndarray, stature: float | None):
        self.scale = self.rt.scale_params(np.asarray(window_pos, float), stature)
        self.reset()

    def step(self, pose: HumanPose, world_yaw: float, world_t: np.ndarray) -> np.ndarray:
        """Robot qpos (7 + ndof) for one corrected frame placed with the world transform."""
        t0 = time.perf_counter()
        Rw = _rot_z(world_yaw)
        qw = yaw_quat(np.array(world_yaw))
        spos = self.rt.apply_scale(pose.pos[None], self.scale)[0] @ Rw.T + np.asarray(world_t, float)
        squat = quat_mul(np.broadcast_to(qw, pose.quat.shape), pose.quat)
        contact = np.asarray(pose.contact, bool)
        if self.flat_feet:
            w = np.clip(self.f_contact(contact.astype(float)), 0.0, 1.0)
            h = self.robot.ankle_flat_height
            for k, s in enumerate(("left", "right")):
                j = J[f"{s}_ankle"]
                spos[j, 2] = (1 - w[k]) * spos[j, 2] + w[k] * h
                squat[j] = slerp(squat[j], yaw_quat(yaw_of(squat[j])), w[k])
        pel = self.f_pelvis(spos[J["pelvis"], :2])
        v = 0.0 if self._pelvis_prev is None else float(np.linalg.norm(pel - self._pelvis_prev)) * self.fps
        self._pelvis_prev = pel
        staticness = float(np.exp(-(v / self.speed_scale) ** 2))
        first = self.config is None
        if first:
            self.config = mink.Configuration(self.robot.model)
            q0 = self.robot.default_qpos.copy()
            q0[:2] = spos[J["pelvis"], :2]
            q0[3:7] = qw
            self.config.update(q0)
            self.rt._com_shift = np.zeros(2)
            self.rt.dt_frame = 1.0 / self.fps
        q = self.rt.solve_frame(self.config, spos, squat, pose.conf, first=first, contact_t=contact,
                                staticness=staticness)
        q = self._post(q, contact)
        self.timing_ms = (time.perf_counter() - t0) * 1e3
        return q

    def _post(self, q, contact):
        rng = self.robot.jnt_range
        q[7:] = np.clip(q[7:], rng[:, 0] + self.margin, rng[:, 1] - self.margin)
        if self.vmax is not None and self.prev is not None:
            q[7:] = self.prev[7:] + np.clip(q[7:] - self.prev[7:], -self.vmax, self.vmax)
        if self.ground != "none":
            sole = self.robot.sole_min_z(self.robot.forward(q))
            if self.ground == "contact" and contact.any():
                self._dz_hold = float(-np.min(sole[contact]))
                target = self._dz_hold
            elif self.ground == "contact":
                target = self._dz_hold
            else:
                target = max(0.0, float(-sole.min()))
            dz = float(self.f_dz(np.array([target]))[0])
            q[2] += max(dz, float(-sole.min()))  # never below the floor
        self.prev = q.copy()
        return q
