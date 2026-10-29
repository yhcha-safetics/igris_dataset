"""Simulated XRoboToolkit stream built from a synthetic clip.

`SimulatedXRTSource` has the same `poll()` interface as `XRTSource` and emits
native-frame (Y-up) packets, so the recorder, the device probe and the correction
stage can be exercised end-to-end without a headset. Error models per profile are
rough approximations of what PICO's body solver and sensors produce (jitter,
correlated drift on estimated joints, outlier spikes, packet drops, foot skating).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..skeleton import J, JOINT_GROUP, NUM_JOINTS
from ..utils.rotations import quat_mul, quat_normalize, rot_to_quat, quat_apply
from .frames import pipe_to_pose7
from .synthetic import SyntheticClip, generate
from .xrt_source import XRFrame

# per joint-group noise: (correlated drift sigma [m], white jitter sigma [m], orient noise [deg])
NOISE = {
    "body_2trk": {"head": (0.003, 0.002, 1.0), "hands": (0.006, 0.003, 2.0), "arms": (0.015, 0.004, 4.0),
                  "spine": (0.02, 0.004, 4.0), "pelvis": (0.02, 0.004, 4.0), "legs": (0.025, 0.005, 5.0),
                  "feet": (0.01, 0.004, 4.0)},
    "body_3trk": {"head": (0.003, 0.002, 1.0), "hands": (0.006, 0.003, 2.0), "arms": (0.015, 0.004, 4.0),
                  "spine": (0.01, 0.003, 3.0), "pelvis": (0.008, 0.003, 2.0), "legs": (0.015, 0.004, 4.0),
                  "feet": (0.01, 0.004, 4.0)},
}

# sensor mounting (pipeline frame, in the parent joint's local frame at T-pose)
HMD_OFFSET = np.array([0.09, 0.0, 0.06])           # eyes in front of / above the head joint
CTRL_OFFSET = {"left": np.array([0.04, 0.10, 0.0]), "right": np.array([0.04, -0.10, 0.0])}
CTRL_ROT = {"left": R.from_euler("x", 20, degrees=True), "right": R.from_euler("x", -20, degrees=True)}
TRACKER_MOUNT = {"pelvis": np.array([-0.12, 0.0, 0.0]),          # on the belt, at the back
                 "left_ankle": np.array([0.03, 0.0, 0.10]),      # above the ankle, on the shin
                 "right_ankle": np.array([0.03, 0.0, 0.10])}


def _correlated_noise(rng, T, dim, sigma, fps, tau=0.8):
    """AR(1) noise with time constant tau (s) and stationary std sigma."""
    a = np.exp(-1.0 / (tau * fps))
    e = rng.normal(size=(T, *dim)) * sigma * np.sqrt(1 - a * a)
    out = np.zeros((T, *dim))
    out[0] = rng.normal(size=dim) * sigma
    for t in range(1, T):
        out[t] = a * out[t - 1] + e[t]
    return out


def _rand_rot(rng, shape, deg):
    rv = rng.normal(size=(*shape, 3)) * np.deg2rad(deg)
    return rot_to_quat(R.from_rotvec(rv.reshape(-1, 3))).reshape(*shape, 4)


class SimulatedXRTSource:
    def __init__(self, profile: str = "body_2trk", clip: SyntheticClip | None = None, rate_hz: float = 72.0,
                 seed: int = 0, drop_prob: float = 0.02, outlier_prob: float = 0.003,
                 solver_scale: float = 0.97, realtime: bool = False):
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self.clip = clip or generate(fps=rate_hz, seed=seed)
        self.rate = rate_hz
        self.realtime = realtime
        self._build(drop_prob, outlier_prob, solver_scale)
        self._i = 0
        self._wall0 = None

    # -- the XRTSource interface -------------------------------------------------------
    def open(self):
        return self

    def close(self):
        pass

    def poll(self) -> XRFrame | None:
        if self.realtime:
            # behave like the live service: packets appear on the clip's clock, a slow consumer
            # gets the newest packet (older ones are skipped), None until the next one is due
            now = time.perf_counter()
            if self._wall0 is None:
                self._wall0 = now
            due = int(np.searchsorted(self.t_ns, self.t_ns[0] + (now - self._wall0) * 1e9, side="right")) - 1
            if due < self._i:
                if self._i >= self.T:
                    raise StopIteration
                return None
            self._i = due
        while self._i < self.T and self.dropped[self._i]:
            self._i += 1
        if self._i >= self.T:
            raise StopIteration
        i = self._i
        self._i += 1
        fr = XRFrame(t_ns=int(self.t_ns[i]), headset=self.hmd[i], left_ctrl=self.ctrl["left"][i],
                     right_ctrl=self.ctrl["right"][i], buttons={"A": False, "B": False, "X": False, "Y": False})
        if self.body is not None and not self.stale[i]:
            fr.body = self.body[i]
            fr.body_joint_ts = np.full(24, self.t_ns[i] // 1_000_000, dtype=np.int64)
        if self.trackers is not None:
            fr.trackers = self.trackers[i]
            fr.tracker_serials = list(self.serials)
        return fr

    # -- error model ------------------------------------------------------------------
    def _build(self, drop_prob, outlier_prob, solver_scale):
        c, rng, fps = self.clip, self.rng, self.rate
        T = c.pos.shape[0]
        self.T = T
        jitter = rng.uniform(-1.5e-3, 1.5e-3, size=T)
        t = c.t + jitter
        self.t_ns = (np.round((1.7e9 + t) * 1e3) * 1e6).astype(np.int64)  # ms resolution, UTC-like
        self.dropped = rng.random(T) < drop_prob
        self.stale = rng.random(T) < 0.01
        gt_pos, gt_q = c.pos, c.quat

        # headset / controllers (accurate sensors)
        hp = gt_pos[:, J["head"]] + quat_apply(gt_q[:, J["head"]], HMD_OFFSET) + rng.normal(size=(T, 3)) * 0.001
        hq = gt_q[:, J["head"]]
        self.hmd = pipe_to_pose7(hp, hq)
        self.ctrl = {}
        for side in ("left", "right"):
            wj = J[f"{side}_wrist"]
            cp = gt_pos[:, wj] + quat_apply(gt_q[:, wj], CTRL_OFFSET[side]) + rng.normal(size=(T, 3)) * 0.002
            cq = quat_mul(gt_q[:, wj], np.broadcast_to(rot_to_quat(CTRL_ROT[side]), (T, 4)))
            self.ctrl[side] = pipe_to_pose7(cp, cq)

        self.body = None
        self.trackers = None
        self.serials = []
        if self.profile in NOISE:
            nz = NOISE[self.profile]
            pelvis = gt_pos[:, :1]
            pos = pelvis + (gt_pos - pelvis) * solver_scale  # solver's avatar proportions
            pos = pos.copy()
            q = gt_q.copy()
            for j in range(NUM_JOINTS):
                drift, white, odeg = nz[JOINT_GROUP[j]]
                pos[:, j] += _correlated_noise(rng, T, (3,), drift, fps) + rng.normal(size=(T, 3)) * white
                q[:, j] = quat_mul(q[:, j], _rand_rot(rng, (T,), odeg))
            # outlier spikes on estimated joints
            spikes = np.nonzero(rng.random(T) < outlier_prob)[0]
            for s in spikes:
                j = rng.choice([J["left_knee"], J["right_knee"], J["left_elbow"], J["right_elbow"], J["spine2"]])
                n = rng.integers(1, 4)
                pos[s:s + n, j] += rng.normal(size=3) * 0.2
            self.body = pipe_to_pose7(pos, quat_normalize(q))
        elif self.profile == "sparse_trackers":
            roles = ["pelvis", "left_ankle", "right_ankle"]
            self.serials = [f"PC2310SIM{k:05d}" for k in range(len(roles))]
            tr = []
            for role in roles:
                j = J[role]
                p = gt_pos[:, j] + quat_apply(gt_q[:, j], TRACKER_MOUNT[role]) + rng.normal(size=(T, 3)) * 0.004
                tr.append(pipe_to_pose7(p, quat_mul(gt_q[:, j], _rand_rot(rng, (T,), 1.5))))
            self.trackers = np.stack(tr, axis=1)
        elif self.profile != "hmd_3pt":
            raise KeyError(self.profile)
