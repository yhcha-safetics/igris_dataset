"""Stage 1-2: tracker-profile-aware calibration and accuracy correction.

Input : an episode with raw/* streams (any sample rate, jitter, drops)
Output: proc/body/{pos,quat,confidence}, proc/contact, proc/valid, proc/t and a
        calibration record (floor, heading, stature, bone lengths, sensor offsets).

Steps (each switchable / parameterised per profile in configs/tracker_profiles.yaml)
  1. resample every stream to a uniform rate; bridge short gaps, invalidate long ones
  2. body profiles: robust outlier removal on PICO solver joints
     sparse/3-point profiles: build the skeleton from sensors (SparseBodySolver)
  3. calibration from a still window: floor, heading, stature, bone lengths,
     controller->wrist / tracker->joint offsets, joint-frame sanity check
  4. zero-phase low-pass per joint group (estimated joints get more smoothing)
  5. bone-length normalisation (constant subject proportions)
  6. sensor anchoring: wrists to controllers (2-bone arm IK), ankles to trackers
  7. foot contact detection and foot-skate cleanup (2-bone leg IK), floor clamp
  8. world alignment (floor z=0, start at origin facing +X) and per-joint confidence
"""
from __future__ import annotations

import copy
import json

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, median_filter

from ..episode import Episode
from ..skeleton import J, JOINT_GROUP, NUM_JOINTS, PARENTS, BONES, PICO_JOINTS, default_bone_length_array
from ..utils.geometry import (fill_invalid, frame_from_two_vectors, lowpass, lowpass_quat, resample_series,
                              swing_quat, two_bone_ik, unit)
from ..utils.rotations import (quat_apply, quat_angle_between, quat_conj, quat_continuity, quat_mul,
                               quat_normalize, yaw_of, yaw_quat)
from .synthetic import REST_OFFSETS

UP = np.array([0.0, 0.0, 1.0])
ANKLE_HEIGHT = 0.08
REST_DIR = unit(REST_OFFSETS + 1e-12)  # T-pose bone direction of each joint relative to its parent
LIMBS = {  # name: (root, mid, end, pole hint in body frame)
    "left_arm": ("left_shoulder", "left_elbow", "left_wrist", np.array([-0.3, 0.3, -1.0])),
    "right_arm": ("right_shoulder", "right_elbow", "right_wrist", np.array([-0.3, -0.3, -1.0])),
    "left_leg": ("left_hip", "left_knee", "left_ankle", np.array([1.0, 0.0, 0.0])),
    "right_leg": ("right_hip", "right_knee", "right_ankle", np.array([1.0, 0.0, 0.0])),
}


def sensor_offsets(cfg_all: dict) -> dict:
    """Sensor mounting offsets for profiles without a body solver (configs: sensor_offsets).

    Measure your own controller offsets once with a body-tracking session: calibration
    prints `controller_to_wrist_<side>` which can be copied into the config.
    """
    so = cfg_all.get("sensor_offsets", {})
    out = {"hmd_to_head": np.array(so.get("hmd_to_head", [0.09, 0.0, 0.06]), dtype=np.float64)}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        c = so.get("controller_to_wrist", {}).get(side, {})
        pos = np.array(c.get("pos", [0.04, 0.10 * sign, 0.0]), dtype=np.float64)
        if "quat_wxyz" in c:
            q = quat_normalize(np.array(c["quat_wxyz"], dtype=np.float64))
        else:
            from scipy.spatial.transform import Rotation as _R
            from ..utils.rotations import rot_to_quat
            q = rot_to_quat(_R.from_euler("xyz", c.get("rot_xyz_deg", [20.0 * sign, 0, 0]), degrees=True))
        out[f"ctrl_{side}"] = (pos, q)
    return out


def profile_config(cfg_all: dict, profile: str) -> dict:
    cfg = copy.deepcopy(cfg_all["defaults"])
    cfg.update(copy.deepcopy(cfg_all["profiles"][profile]))
    cfg["name"] = profile
    cfg["tracker_roles"] = cfg_all.get("tracker_roles") or {}
    return cfg


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _mean_quat(q):
    """Average of unit quaternions (eigenvector method)."""
    q = quat_normalize(np.asarray(q).reshape(-1, 4))
    w, v = np.linalg.eigh(q.T @ q)
    m = v[:, -1]
    return m if m[0] >= 0 else -m


def _still_window(pos_list, fps, window_s, search_s):
    """Index slice of the stillest window in the first search_s seconds."""
    n = max(2, int(window_s * fps))
    T = pos_list[0].shape[0]
    m = min(T, int(search_s * fps))
    speed = np.zeros(T)
    for p in pos_list:
        v = np.linalg.norm(np.gradient(p, 1.0 / fps, axis=0), axis=-1)
        speed += v.reshape(T, -1).mean(axis=1)
    speed /= len(pos_list)
    if m <= n:
        return slice(0, min(T, n)), float(speed[:n].max())
    c = np.convolve(speed[:m], np.ones(n) / n, mode="valid")
    s = int(np.argmin(c))
    return slice(s, s + n), float(speed[s:s + n].max())


def _heading_from_body(pos):
    lat = (pos[..., J["left_hip"], :] - pos[..., J["right_hip"], :]) + \
          (pos[..., J["left_shoulder"], :] - pos[..., J["right_shoulder"], :])
    lat = lat.copy()
    lat[..., 2] = 0
    fwd = np.cross(unit(lat), UP)
    return np.arctan2(fwd[..., 1], fwd[..., 0])


def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _detect_outliers(pos, fps, max_speed, thr=0.06):
    """(T,24,3) -> (T,24) bool, robust to the slow drifts of the body solver."""
    k = max(5, int(0.12 * fps) | 1)
    med = median_filter(pos, size=(k, 1, 1), mode="nearest")
    res = np.linalg.norm(pos - med, axis=-1)
    mad = np.median(res, axis=0, keepdims=True) + 1e-4
    speed = np.linalg.norm(np.gradient(pos, 1.0 / fps, axis=0), axis=-1)
    out = (res > np.maximum(thr, 6 * mad)) | ((speed > max_speed) & (res > 0.5 * thr))
    return binary_closing(out, structure=np.ones((3, 1)))


def _interp_flagged(x, flagged):
    """Linear interpolation over flagged samples per joint. x (T,24,D), flagged (T,24)."""
    x = x.copy()
    T = x.shape[0]
    idx = np.arange(T)
    for j in range(x.shape[1]):
        f = flagged[:, j]
        if not f.any() or f.all():
            continue
        for d in range(x.shape[2]):
            x[f, j, d] = np.interp(idx[f], idx[~f], x[~f, j, d])
    return x


def normalize_bone_lengths(pos, lengths):
    """Rebuild positions from the root keeping raw bone directions but fixed lengths."""
    out = pos.copy()
    for k, (pa, c) in enumerate(BONES):
        d = unit(pos[:, c] - pos[:, pa])
        out[:, c] = out[:, pa] + lengths[k] * d
    return out


def _bone_len(lengths, parent_name, child_name):
    return lengths[BONES.index((J[parent_name], J[child_name]))]


def _apply_swing(quat, j, old_dir, new_dir):
    quat[:, j] = quat_normalize(quat_mul(swing_quat(old_dir, new_dir), quat[:, j]))


def _translate_subtree(pos, j, delta, mask=None):
    """Translate joint j and all descendants by delta (T,3)."""
    kids = [j]
    for i in range(NUM_JOINTS):
        if i != j:
            p = PARENTS[i]
            while p >= 0:
                if p == j:
                    kids.append(i)
                    break
                p = PARENTS[p]
    m = np.ones(pos.shape[0], bool) if mask is None else mask
    for k in kids:
        pos[m, k] += delta[m]


def solve_limb(pos, quat, lengths, limb, target, pole=None, mask=None, target_quat=None,
               straight_deg=(8.0, 25.0)):
    """Move the limb end to `target` (T,3) with constant segment lengths, update orientations.

    Exact 2-bone IK is ill-conditioned for a nearly straight limb: a 1 cm error in the
    root->target distance moves the knee/elbow by ~sqrt(2*l*1cm) ~ 9 cm. So the result is
    blended with a straight-limb solution according to how bent the limb is in the data
    (fully IK above straight_deg[1], fully straight below straight_deg[0]).
    """
    a, b, c, _ = LIMBS[limb]
    ja, jb, jc = J[a], J[b], J[c]
    l1, l2 = _bone_len(lengths, a, b), _bone_len(lengths, b, c)
    m = np.ones(pos.shape[0], bool) if mask is None else mask
    if not m.any():
        return
    pole = pos[:, jb] if pole is None else pole
    old_ab, old_bc = pos[:, jb] - pos[:, ja], pos[:, jc] - pos[:, jb]
    bend = np.degrees(np.arccos(np.clip(np.sum(unit(old_ab) * unit(old_bc), -1), -1, 1)))
    lo, hi = straight_deg
    alpha = np.clip((bend - lo) / (hi - lo), 0.0, 1.0)
    alpha = (alpha * alpha * (3 - 2 * alpha))[:, None]
    mid_ik, end_ik = two_bone_ik(pos[:, ja], target, l1, l2, pole)
    mid_st = pos[:, ja] + (l1 / (l1 + l2)) * (target - pos[:, ja])
    mid = alpha * mid_ik + (1 - alpha) * mid_st
    end = alpha * end_ik + (1 - alpha) * target
    new_ab, new_bc = mid - pos[:, ja], end - mid
    old_end = pos[:, jc].copy()
    pos[m, jb] = mid[m]
    q = quat.copy()
    _apply_swing(q, ja, old_ab, new_ab)
    _apply_swing(q, jb, old_bc, new_bc)
    quat[m, ja], quat[m, jb] = q[m, ja], q[m, jb]
    _translate_subtree(pos, jc, end - old_end, m)
    if target_quat is not None:
        quat[m, jc] = target_quat[m]


# ----------------------------------------------------------------------------------------
# sparse-sensor body solver (3-point / object-mode trackers)
# ----------------------------------------------------------------------------------------
class SparseBodySolver:
    """Builds a full 24-joint skeleton from headset, controllers and optional trackers.

    Estimated joints follow a simple anthropometric model; their confidence is set low
    by the profile so retargeting relies mostly on the sensor-anchored joints.
    """

    def __init__(self, lengths, ankle_height=ANKLE_HEIGHT):
        self.L = lengths
        self.ankle_h = ankle_height

    def solve(self, fps, head_pos, head_quat, wrist_pos, wrist_quat, floor_z, pelvis=None, ankles=None):
        T = head_pos.shape[0]
        L = self.L
        pos = np.zeros((T, NUM_JOINTS, 3))
        quat = np.zeros((T, NUM_JOINTS, 4))
        quat[..., 0] = 1.0

        head_fwd = quat_apply(head_quat, np.array([1.0, 0, 0]))
        head_yaw = np.unwrap(np.arctan2(head_fwd[:, 1], head_fwd[:, 0]))
        trunk_yaw = lowpass(head_yaw, fps, 0.8)
        if pelvis is not None:
            p_yaw = np.unwrap(yaw_of(pelvis[1]))
            trunk_yaw = p_yaw
        head_pitch = np.arcsin(np.clip(-head_fwd[:, 2], -1, 1))  # + = looking down
        trunk_pitch = lowpass(np.clip(0.6 * (head_pitch - np.deg2rad(20)), 0, np.deg2rad(80)), fps, 1.0)
        q_trunk = quat_mul(yaw_quat(trunk_yaw), _quat_axis_angle(np.array([0, 1.0, 0]), trunk_pitch))

        pos[:, J["head"]], quat[:, J["head"]] = head_pos, head_quat
        neck = head_pos - quat_apply(head_quat, _rest_vec(L, "neck", "head"))
        pos[:, J["neck"]] = neck
        spine_chain = ["neck", "spine3", "spine2", "spine1", "pelvis"]
        if pelvis is None:
            cur = neck
            for upper, lower in zip(spine_chain[:-1], spine_chain[1:]):
                v = quat_apply(q_trunk, _rest_vec(L, lower, upper))
                cur = cur - v
                pos[:, J[lower]] = cur
        else:
            pos[:, J["pelvis"]] = pelvis[0]
            for k, n in enumerate(["spine1", "spine2", "spine3"], start=1):
                w = sum(_rest_len(L, a, b) for a, b in zip(spine_chain[::-1][:k], spine_chain[::-1][1:k + 1]))
                tot = sum(_rest_len(L, a, b) for a, b in zip(spine_chain[::-1][:-1], spine_chain[::-1][1:]))
                pos[:, J[n]] = pelvis[0] + (neck - pelvis[0]) * (w / tot)
            d = unit(neck - pelvis[0])
            q_trunk = frame_from_two_vectors(UP, np.array([0, 1.0, 0]), d,
                                             quat_apply(yaw_quat(trunk_yaw), np.array([0, 1.0, 0])))
        for n in ["pelvis", "spine1", "spine2", "spine3", "neck"]:
            quat[:, J[n]] = q_trunk if pelvis is None or n != "pelvis" else pelvis[1]
        for side in ("left", "right"):
            for a, b in ((f"spine3", f"{side}_collar"), (f"{side}_collar", f"{side}_shoulder")):
                pos[:, J[b]] = pos[:, J[a]] + quat_apply(q_trunk, _rest_vec(L, a, b))
                quat[:, J[b]] = q_trunk
            for a, b in ((f"pelvis", f"{side}_hip"),):
                q_p = quat[:, J["pelvis"]]
                pos[:, J[b]] = pos[:, J[a]] + quat_apply(q_p, _rest_vec(L, a, b))

        # arms
        for side in ("left", "right"):
            sh, el, wr, hd = (J[f"{side}_shoulder"], J[f"{side}_elbow"], J[f"{side}_wrist"], J[f"{side}_hand"])
            pole = pos[:, sh] + quat_apply(q_trunk, LIMBS[f"{side}_arm"][3])
            mid, end = two_bone_ik(pos[:, sh], wrist_pos[side], _bone_len(L, f"{side}_shoulder", f"{side}_elbow"),
                                   _bone_len(L, f"{side}_elbow", f"{side}_wrist"), pole)
            pos[:, el], pos[:, wr] = mid, end
            quat[:, wr] = wrist_quat[side]
            pos[:, hd] = end + quat_apply(wrist_quat[side], _rest_vec(L, f"{side}_wrist", f"{side}_hand"))
            quat[:, hd] = wrist_quat[side]
            bend = np.cross(mid - pos[:, sh], end - mid)
            n_ref = quat_apply(q_trunk, np.array([0, 0, -1.0 if side == "left" else 1.0]))
            n = np.where(np.linalg.norm(bend, axis=-1, keepdims=True) > 1e-4, bend, n_ref)
            n0 = np.array([0, 0, -1.0]) if side == "left" else np.array([0, 0, 1.0])
            quat[:, sh] = frame_from_two_vectors(REST_DIR[el], n0, mid - pos[:, sh], n)
            quat[:, el] = frame_from_two_vectors(REST_DIR[wr], n0, end - mid, n)
            quat[:, J[f"{side}_collar"]] = q_trunk

        # legs
        for side in ("left", "right"):
            hip, kn, an, ft = (J[f"{side}_hip"], J[f"{side}_knee"], J[f"{side}_ankle"], J[f"{side}_foot"])
            if ankles is not None:
                target, aq = ankles[side]
            else:
                target = pos[:, hip].copy()
                target[:, 2] = floor_z + self.ankle_h
                aq = yaw_quat(trunk_yaw)
            q_p = quat[:, J["pelvis"]]
            pole = pos[:, hip] + quat_apply(q_p, LIMBS[f"{side}_leg"][3])
            mid, end = two_bone_ik(pos[:, hip], target, _bone_len(L, f"{side}_hip", f"{side}_knee"),
                                   _bone_len(L, f"{side}_knee", f"{side}_ankle"), pole)
            pos[:, kn], pos[:, an] = mid, end
            n = np.cross(mid - pos[:, hip], end - mid)
            n_ref = quat_apply(q_p, np.array([0, 1.0, 0]))
            n = np.where(np.linalg.norm(n, axis=-1, keepdims=True) > 1e-4, n, n_ref)
            n = np.where(np.sum(n * n_ref, -1, keepdims=True) < 0, -n, n)
            quat[:, hip] = frame_from_two_vectors(REST_DIR[kn], np.array([0, 1.0, 0]), mid - pos[:, hip], n)
            quat[:, kn] = frame_from_two_vectors(REST_DIR[an], np.array([0, 1.0, 0]), end - mid, n)
            quat[:, an] = aq
            pos[:, ft] = end + quat_apply(aq, _rest_vec(L, f"{side}_ankle", f"{side}_foot"))
            quat[:, ft] = aq
        return pos, quat_normalize(quat)


def _quat_axis_angle(axis, angle):
    angle = np.asarray(angle, dtype=np.float64)
    return np.concatenate([np.cos(angle / 2)[..., None], np.sin(angle / 2)[..., None] * axis], axis=-1)


def _rest_len(L, a, b):
    try:
        return _bone_len(L, a, b)
    except ValueError:
        return _bone_len(L, b, a)


def _rest_vec(L, a, b):
    """T-pose vector from joint a to joint b (adjacent), scaled to calibrated length."""
    ja, jb = J[a], J[b]
    if PARENTS[jb] == ja:
        return REST_DIR[jb] * _bone_len(L, a, b)
    return -REST_DIR[ja] * _bone_len(L, b, a)


# ----------------------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------------------
def correct_episode(ep: Episode, cfg_all: dict, profile: str | None = None,
                    subject_height: float | None = None) -> dict:
    profile = profile or ep.attrs.get("tracker_profile") or "body_2trk"
    cfg = profile_config(cfg_all, profile)
    fps = float(cfg["fps"])
    max_gap = float(cfg["max_gap_s"])
    subject_height = subject_height or ep.attrs.get("subject_height")
    subject_height = float(subject_height) if subject_height else None
    report: dict = {"profile": profile, "steps": []}

    t = ep["raw/t"]
    tn = np.arange(0.0, t[-1] - t[0] + 1e-9, 1.0 / fps) + t[0]
    T = len(tn)

    def rs(prefix, quat=True):
        p, v, g = resample_series(t, ep[f"{prefix}/pos"], ep[f"{prefix}/valid"], tn, max_gap)
        q = None
        if quat:
            q, _, _ = resample_series(t, ep[f"{prefix}/quat"], ep[f"{prefix}/valid"], tn, max_gap, is_quat=True)
        return p, q, v, g

    hmd_p, hmd_q, hmd_v, _ = rs("raw/headset")
    ctrl = {s: rs(f"raw/controller/{s}") for s in ("left", "right")}

    is_body = profile.startswith("body")
    if is_body:
        raw_valid = ep["raw/body/valid"]
        bp, bv, bg = resample_series(t, ep["raw/body/pos"], raw_valid, tn, max_gap)
        bq, _, _ = resample_series(t, ep["raw/body/quat"], raw_valid, tn, max_gap, is_quat=True)
        if bv.sum() < 2:
            raise ValueError(f"profile {profile!r} needs PICO body-tracking data, but the episode has "
                             f"{int(raw_valid.sum())} valid body frames; use a non-body profile (--profile)")
        report["steps"].append({"resample": {"in_frames": int(len(t)), "out_frames": T, "fps": fps,
                                             "body_valid_in": float(raw_valid.mean()),
                                             "body_valid_out": float(bv.mean()), "gap_filled": int(bg.sum())}})
        # 2) outliers
        outl = _detect_outliers(bp, fps, cfg["max_joint_speed"])
        outl &= bv[:, None]
        bp = _interp_flagged(bp, outl)
        report["steps"].append({"outliers": {"joint_frames": int(outl.sum()),
                                             "per_joint": {PICO_JOINTS[j]: int(outl[:, j].sum())
                                                           for j in range(NUM_JOINTS) if outl[:, j].any()}}})
        bp = fill_invalid(bp, bv)
        bq = fill_invalid(bq, bv, is_quat=True)
        valid = bv
        gapf = bg
    else:
        outl = np.zeros((T, NUM_JOINTS), bool)
        valid = hmd_v & ctrl["left"][2] & ctrl["right"][2]
        gapf = np.zeros(T, bool)

    # 3) calibration ------------------------------------------------------------------
    calib: dict = {"profile": profile}
    if is_body:
        win, still_speed = _still_window([bp], fps, cfg["calibration"]["window_s"], cfg["calibration"]["search_s"])
        ank_z = np.minimum(bp[valid, J["left_ankle"], 2], bp[valid, J["right_ankle"], 2])
        floor = float(np.percentile(ank_z, 5) - ANKLE_HEIGHT)
        lengths = np.median(np.linalg.norm(
            bp[valid][:, [c for _, c in BONES]] - bp[valid][:, [p for p, _ in BONES]], axis=-1), axis=0)
        # median of the headings unwrapped around their circular mean (a plain median of angles
        # wrapped to (-pi, pi] is biased / 180 deg off when the subject faces ~-X)
        hd = _heading_from_body(bp[win])
        hc = np.arctan2(np.sin(hd).mean(), np.cos(hd).mean())
        yaw0 = float(hc + np.median(np.angle(np.exp(1j * (hd - hc)))))
        stature = float(np.median(bp[win, J["head"], 2]) - floor + 0.12)
        # joint-frame sanity: at a T-pose PICO joints should be ~heading-aligned
        rest_dev = quat_angle_between(bq[win].mean(0), np.broadcast_to(yaw_quat(yaw0), (NUM_JOINTS, 4)))
        calib["rest_orientation_deviation_deg"] = {PICO_JOINTS[j]: round(float(np.degrees(rest_dev[j])), 1)
                                                   for j in range(NUM_JOINTS)}
    else:
        h = subject_height or 1.70
        lengths = default_bone_length_array() * (h / 1.70)
        sources = [hmd_p, ctrl["left"][0], ctrl["right"][0]]
        win, still_speed = _still_window(sources, fps, cfg["calibration"]["window_s"], cfg["calibration"]["search_s"])
        hmd_z0 = float(np.median(hmd_p[win, 2]))
        floor = hmd_z0 - (h - 0.11)  # eyes are ~11 cm below the top of the head
        f = quat_apply(hmd_q[win], np.array([1.0, 0, 0]))
        yaw0 = float(np.arctan2(np.median(f[:, 1]), np.median(f[:, 0])))
        stature = h
        if subject_height is None:
            report["steps"].append({"warning": "subject_height not given; assumed 1.70 m for floor / bone lengths"})
    calib.update({"floor_z": floor, "heading_yaw_deg": float(np.degrees(yaw0)), "stature": stature,
                  "still_window": [int(win.start), int(win.stop)], "still_max_speed": still_speed,
                  "bone_lengths": {f"{PICO_JOINTS[p]}->{PICO_JOINTS[c]}": round(float(l), 4)
                                   for (p, c), l in zip(BONES, lengths)}})

    # world alignment transform (applied to every stream): floor->0, start->origin, face +X
    Rz = _rot_z(-yaw0)
    q_align = yaw_quat(np.array(-yaw0))
    if is_body:
        origin = bp[win, J["pelvis"]].mean(0)
    else:
        origin = hmd_p[win].mean(0)
    origin = np.array([origin[0], origin[1], floor])

    def align_p(p):
        return (p - origin) @ Rz.T

    def align_q(q):
        return quat_mul(np.broadcast_to(q_align, q.shape), q)

    hmd_p, hmd_q = align_p(hmd_p), align_q(hmd_q)
    for s in ("left", "right"):
        p, q, v, g = ctrl[s]
        ctrl[s] = (align_p(p), align_q(q), v, g)
    floor_z = 0.0

    # trackers (object mode)
    trk = None
    if "raw/tracker/pos" in ep and ep["raw/tracker/valid"].any():
        K = ep["raw/tracker/pos"].shape[1]
        trk = []
        for k in range(K):
            v_raw = ep["raw/tracker/valid"][:, k]
            if v_raw.sum() < 2:
                continue
            p, v, _ = resample_series(t, ep["raw/tracker/pos"][:, k], v_raw, tn, max_gap)
            q, _, _ = resample_series(t, ep["raw/tracker/quat"][:, k], v_raw, tn, max_gap, is_quat=True)
            trk.append((k, align_p(fill_invalid(p, v)), align_q(fill_invalid(q, v, True)), v))

    if is_body:
        bp = align_p(bp)
        bq = align_q(bq)
    else:
        offs = sensor_offsets(cfg_all)
        wrist = {}
        wrist_q = {}
        for s in ("left", "right"):
            p, q, v, _ = ctrl[s]
            p, q = fill_invalid(p, v), fill_invalid(q, v, True)
            o_p, o_q = offs[f"ctrl_{s}"]
            wrist_q[s] = quat_mul(q, np.broadcast_to(quat_conj(o_q), q.shape))
            wrist[s] = p - quat_apply(wrist_q[s], o_p)
        head_q = fill_invalid(hmd_q, hmd_v, True)
        head_p = fill_invalid(hmd_p, hmd_v) - quat_apply(head_q, offs["hmd_to_head"])
        pelvis_in, ankles_in, roles = None, None, {}
        if trk:
            roles = _assign_tracker_roles(trk, win, ep, cfg)
            calib["tracker_roles"] = {str(k): r for k, r in roles.items()}
            pelvis_in, ankles_in = _tracker_targets(trk, roles, win, head_q, floor_z)
        solver = SparseBodySolver(lengths)
        bp, bq = solver.solve(fps, head_p, head_q, wrist, wrist_q, floor_z, pelvis=pelvis_in, ankles=ankles_in)
        report["steps"].append({"sparse_solver": {"trackers": len(trk or []), "roles": calib.get("tracker_roles")}})

    # 4) zero-phase low-pass per group --------------------------------------------------
    cut = cfg["cutoff_hz"]
    for g in set(JOINT_GROUP):
        idx = [j for j in range(NUM_JOINTS) if JOINT_GROUP[j] == g]
        bp[:, idx] = lowpass(bp[:, idx], fps, cut[g])
        bq[:, idx] = lowpass_quat(bq[:, idx], fps, cut[g])
    report["steps"].append({"lowpass_hz": cut})

    # 5+6) anchor-preserving bone-length normalisation ---------------------------------
    # Joints measured by a sensor (head, wrists via controllers, ankles via trackers) are
    # kept; limb chains are re-solved with constant calibrated lengths by 2-bone IK and the
    # end segments (hand, foot) get their lengths fixed. FK from the pelvis would instead
    # accumulate every solver error onto the most accurate joints.
    if is_body and cfg.get("bone_length_normalize", True):
        bl_std = lambda x: float(np.std(np.linalg.norm(  # noqa: E731
            x[:, [c for _, c in BONES]] - x[:, [p for p, _ in BONES]], axis=-1), 0).mean() * 1e3)
        before = bl_std(bp)
        targets = {}
        for s_ in ("left", "right"):
            targets[f"{s_}_arm"] = (bp[:, J[f"{s_}_wrist"]].copy(), np.ones(T, bool))
            targets[f"{s_}_leg"] = (bp[:, J[f"{s_}_ankle"]].copy(), np.ones(T, bool))
        if cfg.get("anchor", {}).get("hands") == "controllers":
            for s_ in ("left", "right"):
                p, q, v, _ = ctrl[s_]
                ok = v & valid
                if ok.sum() < fps:  # need ~1 s of overlap to calibrate the grip offset
                    continue
                wj = J[f"{s_}_wrist"]
                local = quat_apply(quat_conj(bq[ok, wj]), p[ok] - bp[ok, wj])
                off_p = np.median(local, axis=0)
                off_q = _mean_quat(quat_mul(quat_conj(bq[ok, wj]), q[ok]))
                wq = quat_mul(q, np.broadcast_to(quat_conj(off_q), q.shape))
                wp = p - quat_apply(wq, off_p)
                wp = np.where(ok[:, None], wp, bp[:, wj])
                err = np.linalg.norm(wp[ok] - bp[ok, wj], axis=-1)
                targets[f"{s_}_arm"] = (wp, ok)
                calib[f"controller_to_wrist_{s_}"] = {"pos": off_p.round(4).tolist(),
                                                      "quat_wxyz": off_q.round(4).tolist()}
                report["steps"].append({f"anchor_{s_}_hand": {
                    "frames": int(ok.sum()), "solver_vs_controller_mm": round(float(np.median(err) * 1e3), 1)}})
        for limb, (target, _) in targets.items():
            a, b, c, _ = LIMBS[limb]
            k1, k2 = BONES.index((J[a], J[b])), BONES.index((J[b], J[c]))
            # Median bone lengths of noisy joints are biased long; a limb that is straight in
            # the data (stance, T-pose) would then be forced to bend by several cm. Fit the
            # total length to the straight-limb reach (95th pct of root->end distance).
            reach = np.percentile(np.linalg.norm(target - bp[:, J[a]], axis=-1), 95)
            tot = lengths[k1] + lengths[k2]
            sc = float(np.clip(reach / tot, 0.92, 1.0 if not cfg.get("limb_reach_scaling", False) else 1.08))
            if abs(sc - 1.0) > 1e-3:
                lengths[k1] *= sc
                lengths[k2] *= sc
                report["steps"].append({f"{limb}_length_scale": round(sc, 4)})
            solve_limb(bp, bq, lengths, limb, target)
        for end, child in (("left_wrist", "left_hand"), ("right_wrist", "right_hand"),
                           ("left_ankle", "left_foot"), ("right_ankle", "right_foot")):
            k = BONES.index((J[end], J[child]))
            bp[:, J[child]] = bp[:, J[end]] + lengths[k] * unit(bp[:, J[child]] - bp[:, J[end]])
        calib["bone_lengths"] = {f"{PICO_JOINTS[p]}->{PICO_JOINTS[c]}": round(float(l), 4)
                                 for (p, c), l in zip(BONES, lengths)}
        report["steps"].append({"bone_length_std_mm": {"before": round(before, 2), "after": round(bl_std(bp), 2)}})

    # 7) foot contacts & skate cleanup -----------------------------------------------
    contact = np.zeros((T, 2), bool)
    fc = cfg["foot_contact"]
    for k, s in enumerate(("left", "right")):
        aj = J[f"{s}_ankle"]
        h = bp[:, aj, 2] - floor_z - ANKLE_HEIGHT
        # speed from a 3 Hz smoothed trajectory: raw solver jitter alone reaches ~0.2 m/s
        v_xy = np.linalg.norm(np.gradient(lowpass(bp[:, aj, :2], fps, 3.0), 1.0 / fps, axis=0), axis=-1)
        c = (h < fc["height"]) & (v_xy < fc["speed"])
        n = max(1, int(fc["min_duration_s"] * fps))
        # edge padding: scipy's morphology treats outside samples as False and would drop contact
        # on the first / last frames of a clip that starts / ends standing
        c = binary_opening(binary_closing(np.pad(c, n, mode="edge"), np.ones(n)), np.ones(n))[n:-n]
        contact[:, k] = c
    if cfg.get("foot_skate_cleanup", False):
        skate_before = _foot_skate(bp, contact, fps)
        for k, s in enumerate(("left", "right")):
            aj = J[f"{s}_ankle"]
            target = bp[:, aj].copy()
            target[:, 2] = np.maximum(target[:, 2], floor_z + ANKLE_HEIGHT * 0.8)
            for (a, b) in _runs(contact[:, k]):
                lock = np.median(bp[a:min(b, a + 3), aj, :2], axis=0)
                w = np.ones(b - a)
                r = min(3, (b - a) // 2)
                if r > 0:
                    w[:r] = np.linspace(0.3, 1.0, r)
                    w[-r:] = np.linspace(1.0, 0.3, r)
                target[a:b, :2] = (1 - w[:, None]) * target[a:b, :2] + w[:, None] * lock
            solve_limb(bp, bq, lengths, f"{s}_leg", target)
        report["steps"].append({"foot_skate_mm_per_contact_frame": {
            "before": round(skate_before * 1e3, 2), "after": round(_foot_skate(bp, contact, fps) * 1e3, 2)},
            "contact_ratio": contact.mean(0).round(3).tolist()})

    # 8) confidence -------------------------------------------------------------------
    base = np.array([cfg["confidence"][JOINT_GROUP[j]] for j in range(NUM_JOINTS)])
    conf = np.broadcast_to(base, (T, NUM_JOINTS)).copy()
    conf *= valid[:, None]
    conf *= np.where(gapf[:, None], 0.6, 1.0)
    conf = np.where(outl, conf * 0.4, conf)
    for s, js in (("left", ("left_wrist", "left_hand")), ("right", ("right_wrist", "right_hand"))):
        v = ctrl[s][2]
        for jn in js:
            conf[:, J[jn]] *= np.where(v, 1.0, 0.5)
    conf = lowpass(conf, fps, 2.0).clip(0.0, 1.0)

    out = {
        "proc/t": tn - tn[0], "proc/body/pos": bp, "proc/body/quat": quat_normalize(bq),
        "proc/body/confidence": conf.astype(np.float32), "proc/contact": contact, "proc/valid": valid,
        "proc/headset/pos": hmd_p, "proc/headset/quat": hmd_q,
        "proc/controller/left/pos": ctrl["left"][0], "proc/controller/left/quat": ctrl["left"][1],
        "proc/controller/right/pos": ctrl["right"][0], "proc/controller/right/quat": ctrl["right"][1],
    }
    calib["world_alignment"] = {"origin": origin.round(4).tolist(), "yaw_removed_deg": float(np.degrees(yaw0))}
    return {"arrays": out, "calibration": calib, "report": report, "fps": fps,
            "observed_axes": cfg.get("observed_axes", ["loco", "lower", "upper"])}


def _runs(mask):
    out, s = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    return out


def _foot_skate(pos, contact, fps):
    tot, n = 0.0, 0
    for k, s in enumerate(("left", "right")):
        v = np.linalg.norm(np.diff(pos[:, J[f"{s}_ankle"], :2], axis=0), axis=-1)
        c = contact[1:, k] & contact[:-1, k]
        tot += v[c].sum()
        n += c.sum()
    return float(tot / max(n, 1))


def _assign_tracker_roles(trk, win, ep, cfg):
    """Map object-mode trackers to joints: config by serial, else by height at calibration."""
    serials = json.loads(ep.attrs.get("tracker_serials", "[]")) if isinstance(ep.attrs.get("tracker_serials"), str) else []
    roles = {}
    by_cfg = cfg.get("tracker_roles") or {}
    for k, p, q, v in trk:
        sn = serials[k] if k < len(serials) else None
        if sn and sn in by_cfg:
            roles[k] = by_cfg[sn]
    rest = [(k, float(np.median(p[win, 2])), p[win].mean(0)) for k, p, q, v in trk if k not in roles]
    rest.sort(key=lambda x: x[1])
    low = [r for r in rest if r[1] < 0.4]
    high = [r for r in rest if r[1] >= 0.4]
    if len(low) >= 2:
        a, b = low[0], low[1]
        # left is +Y in the aligned frame
        left, right = (a, b) if a[2][1] > b[2][1] else (b, a)
        roles[left[0]], roles[right[0]] = "left_ankle", "right_ankle"
    if high:
        roles[high[0][0]] = "pelvis"
    return roles


def _tracker_targets(trk, roles, win, head_q, floor_z):
    """Calibrate tracker->joint offsets from the still pose and return joint targets."""
    pelvis_in, ankles = None, {}
    for k, p, q, v in trk:
        role = roles.get(k)
        if role is None:
            continue
        q_rest = _mean_quat(q[win])
        if role in ("left_ankle", "right_ankle"):
            rest_joint = p[win].mean(0).copy()
            rest_joint[2] = floor_z + ANKLE_HEIGHT
            off = quat_apply(quat_conj(q_rest), rest_joint - p[win].mean(0))
            joint_q_off = quat_mul(quat_conj(q_rest), yaw_quat(np.array(yaw_of(_mean_quat(head_q[win])))))
            jq = quat_mul(q, np.broadcast_to(joint_q_off, q.shape))
            ankles[role.split("_")[0]] = (p + quat_apply(q, np.broadcast_to(off, p.shape)), jq)
        elif role == "pelvis":
            yaw_rest = yaw_quat(np.array(yaw_of(_mean_quat(head_q[win]))))
            joint_q_off = quat_mul(quat_conj(q_rest), yaw_rest)
            jq = quat_mul(q, np.broadcast_to(joint_q_off, q.shape))
            # belt tracker sits ~12 cm behind the pelvis joint
            off_world = quat_apply(yaw_rest, np.array([0.12, 0.0, 0.0]))
            off = quat_apply(quat_conj(q_rest), off_world)
            pelvis_in = (p + quat_apply(q, np.broadcast_to(off, p.shape)), jq)
    if len(ankles) < 2:
        ankles = None
    return pelvis_in, ankles


def apply_correction(ep: Episode, cfg_all: dict, profile: str | None = None,
                     subject_height: float | None = None) -> dict:
    res = correct_episode(ep, cfg_all, profile, subject_height)
    for k, v in res["arrays"].items():
        ep[k] = v
    ep.group_attrs["proc"] = {"fps": res["fps"], "profile": res["calibration"]["profile"],
                              "calibration": json.dumps(res["calibration"], ensure_ascii=False),
                              "report": json.dumps(res["report"], ensure_ascii=False),
                              "observed_axes": json.dumps(res["observed_axes"])}
    ep.attrs["tracker_profile"] = res["calibration"]["profile"]
    return res
