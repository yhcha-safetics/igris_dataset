"""Fast regression tests:  .venv\\Scripts\\python -m pytest tests -q"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from igris_pipeline.pico.frames import Q_C, gmr_offset_to_pipeline, pipe_to_pose7, pose7_to_pipe  # noqa: E402
from igris_pipeline.pico.synthetic import generate  # noqa: E402
from igris_pipeline.utils.geometry import two_bone_ik  # noqa: E402
from igris_pipeline.utils.rotations import quat_angle_between  # noqa: E402

CFG = ROOT / "configs"


def _yaml(n):
    return yaml.safe_load(open(CFG / n, encoding="utf-8"))


@pytest.fixture(scope="module")
def clip():
    prog = [("idle", "upright", "t_pose", 2.0), ("idle", "upright", "neutral", 1.5),
            ("walk", "upright", "neutral", 3.0), ("idle", "squat", "neutral", 3.0),
            ("idle", "upright", "arms_up", 2.0), ("idle", "upright", "wave_right", 2.5)]
    return generate(prog, fps=50.0)


def test_frame_roundtrip():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(10, 3))
    q = rng.normal(size=(10, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p2, q2, v = pose7_to_pipe(pipe_to_pose7(p, q))
    assert v.all() and np.allclose(p2, p) and np.allclose(np.abs(np.sum(q2 * q, 1)), 1.0)


def test_native_tpose_is_identity():
    # PICO joints are world-aligned in a forward-facing T-pose: identity in native frame
    _, q, _ = pose7_to_pipe(np.array([0, 1, 0, 0, 0, 0, 1.0]))
    assert quat_angle_between(q, np.array([1.0, 0, 0, 0])) < 1e-9
    # forward in native frame is -Z -> +X in pipeline frame
    p, _, _ = pose7_to_pipe(np.array([0, 0, -1.0, 0, 0, 0, 1.0]))
    assert np.allclose(p, [1, 0, 0])


def test_gmr_offsets_convert_to_expected():
    pelvis = gmr_offset_to_pipeline([-0.5, 0.5, -0.5, -0.5])
    l_arm = gmr_offset_to_pipeline([0.7071, 0.0, 0.7071, 0.0])
    assert quat_angle_between(pelvis, np.array([1.0, 0, 0, 0])) < 1e-3
    assert quat_angle_between(l_arm, np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])) < 1e-3
    assert Q_C.shape == (4,)


def test_two_bone_ik_reaches():
    root = np.zeros((5, 3))
    tgt = np.array([[0.3, 0.2, -0.4]] * 5)
    mid, end = two_bone_ik(root, tgt, 0.4, 0.4, np.array([[1.0, 0, 0]] * 5))
    assert np.allclose(end, tgt, atol=1e-6)
    assert np.allclose(np.linalg.norm(mid - root, axis=1), 0.4)
    assert np.allclose(np.linalg.norm(end - mid, axis=1), 0.4)


def test_labeling_on_synthetic(clip):
    from igris_pipeline.labeling.auto_label import evaluate, label_motion

    out = label_motion(clip.pos, clip.fps, _yaml("labels.yaml"))
    rep = evaluate(out["axes"], clip.labels, clip.fps, tolerance_s=1.0)
    for axis in ("loco", "lower", "upper"):
        assert rep[axis]["acc"] > 0.9, rep


def test_correction_improves_body_profile(tmp_path):
    from igris_pipeline.episode import Episode
    from igris_pipeline.pico.correction import apply_correction
    from igris_pipeline.pico.recorder import record
    from igris_pipeline.pico.simulate import SimulatedXRTSource

    c = generate([("idle", "upright", "t_pose", 3.0), ("idle", "upright", "neutral", 3.0),
                  ("idle", "squat", "neutral", 4.0)], fps=72.0)
    paths = record(SimulatedXRTSource("body_2trk", clip=c, seed=1), tmp_path, {"tracker_profile": "body_2trk"},
                   use_buttons=False, on_status=lambda *a: None)
    ep = Episode.load(paths[0])
    res = apply_correction(ep, _yaml("tracker_profiles.yaml"))
    a = res["arrays"]
    idx = np.clip(np.round(a["proc/t"] * c.fps).astype(int), 0, len(c.t) - 1)
    gt = c.pos[idx]
    # compare in a pelvis-relative frame (calibration re-orients the world)
    rel = lambda x: x - x[:, :1]  # noqa: E731
    traw = ep["raw/t"] - ep["raw/t"][0]
    raw = ep["raw/body/pos"][np.clip(np.searchsorted(traw, a["proc/t"]), 0, len(traw) - 1)]
    yaw = np.radians(res["calibration"]["heading_yaw_deg"])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    est = rel(a["proc/body/pos"]) @ Rz.T  # undo the alignment rotation
    e_corr = np.median(np.linalg.norm(est - rel(gt), axis=-1))
    e_raw = np.median(np.linalg.norm(rel(raw) - rel(gt), axis=-1))
    assert e_corr < e_raw, (e_corr, e_raw)
    assert a["proc/contact"].mean() > 0.8  # standing / squatting: feet on the floor


def test_retarget_tpose_and_squat(clip):
    from igris_pipeline.retarget.retargeter import Retargeter

    r = Retargeter(_yaml("retarget_igris_c.yaml"))
    res = r.retarget(clip.pos[:60], clip.quat[:60], clip.fps, contact=np.ones((60, 2), bool), stature=1.72)
    j = {n: i for i, n in enumerate(res.joint_names)}
    q = res.qpos[50, 7:]
    assert abs(q[j["Joint_Shoulder_Roll_Left"]] - np.pi / 2) < 0.2
    assert abs(q[j["Joint_Shoulder_Roll_Right"]] + np.pi / 2) < 0.2
    assert res.diagnostics["max_penetration_cm"] < 1.0


def test_exports(tmp_path, clip):
    from igris_pipeline.retarget import export as X
    from igris_pipeline.retarget.retargeter import Retargeter

    r = Retargeter(_yaml("retarget_igris_c.yaml"))
    res = r.retarget(clip.pos[:40], clip.quat[:40], clip.fps)
    X.to_tracking_npz(tmp_path / "m.npz", r.robot, res.qpos, clip.fps, 50.0)
    z = np.load(tmp_path / "m.npz")
    for k in ("fps", "joint_names", "joint_pos", "joint_vel", "body_names", "body_pos_w", "body_quat_w",
              "body_lin_vel_w", "body_ang_vel_w"):
        assert k in z.files
    assert z["joint_pos"].shape[1] == 31 and z["body_quat_w"].shape[-1] == 4
    X.to_canonical_pkl(tmp_path / "m.pkl", r.robot, res.qpos, clip.fps)
    d = pickle.load(open(tmp_path / "m.pkl", "rb"))
    assert d["dof_pos"].shape[1] == 31 and len(d["link_body_list"]) == 33
    assert d["root_rot"].dtype == np.float32
    json.dumps({k: v for k, v in d.items() if k == "retarget_meta"})
