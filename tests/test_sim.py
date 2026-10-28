"""Tests for the real-time simulation server (igris_sim)."""
import json
import socket
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from igris_pipeline.episode import Episode  # noqa: E402
from igris_pipeline.pico.frames import pose7_to_pipe  # noqa: E402
from igris_pipeline.pico.recorder import record  # noqa: E402
from igris_pipeline.pico.simulate import SimulatedXRTSource  # noqa: E402
from igris_pipeline.pico.synthetic import generate  # noqa: E402
from igris_pipeline.pico.xrt_source import FrameBuffer  # noqa: E402
from igris_pipeline.retarget.robot import IgrisModel  # noqa: E402
from igris_pipeline.utils.rotations import quat_apply, quat_mul, rot_to_quat  # noqa: E402
from igris_sim import protocol  # noqa: E402
from igris_sim.scene import build_model, list_scenes, load_scene, validate  # noqa: E402
from igris_sim.session import SimSession, load_sim_config  # noqa: E402
from igris_sim.sources import EpisodeReplaySource  # noqa: E402
from igris_sim.streaming import Biquad, StreamingCorrector, StreamingRetargeter  # noqa: E402
from igris_sim.world import SimWorld  # noqa: E402

PROGRAM = [("idle", "upright", "t_pose", 3.0), ("idle", "upright", "neutral", 1.0),
           ("walk", "upright", "neutral", 2.0), ("idle", "upright", "reach_forward", 1.5)]


def sim_frames(profile, program=PROGRAM, seed=3):
    clip = generate(program, fps=72, seed=seed)
    src = SimulatedXRTSource(profile, clip=clip, seed=seed)
    out = []
    while True:
        try:
            out.append(src.poll())
        except StopIteration:
            return out, clip


def frame_times(frames):
    t = np.array([f.t_ns for f in frames]) * 1e-9
    return t - t[0]


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config()


# -- scenes -----------------------------------------------------------------------------
def test_builtin_scenes_compile():
    names = list_scenes()
    assert {"empty", "tabletop", "room"} <= set(names)
    for n in names:
        model, info = build_model(load_scene(n))
        robot = IgrisModel(model=model)
        assert robot.ndof == 31
        assert model.jnt_qposadr[robot.jid(robot.joint_names[-1])] < 7 + robot.ndof  # robot comes first
        assert len(info.weld) == 2 * len(info.graspable)
        for name in info.graspable:
            assert model.body_mass[info.body[name]] > 0


def test_mesh_object_scene(tmp_path):
    v = [(x, y, z) for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)]
    faces = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    obj = "\n".join([f"v {x} {y} {z}" for x, y, z in v] + [f"f {' '.join(map(str, f))}" for f in faces])
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "box.obj").write_text(obj)
    (tmp_path / "meshes" / "box_col.obj").write_text(obj)
    spec = {"format": "igris_scene", "version": 1, "name": "mesh_test", "objects": [
        {"name": "scan", "shape": "mesh", "file": "meshes/box.obj", "pos": [1, 0, 0.05]},
        {"name": "thing", "shape": "mesh", "file": "meshes/box.obj", "collision": ["meshes/box_col.obj"],
         "pos": [1, 0.5, 0.3], "dynamic": True}]}
    (tmp_path / "scene.json").write_text(json.dumps(spec))
    model, info = build_model(load_scene(tmp_path))
    assert info.dynamic["thing"] and not info.dynamic["scan"]
    assert model.body_mass[info.body["thing"]] > 0
    assert len(info.geoms["thing"]) == 1 and len(info.geoms["scan"]) == 1


def test_scene_validation():
    base = {"format": "igris_scene", "version": 1, "name": "x"}
    with pytest.raises(ValueError):
        validate({**base, "objects": [{"name": "a", "shape": "box", "size": [1, 1, 1]}, {"name": "a", "shape": "box"}]})
    with pytest.raises(ValueError):
        validate({**base, "objects": [{"name": "a", "shape": "torus"}]})
    with pytest.raises(ValueError):
        validate({**base, "version": 2})


# -- filters ----------------------------------------------------------------------------
def test_biquad_steady_state_and_attenuation():
    f = Biquad(50.0, 5.0)
    x0 = np.array([1.5, -2.0])
    assert np.allclose(f(x0), x0) and np.allclose(f(x0), x0)
    f = Biquad(50.0, 2.0)
    t = np.arange(500) / 50.0
    y = np.array([f(np.array([np.sin(2 * np.pi * 15 * tt)]))[0] for tt in t])
    assert np.abs(y[100:]).max() < 0.05  # 15 Hz well above a 2 Hz cutoff
    assert np.allclose(Biquad(50.0, None)(x0), x0)


# -- streaming correction + retargeting ------------------------------------------------------
@pytest.mark.parametrize("profile,max_err_cm", [("body_2trk", 8.0), ("body_3trk", 7.0),
                                                ("sparse_trackers", 6.0), ("hmd_3pt", 12.0)])
def test_streaming_profiles(cfg, profile, max_err_cm):
    import yaml
    from igris_pipeline.retarget.ik_config import load as load_ik
    cfg_all = yaml.safe_load(open(ROOT / "configs" / "tracker_profiles.yaml", encoding="utf-8"))
    frames, clip = sim_frames(profile)
    t = frame_times(frames)
    sc = StreamingCorrector(cfg_all, profile, subject_height=1.72)
    res = sc.calibrate([f for f, tt in zip(frames, t) if tt < 3.0])
    rt = StreamingRetargeter(load_ik(ROOT / "configs" / "retarget_igris_c.yaml"), sc.fps)
    rt.calibrate(res["arrays"]["proc/body/pos"], res["calibration"]["stature"])
    ticks = np.arange(3.0, t[-1], 1 / 50)
    idx = np.searchsorted(t, ticks, side="right") - 1
    gt_t = clip.t - clip.t[0]
    errs, qs = [], []
    for i in idx:
        pose = sc.process(frames[i])
        qs.append(rt.step(pose, 0.0, np.zeros(3)))
        g = min(int(np.searchsorted(gt_t, t[i])), len(gt_t) - 1)
        errs.append(np.linalg.norm((clip.pos[g] - clip.pos[g, 0]) - (pose.pos - pose.pos[0]), axis=-1).mean())
    qs = np.array(qs)
    assert np.isfinite(qs).all()
    assert np.mean(errs) * 100 < max_err_cm
    assert 0.5 < qs[:, 2].min() and qs[:, 2].max() < 1.1          # pelvis height stays plausible
    assert qs[-1, 0] - qs[0, 0] > 0.5                              # follows the walk forward
    rng = rt.robot.jnt_range
    assert np.all(qs[:, 7:] >= rng[:, 0] - 1e-6) and np.all(qs[:, 7:] <= rng[:, 1] + 1e-6)


# -- physics / grasping ---------------------------------------------------------------------
def _place_near_palm(w, obj, gap=0.03):
    """Put a dynamic object below the left palm with ~gap between the surfaces."""
    d, adr = w.data, w.info.qposadr[obj]
    palm = w.palm_pos("left")
    for dz in np.arange(0.02, 0.3, 0.005):
        d.qpos[adr:adr + 3] = palm - np.array([0.0, 0.0, dz])
        d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(w.model, d)
        dist = min(mujoco.mj_geomDistance(w.model, d, w.palm["left"], g, 1.0, None) for g in w.info.geoms[obj])
        if dist >= gap:
            return
    raise AssertionError("could not place the object")


@pytest.mark.parametrize("mode", ["kinematic", "harness"])
def test_grasp_follows_hand_and_releases(cfg, mode):
    w = SimWorld(load_scene("tabletop"), cfg, mode)
    _place_near_palm(w, "cube_red")
    assert w.update_grasp({"left": 1.0, "right": 0.0}) == [("left", "grasp", "cube_red")]
    body, hand = w.info.body["cube_red"], w.hand_body["left"]

    def in_hand():  # object position in the hand frame (what the weld keeps constant)
        d = w.data
        return d.xmat[hand].reshape(3, 3).T @ (d.xpos[body] - d.xpos[hand])

    rel0, palm0 = in_hand(), w.palm_pos("left")
    q = w.robot_qpos()
    j = w.robot.joint_names.index("Joint_Shoulder_Pitch_Left")
    for k in range(40):  # raise the arm forward
        qt = q.copy()
        qt[7 + j] = q[7 + j] - 0.8 * min(1.0, (k + 1) / 20)
        w.step(qt, 0.02)
    assert np.linalg.norm(w.palm_pos("left") - palm0) > 0.2   # the hand really moved
    assert np.linalg.norm(in_hand() - rel0) < 0.02
    assert w.update_grasp({"left": 0.0}) == [("left", "release", "cube_red")]
    p_release = w.data.xpos[body].copy()
    for k in range(40):  # lower the arm again: a released object must not follow
        qt2 = qt.copy()
        qt2[7 + j] = qt[7 + j] + 0.8 * min(1.0, (k + 1) / 20)
        w.step(qt2, 0.02)
    assert np.linalg.norm(w.data.xpos[body] - w.palm_pos("left")) > 0.15
    assert np.linalg.norm(w.data.xpos[body][:2] - p_release[:2]) < 0.1  # stays where it was put down
    v = w.data.qvel[w.dofadr["cube_red"]:w.dofadr["cube_red"] + 3]
    assert np.linalg.norm(v) < 0.05  # and comes to rest (on the table or the floor)


def test_grasp_needs_proximity(cfg):
    w = SimWorld(load_scene("tabletop"), cfg)
    assert w.update_grasp({"left": 1.0, "right": 1.0}) == []  # nothing within reach at the spawn pose


# -- protocol ------------------------------------------------------------------------------
def test_tracking_message_roundtrip():
    frames, _ = sim_frames("sparse_trackers", PROGRAM[:1])
    fr = frames[10]
    fr.buttons = {"A": True, "left_grip": 0.7}
    fr.axes = {"left": np.array([0.1, -0.5])}
    back = protocol.frame_from_tracking(protocol.decode(protocol.encode(protocol.tracking_from_frame(fr))))
    assert back.t_ns == fr.t_ns and back.buttons == {"A": True, "left_grip": 0.7}
    assert np.allclose(back.headset, fr.headset, atol=1e-6)
    assert np.allclose(back.trackers, fr.trackers, atol=1e-6) and back.tracker_serials == fr.tracker_serials
    assert np.allclose(back.axes["left"], [0.1, -0.5])
    with pytest.raises(ValueError):
        protocol.frame_from_tracking({"t_ns": 1, "head": [0, 0, 0]})
    with pytest.raises(ValueError):
        protocol.decode(b'{"v": 9, "type": "state"}')


def test_unity_conversions_are_consistent():
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation as R
    for _ in range(20):
        q = rot_to_quat(R.random(random_state=rng))
        p = rng.normal(size=3)
        v = rng.normal(size=3)
        # rotating then converting == converting then rotating (Unity applies q*v with its own axes)
        qu = protocol.unity_from_mujoco_quat(q)
        lhs = protocol.unity_from_mujoco_pos(quat_apply(q, v))
        rhs = quat_apply(np.array([qu[3], qu[0], qu[1], qu[2]]), protocol.unity_from_mujoco_pos(v))
        assert np.allclose(lhs, rhs)
        # Unity -> native -> pipeline (server side) -> Unity is the identity
        pu = protocol.unity_from_mujoco_pos(p)
        pp, qp, _ = pose7_to_pipe(protocol.native_from_unity(pu, qu))
        assert np.allclose(pp, p) and np.allclose(np.abs(np.dot(qp, q)), 1.0)


def test_udp_session_roundtrip(cfg):
    hub = protocol.Hub(0, host="127.0.0.1")
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(1.0)
    try:
        s = SimSession(cfg, "empty")
        addr = ("127.0.0.1", hub.port)
        client.sendto(protocol.encode({"v": 1, "type": "cmd", "cmd": "hello"}), addr)
        frames, _ = sim_frames("body_2trk", PROGRAM[:2])
        replies = []
        for k, fr in enumerate(frames):
            client.sendto(protocol.encode(protocol.tracking_from_frame(fr)), addr)
            if k % 3 == 0:  # ~24 Hz ticks are plenty here
                got, cmds = hub.poll()
                for f, _ in got:
                    s.push_frame(f)
                for msg, a in cmds:
                    hub.send(s.command(msg), a)
                s.tick()
                hub.broadcast(s.state_message())
        last = None
        client.setblocking(False)
        while True:
            try:
                m = protocol.decode(client.recvfrom(65507)[0])
            except (BlockingIOError, ConnectionResetError):
                break
            replies.append(m["type"])
            last = m if m["type"] == "state" else last
        assert "scene" in replies and last is not None
        assert last["status"] == "running" and last["profile"] == "body_2trk"
        assert len(last["robot"]["qpos"]) == 38 and len(last["objects"]["pos"]) == 1
    finally:
        client.close()
        hub.close()


# -- session: recording, locomotion, replay --------------------------------------------------
def _run(s, frames, until=None, axes=None, offset=0.0):
    t = frame_times(frames)
    end = t[-1] if until is None else until
    tick = 0.0
    i = 0
    while tick <= end:
        while i < len(frames) and t[i] <= tick:
            if axes is not None:
                frames[i].axes = axes
            s.push_frame(frames[i])
            i += 1
        s.tick(now=offset + tick)
        tick += 1 / 50


def test_recording_is_pipeline_compatible(cfg, tmp_path):
    from igris_pipeline.pipeline import stage_correct
    s = SimSession(cfg, "tabletop", record_dir=tmp_path / "raw")
    frames, _ = sim_frames("body_2trk")
    t = frame_times(frames)
    _run(s, [f for f, tt in zip(frames, t) if tt < 3.6])
    assert s.status == "running"
    s.set_recording(True, now=3.6)
    _run(s, [f for f, tt in zip(frames, t) if tt >= 3.6], offset=3.6)
    s.set_recording(False)
    assert len(s.saved) == 1
    ep = Episode.load(s.saved[0])
    assert "raw/body/pos" in ep and ep["sim/robot/qpos"].shape[1] == 38
    assert ep["sim/objects/pos"].shape[1] == len(s.world.dynamic)
    assert json.loads(ep.group_attrs["sim"]["scene_json"])["name"] == "tabletop"
    stage_correct(s.saved[0])  # the offline pipeline accepts it


def test_locomotion_moves_and_turns_about_the_robot(cfg):
    s = SimSession(cfg, "empty")
    frames, _ = sim_frames("body_2trk", [("idle", "upright", "t_pose", 3.0), ("idle", "upright", "neutral", 4.0)])
    t = frame_times(frames)
    _run(s, [f for f, tt in zip(frames, t) if tt < 3.5])
    assert s.status == "running"
    x0 = s.q_target[:2].copy()
    _run(s, [f for f, tt in zip(frames, t) if 3.5 <= tt < 4.5], axes={"left": np.array([0.0, 1.0])}, offset=3.5)
    moved = s.q_target[:2] - x0
    assert 0.6 < moved[0] < 1.3 and abs(moved[1]) < 0.3  # ~1 m forward at full deflection for 1 s
    p0, yaw0 = s.q_target[:2].copy(), s.loco[2]
    _run(s, [f for f, tt in zip(frames, t) if 4.5 <= tt < 5.5], axes={"right": np.array([1.0, 0.0])}, offset=4.5)
    assert s.loco[2] - yaw0 < -np.radians(60)               # stick right turns right (clockwise)
    assert np.linalg.norm(s.q_target[:2] - p0) < 0.15        # ... about the robot, not the world origin


def test_episode_replay_source_roundtrip(tmp_path):
    frames, clip = sim_frames("sparse_trackers", PROGRAM[:2])
    src = SimulatedXRTSource("sparse_trackers", clip=clip, seed=3)
    (path,) = record(src, tmp_path, {"tracker_profile": "sparse_trackers"}, use_buttons=False, on_status=lambda *_: None)
    rep = EpisodeReplaySource(path)
    buf = FrameBuffer()
    while True:
        try:
            buf.append(rep.poll())
        except StopIteration:
            break
    a = buf.to_arrays()
    ep = Episode.load(path)
    for k in ("raw/headset/pos", "raw/controller/left/quat", "raw/tracker/pos"):
        assert np.allclose(a[k], ep[k], atol=1e-9), k
    assert np.array_equal(a["raw/t_ns"], ep["raw/t_ns"])
