"""Synthetic RGB-D captures rendered with MuJoCo, with ground truth (tests and demos of the scan pipeline).

The default script is the recording protocol of docs/scan_pipeline.md in a 5 x 5 m room: the camera
sweeps an arc around a table (scan phase), comes to rest, and a person walks in, moves one object on
the table to another place and leaves (interaction phase). Depth gets the noise of a structured
light / stereo camera (sigma = 1.5 mm x d^2) and 1 mm quantisation.

    gt.json  {"split_s": t, "objects": {name: {"size": [sx, sy, sz] | [r, h], "shape", "start": [x, y, z],
              "end": [x, y, z], "moved": bool, "support": "floor" | "table"}}, "table_top": z,
              "camera_poses": [...4x4 camera-to-world (OpenCV axes), world = the MuJoCo frame]}
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .rgbd import CaptureWriter, Intrinsics

TABLE = {"pos": (1.6, 0.0), "size": (0.8, 1.2), "top": 0.75}
OBJECTS = {  # name: shape, size (box: full extents; cylinder: radius, height), start xy, rgba, support
    "box_red": ("box", (0.20, 0.15, 0.12), (1.45, 0.25), (0.85, 0.15, 0.12, 1), "table"),
    "can_blue": ("cylinder", (0.04, 0.12), (1.75, -0.25), (0.15, 0.3, 0.85, 1), "table"),
    "box_green": ("box", (0.08, 0.08, 0.08), (1.80, 0.25), (0.2, 0.7, 0.25, 1), "table"),
    "crate": ("box", (0.40, 0.30, 0.30), (1.2, -1.2), (0.72, 0.52, 0.32, 1), "floor"),
}
MOVED = {"box_red": (1.45, -0.20)}  # end xy of the objects the person moves


def _z(name):
    shape, size, _, _, support = OBJECTS[name]
    h = size[2] if shape == "box" else size[1]
    return (TABLE["top"] if support == "table" else 0.0) + h / 2


def scene_xml() -> str:
    tx, ty = TABLE["pos"]
    sx, sy = TABLE["size"]
    top = TABLE["top"]
    legs = "".join(
        f'<geom type="box" size="0.03 0.03 {(top - 0.04) / 2:.3f}" pos="{tx + a * (sx / 2 - 0.06):.3f} '
        f'{ty + b * (sy / 2 - 0.06):.3f} {(top - 0.04) / 2:.3f}" rgba="0.45 0.32 0.22 1"/>'
        for a in (-1, 1) for b in (-1, 1))
    objs = []
    for n, (shape, size, xy, rgba, _) in OBJECTS.items():
        s = (" ".join(f"{v / 2:.4f}" for v in size) if shape == "box" else f"{size[0]:.4f} {size[1] / 2:.4f}")
        objs.append(f'<body name="{n}" mocap="true" pos="{xy[0]} {xy[1]} {_z(n):.4f}">'
                    f'<geom type="{shape}" size="{s}" rgba="{" ".join(map(str, rgba))}"/></body>')
    return f"""<mujoco>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/>
          <headlight ambient="0.45 0.45 0.45" diffuse="0.4 0.4 0.4"/></visual>
  <worldbody>
    <light pos="1.5 0 2.8" dir="0 0 -1" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="1.5 0 0" rgba="0.62 0.6 0.56 1"/>
    <geom type="box" size="2.5 0.05 1.3" pos="1.5 2.55 1.3" rgba="0.85 0.85 0.8 1"/>
    <geom type="box" size="2.5 0.05 1.3" pos="1.5 -2.55 1.3" rgba="0.8 0.85 0.85 1"/>
    <geom type="box" size="0.05 2.5 1.3" pos="4.05 0 1.3" rgba="0.82 0.8 0.86 1"/>
    <geom type="box" size="0.6 0.25 0.45" pos="3.6 1.8 0.45" rgba="0.4 0.4 0.45 1"/>
    <geom type="box" size="{sx / 2} {sy / 2} 0.02" pos="{tx} {ty} {top - 0.02}" rgba="0.55 0.4 0.28 1"/>
    {legs}
    {''.join(objs)}
    <body name="person" mocap="true" pos="1.2 -4 0">
      <geom type="capsule" fromto="0 0 0.25 0 0 1.45" size="0.18" rgba="0.3 0.35 0.5 1"/>
      <geom type="sphere" pos="0 0 1.62" size="0.11" rgba="0.85 0.7 0.6 1"/>
    </body>
    <body name="arm" mocap="true" pos="1.2 -4 0">
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" rgba="0.85 0.7 0.6 1"/>
    </body>
    <body name="cam" mocap="true"><camera name="rgbd" fovy="58"/></body>
  </worldbody>
</mujoco>"""


def look_at(eye, target) -> np.ndarray:
    """Camera-to-world 4x4 (OpenCV axes) at eye looking at target, image up = world up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    T[:3, 3] = eye
    return T


def _smooth(a):
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3 - 2 * a)


def default_script(fps: float = 30.0, scan_s: float = 8.0, interact_s: float = 4.0):
    """(times, camera poses, {body: (pos, quat) per frame}, split time). The sweep (4 m of arc) at
    scan_s 8 is a careful handheld pace (0.5 m/s)."""
    n = int(round((scan_s + interact_s) * fps))
    t = np.arange(n) / fps
    center = np.array([*TABLE["pos"], TABLE["top"]])
    poses, bodies = [], {k: [] for k in ("person", "arm", *MOVED)}
    for ti in t:
        if ti < scan_s:  # sweep +40 deg -> -40 deg in 2/3 of the scan, back to 0
            u = ti / scan_s
            az = np.radians(40 - 120 * u) if u < 2 / 3 else np.radians(-40 + 120 * (u - 2 / 3))
        else:
            az = 0.0
        eye = center + np.array([-1.9 * np.cos(az), -1.9 * np.sin(az), 0.65])
        poses.append(look_at(eye, center + [0.1, 0, -0.1]))
        # interaction: the person walks in (0..1.2 s), reaches and moves the box (1.2..2.8 s), leaves
        s = ti - scan_s
        walk = _smooth(s / 1.2) - _smooth((s - 2.9) / 1.0)
        pp = np.array([1.25, -3.2 + walk * 2.6, 0.0])
        bodies["person"].append((pp, np.array([1.0, 0, 0, 0])))
        for name, (ex, ey) in MOVED.items():
            sx, sy = OBJECTS[name][2]
            m = _smooth((s - 1.4) / 1.2)
            lift = 0.08 * np.sin(np.pi * m)
            bp = np.array([sx + m * (ex - sx), sy + m * (ey - sy), _z(name) + lift])
            bodies[name].append((bp, np.array([1.0, 0, 0, 0])))
            reach = (1.2 < s < 2.8)
            hand = bp + [0.0, -0.05, 0.03]
            shoulder = pp + [0.0, 0.12, 1.35]
            v = hand - shoulder
            ang = np.arctan2(v[1], v[0])
            pitch = -np.arctan2(v[2], np.linalg.norm(v[:2]))
            q = _quat_zy(ang, pitch)
            bodies["arm"].append((shoulder if reach else np.array([0, -6.0, 0]), q))
    return t, np.array(poses), bodies, scan_s


def _quat_zy(yaw, pitch):
    cz, sz, cy, sy = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([cz * cy, -sz * sy, cz * sy, sz * cy])


def render_capture(out: str | Path, width: int = 320, height: int = 240, fps: float = 30.0,
                   noise: float = 0.0015, seed: int = 0, write_poses: bool = False, script=None) -> Path:
    """Render the scripted capture to `out` (igris_rgbd folder + gt.json)."""
    model = mujoco.MjModel.from_xml_string(scene_xml())
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    t, poses, bodies, split = script or default_script(fps)
    cam = model.body("cam").mocapid[0]
    fy = (height / 2) / np.tan(np.radians(58) / 2)
    intr = Intrinsics(width, height, fy, fy, (width - 1) / 2, (height - 1) / 2)
    renderer = mujoco.Renderer(model, height, width)
    flip = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> MuJoCo camera axes (looks along -z, y up)
    try:
        with CaptureWriter(out, intr, 1000.0, source="synthetic", device="mujoco", fps=fps) as w:
            for k, ti in enumerate(t):
                for name, seq in bodies.items():
                    mid = model.body(name).mocapid[0]
                    data.mocap_pos[mid], data.mocap_quat[mid] = seq[k]
                T = poses[k]
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, (T[:3, :3] @ flip).ravel())
                data.mocap_pos[cam], data.mocap_quat[cam] = T[:3, 3], q
                mujoco.mj_forward(model, data)
                renderer.disable_depth_rendering()
                renderer.update_scene(data, "rgbd")
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                renderer.update_scene(data, "rgbd")
                d = renderer.render().astype(np.float64)
                d[d > 6.0] = 0.0
                d = np.where(d > 0, d + rng.normal(0, 1, d.shape) * noise * d * d, 0.0)
                w.add(rgb, np.round(d * 1000) / 1000, ti, T if write_poses else None)
    finally:
        renderer.close()
    gt = {"split_s": split, "table_top": TABLE["top"], "camera_poses": poses.tolist(), "objects": {}}
    for name, (shape, size, xy, _, support) in OBJECTS.items():
        start = [xy[0], xy[1], _z(name)]
        end = [*MOVED[name], _z(name)] if name in MOVED else start
        gt["objects"][name] = {"shape": shape, "size": list(size), "start": start, "end": end,
                               "moved": name in MOVED, "support": support}
    (Path(out) / "gt.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    return Path(out)
