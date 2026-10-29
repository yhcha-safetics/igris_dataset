"""Robot description for the PICO app: body tree, joint axes and visual meshes of the compiled model.

The app draws IGRIS-C from `state.robot.qpos` with its own forward kinematics, so it needs the same
kinematic tree and meshes as the simulator. They are taken from the compiled MuJoCo model: MuJoCo
re-centres meshes when it compiles them, so the MJCF and STL files alone would place them wrongly.
The server publishes the result under `_robot/` on the asset HTTP server.

    robot.json
    {"format": "igris_robot", "version": 1, "nq": 38,
     "joint_names": [...],                            order of qpos[7:]
     "bodies": [{"name", "parent": index (-1 = world), "pos": [3], "quat": [w, x, y, z],
                 "joint": null | {"type": "free"}
                               | {"type": "hinge", "qpos": i, "axis": [3], "pos": [3], "ref": r}}],
     "geoms":  [{"name", "body": index, "pos", "quat", "rgba",
                 "type": "mesh", "mesh": "meshes/<name>.obj"}
                | {..., "type": "box" | "sphere" | "capsule" | "cylinder", "size": MuJoCo half sizes}]}
    meshes/<name>.obj   vertices and triangles in the geom frame (MuJoCo axes, metres)

Bodies are listed parents first. Forward kinematics, as in mj_kinematics:
    free joint   x_pos = qpos[0:3], x_quat = qpos[3:7]
    otherwise    x_pos = parent_pos + R(parent_quat) pos, x_quat = parent_quat * quat
                 hinge: anchor = x_pos + R(x_quat) jpos
                        x_quat = x_quat * axis_angle(axis, qpos[i] - ref), x_pos = anchor - R(x_quat) jpos
    geom         world pose = body pose * (pos, quat)
"""
from __future__ import annotations

import json

import mujoco
import numpy as np

from igris_pipeline.utils.rotations import quat_apply, quat_mul, quat_normalize

FORMAT = "igris_robot"
VERSION = 1
VISUAL_GROUP = 1
_PRIMITIVES = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
               mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder"}


def _r(a, n=6):
    return np.round(np.asarray(a, float), n).tolist()


def robot_description(world) -> tuple[dict, dict[str, bytes]]:
    """(robot.json content, {relative path: file bytes}) for the robot of a SimWorld.

    With hand overlays (igris_sim/hand_model.py) their bodies are published as children of the
    robot hands (the overlay root sits exactly on the hand) and their joints read
    qpos[world.nq_r + k] for the k-th of world.joint_names_ext()[31:], i.e. the layout of
    SimWorld.qpos_ext() that state.robot.qpos carries."""
    m = world.model
    bodies = [b for b in range(1, m.nbody) if world._is_robot_body(b)]  # body ids are parents-first
    overlay = getattr(world, "overlay", None)
    roots = {}
    if overlay is not None:
        roots = {r: world.hand_body[s] for s, r in overlay.root.items()}
        bodies += [b for b in range(1, m.nbody) if overlay.is_overlay_body(m, b)]
        ext_names = world.joint_names_ext()
        ext_qpos = {n: world.nq_r + k for k, n in enumerate(ext_names[len(world.robot.joint_names):])}
    index = {b: k for k, b in enumerate(bodies)}
    out_bodies = []
    for b in bodies:
        if b in roots:  # overlay root: rigidly on the hand (its mocap pose is the hand pose)
            out_bodies.append({"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), "parent": index[roots[b]],
                               "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "joint": None})
            continue
        if m.body_jntnum[b] > 1:
            raise ValueError(f"body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)}: more than one joint")
        joint = None
        if m.body_jntnum[b] == 1:
            j = m.body_jntadr[b]
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                joint = {"type": "free"}
            elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                a = int(m.jnt_qposadr[j])
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                qa = ext_qpos[name] if overlay is not None and overlay.is_overlay_body(m, b) else a
                joint = {"type": "hinge", "qpos": qa, "axis": _r(m.jnt_axis[j]), "pos": _r(m.jnt_pos[j]),
                         "ref": float(m.qpos0[a])}
            else:
                raise ValueError(f"joint {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)}: unsupported type")
        out_bodies.append({"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                           "parent": index.get(int(m.body_parentid[b]), -1),
                           "pos": _r(m.body_pos[b]), "quat": _r(m.body_quat[b]), "joint": joint})
    geoms, files = [], {}
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in index or m.geom_group[g] != VISUAL_GROUP:
            continue
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        e = {"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}", "body": index[b],
             "pos": _r(m.geom_pos[g]), "quat": _r(m.geom_quat[g]), "rgba": _r(rgba, 4)}
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            k = int(m.geom_dataid[g])
            path = f"meshes/{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, k) or f'mesh{k}'}.obj"
            if path not in files:
                files[path] = _obj_bytes(m, k)
            e.update(type="mesh", mesh=path)
        elif m.geom_type[g] in _PRIMITIVES:
            e.update(type=_PRIMITIVES[m.geom_type[g]], size=_r(m.geom_size[g]))
        else:
            continue
        geoms.append(e)
    names = world.joint_names_ext() if overlay is not None else list(world.robot.joint_names)
    desc = {"format": FORMAT, "version": VERSION, "nq": 7 + len(names),
            "joint_names": names, "bodies": out_bodies, "geoms": geoms}
    files["robot.json"] = json.dumps(desc, separators=(",", ":")).encode("utf-8")
    return desc, files


def _obj_bytes(m: mujoco.MjModel, k: int) -> bytes:
    v0, nv = m.mesh_vertadr[k], m.mesh_vertnum[k]
    f0, nf = m.mesh_faceadr[k], m.mesh_facenum[k]
    lines = ["# IGRIS-C visual mesh: MuJoCo geom frame, metres"]
    lines += [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in m.mesh_vert[v0:v0 + nv]]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in m.mesh_face[f0:f0 + nf]]  # indices are per mesh
    return ("\n".join(lines) + "\n").encode("ascii")


def forward_kinematics(desc: dict, qpos) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation of the app's FK: world body positions (n, 3) and quaternions wxyz (n, 4)."""
    qpos = np.asarray(qpos, float)
    n = len(desc["bodies"])
    xpos, xquat = np.zeros((n, 3)), np.zeros((n, 4))
    for k, b in enumerate(desc["bodies"]):
        j = b["joint"]
        if j is not None and j["type"] == "free":
            xpos[k], xquat[k] = qpos[0:3], quat_normalize(qpos[3:7])
            continue
        pp, pq = (xpos[b["parent"]], xquat[b["parent"]]) if b["parent"] >= 0 else (np.zeros(3), np.array([1.0, 0, 0, 0]))
        p = pp + quat_apply(pq, np.asarray(b["pos"], float))
        q = quat_mul(pq, np.asarray(b["quat"], float))
        if j is not None:
            jp = np.asarray(j["pos"], float)
            anchor = p + quat_apply(q, jp)
            ang = qpos[j["qpos"]] - j["ref"]
            q = quat_normalize(quat_mul(q, np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * np.asarray(j["axis"])])))
            p = anchor - quat_apply(q, jp)
        xpos[k], xquat[k] = p, q
    return xpos, xquat
