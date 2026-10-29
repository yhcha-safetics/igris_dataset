"""Simulation scenes: one JSON description for the PC simulator (MuJoCo) and the PICO app (Unity).

    scenes/<name>/scene.json
    {
      "format": "igris_scene", "version": 1,
      "name": "tabletop", "description": "...",
      "spawn": {"pos": [x, y], "yaw_deg": 0},     robot / user start on the floor
      "floor": {"rgba": [r, g, b, a]} | false,     false hides the MJCF floor (e.g. a scanned floor mesh)
      "objects": [object, ...]
    }

    object
      name       unique; MuJoCo body "obj_<name>"
      pos        [x, y, z] body origin in metres (Z up; +X is the spawn heading at yaw_deg 0)
      rot        {"euler_deg": [roll, pitch, yaw]} (about fixed X, then Y, then Z) or {"quat": [w, x, y, z]}
      dynamic    false: fixed to the world (default); true: free rigid body
      graspable  default = dynamic
      mass       kg, dynamic only (default: density 500 kg/m^3)
      rgba, friction [slide, spin, roll], collide (default true)
      geoms      parts in the body frame; an object without "geoms" is a single part given by
                 its own shape fields:
        shape    box | sphere | cylinder | capsule | mesh
        size     box [sx, sy, sz] full extents; sphere [r]; cylinder / capsule [r, height]
        file     mesh only: OBJ / STL for MuJoCo, relative to the scene folder
        collision  mesh only: convex parts used for contacts (default: `file`, convexified)
        render   mesh only: display model for the app (OBJ, vertex colours "v x y z r g b" allowed,
                 or STL; default `file`); MuJoCo ignores it
        scale    mesh only: [sx, sy, sz]
        pos, rot, rgba, friction, collide   override the object's

Sizes are full extents (as in Unity) and are halved for MuJoCo. Unity rebuilds primitives from
the same fields; see docs/sim_protocol.md for the axis conversion.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from igris_pipeline.retarget.robot import DEFAULT_MJCF, ROOT
from igris_pipeline.utils.rotations import quat_normalize, rot_to_quat

SCENES_DIR = ROOT / "scenes"
HANDS = {"left": "Left_Hand", "right": "Right_Hand"}
OBJ_PREFIX = "obj_"
DEFAULT_DENSITY = 500.0
GEOM_TYPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
              "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
              "mesh": mujoco.mjtGeom.mjGEOM_MESH}


@dataclass
class Scene:
    spec: dict
    folder: Path

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def spawn(self) -> tuple[np.ndarray, float]:
        s = self.spec.get("spawn", {})
        return np.array(s.get("pos", [0.0, 0.0])[:2], float), float(np.radians(s.get("yaw_deg", 0.0)))


@dataclass
class SceneInfo:
    """Ids of the scene elements in the compiled model."""
    objects: list                      # object names, in scene order
    body: dict                         # name -> body id
    dynamic: dict                      # name -> bool
    graspable: list                    # names
    geoms: dict                        # name -> collision geom ids
    qposadr: dict                      # dynamic name -> qpos address of its free joint
    weld: dict = field(default_factory=dict)  # (side, name) -> equality id


def list_scenes(folder: Path = SCENES_DIR) -> list[str]:
    return sorted(p.parent.name for p in Path(folder).glob("*/scene.json"))


def load_scene(name_or_path: str | Path, folder: Path = SCENES_DIR) -> Scene:
    """A scene by name (scenes/<name>/scene.json), folder or scene.json path."""
    p = Path(name_or_path)
    if not p.exists():
        p = Path(folder) / str(name_or_path)
    if p.is_dir():
        p = p / "scene.json"
    if not p.exists():
        raise FileNotFoundError(f"scene {name_or_path!r} not found (available: {', '.join(list_scenes(folder))})")
    spec = json.loads(p.read_text(encoding="utf-8"))
    validate(spec)
    return Scene(spec, p.parent)


def validate(spec: dict) -> None:
    if spec.get("format") != "igris_scene":
        raise ValueError("not an igris_scene file (format must be 'igris_scene')")
    if int(spec.get("version", 0)) != 1:
        raise ValueError(f"unsupported scene version {spec.get('version')}")
    names = [o.get("name") for o in spec.get("objects", [])]
    if None in names or len(set(names)) != len(names):
        raise ValueError("every object needs a unique name")
    for o in spec.get("objects", []):
        for g in _parts(o):
            if g.get("shape") not in GEOM_TYPES:
                raise ValueError(f"object {o['name']!r}: unknown shape {g.get('shape')!r}")
            if g["shape"] == "mesh" and not g.get("file"):
                raise ValueError(f"object {o['name']!r}: mesh part needs 'file'")
            n_size = {"box": 3, "sphere": 1, "cylinder": 2, "capsule": 2}.get(g["shape"], 0)
            if len(g.get("size") or []) < n_size:
                raise ValueError(f"object {o['name']!r}: {g['shape']} part needs 'size' with {n_size} values")


def rot_quat(rot: dict | None) -> np.ndarray:
    """Scene rotation -> MuJoCo quaternion wxyz."""
    if not rot:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if "quat" in rot:
        return quat_normalize(np.asarray(rot["quat"], float))
    return rot_to_quat(R.from_euler("xyz", rot.get("euler_deg", [0, 0, 0]), degrees=True))


def _parts(obj: dict) -> list[dict]:
    """Geom parts of an object, with object-level defaults filled in."""
    keys = ("rgba", "friction", "collide")
    if "geoms" in obj:
        parts = []
        for g in obj["geoms"]:
            p = copy.deepcopy(g)
            for k in keys:
                if k not in p and k in obj:
                    p[k] = obj[k]
            parts.append(p)
        return parts
    p = {k: obj[k] for k in ("shape", "size", "file", "collision", "render", "scale") + keys if k in obj}
    return [p]


def _half_size(shape: str, size) -> list[float]:
    s = [float(v) for v in size]
    if shape == "box":
        return [s[0] / 2, s[1] / 2, s[2] / 2]
    if shape == "sphere":
        return [s[0], 0.0, 0.0]
    return [s[0], s[1] / 2, 0.0]  # cylinder / capsule: radius, half height


def build_model(scene: Scene, robot_mjcf: str | Path = DEFAULT_MJCF,
                hands: bool = False) -> tuple[mujoco.MjModel, SceneInfo]:
    """Compile the IGRIS-C MJCF with the scene's objects and the (inactive) grasp welds.

    hands: add the articulated finger overlays (igris_sim/hand_model.py) after the objects."""
    spec = mujoco.MjSpec.from_file(str(robot_mjcf))
    sp = scene.spec
    floor_cfg = sp.get("floor", {})
    floor = next((g for g in spec.geoms if g.name == "floor"), None)
    if floor is not None:
        if floor_cfg is False:
            floor.contype, floor.conaffinity = 0, 0
            floor.rgba = [0, 0, 0, 0]
        elif isinstance(floor_cfg, dict) and "rgba" in floor_cfg:
            floor.material = ""
            floor.rgba = floor_cfg["rgba"]

    mesh_id = 0
    for obj in sp.get("objects", []):
        body = spec.worldbody.add_body(name=OBJ_PREFIX + obj["name"], pos=list(map(float, obj.get("pos", [0, 0, 0]))),
                                       quat=list(rot_quat(obj.get("rot"))))
        dynamic = bool(obj.get("dynamic", False))
        if dynamic:
            body.add_freejoint(name=OBJ_PREFIX + obj["name"] + "_free")
        parts = _parts(obj)
        total_vol = 0.0
        geoms = []
        for k, g in enumerate(parts):
            common = {"pos": list(map(float, g.get("pos", [0, 0, 0]))), "quat": list(rot_quat(g.get("rot"))),
                      "rgba": list(map(float, g.get("rgba", [0.7, 0.7, 0.7, 1.0])))}
            if "friction" in g:
                common["friction"] = list(map(float, g["friction"]))
            collide = bool(g.get("collide", True))
            if g["shape"] == "mesh":
                scale = list(map(float, g.get("scale", [1, 1, 1])))
                vis = f"mesh{mesh_id}"
                mesh_id += 1
                spec.add_mesh(name=vis, file=str((scene.folder / g["file"]).resolve()), scale=scale)
                col_files = g.get("collision") or [g["file"]]
                same = col_files == [g["file"]]
                body.add_geom(name=f"{OBJ_PREFIX}{obj['name']}_vis{k}", type=GEOM_TYPES["mesh"], meshname=vis,
                              contype=0 if not same or not collide else 1, conaffinity=0 if not same or not collide else 1,
                              group=2 if not same else 0, density=0.0 if not same else DEFAULT_DENSITY, **common)
                if same:
                    geoms.append(f"{OBJ_PREFIX}{obj['name']}_vis{k}")
                else:
                    for c, f in enumerate(col_files):
                        cn = f"mesh{mesh_id}"
                        mesh_id += 1
                        spec.add_mesh(name=cn, file=str((scene.folder / f).resolve()), scale=scale)
                        name = f"{OBJ_PREFIX}{obj['name']}_col{k}_{c}"
                        body.add_geom(name=name, type=GEOM_TYPES["mesh"], meshname=cn, group=3,
                                      contype=int(collide), conaffinity=int(collide), density=DEFAULT_DENSITY,
                                      rgba=[0.5, 0.5, 0.5, 0.0], pos=common["pos"], quat=common["quat"],
                                      **({"friction": common["friction"]} if "friction" in common else {}))
                        geoms.append(name)
            else:
                hs = _half_size(g["shape"], g["size"])
                name = f"{OBJ_PREFIX}{obj['name']}_g{k}"
                body.add_geom(name=name, type=GEOM_TYPES[g["shape"]], size=hs, density=DEFAULT_DENSITY,
                              contype=int(collide), conaffinity=int(collide), **common)
                total_vol += _volume(g["shape"], hs)
                if collide:
                    geoms.append(name)
        if dynamic and "mass" in obj:
            # spread the given mass by volume over the primitive parts (mesh parts keep their density)
            for gs in body.geoms:
                if gs.type != GEOM_TYPES["mesh"] and total_vol > 0:
                    gs.mass = float(obj["mass"]) * _volume_from_spec(gs) / total_vol

    names = [o["name"] for o in sp.get("objects", [])]
    graspable = [o["name"] for o in sp.get("objects", []) if bool(o.get("graspable", o.get("dynamic", False)))
                 and bool(o.get("dynamic", False))]
    for side, hand in HANDS.items():
        for n in graspable:
            eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=hand, name2=OBJ_PREFIX + n,
                                   objtype=mujoco.mjtObj.mjOBJ_BODY, active=False)
            eq.name = f"grasp_{side}_{n}"
            eq.solref = [0.02, 1.0]
    if hands:
        from .hand_model import add_hand_overlays
        add_hand_overlays(spec, HANDS)

    model = spec.compile()
    info = SceneInfo(objects=names, body={}, dynamic={}, graspable=graspable, geoms={}, qposadr={})
    for o in sp.get("objects", []):
        n = o["name"]
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_PREFIX + n)
        info.body[n] = b
        info.dynamic[n] = bool(o.get("dynamic", False))
        info.geoms[n] = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_contype[g]]
        if info.dynamic[n]:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJ_PREFIX + n + "_free")
            info.qposadr[n] = int(model.jnt_qposadr[j])
    for side in HANDS:
        for n in graspable:
            info.weld[(side, n)] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{side}_{n}")
    return model, info


def _volume(shape, hs) -> float:
    if shape == "box":
        return 8.0 * hs[0] * hs[1] * hs[2]
    if shape == "sphere":
        return 4.0 / 3.0 * np.pi * hs[0] ** 3
    if shape == "cylinder":
        return np.pi * hs[0] ** 2 * 2 * hs[1]
    return np.pi * hs[0] ** 2 * 2 * hs[1] + 4.0 / 3.0 * np.pi * hs[0] ** 3  # capsule


def _volume_from_spec(gs) -> float:
    inv = {v: k for k, v in GEOM_TYPES.items()}
    return _volume(inv[gs.type], list(gs.size))
