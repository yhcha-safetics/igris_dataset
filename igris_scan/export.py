"""A segmented scan -> simulation scene (scenes/<name>/scene.json, igris_sim/scene.py format).

    scenes/<name>/
      scene.json              floor colour, spawn at the scan origin, the objects below
      background.obj          the scanned surroundings (walls, furniture, clutter) with vertex colours,
                              without the floor, the support tops and the objects; visual only
      collision/*.obj         convex hulls of large structures (invisible colliders)
      objects/<obj>.obj       scanned object surface with vertex colours (what the app draws)
      objects/<obj>_hull.obj  its collider: convex hull of the surface extruded down to its support
      scan_report.json        alignment, surfaces, objects, interaction evidence, settings
      preview.png             top view of the segmentation

    floor     the MJCF floor at z = 0 in the scanned floor colour (flat, collides)
    supports  boxes whose top is the measured surface (visual + collision): objects rest on a flat top
    walls     thin boxes behind the measured wall planes (collision only; the scan shows them)
    objects   free rigid bodies if interacted (or all / none, see ExportParams.dynamic), else fixed;
              mass = hull volume x object_density, clamped
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, cKDTree

from .segment import Segmentation

INVISIBLE = 0.0


@dataclass
class ExportParams:
    max_triangles: int = 300000        # of the background mesh (quadric decimation above)
    object_density: float = 250.0      # kg/m^3 over the hull volume (boxes and packaging are mostly air)
    min_mass: float = 0.05
    max_mass: float = 5.0
    dynamic: str = "interacted"        # interacted | all | none: which objects become free bodies
    structure_min_points: int = 300    # smaller structure clusters get no collider
    wall_thickness: float = 0.05
    support_thickness: float = 0.03
    top_face_ratio: float = 0.5        # top face area / footprint area above which the top face sets the footprint


def write_obj(path: Path, V: np.ndarray, F: np.ndarray, C: np.ndarray | None = None, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"# {header}\n" if header else "")
        if C is not None:
            np.savetxt(f, np.c_[V, np.clip(C, 0, 1)], fmt="v %.5f %.5f %.5f %.3f %.3f %.3f")
        else:
            np.savetxt(f, V, fmt="v %.5f %.5f %.5f")
        np.savetxt(f, F + 1, fmt="f %d %d %d")


def _rect(xy):
    import cv2
    (cx, cy), (w, h), a = cv2.minAreaRect(xy.astype(np.float32))
    return np.array([cx, cy]), np.array([w, h]), np.radians(a)


def _in_rect(xy, c, s, yaw, margin):
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    loc = (xy - c) @ R.T
    return np.all(np.abs(loc) <= s / 2 + margin, axis=1)


def _hull(points: np.ndarray):
    """Convex hull (vertices, outward faces, volume) or None if degenerate."""
    try:
        h = ConvexHull(points)
    except Exception:
        return None
    V = points[h.vertices]
    remap = {v: k for k, v in enumerate(h.vertices)}
    F = np.array([[remap[a] for a in s] for s in h.simplices])
    ctr = V.mean(0)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = np.einsum("ij,ij->i", n, V[F].mean(1) - ctr) < 0
    F[flip] = F[flip][:, ::-1]
    return V, F, float(h.volume)


def _submesh(V, F, keep_v):
    keep_f = keep_v[F].all(axis=1)
    used = np.unique(F[keep_f])
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    return used, remap[F[keep_f]]


def object_geometry(seg: Segmentation, obj, mesh_V, mesh_F, mesh_C, voxel: float, p: ExportParams) -> dict | None:
    """Origin, footprint, collider hull and display mesh of one object (scene frame)."""
    P, N = seg.points[obj.index], seg.normals[obj.index]
    z1, hs = float(P[:, 2].max()), obj.support_height
    c, s, yaw = _rect(P[:, :2])
    top = (N[:, 2] > 0.8) & (P[:, 2] > z1 - 0.02)
    if top.sum() >= 30:
        ct, st, yt = _rect(P[top, :2])
        if st.prod() >= p.top_face_ratio * s.prod():  # flat top (box, can): sharper than the TSDF-fattened sides
            c, s, yaw = ct, st + 0.5 * voxel, yt
    inside = _in_rect(P[:, :2], c, s, yaw, voxel)
    P = P[inside]
    base = np.c_[P[:, :2], np.full(len(P), hs)]
    hull = _hull(np.r_[P, base])
    if hull is None or min(s) < 0.01 or z1 - hs < 0.01:
        return None
    origin = np.array([c[0], c[1], 0.5 * (hs + z1)])
    near = cKDTree(P).query(mesh_V, k=1, distance_upper_bound=1.5 * voxel)[0] < np.inf
    near &= mesh_V[:, 2] > hs + 0.002
    used, F = _submesh(mesh_V, mesh_F, near)
    return {"origin": origin, "size": [float(s[0]), float(s[1]), z1 - hs], "yaw": float(yaw),
            "hull": (hull[0] - origin, hull[1]), "volume": hull[2],
            "render": (mesh_V[used] - origin, F, mesh_C[used] if mesh_C is not None else None),
            "mesh_vertices": near}


def background_mesh(seg: Segmentation, mesh: o3d.geometry.TriangleMesh, remove: np.ndarray, voxel: float,
                    p: ExportParams) -> o3d.geometry.TriangleMesh:
    """The scan mesh without the floor, the support tops and the `remove` vertices (objects)."""
    V = np.asarray(mesh.vertices)
    drop = remove.copy()
    drop |= np.abs(V[:, 2]) < 0.02
    for s in seg.surfaces:
        if s.kind == "support":
            d = cKDTree(seg.points[s.index]).query(V, k=1, distance_upper_bound=1.5 * voxel)[0]
            drop |= d < np.inf
    bg = o3d.geometry.TriangleMesh(mesh)
    bg.remove_vertices_by_mask(drop)
    tri, n_tri, _ = bg.cluster_connected_triangles()
    tri, n_tri = np.asarray(tri), np.asarray(n_tri)
    if len(n_tri):
        bg.remove_triangles_by_mask(n_tri[tri] < 50)  # specks
        bg.remove_unreferenced_vertices()
    if len(bg.triangles) > p.max_triangles:
        bg = bg.simplify_quadric_decimation(p.max_triangles)
    return bg


def export_scene(out: Path, name: str, seg: Segmentation, mesh_scene: o3d.geometry.TriangleMesh,
                 moved: dict, finals: dict, voxel: float, p: ExportParams | None = None,
                 description: str = "", report: dict | None = None) -> dict:
    """Write the scene folder; returns the scene spec."""
    p = p or ExportParams()
    out = Path(out)
    (out / "objects").mkdir(parents=True, exist_ok=True)
    (out / "collision").mkdir(parents=True, exist_ok=True)
    V = np.asarray(mesh_scene.vertices)
    F = np.asarray(mesh_scene.triangles)
    C = np.asarray(mesh_scene.vertex_colors) if mesh_scene.has_vertex_colors() else None
    floor = seg.surface("floor")
    objects = []
    remove = np.zeros(len(V), bool)
    obj_report = []
    for o in seg.objects:
        g = object_geometry(seg, o, V, F, C, voxel, p)
        if g is None:
            continue
        remove |= g["mesh_vertices"]
        interacted = bool(moved.get(o.name, {}).get("moved", False))
        dynamic = p.dynamic == "all" or (p.dynamic == "interacted" and interacted)
        mass = float(np.clip(g["volume"] * p.object_density, p.min_mass, p.max_mass))
        write_obj(out / "objects" / f"{o.name}.obj", *g["render"], header=f"{o.name}: scanned surface, body frame")
        write_obj(out / "objects" / f"{o.name}_hull.obj", *g["hull"], header=f"{o.name}: collider")
        e = {"name": o.name, "shape": "mesh", "file": f"objects/{o.name}_hull.obj",
             "render": f"objects/{o.name}.obj", "pos": np.round(g["origin"], 4).tolist(),
             "dynamic": dynamic, "rgba": o.rgba}  # MuJoCo draws this; the app the vertex colours
        if dynamic:
            e["mass"] = round(mass, 3)
        objects.append(e)
        rep = {"name": o.name, "interacted": interacted, "dynamic": dynamic, "support": o.support,
               "center": np.round(g["origin"], 3).tolist(), "size": np.round(g["size"], 3).tolist(),
               "yaw_deg": round(float(np.degrees(g["yaw"])), 1), "volume_m3": round(g["volume"], 5),
               "mass": round(mass, 3), "rgba": o.rgba, "points": int(len(o.index))}
        if o.name in moved:
            rep["moved_at_s"] = moved[o.name].get("t_moved")
        if o.name in finals:
            rep["final"] = {k: (np.round(v, 3).tolist() if isinstance(v, (list, np.ndarray)) else v)
                            for k, v in finals[o.name].items()}
        obj_report.append(rep)
    bg = background_mesh(seg, mesh_scene, remove, voxel, p)
    bv, bf = np.asarray(bg.vertices), np.asarray(bg.triangles)
    bc = np.asarray(bg.vertex_colors) if bg.has_vertex_colors() else None
    write_obj(out / "background.obj", bv, bf, bc, header=f"{name}: scanned background, scene frame")
    fixed = [{"name": "scan_background", "shape": "mesh", "file": "background.obj", "collide": False,
              "rgba": [0.75, 0.75, 0.75, 1.0]}]
    for s in seg.surfaces:
        if s.kind == "support":
            t = p.support_thickness
            fixed.append({"name": s.name, "shape": "box", "size": [round(float(s.size[0]), 3), round(float(s.size[1]), 3), t],
                          "pos": [round(float(s.center[0]), 4), round(float(s.center[1]), 4), round(s.height - t / 2, 4)],
                          "rot": {"euler_deg": [0, 0, round(float(np.degrees(s.yaw)), 2)]}, "rgba": s.rgba})
        elif s.kind == "wall":
            nrm = s.normal if (np.zeros(3) - s.center) @ s.normal >= 0 else -s.normal  # towards the scan origin
            c = s.center - nrm * p.wall_thickness / 2
            fixed.append({"name": s.name, "shape": "box",
                          "size": [p.wall_thickness, round(float(s.size[0]), 3), round(float(s.size[1]), 3)],
                          "pos": [round(float(c[0]), 4), round(float(c[1]), 4), round(float(s.size[1] / 2), 4)],
                          "rot": {"euler_deg": [0, 0, round(float(np.degrees(np.arctan2(nrm[1], nrm[0]))), 2)]},
                          "rgba": s.rgba[:3] + [INVISIBLE]})
    for cl in seg.clusters:
        if cl.kind != "structure" or len(cl.index) < p.structure_min_points:
            continue
        h = _hull(seg.points[cl.index])
        if h is None or min(cl.size) < 0.01:
            continue
        ctr = h[0].mean(0)
        write_obj(out / "collision" / f"{cl.name}.obj", h[0] - ctr, h[1], header=f"{cl.name}: collider")
        fixed.append({"name": cl.name, "shape": "mesh", "file": f"collision/{cl.name}.obj",
                      "pos": np.round(ctr, 4).tolist(), "rgba": cl.rgba[:3] + [INVISIBLE]})
    spec = {"format": "igris_scene", "version": 1, "name": name,
            "description": description or "scanned with a depth camera (igris_scan)",
            "spawn": {"pos": [0.0, 0.0], "yaw_deg": 0},
            "floor": {"rgba": floor.rgba},
            "objects": fixed + objects}
    (out / "scene.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    rep = {**(report or {}), "objects": obj_report, "export": asdict(p),
           "surfaces": [{"name": s.name, "kind": s.kind, "height": round(s.height, 3),
                         "size": None if s.size is None else np.round(s.size, 3).tolist(), "rgba": s.rgba}
                        for s in seg.surfaces],
           "background_triangles": int(len(bf))}
    (out / "scan_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    preview(out / "preview.png", seg, obj_report)
    return spec


def preview(path: Path, seg: Segmentation, objects: list[dict]) -> None:
    """Top view: background grey, supports, walls, objects (interacted in orange, with their final place)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.transforms import Affine2D

    rng = np.random.default_rng(0)
    pts = seg.points[rng.choice(len(seg.points), min(60000, len(seg.points)), replace=False)]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c=np.clip(pts[:, 2], 0, 2), cmap="Greys", alpha=0.4)
    for s in seg.surfaces:
        if s.kind == "support":
            ax.add_patch(Polygon(s.hull, closed=True, fc=s.rgba[:3], ec="k", alpha=0.6))
            ax.text(*s.center, f"{s.name}\n{s.height:.2f} m", ha="center", fontsize=7)
        elif s.kind == "wall":
            along = np.array([-s.normal[1], s.normal[0], 0])
            a, b = s.center - along * s.size[0] / 2, s.center + along * s.size[0] / 2
            ax.plot([a[0], b[0]], [a[1], b[1]], "b-", lw=3, alpha=0.6)
    for o in objects:
        col = "tab:orange" if o["interacted"] else "tab:blue"
        cx, cy = o["center"][:2]
        sx, sy = o["size"][:2]
        r = Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fc=col, ec="k", alpha=0.8,
                      transform=Affine2D().rotate_deg_around(cx, cy, o["yaw_deg"]) + ax.transData)
        ax.add_patch(r)
        ax.text(cx, cy + sy / 2 + 0.03, o["name"] + (" (moved)" if o["interacted"] else ""), ha="center", fontsize=7)
        if "final" in o:
            fx, fy = o["final"]["center"][:2]
            ax.annotate("", (fx, fy), (cx, cy), arrowprops={"arrowstyle": "->", "color": "tab:orange", "lw": 2})
    ax.annotate("", (0.4, 0), (0, 0), arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 3})
    ax.text(0, -0.12, "spawn", color="tab:green", ha="center")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("scan segmentation: supports, walls (blue lines), objects (orange: interacted)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
