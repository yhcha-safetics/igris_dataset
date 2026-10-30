"""Background / object segmentation of an aligned scan (scene frame: Z up, floor at z = 0).

  floor       points within floor_tolerance of z = 0
  supports    horizontal surfaces (upward normals, one height, connected) of at least
              support_min_area: tables, counters, shelves. Smaller horizontal patches stay with
              whatever they belong to (the top of a box is part of the box).
  walls       large vertical planes
  clusters    everything else, split into connected parts (DBSCAN):
                object     rests on the floor or a support, fits in max_object_size x
                           max_object_height and is not a leg under a support
                structure  the rest (furniture parts, clutter, wall fixtures): static background
Objects that the interaction phase shows moving are the interacted ones (igris_scan.changes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass
class SegmentParams:
    voxel: float = 0.01
    floor_tolerance: float = 0.02      # m
    plane_tolerance: float = 0.015     # m
    support_min_area: float = 0.3      # m^2
    support_min_height: float = 0.15   # m above the floor
    wall_min_area: float = 1.0         # m^2
    wall_tolerance: float = 0.03       # m, RANSAC inliers of walls (far surfaces are noisy)
    surface_layer: float = 0.015       # m above a support that still belongs to it (TSDF rounds the
                                       # foot of objects into a skirt that would widen them)
    cluster_eps: float = 0.025         # m, DBSCAN neighbourhood
    min_object_points: int = 60
    max_object_size: float = 0.8       # m, horizontal extent
    max_object_height: float = 1.0     # m
    rest_tolerance: float = 0.04       # m between an object's bottom and its support
    min_thickness: float = 0.02        # m: thinner clusters are fragments of planes
    leg_reach: float = 0.25            # m: a cluster under a support reaching this close to it is a leg


@dataclass
class Surface:
    name: str
    kind: str                  # floor | support | wall
    height: float = 0.0        # horizontal: z of the surface
    normal: np.ndarray = None  # wall: unit normal (horizontal)
    offset: float = 0.0        # wall: n . x = offset
    center: np.ndarray = None  # (x, y) / wall: 3D centre
    size: np.ndarray = None    # (sx, sy) of the oriented footprint / wall: (length, height)
    yaw: float = 0.0           # rad, footprint orientation
    hull: np.ndarray = None    # (k, 2) footprint polygon
    rgba: list = field(default_factory=lambda: [0.6, 0.6, 0.6, 1.0])
    index: np.ndarray = None   # point indices

    def contains(self, xy, margin=0.0) -> bool:
        return cv2.pointPolygonTest(self.hull.astype(np.float32), (float(xy[0]), float(xy[1])), True) >= -margin


@dataclass
class Cluster:
    name: str
    kind: str                  # object | structure
    index: np.ndarray          # point indices
    center: np.ndarray         # OBB centre (x, y, z)
    size: np.ndarray           # OBB extents (sx, sy, sz)
    yaw: float
    support: str | None = None
    support_height: float = 0.0
    rgba: list = field(default_factory=lambda: [0.6, 0.6, 0.6, 1.0])
    moved: bool = False        # set by igris_scan.changes
    evidence: dict = field(default_factory=dict)


@dataclass
class Segmentation:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    surfaces: list
    clusters: list
    label: np.ndarray          # per point: "floor" | surface name | cluster name | ""

    @property
    def objects(self):
        return [c for c in self.clusters if c.kind == "object"]

    def surface(self, name):
        return next(s for s in self.surfaces if s.name == name)


def _rgba(colors, idx):
    c = np.median(colors[idx], axis=0) if len(idx) else np.array([0.6, 0.6, 0.6])
    return [round(float(v), 3) for v in c] + [1.0]


def _footprint(xy):
    xy = xy.astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(xy)
    return np.array([cx, cy]), np.array([w, h]), np.radians(ang), cv2.convexHull(xy)[:, 0, :]


def _components(pts, eps, min_points):
    if len(pts) == 0:
        return []
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    lab = np.asarray(pc.cluster_dbscan(eps, min_points))
    return [np.nonzero(lab == k)[0] for k in range(lab.max() + 1)] if lab.size and lab.max() >= 0 else []


def segment(points: np.ndarray, colors: np.ndarray, normals: np.ndarray, p: SegmentParams | None = None) -> Segmentation:
    p = p or SegmentParams()
    n = len(points)
    label = np.array([""] * n, dtype=object)
    z = points[:, 2]
    surfaces = []
    # floor
    fl = np.nonzero(np.abs(z) < p.floor_tolerance)[0]
    label[fl] = "floor"
    floor = Surface("floor", "floor", 0.0, rgba=_rgba(colors, fl), index=fl)
    if len(fl) > 3:
        floor.center, floor.size, floor.yaw, floor.hull = _footprint(points[fl, :2])
    surfaces.append(floor)
    # supports: peaks of the height histogram of upward-facing points, split into connected surfaces
    up = (normals[:, 2] > 0.85) & (z > p.support_min_height) & (label == "")
    if up.any():
        bins = np.arange(p.support_min_height, z[up].max() + 0.02, 0.01)
        hist, edges = np.histogram(z[up], bins)
        for k in np.argsort(hist)[::-1]:
            if hist[k] * p.voxel ** 2 < p.support_min_area * 0.5:
                break
            h0 = 0.5 * (edges[k] + edges[k + 1])
            cand = np.nonzero(up & (np.abs(z - h0) < p.plane_tolerance) & (label == ""))[0]
            if len(cand) == 0:
                continue
            h = float(np.median(z[cand]))
            cand = np.nonzero(up & (np.abs(z - h) < p.plane_tolerance) & (label == ""))[0]
            for comp in _components(np.c_[points[cand, :2], np.zeros(len(cand))], 2.5 * p.voxel, 10):
                idx = cand[comp]
                if len(idx) * p.voxel ** 2 < p.support_min_area:
                    continue
                c, s, yaw, hull = _footprint(points[idx, :2])
                name = f"support_{sum(x.kind == 'support' for x in surfaces) + 1}"
                hs = float(np.median(z[idx]))
                sup = Surface(name, "support", hs, center=c, size=s, yaw=yaw, hull=hull, rgba=_rgba(colors, idx))
                # the thin layer on it (whatever the normals) belongs to it too
                near = np.nonzero((label == "") & (z > hs - p.plane_tolerance) & (z < hs + p.surface_layer))[0]
                near = near[[sup.contains(xy) for xy in points[near, :2]]] if len(near) else near
                sup.index = np.union1d(idx, near)
                surfaces.append(sup)
                label[sup.index] = name
    # walls: big vertical planes among the rest
    rest = np.nonzero((label == "") & (np.abs(normals[:, 2]) < 0.3))[0]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[rest]))
    o3d.utility.random.seed(0)
    for _ in range(6):
        if len(rest) < 200:
            break
        model, inl = pc.segment_plane(p.wall_tolerance, 3, 1000)
        nrm = np.array(model[:3]) / np.linalg.norm(model[:3])
        if len(inl) * p.voxel ** 2 < p.wall_min_area or abs(nrm[2]) > 0.15:
            break
        idx = rest[np.asarray(inl)]
        nrm = nrm - np.array([0, 0, nrm[2]])
        nrm /= np.linalg.norm(nrm)
        along = np.array([-nrm[1], nrm[0], 0.0])
        t = points[idx] @ along
        zz = points[idx, 2]
        off = float(np.median(points[idx] @ nrm))
        c3 = nrm * off + along * 0.5 * (t.min() + t.max()) + np.array([0, 0, 0.5 * (zz.min() + zz.max())])
        name = f"wall_{sum(x.kind == 'wall' for x in surfaces) + 1}"
        surfaces.append(Surface(name, "wall", normal=nrm, offset=off, center=c3,
                                size=np.array([t.max() - t.min(), zz.max()]), rgba=_rgba(colors, idx), index=idx))
        label[idx] = name
        keep = np.setdiff1d(np.arange(len(rest)), inl)
        rest = rest[keep]
        pc = pc.select_by_index(keep)
    # clusters of what is left
    left = np.nonzero(label == "")[0]
    left = left[z[left] > p.floor_tolerance * 0.5]
    supports = [s for s in surfaces if s.kind == "support"]
    walls = [s for s in surfaces if s.kind == "wall"]
    clusters = []
    for comp in _components(points[left], p.cluster_eps, 10):
        idx = left[comp]
        if len(idx) < 10:
            continue
        pts = points[idx]
        c2, s2, yaw, _ = _footprint(pts[:, :2])
        z0, z1 = float(pts[:, 2].min()), float(pts[:, 2].max())
        cl = Cluster("", "structure", idx, np.array([c2[0], c2[1], 0.5 * (z0 + z1)]), np.array([s2[0], s2[1], z1 - z0]),
                     yaw, rgba=_rgba(colors, idx))
        base = [floor] + supports
        rest_on = [s for s in base if abs(z0 - s.height) < p.rest_tolerance
                   and (s.kind == "floor" or s.contains(c2, margin=0.02))]
        under = any(s.contains(c2, margin=0.05) and -p.leg_reach < z1 - s.height < 0.02 for s in supports)
        at_wall = any(np.median(np.abs(pts @ w.normal - w.offset)) < 2 * p.wall_tolerance for w in walls)
        if rest_on:
            sup = min(rest_on, key=lambda s: abs(z0 - s.height))
            cl.support, cl.support_height = sup.name, sup.height
            cl.center[2] = 0.5 * (sup.height + z1)  # from the support up: the foot was cut off with the surface
            cl.size[2] = z1 - sup.height
        if (rest_on and not under and not at_wall and len(idx) >= p.min_object_points
                and max(s2) <= p.max_object_size and min(s2) >= p.min_thickness
                and z1 - cl.support_height <= p.max_object_height):
            cl.kind = "object"
        clusters.append(cl)
    k = {"object": 0, "structure": 0}
    for cl in sorted(clusters, key=lambda c: (c.kind, c.center[0], c.center[1])):
        k[cl.kind] += 1
        cl.name = f"{cl.kind}_{k[cl.kind]}"
        label[cl.index] = cl.name
    return Segmentation(points, colors, normals, surfaces, clusters, label)


def nearest(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Distance from each of points_a to the nearest of points_b."""
    if len(points_b) == 0:
        return np.full(len(points_a), np.inf)
    return cKDTree(points_b).query(points_a, k=1)[0]
