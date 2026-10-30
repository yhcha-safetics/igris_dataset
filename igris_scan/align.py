"""Correction of a reconstruction into the scene frame: gravity, floor, heading, origin, scale.

The reconstruction lives in the first camera's frame (OpenCV axes, arbitrary tilt). The scene frame
is the simulator's: Z up with the floor at z = 0, +X the heading, metres.

  up       the normal of the floor: the lowest large horizontal plane among the RANSAC planes whose
           normal is near the gravity estimate. Gravity from the cameras: people do not roll a
           handheld camera, so its x axis (image right) stays horizontal; over a sweep these axes span
           the horizontal plane and up is perpendicular to all of them, whatever the pitch. A camera
           that did not turn only gives the image "up" (-y), off by its pitch: a wider tolerance.
           With no floor in view, the largest horizontal plane is a support of known height
           (support_height)
  heading  the first camera's viewing direction projected on the floor, snapped to the dominant
           wall / table edge direction within manhattan_snap_deg (rooms are rectangular)
  origin   below the first camera: where the operator / robot stands (the scene spawn), plus an
           optional offset
  scale    1, or support_height / measured height of the main support when the true height of a
           table is known (stereo cameras can be off by a percent or two)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d


@dataclass
class AlignParams:
    plane_distance: float = 0.015      # m, RANSAC inlier distance
    max_planes: int = 8
    min_plane_fraction: float = 0.03   # of the (downsampled) points
    horizontal_deg: float = 25.0       # plane normal from the gravity estimate of a sweep
    static_horizontal_deg: float = 50.0  # ... of a camera that did not turn (pitch unknown)
    manhattan_snap_deg: float = 45.0   # 45: always the room axis nearest to the camera heading
    support_height: float | None = None  # m: true height of the main support (scale / no floor in view)
    yaw_offset_deg: float = 0.0
    origin_offset: tuple = (0.0, 0.0)  # m, added to the origin below the first camera (scene frame)


@dataclass
class Alignment:
    T: np.ndarray                      # 4x4: scene point = T @ reconstruction point (incl. scale)
    scale: float = 1.0
    floor_visible: bool = True
    floor_rms: float = 0.0             # m, residual of the floor plane
    camera_height: float = 0.0         # m, first camera above the floor
    heading_source: str = "camera"     # camera | manhattan
    notes: list = field(default_factory=list)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.T[:3, :3].T + self.T[:3, 3]

    def pose(self, T_cam: np.ndarray) -> np.ndarray:
        """Camera-to-reconstruction pose -> camera-to-scene pose (rotation without the scale)."""
        out = self.T @ T_cam
        out[:3, :3] /= self.scale
        return out

    def to_dict(self) -> dict:
        return {"T": np.round(self.T, 7).tolist(), "scale": self.scale, "floor_visible": self.floor_visible,
                "floor_rms_m": round(self.floor_rms, 4), "camera_height_m": round(self.camera_height, 3),
                "heading_source": self.heading_source, "notes": self.notes}


def _rot_to(a, b) -> np.ndarray:
    """Smallest rotation taking unit vector a to b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def find_planes(points: np.ndarray, p: AlignParams, seed: int = 0):
    """Dominant planes [(normal, d, inlier indices)] by iterated RANSAC (n . x + d = 0)."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    idx = np.arange(len(points))
    planes = []
    o3d.utility.random.seed(seed)
    for _ in range(p.max_planes):
        if len(idx) < max(100, p.min_plane_fraction * len(points)):
            break
        sub = pc.select_by_index(idx)
        model, inl = sub.segment_plane(p.plane_distance, 3, 1000)
        if len(inl) < p.min_plane_fraction * len(points):
            break
        n, d = np.array(model[:3]), float(model[3])
        s = np.linalg.norm(n)
        planes.append((n / s, d / s, idx[np.asarray(inl)]))
        idx = np.delete(idx, inl)
    return planes


def _yaw_of(v) -> float:
    return float(np.arctan2(v[1], v[0]))


def _wrap90(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def gravity_estimate(camera_poses: list[np.ndarray], min_spread: float = 0.15) -> tuple[np.ndarray, bool]:
    """Up direction from the camera axes -> (unit vector, from a sweep?)."""
    image_up = -np.mean([T[:3, 1] for T in camera_poses], axis=0)
    image_up /= np.linalg.norm(image_up)
    X = np.array([T[:3, 0] for T in camera_poses])
    _, sv, Vt = np.linalg.svd(X, full_matrices=False)
    if len(X) >= 3 and sv[1] / sv[0] > min_spread:  # the image x axes turned: up is normal to all of them
        up = Vt[2] if Vt[2] @ image_up > 0 else -Vt[2]
        return up, True
    return image_up, False


def align(points: np.ndarray, camera_poses: list[np.ndarray], p: AlignParams | None = None) -> Alignment:
    """Alignment of a reconstruction (points and camera-to-world poses in its frame)."""
    p = p or AlignParams()
    notes = []
    up_hint, swept = gravity_estimate(camera_poses)
    planes = find_planes(points, p)
    cos_h = np.cos(np.radians(p.horizontal_deg if swept else p.static_horizontal_deg))
    horiz = []
    for n, d, inl in planes:
        if abs(n @ up_hint) >= cos_h:
            if n @ up_hint < 0:
                n, d = -n, -d
            horiz.append((n, d, inl, -d))  # height along up: n . x = -d
    if not horiz:
        raise ValueError("no horizontal surface found: is the floor or a table in view?")
    cam0 = camera_poses[0][:3, 3]
    # the floor: the lowest horizontal plane that is clearly below the first camera
    below = [h for h in horiz if h[0] @ cam0 - h[3] > 0.3]
    floor_visible = bool(below)
    if floor_visible:
        n, d, inl, h = min(below, key=lambda h: h[3])
    else:
        n, d, inl, h = max(horiz, key=lambda h: len(h[2]))
        if p.support_height is None:
            notes.append("no floor in view: the largest horizontal surface is taken as the floor "
                         "(give support_height for a table)")
    R1 = _rot_to(n, np.array([0.0, 0.0, 1.0]))
    area = cv2.contourArea(cv2.convexHull((points[inl] @ R1.T)[:, :2].astype(np.float32)))
    levels = {round(hh[3], 1) for hh in horiz}
    if floor_visible and area < 1.5 and len(levels) == 1:
        # one small horizontal level: as likely a table top seen from above as a floor
        if p.support_height is not None:
            floor_visible = False
        else:
            notes.append(f"the floor plane is small ({area:.2f} m^2): if it is a table top, give support_height")
    rms = float(np.sqrt(np.mean((points[inl] @ n + d) ** 2)))
    # heading: first camera's view direction on the floor, snapped to walls / table edges
    f = R1 @ camera_poses[0][:3, 2]
    yaw = _yaw_of(f)
    src = "camera"
    dirs, weights = [], []
    for n2, _, inl2 in planes:
        v = R1 @ n2
        if abs(v[2]) < 0.2:  # wall
            dirs.append(_yaw_of(v))
            weights.append(len(inl2))
    for n2, _, inl2, _ in horiz:  # long edges of horizontal surfaces (tables)
        xy = (points[inl2] @ R1.T)[:, :2].astype(np.float32)
        if len(xy) > 50:
            (_, _), (w, l), ang = cv2.minAreaRect(xy)
            if max(w, l) > 0.3:
                dirs.append(np.radians(ang))
                weights.append(len(inl2) * 0.5)
    if dirs:
        # dominant direction modulo 90 deg (circular mean of 4x the angle)
        a4 = np.array(dirs) * 4
        dom = np.arctan2(np.average(np.sin(a4), weights=weights), np.average(np.cos(a4), weights=weights)) / 4
        delta = _wrap90(dom - yaw)
        if abs(delta) <= np.radians(p.manhattan_snap_deg):
            yaw, src = yaw + delta, "manhattan"
    yaw += np.radians(p.yaw_offset_deg)
    Rz = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw), np.cos(-yaw), 0], [0, 0, 1]])
    R = Rz @ R1
    floor_z = float((R @ (points[inl].mean(0)))[2])
    c = R @ cam0
    t = -np.array([c[0], c[1], floor_z])
    scale = 1.0
    cam_h = float(c[2] - floor_z)
    if p.support_height is not None:
        if floor_visible:
            sup = [hh for hh in horiz if 0.2 < hh[3] - h < 1.6]
            if sup:
                hs = max(sup, key=lambda hh: len(hh[2]))[3] - h
                scale = p.support_height / hs
                notes.append(f"scale {scale:.4f} from the main support ({hs:.3f} m measured, {p.support_height} m given)")
            else:
                notes.append("support_height given but no support found: scale 1")
        else:  # the plane found is the table top: put it at support_height
            t[2] += p.support_height
            cam_h += p.support_height
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = scale * t + np.array([*p.origin_offset, 0.0])
    if not 0.3 < cam_h * scale < 2.5:
        notes.append(f"first camera {cam_h * scale:.2f} m above the floor: check the floor detection")
    return Alignment(T, scale, floor_visible, rms, cam_h * scale, src, notes)
