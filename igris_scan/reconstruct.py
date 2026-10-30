"""Camera tracking and TSDF fusion of a capture.

Tracking is frame-to-keyframe: each depth frame (cleaned, rgbd.clean_depth) is registered to the
current keyframe's depth by multi-scale point-to-plane ICP (Open3D rgbd_odometry_multi_scale), starting
from the pose a constant-velocity motion model predicts; a frame that moved more than
keyframe_distance / keyframe_angle from its keyframe becomes the next keyframe. Registration against
measured depth, not against a TSDF raycast: Open3D's raycast places surfaces seen at grazing angles
1-3 cm too far (about one voxel), which biases every frame-to-model step the same way.
A frame whose registration fails (few inliers, implausible jump) keeps the previous pose and is not
fused (`tracked` false). Poses from the capture (poses.txt) or a fixed camera skip the tracking.

The world frame of a Model is the camera frame of its first frame (OpenCV axes); igris_scan.align
maps it to the scene frame. `localize` registers further frames to the model's keyframes without
fusing them (the interaction phase, where people and moved objects must not enter the scan).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
import open3d.core as o3c

from .rgbd import Capture, Intrinsics

DEVICE = o3c.Device("CPU:0")


@dataclass
class ScanParams:
    voxel: float = 0.01            # m, TSDF voxel size
    depth_min: float = 0.15
    depth_max: float = 3.5         # m, farther depth is dropped (noise grows with d^2)
    trunc_voxels: float = 6.0      # TSDF truncation in voxels
    block_count: int = 60000       # voxel blocks (16^3) the hash map can hold
    odometry_distance: float = 0.07  # m, ICP outlier distance (moving people are outliers)
    icp_iterations: tuple = (20, 10, 5)  # per pyramid level, fine to coarse
    min_fitness: float = 0.25      # ICP inlier fraction below which a frame is not tracked
    max_jump: float = 0.15         # m / rad between consecutive frames: more is a tracking failure
    keyframe_distance: float = 0.12  # m ...
    keyframe_angle: float = 0.17   # ... rad from the keyframe: start a new one
    weight_threshold: float = 3.0  # TSDF weight (observations) for surface extraction


@dataclass
class Keyframe:
    frame: int
    pose: np.ndarray
    rgbd: o3d.t.geometry.RGBDImage


@dataclass
class Model:
    params: ScanParams
    intr: Intrinsics
    vbg: o3d.t.geometry.VoxelBlockGrid
    frames: list = field(default_factory=list)     # capture frame numbers fused
    poses: dict = field(default_factory=dict)      # frame -> camera-to-world 4x4 (every frame seen)
    tracked: dict = field(default_factory=dict)    # frame -> bool
    keyframes: list = field(default_factory=list)  # Keyframe

    def point_cloud(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Surface points, colours (0..1) and normals of the TSDF."""
        return cloud_arrays(self.vbg, self.params)

    def mesh(self) -> o3d.geometry.TriangleMesh:
        m = self.vbg.extract_triangle_mesh(self.params.weight_threshold).to_legacy()
        m.compute_vertex_normals()
        return m


def cloud_arrays(vbg, params: ScanParams, weight_threshold: float | None = None):
    w = params.weight_threshold if weight_threshold is None else weight_threshold
    pc = vbg.extract_point_cloud(w).to_legacy()
    if len(pc.points) and not pc.has_normals():
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * params.voxel, max_nn=30))
    colors = np.asarray(pc.colors) if pc.has_colors() else np.zeros((len(pc.points), 3))
    return np.asarray(pc.points), colors, np.asarray(pc.normals)


def new_grid(params: ScanParams) -> o3d.t.geometry.VoxelBlockGrid:
    return o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight", "color"), (o3c.float32, o3c.float32, o3c.float32),
                                         ((1), (1), (3)), params.voxel, 16, params.block_count, DEVICE)


def _K(intr: Intrinsics):
    return o3c.Tensor(intr.K, o3c.Dtype.Float64)


def load_rgbd(capture: Capture, i: int, p: ScanParams) -> o3d.t.geometry.RGBDImage:
    """Frame i as an Open3D RGBD image: cleaned depth in mm (uint16), colour."""
    d = capture.depth(i, min_depth=p.depth_min, max_depth=p.depth_max)
    depth = o3d.t.geometry.Image(o3c.Tensor(np.round(d * 1000.0).astype(np.uint16)))
    color = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(capture.color(i))))
    return o3d.t.geometry.RGBDImage(color, depth)


def integrate(vbg, rgbd, K, T: np.ndarray, p: ScanParams) -> None:
    ext = o3c.Tensor(np.linalg.inv(T))
    blocks = vbg.compute_unique_block_coordinates(rgbd.depth, K, ext, 1000.0, p.depth_max, p.trunc_voxels)
    vbg.integrate(blocks, rgbd.depth, rgbd.color, K, ext, 1000.0, p.depth_max, p.trunc_voxels)


def register(src, tgt, K, init: np.ndarray, p: ScanParams):
    """Pose of the source camera in the target camera frame -> (transform, fitness) or (None, 0)."""
    odo = o3d.t.pipelines.odometry
    try:
        res = odo.rgbd_odometry_multi_scale(
            src, tgt, K, o3c.Tensor(init), 1000.0, p.depth_max,
            [odo.OdometryConvergenceCriteria(n) for n in p.icp_iterations], odo.Method.PointToPlane,
            odo.OdometryLossParams(p.odometry_distance))
    except RuntimeError:
        return None, 0.0
    return res.transformation.cpu().numpy(), float(res.fitness)


def _motion(a, b) -> tuple[float, float]:
    d = np.linalg.inv(a) @ b
    return float(np.linalg.norm(d[:3, 3])), float(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))


def _track(rgbd, kf: Keyframe, K, pred, T_last, p: ScanParams):
    """Register a frame to a keyframe starting from the predicted pose -> (ok, pose, fitness)."""
    rel, fit = register(rgbd, kf.rgbd, K, np.linalg.inv(kf.pose) @ pred, p)
    if rel is None or fit < p.min_fitness:
        return False, T_last, fit
    T = kf.pose @ rel
    if max(_motion(T_last, T)) > p.max_jump:
        return False, T_last, fit
    return True, T, fit


def fuse(capture: Capture, frames: list[int], params: ScanParams | None = None, mode: str = "track",
         log=print) -> Model:
    """TSDF of `frames`. mode: track (moving camera) | static (fixed camera) | poses (poses.txt)."""
    p = params or ScanParams()
    K = _K(capture.intr)
    if mode == "poses" and capture.poses is None:
        raise ValueError("mode 'poses' needs poses.txt in the capture")
    model = Model(p, capture.intr, new_grid(p))
    T = capture.poses[frames[0]].copy() if mode == "poses" else np.eye(4)
    vel = np.eye(4)  # motion of the last tracked step, camera frame
    lost = 0
    for k, i in enumerate(frames):
        rgbd = load_rgbd(capture, i, p)
        ok = True
        if mode == "poses":
            ok = bool(np.all(np.isfinite(capture.poses[i])))
            T = capture.poses[i].copy() if ok else T
        elif mode == "track" and k > 0:
            ok, Tn, _ = _track(rgbd, model.keyframes[-1], K, T @ vel, T, p)
            vel = np.linalg.inv(T) @ Tn if ok else np.eye(4)
            T = Tn
        model.poses[i], model.tracked[i] = T.copy(), ok
        if not ok:
            lost += 1
            continue
        if not model.keyframes or max(np.divide(_motion(model.keyframes[-1].pose, T),
                                                (p.keyframe_distance, p.keyframe_angle))) > 1.0:
            model.keyframes.append(Keyframe(i, T.copy(), rgbd))
        integrate(model.vbg, rgbd, K, T, p)
        model.frames.append(i)
    if lost:
        log(f"fuse: {lost} of {len(frames)} frames not tracked (left out)")
    return model


def localize(model: Model, capture: Capture, frames: list[int], T_start: np.ndarray | None = None,
             static: bool = False) -> dict:
    """Camera poses of `frames` against the model's keyframes, without fusing: {frame: (pose, tracked)}.

    static: the camera did not move (tripod): every frame gets T_start (default: the last fused pose)."""
    p = model.params
    T = model.poses[model.frames[-1]].copy() if T_start is None else np.asarray(T_start, float).copy()
    if static:
        return {i: (T.copy(), True) for i in frames}
    K = _K(model.intr)
    out, vel = {}, np.eye(4)
    for i in frames:
        pred = T @ vel
        kf = min(model.keyframes, key=lambda f: sum(np.divide(_motion(f.pose, pred),
                                                              (p.keyframe_distance, p.keyframe_angle))))
        ok, Tn, _ = _track(load_rgbd(capture, i, p), kf, K, pred, T, p)
        vel = np.linalg.inv(T) @ Tn if ok else np.eye(4)
        T = Tn
        out[i] = (T.copy(), ok)
    return out


def fuse_with_poses(capture: Capture, poses: dict, params: ScanParams) -> o3d.t.geometry.VoxelBlockGrid:
    """A separate TSDF of frames with known poses {frame: camera-to-world} (e.g. the final state)."""
    vbg = new_grid(params)
    K = _K(capture.intr)
    for i, T in poses.items():
        integrate(vbg, load_rgbd(capture, i, params), K, T, params)
    return vbg


def project(points_world: np.ndarray, T_cam: np.ndarray, intr: Intrinsics):
    """World points -> (u, v) integer pixels, depth z and an in-image mask, for camera-to-world T_cam."""
    Tw = np.linalg.inv(T_cam)
    pc = points_world @ Tw[:3, :3].T + Tw[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(intr.fx * pc[:, 0] / z + intr.cx)
        v = np.round(intr.fy * pc[:, 1] / z + intr.cy)
    ok = (z > 0.05) & (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    return np.where(ok, u, 0).astype(np.int64), np.where(ok, v, 0).astype(np.int64), z, ok
