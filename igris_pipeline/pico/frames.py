"""PICO / XRoboToolkit coordinate conventions -> pipeline convention.

XRoboToolkit streams every pose (headset, controllers, body joints, motion trackers) as
``[x, y, z, qx, qy, qz, qw]`` in the PICO runtime's native frame: right-handed, Y up,
-Z = the direction the user faced when the app started, metres.

Pipeline convention (see utils/rotations.py): right-handed, Z up, +X forward, +Y left,
quaternions wxyz. We apply a *change of basis* to both the world and each joint's local
frame, so that the PICO body skeleton in T-pose facing forward has identity orientation
for every joint (PICO uses Unity-humanoid joint frames that are world-aligned in T-pose).

    p_pipe = C p_native
    q_pipe = q_C * q_native * q_C^-1

GMR's xrobot IK configs instead use q_gmr = q_M * q_native (world-only rotation with
M = [[1,0,0],[0,0,-1],[0,1,0]]); `gmr_offset_to_pipeline()` converts their rotation
offsets so that target_link_quat = q_pipe * offset_pipe.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..utils.rotations import quat_conj, quat_mul, quat_normalize, rot_to_quat, xyzw_to_wxyz

# native (X right, Y up, Z back) -> pipeline (X fwd, Y left, Z up)
C_NATIVE_TO_PIPE = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
Q_C = rot_to_quat(R.from_matrix(C_NATIVE_TO_PIPE))
# GMR's world rotation for xrobot data, and the extra yaw that makes its +Y-forward world +X-forward
M_GMR = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def pose7_to_pipe(pose7: np.ndarray, zero_is_invalid: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(...,7) native [x,y,z,qx,qy,qz,qw] -> pos (...,3), quat wxyz (...,4), valid (...,)."""
    a = np.asarray(pose7, dtype=np.float64)
    pos = a[..., :3] @ C_NATIVE_TO_PIPE.T
    q = xyzw_to_wxyz(a[..., 3:7])
    norm = np.linalg.norm(q, axis=-1)
    valid = norm > 0.5 if zero_is_invalid else np.ones(a.shape[:-1], dtype=bool)
    q = np.where(valid[..., None], q, np.array([1.0, 0.0, 0.0, 0.0]))
    q = quat_normalize(q)
    qc = np.broadcast_to(Q_C, q.shape)
    q = quat_mul(quat_mul(qc, q), quat_conj(qc))
    return pos, q, valid


def pipe_to_pose7(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of pose7_to_pipe (used by the simulator to emit native-frame data)."""
    p = np.asarray(pos) @ C_NATIVE_TO_PIPE  # C is orthonormal: C^-1 = C^T
    qc = np.broadcast_to(Q_C, np.asarray(quat_wxyz).shape)
    q = quat_mul(quat_mul(quat_conj(qc), quat_wxyz), qc)
    return np.concatenate([p, q[..., 1:4], q[..., 0:1]], axis=-1)


def gmr_offset_to_pipeline(offset_wxyz) -> np.ndarray:
    """Convert a GMR xrobot rot-offset (wxyz) into this pipeline's convention.

    GMR target = q_M * q_native * off.  With q_native = q_C^-1 * q_pipe * q_C and the
    extra world yaw Rz(-90) (GMR humans face +Y, ours face +X):
        target = Rz(-90) * q_M * q_C^-1 * q_pipe * q_C * off = q_pipe * (q_C * off)
    because Rz(-90) * M == C.
    """
    off = quat_normalize(np.asarray(offset_wxyz, dtype=np.float64))
    return quat_normalize(quat_mul(Q_C, off))
