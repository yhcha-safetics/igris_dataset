"""IGRIS-C MuJoCo model wrapper (Robros MJCF, 31 actuated DOF + free base)."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = ROOT / "third_party" / "robros_GMR" / "assets" / "robros_igris_c_v2" / "igris_c_v2.xml"

# Joints that the Robros Isaac Lab asset keeps actuated (wrists and neck are fixed there).
ISAAC_23_JOINTS = [
    "Joint_Waist_Yaw", "Joint_Waist_Roll", "Joint_Waist_Pitch",
    "Joint_Hip_Pitch_Left", "Joint_Hip_Roll_Left", "Joint_Hip_Yaw_Left", "Joint_Knee_Pitch_Left",
    "Joint_Ankle_Pitch_Left", "Joint_Ankle_Roll_Left",
    "Joint_Hip_Pitch_Right", "Joint_Hip_Roll_Right", "Joint_Hip_Yaw_Right", "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Right", "Joint_Ankle_Roll_Right",
    "Joint_Shoulder_Pitch_Left", "Joint_Shoulder_Roll_Left", "Joint_Shoulder_Yaw_Left", "Joint_Elbow_Pitch_Left",
    "Joint_Shoulder_Pitch_Right", "Joint_Shoulder_Roll_Right", "Joint_Shoulder_Yaw_Right", "Joint_Elbow_Pitch_Right",
]

# Default standing pose used by Robros' Isaac Lab locomotion config.
DEFAULT_POSE = {
    "Joint_Hip_Pitch_Left": -0.2, "Joint_Hip_Pitch_Right": -0.2,
    "Joint_Knee_Pitch_Left": 0.3, "Joint_Knee_Pitch_Right": 0.3,
    "Joint_Ankle_Pitch_Left": -0.1, "Joint_Ankle_Pitch_Right": -0.1,
    "Joint_Shoulder_Pitch_Left": 0.13, "Joint_Shoulder_Pitch_Right": 0.13,
    "Joint_Shoulder_Roll_Left": 0.13, "Joint_Shoulder_Roll_Right": -0.13,
    "Joint_Elbow_Pitch_Left": -0.3, "Joint_Elbow_Pitch_Right": -0.3,
}


def urdf_joint_limits(path: str | Path) -> dict:
    """Revolute joint limits {name: (lower, upper)} from a URDF."""
    import xml.etree.ElementTree as ET

    out = {}
    for j in ET.parse(str(path)).getroot().iter("joint"):
        lim = j.find("limit")
        if j.get("type") == "revolute" and lim is not None and lim.get("lower") not in (None, "", "-"):
            try:
                out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
            except ValueError:
                pass
    return out


class IgrisModel:
    """MJCF from Robros' GMR fork, made consistent with Robros' Isaac Lab URDF by default.

    feet="urdf"        foot boxes replaced by the URDF ones (sole 9 mm higher than the MJCF's)
                       and ankle-roll mesh colliders disabled, as in Isaac and the sim2sim runner
    limits="intersect" joint ranges = MJCF range intersected with the Isaac URDF limits
    """

    def __init__(self, mjcf: str | Path = DEFAULT_MJCF, feet: str = "urdf", limits: str = "intersect"):
        self.path = Path(mjcf)
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        from ..isaaclab_ext import constants as IC  # pure python, no Isaac import

        if feet == "urdf":
            for side in ("left", "right"):
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_box_{side}")
                m.geom_size[g] = np.array(IC.URDF_FOOT_BOX_HALF_SIZE)
                m.geom_pos[g] = np.array(IC.URDF_FOOT_BOX_POS)
                m.geom_quat[g] = np.array([1.0, 0.0, 0.0, 0.0])
            for name in ("Link_Ankle_Roll_Left_collision", "Link_Ankle_Roll_Right_collision"):
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                if g >= 0:
                    m.geom_contype[g] = 0
                    m.geom_conaffinity[g] = 0
        self.urdf_limits = urdf_joint_limits(IC.IGRIS_C_URDF_PATH) if limits == "intersect" else {}
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
                            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        self.nq, self.nv, self.ndof = m.nq, m.nv, len(self.joint_names)
        for n, (lo, hi) in self.urdf_limits.items():
            j = self.jid(n)
            if j >= 0:
                m.jnt_range[j] = [max(m.jnt_range[j][0], lo), min(m.jnt_range[j][1], hi)]
        self.jnt_range = np.array([m.jnt_range[self.jid(n)] for n in self.joint_names])
        self.default_qpos = self.make_qpos(np.array([0, 0, 0.96]), np.array([1.0, 0, 0, 0]),
                                           np.array([DEFAULT_POSE.get(n, 0.0) for n in self.joint_names]))
        self.foot_geoms = [self.gid("foot_box_left"), self.gid("foot_box_right")]
        # height of the ankle-roll body origin above the floor when the sole is flat on it
        d = self.forward(self.make_qpos(np.array([0, 0, 1.0]), np.array([1.0, 0, 0, 0]), np.zeros(self.ndof)))
        self.ankle_flat_height = float(np.mean([d.xpos[self.bid(f"Link_Ankle_Roll_{s}")][2] for s in ("Left", "Right")])
                                       - self.sole_min_z(d).min())

    def reference_range(self, margin: float = 0.0, soft_factor: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Joint range (lower, upper) in joint_names order that exported references are clipped to.

        Hard range (MJCF ∩ URDF) shrunk by `margin`, and with `soft_factor` additionally Isaac's soft
        limits (mid +- 0.5 * URDF range * factor) for the joints Isaac simulates: a reference outside
        them is penalised every step and clipped at reset, so the policy could never match it.
        """
        lo = self.jnt_range[:, 0] + margin
        hi = self.jnt_range[:, 1] - margin
        if soft_factor:
            for i, n in enumerate(self.joint_names):
                if n in self.urdf_limits:
                    ul, uh = self.urdf_limits[n]
                    mid, half = 0.5 * (ul + uh), 0.5 * (uh - ul) * float(soft_factor)
                    lo[i], hi[i] = max(lo[i], mid - half), min(hi[i], mid + half)
        return lo, hi

    # -- ids -------------------------------------------------------------------------
    def jid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def bid(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise KeyError(name)
        return i

    def gid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    # -- kinematics ------------------------------------------------------------------
    def make_qpos(self, root_pos, root_quat, dof):
        return np.concatenate([root_pos, root_quat, dof])

    def forward(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return self.data

    def fk_sequence(self, qpos_seq, bodies=None):
        """(T,nq) -> body pos (T,B,3), quat wxyz (T,B,4) for `bodies` (names) or all."""
        ids = list(range(self.model.nbody)) if bodies is None else [self.bid(b) for b in bodies]
        T = qpos_seq.shape[0]
        P = np.zeros((T, len(ids), 3))
        Q = np.zeros((T, len(ids), 4))
        for t in range(T):
            d = self.forward(qpos_seq[t])
            P[t] = d.xpos[ids]
            Q[t] = d.xquat[ids]
        return P, Q

    def sole_min_z(self, data=None):
        """Lowest z of each foot sole box (left, right)."""
        d = data or self.data
        out = []
        for g in self.foot_geoms:
            half = self.model.geom_size[g]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
            world = d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T
            out.append(world[:, 2].min())
        return np.array(out)

    def segment_lengths(self, qpos=None) -> dict:
        """Characteristic robot segment lengths at the zero pose (for kinematic scaling)."""
        q = self.make_qpos(np.array([0, 0, 1.0]), np.array([1.0, 0, 0, 0]), np.zeros(self.ndof)) if qpos is None else qpos
        d = self.forward(q)

        def p(n):
            return d.xpos[self.bid(n)].copy()

        out = {}
        for s in ("Left", "Right"):
            hip, knee, ank = p(f"Link_Hip_Pitch_{s}"), p(f"Link_Knee_Pitch_{s}"), p(f"Link_Ankle_Roll_{s}")
            out[f"thigh_{s}"] = float(np.linalg.norm(knee - hip))
            out[f"shank_{s}"] = float(np.linalg.norm(ank - knee))
            sh, el, wr = p(f"Link_Shoulder_Pitch_{s}"), p(f"Link_Elbow_Pitch_{s}"), p(f"Link_Wrist_Pitch_{s}")
            out[f"upper_arm_{s}"] = float(np.linalg.norm(el - sh))
            out[f"forearm_{s}"] = float(np.linalg.norm(wr - el))
        pelvis, neck = p("Link_Waist_Pitch"), p("Link_Neck_Pitch")
        out["trunk"] = float(np.linalg.norm(neck - pelvis))
        out["hip_width"] = float(np.linalg.norm(p("Link_Hip_Pitch_Left") - p("Link_Hip_Pitch_Right")))
        out["shoulder_width"] = float(np.linalg.norm(p("Link_Shoulder_Pitch_Left") - p("Link_Shoulder_Pitch_Right")))
        out["pelvis_height"] = float(pelvis[2] - self.sole_min_z(d).min())
        out["height_neck"] = float(neck[2] - self.sole_min_z(d).min())
        return out
