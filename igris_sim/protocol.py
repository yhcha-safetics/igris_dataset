"""UDP JSON protocol between the PICO app and the simulation server (spec: docs/sim_protocol.md).

One UDP port carries everything; every datagram is one UTF-8 JSON object with "v" (protocol
version) and "type":

  app -> server   "tracking"  headset / controller / body / tracker poses + buttons, every frame
                  "cmd"       hello, list_scenes, load_scene, set_mode, reset, recalibrate, record, marker
  server -> app   "state"     robot qpos, object poses, grasps, rig transform, status (rate_hz)
                  "scene"     scene description (reply to hello / load_scene)
                  "scenes", "ack", "error"

Poses in "tracking" use the XRoboToolkit native convention (right-handed, Y up, Z back, metres,
[x, y, z, qx, qy, qz, qw]) so the server handles the PICO app and the XRoboToolkit PC service the
same way. Poses in "state" / "scene" use the MuJoCo world (Z up, +X forward, quaternions wxyz).
The Unity conversions are native_from_unity() and unity_from_mujoco_*() below.
"""
from __future__ import annotations

import functools
import http.server
import io
import json
import socket
import threading
import time
import urllib.parse
from pathlib import Path

import numpy as np

from igris_pipeline.pico.xrt_source import XRFrame

from .scene import SCENES_DIR

VERSION = 1
MAX_DATAGRAM = 65507
ROBOT_ASSETS = "_robot/"  # asset server folder of the robot description (igris_sim.robot_export)
BUTTONS = ("A", "B", "X", "Y", "left_trigger", "right_trigger", "left_grip", "right_grip", "left_menu", "right_menu")


def encode(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode(data: bytes) -> dict:
    msg = json.loads(data.decode("utf-8"))
    if not isinstance(msg, dict) or "type" not in msg:
        raise ValueError("message must be a JSON object with a 'type'")
    if int(msg.get("v", VERSION)) != VERSION:
        raise ValueError(f"protocol version {msg.get('v')} != {VERSION}")
    return msg


# -- tracking <-> XRFrame ------------------------------------------------------------------
def _pose7(x, name) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.shape != (7,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: expected 7 finite numbers [x, y, z, qx, qy, qz, qw]")
    return a


def frame_from_tracking(msg: dict) -> XRFrame:
    ctrl = msg.get("controllers", {})
    fr = XRFrame(t_ns=int(msg["t_ns"]), headset=_pose7(msg["head"], "head"),
                 left_ctrl=_pose7(ctrl.get("left", [0] * 7), "controllers.left"),
                 right_ctrl=_pose7(ctrl.get("right", [0] * 7), "controllers.right"))
    b = msg.get("buttons", {})
    fr.buttons = {k: (float(b[k]) if k.endswith(("trigger", "grip")) else bool(b[k])) for k in BUTTONS if k in b}
    ax = msg.get("axes", {})
    fr.axes = {s: np.asarray(ax[s], dtype=np.float64).reshape(2) for s in ("left", "right") if s in ax}
    if msg.get("body") is not None:
        body = np.asarray(msg["body"], dtype=np.float64)
        if body.shape != (24, 7) or not np.all(np.isfinite(body)):
            raise ValueError("body: expected 24 x 7 finite numbers")
        fr.body = body
        ts = msg.get("body_ts")
        fr.body_joint_ts = np.asarray(ts, dtype=np.int64) if ts is not None else np.full(24, fr.t_ns // 1_000_000)
    trk = msg.get("trackers") or []
    if trk:
        fr.trackers = np.stack([_pose7(t["pose"], "trackers.pose") for t in trk])
        fr.tracker_serials = [str(t.get("sn", k)) for k, t in enumerate(trk)]
    hands = msg.get("hands") or {}
    for s in ("left", "right"):
        h = hands.get(s)
        if h is not None:
            setattr(fr, f"{s}_hand", np.asarray(h["joints"], dtype=np.float64).reshape(26, 7))
            setattr(fr, f"{s}_hand_active", int(h.get("active", 1)))
    return fr


def tracking_from_frame(fr: XRFrame) -> dict:
    """The tracking message for a frame (used by tests and relay tools)."""
    msg = {"v": VERSION, "type": "tracking", "t_ns": int(fr.t_ns), "head": _l(fr.headset),
           "controllers": {"left": _l(fr.left_ctrl), "right": _l(fr.right_ctrl)},
           "buttons": {k: (float(v) if isinstance(v, (float, np.floating)) else bool(v)) for k, v in fr.buttons.items()},
           "axes": {s: _l(v) for s, v in fr.axes.items()},
           "body": None if fr.body is None else _l(fr.body),
           "trackers": [] if fr.trackers is None else [{"sn": sn, "pose": _l(p)} for sn, p in
                                                       zip(fr.tracker_serials, fr.trackers)]}
    return msg


def _l(a):
    return np.round(np.asarray(a, dtype=np.float64), 6).tolist()


# -- coordinate conversions (reference for the Unity app) ------------------------------------
def native_from_unity(pos_xyz, quat_xyzw) -> np.ndarray:
    """Unity pose (left-handed, Y up, Z forward) -> native pose7 (right-handed, Y up, Z back)."""
    x, y, z = pos_xyz
    qx, qy, qz, qw = quat_xyzw
    return np.array([x, y, -z, -qx, -qy, qz, qw], dtype=np.float64)


def unity_from_mujoco_pos(p) -> np.ndarray:
    """MuJoCo world (Z up, X forward, Y left) -> Unity (Y up, Z forward, X right)."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([-p[..., 1], p[..., 2], p[..., 0]], axis=-1)


def unity_from_mujoco_quat(q_wxyz) -> np.ndarray:
    """MuJoCo quaternion wxyz -> Unity Quaternion (x, y, z, w)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.stack([q[..., 2], -q[..., 3], -q[..., 1], q[..., 0]], axis=-1)


# -- UDP hub -----------------------------------------------------------------------------
class Hub:
    """Non-blocking UDP endpoint: receives tracking / commands, sends state to active clients."""

    def __init__(self, port: int, host: str = "0.0.0.0", client_timeout: float = 3.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, "SIO_UDP_CONNRESET"):  # Windows: ignore ICMP "port unreachable" from gone clients
            self.sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.timeout = float(client_timeout)
        self.clients: dict[tuple, float] = {}
        self.errors = 0
        self.last_error = ""

    def poll(self):
        """-> (frames [(XRFrame, addr)], commands [(msg, addr)]) received since the last call."""
        frames, cmds = [], []
        while True:
            try:
                data, addr = self.sock.recvfrom(MAX_DATAGRAM)
            except (BlockingIOError, InterruptedError):
                break
            except ConnectionResetError:
                continue
            self.clients[addr] = time.time()
            try:
                msg = decode(data)
                if msg["type"] == "tracking":
                    frames.append((frame_from_tracking(msg), addr))
                elif msg["type"] == "cmd":
                    cmds.append((msg, addr))
            except (ValueError, KeyError, TypeError, AttributeError) as e:  # e.g. "controllers": null
                self.errors += 1
                self.last_error = f"{addr}: {e}"
        return frames, cmds

    def _encode(self, msg: dict) -> bytes | None:
        """Datagram for msg, or None (counted in errors / last_error) if it cannot be sent."""
        try:
            data = encode(msg)
        except ValueError as e:  # NaN / inf
            data, err = None, f"cannot encode {msg.get('type')!r}: {e}"
        else:
            err = (f"{msg.get('type')!r} message too large for one datagram ({len(data)} bytes)"
                   if len(data) > MAX_DATAGRAM else None)
        if err is not None:
            self.errors += 1
            self.last_error = err
            return None
        return data

    def send(self, msg: dict | bytes, addr):
        data = msg if isinstance(msg, bytes) else self._encode(msg)
        if data is None:
            return
        try:
            self.sock.sendto(data, addr)
        except OSError:
            pass

    def broadcast(self, msg: dict):
        now = time.time()
        for addr, seen in list(self.clients.items()):
            if now - seen > self.timeout:
                del self.clients[addr]
        if not self.clients:
            return
        data = self._encode(msg)  # once for all clients
        if data is not None:
            for addr in self.clients:
                self.send(data, addr)

    def close(self):
        self.sock.close()


class AssetServer:
    """Read-only HTTP server for scene files (meshes, textures) under `folder`, in a daemon thread.

    Generated files (the robot description, ROBOT_ASSETS) are served from memory with add_files().
    """

    def __init__(self, port: int, folder: Path = SCENES_DIR, host: str = "0.0.0.0"):
        handler = functools.partial(_QuietHandler, directory=str(folder))
        self.httpd = http.server.ThreadingHTTPServer((host, port), handler)
        self.httpd.files = {}
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def add_files(self, files: dict[str, bytes], prefix: str = ""):
        self.httpd.files = {**self.httpd.files, **{prefix + k: v for k, v in files.items()}}

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_head(self):
        path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).lstrip("/")
        data = self.server.files.get(path)
        if data is None:
            return super().send_head()
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        return io.BytesIO(data)

    def do_PUT(self):  # read-only
        self.send_error(405)

    do_POST = do_DELETE = do_PUT


def local_ip() -> str:
    """LAN address of this PC (the one a headset on the same network would use)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
