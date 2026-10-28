"""Talk to a running sim_server.py over its UDP port (stands in for the PICO app while testing).

    python scripts/sim_ctl.py cmd list_scenes
    python scripts/sim_ctl.py cmd load_scene scene=room
    python scripts/sim_ctl.py cmd set_mode mode=harness
    python scripts/sim_ctl.py cmd record on=true
    python scripts/sim_ctl.py watch                                 # print the state stream
    python scripts/sim_ctl.py relay --source sim:body_2trk          # send tracking like the app would
    python scripts/sim_ctl.py relay --source replay:data/s01/raw/ep_0000.h5 --host 192.168.0.10
"""
import argparse
import json
import socket
import time

import _common  # noqa: F401

from igris_sim.protocol import VERSION, decode, encode, tracking_from_frame
from igris_sim.sources import PacedSource, make_source

ap = argparse.ArgumentParser()
ap.add_argument("action", choices=["cmd", "watch", "relay"])
ap.add_argument("args", nargs="*", help="cmd: name [key=value ...]")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=9870)
ap.add_argument("--source", default="sim:body_2trk", help="relay: sim:<profile> | replay:<h5> | xrt")
ap.add_argument("--seconds", type=float, default=None, help="watch / relay: stop after this long")
a = ap.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
if hasattr(socket, "SIO_UDP_CONNRESET"):
    sock.ioctl(socket.SIO_UDP_CONNRESET, False)
sock.settimeout(0.2)
server = (a.host, a.port)


def value(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def receive(until_type=None, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            msg = decode(sock.recvfrom(65507)[0])
        except (socket.timeout, ConnectionResetError):
            continue
        if until_type is None or msg["type"] in until_type:
            return msg
    return None


def latest_state(timeout=1.0):
    """Newest state message: drain everything queued (states arrive at rate_hz), keep the last."""
    st = receive({"state"}, timeout=timeout)
    sock.setblocking(False)
    try:
        while True:
            try:
                msg = decode(sock.recvfrom(65507)[0])
            except (BlockingIOError, ConnectionResetError):
                break
            if msg["type"] == "state":
                st = msg
    finally:
        sock.settimeout(0.2)
    return st


def short_state(st):
    return (f"[{st['status']}] scene={st['scene']} profile={st['profile']} mode={st['mode']} "
            f"root=({st['robot']['qpos'][0]:.2f},{st['robot']['qpos'][1]:.2f},{st['robot']['qpos'][2]:.2f}) "
            f"grasp={st['grasp']} rec={st['recording']} timing={st['timing_ms']}"
            + (f" | {st['message']}" if st["message"] else ""))


if a.action == "cmd":
    if not a.args:
        raise SystemExit("cmd needs a command name")
    msg = {"v": VERSION, "type": "cmd", "cmd": a.args[0]}
    for kv in a.args[1:]:
        k, _, v = kv.partition("=")
        msg[k] = value(v)
    sock.sendto(encode(msg), server)
    reply = receive({"ack", "error", "scene", "scenes"})
    if reply is None:
        raise SystemExit(f"no reply from {server} (is sim_server.py running?)")
    if reply["type"] == "scene":
        reply = {k: reply[k] for k in ("type", "name", "dynamic", "assets", "rate_hz")}
    print(json.dumps(reply, ensure_ascii=False, indent=1))

elif a.action == "watch":
    sock.sendto(encode({"v": VERSION, "type": "cmd", "cmd": "hello"}), server)
    t0, last = time.time(), 0.0
    while a.seconds is None or time.time() - t0 < a.seconds:
        if time.time() - last > 1.0:  # keep-alive: the server drops silent clients
            sock.sendto(encode({"v": VERSION, "type": "cmd", "cmd": "hello"}), server)
            last = time.time()
        st = latest_state(timeout=1.0)
        if st is not None:
            print(short_state(st), flush=True)
            time.sleep(0.5)

else:  # relay
    src = make_source(a.source)
    if not a.source.startswith("xrt"):
        src = PacedSource(src)
    t0, last, n = time.time(), 0.0, 0
    try:
        while a.seconds is None or time.time() - t0 < a.seconds:
            try:
                fr = src.poll()
            except StopIteration:
                break
            if fr is None:
                time.sleep(0.002)
                continue
            sock.sendto(encode(tracking_from_frame(fr)), server)
            n += 1
            if time.time() - last > 2.0:
                last = time.time()
                st = latest_state(timeout=0.3)
                print(f"sent {n} frames | " + (short_state(st) if st else "no state yet"), flush=True)
    finally:
        src.close()
