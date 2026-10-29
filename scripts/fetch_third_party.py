"""Clone the Robros IGRIS-C assets this project uses into third_party/ at the tested commits.

    python scripts/fetch_third_party.py

They are not redistributed in this repository; see each project for its license.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    # MJCF (assets/robros_igris_c_v2) + PICO->IGRIS-C IK table (ik_configs/xrobot_to_igris_c.json), MIT
    # (the previous pin e47d4f8 is no longer in upstream history)
    "robros_GMR": ("https://github.com/robrosinc/GMR.git", "1281a40"),
    # Isaac Lab URDF (wrists/neck fixed, 23 DOF) + actuator parameters (robros.py)
    "robros_lab_public": ("https://github.com/robrosinc/robros_lab_public.git", "b72d871"),
}


def run(*cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    tp = ROOT / "third_party"
    tp.mkdir(exist_ok=True)
    for name, (url, rev) in REPOS.items():
        dst = tp / name
        if not dst.exists():
            run("git", "clone", "-c", "core.longpaths=true", "--filter=blob:none", url, str(dst))
        run("git", "fetch", "--quiet", "origin", cwd=dst)
        run("git", "checkout", "--quiet", rev, cwd=dst)
    print("done:", ", ".join(REPOS))


if __name__ == "__main__":
    main()
