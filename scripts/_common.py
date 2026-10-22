"""Shared CLI helpers (scripts can be run from the project root or anywhere)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def episode_paths(args):
    """Expand files / session dirs into episode .h5 paths."""
    from igris_pipeline.episode import list_episodes

    out = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            out += list_episodes(p / "raw" if (p / "raw").is_dir() else p)
        else:
            out.append(p)
    return out
