"""Extract only the needed files from DMD archives (face/body video + JSON), skipping the rest.

    python scripts/prepare_dmd.py --bundle drowsiness --videos face
    python scripts/prepare_dmd.py --bundle distractions --videos face body

Output: data/interim/dmd/<bundle>/<group>_<subject>_<session>_<timestamp>/{rgb_face.mp4, ..., ann.json}
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from dms.utils.config import PROJECT_ROOT

RAW = PROJECT_ROOT / "data/raw/cabin/dmd"
OUT = PROJECT_ROOT / "data/interim/dmd"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, choices=["drowsiness", "distractions"])
    ap.add_argument("--videos", nargs="+", default=["face"], choices=["face", "body", "hands", "mosaic"])
    args = ap.parse_args()

    wanted = tuple(f"_rgb_{v}.mp4" for v in args.videos) + tuple(f"_rgb_{v}.avi" for v in args.videos)
    archives = sorted((RAW / args.bundle).glob("*.tar.gz"))
    if not archives:
        raise SystemExit(f"No archives found in {RAW / args.bundle}")

    for arc in archives:
        with tarfile.open(arc) as tar:
            for m in tar.getmembers():
                name = Path(m.name).name
                if not m.isfile() or not (name.endswith(wanted) or name.endswith(".json")):
                    continue
                recording, _, suffix = name.partition("_rgb_")
                dest_dir = OUT / args.bundle / recording.replace(";", "-").replace("+", "p")
                dest = dest_dir / ("ann.json" if suffix.startswith("ann") else f"rgb_{suffix}")
                if dest.exists() and dest.stat().st_size == m.size:
                    continue
                dest_dir.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(dest, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                print(f"{arc.name}: {dest.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
