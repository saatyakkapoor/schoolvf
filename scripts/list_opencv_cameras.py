#!/usr/bin/env python3
"""
List local cameras as OpenCV sees them (indices 0, 1, …). Run on the same machine as the vision worker.

Usage:
  python scripts/list_opencv_cameras.py

Windows / macOS / Linux: uses the same backends as apps.vision_worker.app.webcam_capture.open_webcam.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_probe_local_cameras():
    """Load webcam_capture without PYTHONPATH (repo folder is apps/vision-worker, not vision_worker)."""
    wc = _ROOT / "apps" / "vision-worker" / "app" / "webcam_capture.py"
    spec = importlib.util.spec_from_file_location("webcam_capture_standalone", wc)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {wc}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.probe_local_cameras


def main() -> None:
    ap = argparse.ArgumentParser(description="List OpenCV camera indices on this machine.")
    ap.add_argument(
        "--max",
        type=int,
        default=10,
        metavar="N",
        help="Try indices 0..N-1 (default: 10)",
    )
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout.")
    args = ap.parse_args()

    probe_local_cameras = _load_probe_local_cameras()
    rows = probe_local_cameras(max_index=max(1, args.max))
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    for r in rows:
        if r.get("opened"):
            print(
                f"index={r['index']}\t{r['width']}x{r['height']}\t"
                f"platform={r['platform']}\tOK",
            )
        else:
            print(f"index={r['index']}\t—\tplatform={r['platform']}\t(not opened)")
    opened = [r for r in rows if r.get("opened")]
    print(f"\n{len(opened)} camera(s) reachable.", file=sys.stderr)


if __name__ == "__main__":
    main()
