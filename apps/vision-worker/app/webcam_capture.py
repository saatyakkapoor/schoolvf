"""Cross-platform webcam capture (Windows DirectShow, macOS AVFoundation, Linux V4L2)."""

from __future__ import annotations

import logging
import sys

import cv2

log = logging.getLogger(__name__)

# Typical laptop HD; device may negotiate a nearby mode.
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720


def open_webcam(
    device_index: int,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> cv2.VideoCapture:
    """
    Open a local camera with a backend that works well on Windows and macOS, request HD, then fall back.
    """
    if sys.platform == "win32":
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    elif sys.platform == "darwin":
        cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(device_index)

    if not cap.isOpened() and sys.platform in ("win32", "darwin"):
        cap.release()
        cap = cv2.VideoCapture(device_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        log.info(
            "Webcam index %s: requested %sx%s, actual %sx%s",
            device_index,
            width,
            height,
            aw,
            ah,
        )
    return cap


def probe_local_cameras(max_index: int = 8) -> list[dict[str, int | str | bool]]:
    """
    Try OpenCV indices 0..max_index-1 on this machine (for CLI / troubleshooting).
    Returns entries with index, opened, and negotiated width/height when opened.
    """
    out: list[dict[str, int | str | bool]] = []
    for i in range(max_index):
        cap = open_webcam(i)
        try:
            ok = cap.isOpened()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if ok else 0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if ok else 0
            out.append(
                {
                    "index": i,
                    "opened": ok,
                    "width": w,
                    "height": h,
                    "platform": sys.platform,
                }
            )
        finally:
            cap.release()
    return out
