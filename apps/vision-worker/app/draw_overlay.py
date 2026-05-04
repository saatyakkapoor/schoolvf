"""
Per-frame detection overlay collector.

The worker already runs vehicle YOLO -> plate YOLO -> OCR + a separate route OCR.
Each of those passes knows the geometry it found, but until now that geometry
was thrown away once the textual result was returned. The user requested
"detection boxes" — visible proof of work — so we collect every box during
the OCR pass into a per-thread bag, then drain + render it onto the
snapshot just before the JPEG encode.

Usage from a worker thread:
    overlay_reset()
    overlay_add_vehicle((x1,y1,x2,y2), conf=0.83)
    overlay_add_plate_region((x1,y1,x2,y2), text="HR55BC2973", conf=0.71, accepted=True)
    boxes = overlay_drain()                # returns list[BoxRecord], clears bag

From the main thread:
    annotated = render_overlay(frame, plate_boxes + route_boxes)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# BGR colours so they survive jpeg encoding well.
_COLOR_VEHICLE = (60, 220, 60)       # green
_COLOR_PLATE_HIT = (0, 0, 255)       # red — OCR produced an accepted plate string
_COLOR_PLATE_REGION = (0, 165, 255)  # orange — detector only; OCR/validator did not accept a read
_COLOR_ROUTE = (0, 230, 230)         # yellow — route placard
_COLOR_BUMPER = (255, 180, 80)       # blue — bumper crop region


@dataclass
class BoxRecord:
    bbox: tuple[int, int, int, int]
    color: tuple[int, int, int]
    label: Optional[str]
    conf: Optional[float]
    thickness: int = 2


_local = threading.local()


def _bag() -> list[BoxRecord]:
    if not hasattr(_local, "boxes"):
        _local.boxes = []
    return _local.boxes


def overlay_reset() -> None:
    """Clear the per-thread overlay bag at the start of a frame cycle."""
    _local.boxes = []


def overlay_drain() -> list[BoxRecord]:
    """Return the collected boxes and clear the bag in one shot."""
    boxes = _bag()
    _local.boxes = []
    return boxes


def overlay_add(
    bbox: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    label: str | None = None,
    conf: float | None = None,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return
    _bag().append(BoxRecord((x1, y1, x2, y2), color, label, conf, thickness))


def overlay_add_vehicle(bbox, conf: float | None = None, label: str | None = "veh") -> None:
    overlay_add(bbox, color=_COLOR_VEHICLE, label=label, conf=conf, thickness=2)


def overlay_add_bumper(bbox) -> None:
    """Faint blue rectangle around the bumper crop (debug hint)."""
    overlay_add(bbox, color=_COLOR_BUMPER, label=None, conf=None, thickness=1)


def overlay_add_plate_region(bbox, *, text: str | None = None, conf: float | None = None,
                             accepted: bool = False) -> None:
    color = _COLOR_PLATE_HIT if accepted else _COLOR_PLATE_REGION
    overlay_add(bbox, color=color, label=text, conf=conf, thickness=2 if accepted else 1)


def overlay_add_route_box(bbox, *, text: str | None = None, conf: float | None = None) -> None:
    overlay_add(bbox, color=_COLOR_ROUTE, label=text, conf=conf, thickness=2)


def make_route_box(bbox, *, text: str | None = None, conf: float | None = None) -> BoxRecord:
    """Construct a yellow placard `BoxRecord` for inclusion in a render list.

    Use this from the main loop where the placard bbox came back as a
    plain tuple from the route-OCR future (different thread).
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return BoxRecord((x1, y1, x2, y2), _COLOR_ROUTE, text, conf, 2)


def render_overlay(frame: np.ndarray, boxes: list[BoxRecord]) -> np.ndarray:
    """Return a copy of `frame` with every box drawn on it.

    Boxes are layered: bumper first, then vehicle, then plate, then route,
    so the most informative outline ends up on top. If `boxes` is empty
    we still return a copy (cheap, keeps the contract uniform).
    """
    if not boxes:
        return frame  # let caller skip the copy when there's nothing to draw

    out = frame.copy()
    fh, fw = out.shape[:2]

    importance = {
        _COLOR_BUMPER: 0,
        _COLOR_VEHICLE: 1,
        _COLOR_PLATE_REGION: 2,
        _COLOR_PLATE_HIT: 3,
        _COLOR_ROUTE: 3,
    }
    boxes = sorted(boxes, key=lambda b: importance.get(b.color, 0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(0.9, fw / 1600))
    text_thickness = 1 if fw < 1200 else 2

    for box in boxes:
        x1, y1, x2, y2 = box.bbox
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(fw - 1, x2), min(fh - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        cv2.rectangle(out, (x1c, y1c), (x2c, y2c), box.color, box.thickness)

        parts: list[str] = []
        if box.label:
            parts.append(box.label)
        if box.conf is not None:
            parts.append(f"{box.conf:.2f}")
        caption = " ".join(parts)
        if not caption:
            continue
        (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, text_thickness)
        ty1 = y1c - th - baseline - 4
        place_above = ty1 >= 0
        if place_above:
            tl = (x1c, ty1)
            br = (x1c + tw + 8, y1c)
            text_org = (x1c + 4, y1c - 4)
        else:
            tl = (x1c, y2c)
            br = (x1c + tw + 8, y2c + th + baseline + 4)
            text_org = (x1c + 4, y2c + th + 2)
        cv2.rectangle(out, tl, br, box.color, -1)
        cv2.putText(out, caption, text_org, font, font_scale, (0, 0, 0),
                    text_thickness, cv2.LINE_AA)

    return out
