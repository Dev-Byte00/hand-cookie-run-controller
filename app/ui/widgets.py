"""Small reusable drawing widgets shared across HUD screens."""
from __future__ import annotations

from collections.abc import Mapping

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.ui.theme import C_CYAN, C_GREEN, C_GREY


def draw_bar(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    value: float, on_ratio: float, off_ratio: float, up: bool,
) -> None:
    """A small meter showing the raw value against its on/off thresholds —
    at a glance this shows *why* a finger is or isn't counted as raised,
    instead of making you read and compare two raw numbers."""
    lo = off_ratio - 0.15
    hi = on_ratio + 0.30
    span = max(hi - lo, 0.3)
    frac = max(0.0, min(1.0, (value - lo) / span))
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_GREY, 1)
    fill_w = int(w * frac)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), C_GREEN if up else C_GREY, -1)
    tick_x = x + int(w * max(0.0, min(1.0, (on_ratio - lo) / span)))
    cv2.line(frame, (tick_x, y), (tick_x, y + h), C_CYAN, 2)


def role_of(finger: int, chosen: Mapping[str, frozenset[int]]) -> str:
    """Which already-assigned action (if any) already uses ``finger``."""
    for role, fingers in chosen.items():
        if finger in fingers:
            return role
    return ""
