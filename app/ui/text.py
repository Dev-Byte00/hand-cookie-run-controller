"""Small text-rendering helpers shared by every HUD screen."""
from __future__ import annotations

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.ui.theme import FONT


def wrap_text(text: str, max_width: int, font_scale: float, thickness: int = 1) -> list[str]:
    """Greedy word-wrap so instruction text fits a fixed pixel width."""
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        (tw, _), _ = cv2.getTextSize(trial, FONT, font_scale, thickness)
        if tw > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(
    frame: np.ndarray, text: str, x: int, y: int, max_width: int,
    font_scale: float, colour: tuple[int, int, int], thickness: int = 1, line_h: int = 18,
) -> int:
    """Draw word-wrapped text starting at (x, y); returns the y just below it."""
    for line in wrap_text(text, max_width, font_scale, thickness):
        cv2.putText(frame, line, (x, y), FONT, font_scale, colour, thickness)
        y += line_h
    return y
