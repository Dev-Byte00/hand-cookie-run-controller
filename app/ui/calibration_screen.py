"""Calibration-wizard screen rendering."""
from __future__ import annotations

from collections.abc import Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.domain.models import FINGER_NAMES
from app.ui.text import draw_wrapped
from app.ui.theme import (
    CANVAS_H,
    C_CYAN,
    C_GREEN,
    C_GREY,
    C_WHITE,
    FONT,
    FS_L,
    FS_M,
    FS_S,
    SIDEBAR_MARGIN,
    SIDEBAR_TEXT_W,
    SIDEBAR_X,
    UI_PADDING,
)


def draw_calibration(
    canvas: np.ndarray,
    step_name: str,
    step_hint: str,
    active: frozenset[int],
    ext: Sequence[float],
    step_index: int,
    step_count: int,
    fps: float,
) -> None:
    """Render one frame of the guided calibration wizard."""
    x0 = SIDEBAR_X + SIDEBAR_MARGIN
    y = UI_PADDING + 34

    cv2.putText(canvas, "CALIBRATION", (x0, y), FONT, FS_L * 0.8, C_WHITE, 2)
    y += 34
    cv2.putText(canvas, f"Step {step_index + 1}/{step_count}: {step_name}",
                (x0, y), FONT, FS_M, C_CYAN, 2)
    y += 26
    y = draw_wrapped(canvas, step_hint, x0, y, SIDEBAR_TEXT_W, FS_S, C_WHITE, line_h=18) + 16

    targets = active if active else set(range(5))
    for i in sorted(targets):
        col = C_GREEN if (not active or i in active) else C_GREY
        cv2.putText(canvas, f"{FINGER_NAMES[i]:<8}{ext[i]:+.2f}", (x0, y), FONT, FS_M, col, 2)
        y += 26

    fps_str = f"FPS:{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, FONT, FS_S, 1)
    cv2.putText(canvas, fps_str, (SIDEBAR_X + SIDEBAR_TEXT_W + SIDEBAR_MARGIN - tw - 12,
                                   UI_PADDING + 16), FONT, FS_S, C_GREEN, 1)

    cv2.putText(canvas, "space:capture  esc:redo step", (x0, CANVAS_H - UI_PADDING - 32),
                FONT, FS_S, C_WHITE, 1)
    cv2.putText(canvas, "c:restart  q:quit", (x0, CANVAS_H - UI_PADDING - 12),
                FONT, FS_S, C_WHITE, 1)
