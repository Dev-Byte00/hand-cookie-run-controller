"""In-game HUD rendering: gesture state, signal quality, finger bars, keys."""
from __future__ import annotations

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.domain.finger_state import FingerState
from app.domain.models import FINGER_NAMES, FingerConfig
from app.ui.theme import (
    CAM_DISPLAY_H,
    CAM_DISPLAY_W,
    CAM_X,
    CAM_Y,
    CANVAS_H,
    C_CYAN,
    C_GREEN,
    C_GREY,
    C_RED,
    C_WHITE,
    C_YELLOW,
    FONT,
    FS_L,
    FS_M,
    FS_S,
    SIDEBAR_MARGIN,
    SIDEBAR_TEXT_W,
    SIDEBAR_WIDTH,
    SIDEBAR_X,
    TRACK_BTN_H,
    TRACK_BTN_W,
    TRACK_BTN_X0,
    TRACK_BTN_X1,
    TRACK_BTN_Y0,
    TRACK_BTN_Y1,
    UI_PADDING,
)
from app.ui.widgets import draw_bar


def draw_hud(
    canvas: np.ndarray,
    state: str,
    gesture: str,
    finger_state: FingerState,
    config: FingerConfig,
    fps: float,
    *,
    confidence: str,
    confidence_colour: tuple[int, int, int],
    minimal: bool,
    jump_flash: int,
    slide_holding: bool,
    tracking_on: bool,
) -> None:
    """Draw the full in-game HUD onto an already-built canvas."""
    # Colour flash around the camera preview only — instant feedback readable
    # at a glance, without wandering into the sidebar's text.
    if jump_flash > 0:
        cv2.rectangle(canvas, (CAM_X - 2, CAM_Y - 2),
                      (CAM_X + CAM_DISPLAY_W + 2, CAM_Y + CAM_DISPLAY_H + 2), C_GREEN, 6)
    elif slide_holding:
        cv2.rectangle(canvas, (CAM_X - 2, CAM_Y - 2),
                      (CAM_X + CAM_DISPLAY_W + 2, CAM_Y + CAM_DISPLAY_H + 2), C_RED, 6)

    # Tracking on/off button — pinned at the top of the sidebar, click or 't'.
    fill = (24, 58, 24) if tracking_on else (26, 26, 70)
    border_col = C_GREEN if tracking_on else C_RED
    label = "TRACKING: ON" if tracking_on else "TRACKING: OFF"
    cv2.rectangle(canvas, (TRACK_BTN_X0, TRACK_BTN_Y0), (TRACK_BTN_X1, TRACK_BTN_Y1), fill, -1)
    cv2.rectangle(canvas, (TRACK_BTN_X0, TRACK_BTN_Y0), (TRACK_BTN_X1, TRACK_BTN_Y1), border_col, 2)
    (ltw, lth), _ = cv2.getTextSize(label, FONT, FS_M, 2)
    cv2.putText(canvas, label,
                (TRACK_BTN_X0 + (TRACK_BTN_W - ltw) // 2, TRACK_BTN_Y0 + (TRACK_BTN_H + lth) // 2),
                FONT, FS_M, C_WHITE, 2)

    x0 = SIDEBAR_X + SIDEBAR_MARGIN
    y = TRACK_BTN_Y1 + 34

    cv2.putText(canvas, state, (x0, y), FONT, FS_L * 0.85, C_WHITE, 2)
    y += 30

    colour = C_GREEN if gesture == "JUMP" else C_RED if gesture == "SLIDE" else C_YELLOW
    cv2.putText(canvas, f"GESTURE: {gesture}", (x0, y), FONT, FS_M, colour, 2)
    y += 26

    cv2.putText(canvas, f"SIGNAL: {confidence}", (x0, y), FONT, FS_S, confidence_colour, 1)
    y += 24

    if not minimal:
        cv2.putText(canvas, config.label, (x0, y), FONT, FS_S, C_CYAN, 1)
        y += 24
        for i, name in enumerate(FINGER_NAMES):
            role = ""
            if i in config.jump_fingers:
                role = "JUMP"
            elif i in config.slide_fingers:
                role = "SLIDE"
            elif i in config.ready_fingers:
                role = "READY"
            up = finger_state.extended[i]
            col = C_GREEN if up else C_GREY
            cv2.putText(canvas, f"{name:<7}{role}", (x0, y + 5), FONT, FS_S, col, 1)
            on_r, off_r = finger_state.threshold(i)
            draw_bar(canvas, x0 + 130, y - 9, SIDEBAR_TEXT_W - 130, 14,
                     finger_state.extension[i], on_r, off_r, up)
            y += 24

    fps_str = f"FPS:{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, FONT, FS_S, 1)
    cv2.putText(canvas, fps_str, (SIDEBAR_X + SIDEBAR_WIDTH - tw - 12, CANVAS_H - UI_PADDING - 54),
                FONT, FS_S, C_GREEN, 1)

    cv2.putText(canvas, "r:setup  c:swap-keys  l:skeleton", (x0, CANVAS_H - UI_PADDING - 32),
                FONT, FS_S, C_WHITE, 1)
    cv2.putText(canvas, "h:toggle-hud  t:tracking  q:quit", (x0, CANVAS_H - UI_PADDING - 12),
                FONT, FS_S, C_WHITE, 1)
