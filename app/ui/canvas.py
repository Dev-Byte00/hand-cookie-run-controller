"""Builds the base black-theme window canvas (camera preview + sidebar card)."""
from __future__ import annotations

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.ui.theme import (
    BG_CAM_OFF,
    BG_CANVAS,
    BG_PANEL,
    BORDER,
    CAM_DISPLAY_H,
    CAM_DISPLAY_W,
    CAM_X,
    CAM_Y,
    CANVAS_H,
    CANVAS_W,
    C_GREY,
    FONT,
    FS_L,
    FS_S,
    SIDEBAR_WIDTH,
    SIDEBAR_X,
    UI_PADDING,
)


def make_canvas(cam_frame: np.ndarray | None) -> np.ndarray:
    """Build the black-theme window: an enlarged camera preview on the left,
    a solid sidebar card on the right for all HUD text.

    Text is drawn ONLY inside the sidebar region, so it can never overlap
    the camera image — that's a layout guarantee, not something each drawing
    function has to be careful about. ``cam_frame=None`` renders a "camera
    off" placeholder instead (used while tracking is toggled off).
    """
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_CANVAS, dtype=np.uint8)

    if cam_frame is not None:
        disp = cv2.resize(cam_frame, (CAM_DISPLAY_W, CAM_DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        canvas[CAM_Y:CAM_Y + CAM_DISPLAY_H, CAM_X:CAM_X + CAM_DISPLAY_W] = disp
    else:
        cv2.rectangle(canvas, (CAM_X, CAM_Y),
                      (CAM_X + CAM_DISPLAY_W, CAM_Y + CAM_DISPLAY_H), BG_CAM_OFF, -1)
        msg = "CAMERA OFF"
        (tw, th), _ = cv2.getTextSize(msg, FONT, FS_L, 2)
        tx = CAM_X + (CAM_DISPLAY_W - tw) // 2
        ty = CAM_Y + (CAM_DISPLAY_H + th) // 2
        cv2.putText(canvas, msg, (tx, ty), FONT, FS_L, C_GREY, 2)
        hint = "click TRACKING or press 't' to resume"
        (hw, _), _ = cv2.getTextSize(hint, FONT, FS_S, 1)
        cv2.putText(canvas, hint, (CAM_X + (CAM_DISPLAY_W - hw) // 2, ty + 32),
                    FONT, FS_S, C_GREY, 1)

    cv2.rectangle(canvas, (CAM_X, CAM_Y),
                  (CAM_X + CAM_DISPLAY_W, CAM_Y + CAM_DISPLAY_H), BORDER, 2)
    cv2.rectangle(canvas, (SIDEBAR_X, UI_PADDING),
                  (SIDEBAR_X + SIDEBAR_WIDTH, CANVAS_H - UI_PADDING), BG_PANEL, -1)
    cv2.rectangle(canvas, (SIDEBAR_X, UI_PADDING),
                  (SIDEBAR_X + SIDEBAR_WIDTH, CANVAS_H - UI_PADDING), BORDER, 1)
    return canvas
