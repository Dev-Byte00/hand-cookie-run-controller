"""Window layout + black-theme colour palette.

The camera preview and all HUD text live in separate regions of one big
canvas, so text can never overlap the video the way it might if drawn
directly on top of the camera frame — that's a layout guarantee enforced by
:func:`app.ui.canvas.make_canvas`, not something each drawing routine has to
be careful about.
"""
from __future__ import annotations

import cv2  # type: ignore

# ---------------------------------------------------------------------------
# Frame / camera geometry
# ---------------------------------------------------------------------------
FRAME_WIDTH: int = 480
FRAME_HEIGHT: int = 360
CAM_DISPLAY_SCALE: float = 1.5  # camera preview enlarged for visibility
CAM_DISPLAY_W: int = int(FRAME_WIDTH * CAM_DISPLAY_SCALE)   # 720
CAM_DISPLAY_H: int = int(FRAME_HEIGHT * CAM_DISPLAY_SCALE)  # 540

UI_PADDING: int = 20
SIDEBAR_WIDTH: int = 380
CAM_X: int = UI_PADDING
CAM_Y: int = UI_PADDING
SIDEBAR_X: int = UI_PADDING + CAM_DISPLAY_W + UI_PADDING
CANVAS_W: int = SIDEBAR_X + SIDEBAR_WIDTH + UI_PADDING
CANVAS_H: int = CAM_DISPLAY_H + UI_PADDING * 2

SIDEBAR_MARGIN: int = 18
SIDEBAR_TEXT_W: int = SIDEBAR_WIDTH - 2 * SIDEBAR_MARGIN

# "TRACKING: ON/OFF" button — centered at the top of the sidebar.
TRACK_BTN_W: int = 220
TRACK_BTN_H: int = 44
TRACK_BTN_X0: int = SIDEBAR_X + (SIDEBAR_WIDTH - TRACK_BTN_W) // 2
TRACK_BTN_Y0: int = UI_PADDING + 14
TRACK_BTN_X1: int = TRACK_BTN_X0 + TRACK_BTN_W
TRACK_BTN_Y1: int = TRACK_BTN_Y0 + TRACK_BTN_H

# ---------------------------------------------------------------------------
# Black theme palette (BGR)
# ---------------------------------------------------------------------------
BG_CANVAS = (16, 16, 16)     # window background
BG_PANEL = (28, 28, 30)      # sidebar card background
BG_CAM_OFF = (30, 30, 30)    # placeholder when the camera is turned off
BORDER = (70, 70, 74)        # subtle card/preview border

C_YELLOW = (0, 255, 255)
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_WHITE = (255, 255, 255)
C_CYAN = (255, 255, 0)
C_GREY = (128, 128, 128)

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
FS_S = 0.45
FS_M = 0.6
FS_L = 1.0
