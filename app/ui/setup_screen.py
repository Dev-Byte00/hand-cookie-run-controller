"""Setup-wizard (finger-assignment) screen rendering."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.config.settings import SetupSettings
from app.domain.finger_metrics import is_fist
from app.domain.models import FINGER_NAMES
from app.ui.text import draw_wrapped
from app.ui.theme import (
    CANVAS_H,
    C_CYAN,
    C_GREEN,
    C_GREY,
    C_WHITE,
    C_YELLOW,
    FONT,
    FS_L,
    FS_M,
    FS_S,
    SIDEBAR_MARGIN,
    SIDEBAR_TEXT_W,
    SIDEBAR_X,
    UI_PADDING,
)
from app.ui.widgets import role_of

#: Clickable-region registry type: key → (x0, y0, x1, y1).
ClickRects = dict[object, tuple[int, int, int, int]]


def draw_setup(
    canvas: np.ndarray,
    step_name: str,
    step_hint: str,
    detected: Sequence[int],
    chosen: Mapping[str, frozenset[int]],
    ext: Sequence[float],
    fps: float,
    current_fingers: Sequence[int],
    click_rects: ClickRects,
    setup_settings: SetupSettings,
) -> None:
    """Draw the setup screen and populate ``click_rects`` (cleared and
    refilled every call) with this frame's clickable regions, so the caller's
    mouse callback always hit-tests against exactly what's currently shown.
    """
    click_rects.clear()
    x0 = SIDEBAR_X + SIDEBAR_MARGIN
    y = UI_PADDING + 34

    cv2.putText(canvas, "FINGER SETUP", (x0, y), FONT, FS_L * 0.8, C_WHITE, 2)
    y += 34
    cv2.putText(canvas, f"STEP: {step_name}", (x0, y), FONT, FS_M, C_CYAN, 2)
    y += 26
    y = draw_wrapped(canvas, step_hint, x0, y, SIDEBAR_TEXT_W, FS_S, C_WHITE, line_h=18) + 14

    if step_name == "FIST":
        # No fingers to pick here — either make a real fist for the camera,
        # or just confirm the empty set manually (a fist is the pose the
        # camera struggles with most, so this needs a reliable fallback).
        fist_now = is_fist(ext, setup_settings)
        status = "Camera sees a fist" if fist_now else "Camera: not curled enough"
        cv2.putText(canvas, status, (x0, y), FONT, FS_S, C_GREEN if fist_now else C_YELLOW, 1)
        y += 30

        btn_w, btn_h = SIDEBAR_TEXT_W, 42
        rect = (x0, y, x0 + btn_w, y + btn_h)
        click_rects["FIST"] = rect
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (24, 50, 24), -1)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), C_GREEN, 2)
        label = "CONFIRM: NO FINGERS (click)"
        (lw, lth), _ = cv2.getTextSize(label, FONT, FS_S, 1)
        cv2.putText(canvas, label, (rect[0] + (btn_w - lw) // 2, rect[1] + (btn_h + lth) // 2),
                    FONT, FS_S, C_WHITE, 1)
        y += btn_h + 16
    else:
        if detected:
            names = "+".join(FINGER_NAMES[i] for i in detected)
            colour = C_GREEN
            y = draw_wrapped(canvas, f"DETECTED: {names}", x0, y, SIDEBAR_TEXT_W,
                              FS_M, colour, thickness=2, line_h=24)
            cv2.putText(canvas, "[space] to add", (x0, y), FONT, FS_S, colour, 1)
            y += 26
        else:
            colour = C_YELLOW
            cv2.putText(canvas, "DETECTED: ---", (x0, y), FONT, FS_M, colour, 2)
            y += 24
            cv2.putText(canvas, "(raise the pose, then press space)", (x0, y), FONT, FS_S, colour, 1)
            y += 26

        cv2.putText(canvas, "or click a finger to toggle it:", (x0, y), FONT, FS_S, C_WHITE, 1)
        y += 24

        row_h = 26
        for i, name in enumerate(FINGER_NAMES):
            checked = i in current_fingers
            role = role_of(i, chosen)
            box = "[x]" if checked else "[ ]"
            if checked:
                col = C_GREEN
            elif i in detected:
                col = C_CYAN
            else:
                col = C_GREY
            label = f"{box} {name}"
            if role:
                label += f" (also {role})"
            click_rects[i] = (x0 - 6, y - 17, x0 + SIDEBAR_TEXT_W + 6, y + 7)
            cv2.putText(canvas, label, (x0, y), FONT, FS_S, col, 1)
            ext_str = f"{ext[i]:+.2f}"
            (ew, _), _ = cv2.getTextSize(ext_str, FONT, FS_S, 1)
            cv2.putText(canvas, ext_str, (x0 + SIDEBAR_TEXT_W - ew, y), FONT, FS_S, col, 1)
            y += row_h

    fps_str = f"FPS:{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, FONT, FS_S, 1)
    cv2.putText(canvas, fps_str, (SIDEBAR_X + SIDEBAR_TEXT_W + SIDEBAR_MARGIN - tw - 12,
                                   UI_PADDING + 16), FONT, FS_S, C_GREEN, 1)

    cv2.putText(canvas, "space:capture  enter:done", (x0, CANVAS_H - UI_PADDING - 32),
                FONT, FS_S, C_WHITE, 1)
    cv2.putText(canvas, "esc:undo  c:restart  q:quit", (x0, CANVAS_H - UI_PADDING - 12),
                FONT, FS_S, C_WHITE, 1)
