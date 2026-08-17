"""Interactive finger-assignment wizard (JUMP → SLIDE → READY → FIST)."""
from __future__ import annotations

import time

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.config.settings import AppSettings
from app.domain.finger_metrics import (
    detect_pose,
    extract_extensions,
    is_fist,
    palm_size,
)
from app.domain.models import FINGER_NAMES, FingerConfig
from app.ui.canvas import make_canvas
from app.ui.setup_screen import ClickRects, draw_setup
from app.ui.skeleton_overlay import draw_skeleton
from app.vision.camera_feed import CameraFeed
from app.vision.hand_estimator import HandEstimator

_SETUP_STEPS: tuple[tuple[str, str], ...] = (
    ("JUMP", "Raise the JUMP pose (1+ fingers) — SPACE captures it, ENTER when done"),
    ("SLIDE", "Raise the SLIDE pose (held) — SPACE captures it, ENTER when done"),
    ("READY", "Raise the READY pose (neutral) — SPACE captures it, ENTER when done"),
    ("FIST", "Make a FIST (no fingers raised) — SPACE captures it, ENTER when done"),
)


def run_setup(
    feed: CameraFeed,
    estimator: HandEstimator,
    settings: AppSettings,
    *,
    current: FingerConfig | None = None,
    show_window: bool = True,
    window_title: str = "Hand Cookie Run — Setup",
) -> FingerConfig | None:
    """Interactive finger-assignment screen.

    Walks the user through JUMP → SLIDE → READY → FIST and returns the
    resulting :class:`FingerConfig`, or None if the user quits. The
    ``current`` config (if any) is used to carry over the user's key
    bindings across reconfigures. The feed must already be started
    (``feed.start()`` is idempotent).

    A finger CAN be reused across different actions (e.g. JUMP =
    INDEX+MIDDLE, SLIDE = MIDDLE alone) — but two actions can't end up with
    the exact same finger set, since that would make them indistinguishable
    at runtime. Pressing ENTER on an exact duplicate is rejected; add or
    remove a finger and try again.

    Every finger can also be toggled by CLICKING it in the sidebar instead of
    raising it for the camera — useful since the camera can be flaky
    mid-pose, and the actual detection thresholds get tuned properly in the
    calibration step afterward anyway, so the assignment itself doesn't need
    a perfect camera read.
    """
    feed.start()
    setup_settings = settings.setup

    click_rects: ClickRects = {}
    click_events: list[object] = []

    def _on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for key, (rx0, ry0, rx1, ry1) in click_rects.items():
            if rx0 <= x <= rx1 and ry0 <= y <= ry1:
                click_events.append(key)
                break

    if show_window:
        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, _on_click)

    deadline = time.perf_counter() + settings.camera.warmup_sec
    frame: np.ndarray | None = None
    while frame is None:
        frame, ok = feed.latest()
        if not ok and time.perf_counter() >= deadline:
            print("[FATAL] Camera disconnected.")
            return None
        time.sleep(0.01)

    chosen: dict[str, frozenset[int]] = {}
    current_fingers: list[int] = []  # fingers added to the action being configured
    step = 0
    show_skeleton = True

    disp = [0.0] * 5  # lightly smoothed raw extensions for the live read-out
    fps = 0.0
    prev_t = time.perf_counter()
    frame_failures = 0

    try:
        while True:
            frame, ok = feed.latest()
            if not ok or frame is None:
                frame_failures += 1
                if frame_failures >= settings.camera.max_frame_failures:
                    print("[FATAL] Camera disconnected.")
                    return None
                time.sleep(0.005)
                continue
            frame_failures = 0

            landmarks = estimator.process(frame)
            size = palm_size(landmarks) if landmarks is not None else None
            raw = extract_extensions(landmarks, size)
            for i in range(5):
                disp[i] += setup_settings.smooth_alpha * (raw[i] - disp[i])

            # Fingers already added to the action currently being built can't
            # be re-picked mid-build (that's just duplicate-add protection).
            # Fingers used by OTHER, already-finished actions are fine to
            # reuse here — chords are told apart by ENTER's duplicate-chord
            # check below, not by reserving whole fingers per action.
            unavailable = frozenset(current_fingers)
            # Use the RAW extension for detection (not the smoothed ``disp``)
            # so a pose is captured immediately on the first frame it's
            # raised, instead of waiting for the display smoothing to catch
            # up.
            detected = detect_pose(raw, setup_settings, unavailable)

            if show_skeleton and landmarks is not None:
                draw_skeleton(frame, landmarks)

            now = time.perf_counter()
            dt = now - prev_t
            if dt > 0:
                fps = fps * (1.0 - settings.fps_alpha) + (1.0 / dt) * settings.fps_alpha
            prev_t = now

            step_name, step_hint = _SETUP_STEPS[step]
            canvas = make_canvas(frame)
            draw_setup(canvas, step_name, step_hint, detected, chosen, disp, fps,
                       current_fingers, click_rects, setup_settings)

            if show_window:
                cv2.imshow(window_title, canvas)
                k = cv2.waitKey(1) & 0xFF
            else:
                k = 0
                time.sleep(0.001)  # don't busy-spin in headless mode

            while click_events:
                ev = click_events.pop(0)
                if ev == "FIST" and step == 3:
                    current_fingers = []
                    print(f"{step_name} = (fist/punch) [manual]")
                elif isinstance(ev, int) and step != 3:
                    if ev in current_fingers:
                        current_fingers.remove(ev)
                        print(f"Removed {FINGER_NAMES[ev]} from {step_name} (click) — "
                              f"{len(current_fingers)} finger(s) in this action.")
                    else:
                        current_fingers.append(ev)
                        print(f"{step_name} += {FINGER_NAMES[ev]} (click) — "
                              f"{len(current_fingers)} finger(s) in this action.")

            if k == ord("q"):
                return None
            elif k == ord("c"):
                step = 0
                chosen.clear()
                current_fingers.clear()
                print("Setup restarted.")
            elif k == 27:  # ESC → remove the last finger added, or step back
                if step == 3 and current_fingers is not None:
                    current_fingers = []
                    print("Cleared FIST capture.")
                elif current_fingers:
                    removed = current_fingers.pop()
                    print(f"Removed {FINGER_NAMES[removed]} from {step_name} — "
                          f"{len(current_fingers)} finger(s) in this action.")
                elif step > 0:
                    step -= 1
                    current_fingers = list(chosen.popitem()[1])
                    print(f"Editing {_SETUP_STEPS[step][0]} again — "
                          f"{len(current_fingers)} finger(s) already added.")
            elif k == ord(" "):  # SPACE → capture the raised pose now
                if step == 3:  # FIST step — capture a genuinely curled fist
                    if is_fist(raw, setup_settings):
                        current_fingers = []
                        print(f"{step_name} = (fist/punch)")
                    else:
                        print("Make a tighter fist — all fingers curled in.")
                elif detected:
                    added: list[int] = []
                    for f in sorted(detected):
                        if f not in current_fingers:
                            current_fingers.append(f)
                            added.append(f)
                    if added:
                        print(f"{step_name} += "
                              f"{'+'.join(FINGER_NAMES[f] for f in added)} "
                              f"({len(current_fingers)} finger(s))")
                    else:
                        print(f"All detected fingers already added to {step_name}.")
                else:
                    print("No fingers detected — raise at least one finger.")
            elif k == 13:  # ENTER → finish this action (FIST allows empty set)
                if step != 3 and not current_fingers:
                    print("No fingers added yet — raise a finger and press SPACE.")
                    continue
                new_chord = frozenset(current_fingers)
                clash = next(
                    (name for name, fs in chosen.items() if step != 3 and fs == new_chord),
                    None,
                )
                if clash is not None:
                    chord_names = "+".join(FINGER_NAMES[i] for i in sorted(new_chord))
                    print(f"{step_name} can't use the exact same fingers as {clash} "
                          f"({chord_names}) — add one more finger (or remove one) "
                          f"so the two poses are distinguishable.")
                    continue
                chosen[step_name] = new_chord
                names = ("+".join(FINGER_NAMES[i] for i in sorted(current_fingers))
                         if current_fingers else "(fist)")
                print(f"{step_name} = {names}")
                current_fingers = []
                step += 1
                if step >= len(_SETUP_STEPS):
                    cfg = FingerConfig(
                        chosen["JUMP"], chosen["SLIDE"], chosen["READY"],
                    )
                    if current is not None:  # carry over key bindings
                        cfg.jump_key = current.jump_key
                        cfg.slide_key = current.slide_key
                    print(f"Setup complete  |  {cfg.label}")
                    return cfg

    finally:
        if show_window:
            cv2.destroyAllWindows()
