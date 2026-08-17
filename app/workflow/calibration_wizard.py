"""Guided per-user threshold calibration."""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.config.settings import AppSettings, CalibrationSettings
from app.domain.finger_metrics import extract_extensions, palm_size
from app.domain.models import FINGER_NAMES, FingerCalibration, FingerConfig
from app.ui.calibration_screen import draw_calibration
from app.ui.canvas import make_canvas
from app.ui.skeleton_overlay import draw_skeleton
from app.vision.camera_feed import CameraFeed
from app.vision.hand_estimator import HandEstimator


def _calib_steps_for(config: FingerConfig) -> list[tuple[str, str, frozenset[int]]]:
    """Baseline first, then one step per distinct pose the config actually
    uses — a finger config never exercises a finger it doesn't assign, so
    unused fingers are simply left on their tuned global defaults."""
    steps: list[tuple[str, str, frozenset[int]]] = [
        ("BASELINE", "Relax your hand (nothing raised) — then press SPACE", frozenset())
    ]
    seen: set[frozenset[int]] = set()
    for role, fingers in (
        ("JUMP", config.jump_fingers),
        ("SLIDE", config.slide_fingers),
        ("READY", config.ready_fingers),
    ):
        if fingers and fingers not in seen:
            names = "+".join(FINGER_NAMES[i] for i in sorted(fingers))
            steps.append((role, f"Hold {names} up ({role} pose) — then press SPACE", fingers))
            seen.add(fingers)
    return steps


def _derive_calibration(
    down: Mapping[int, float],
    up: Mapping[int, Sequence[float]],
    calib: CalibrationSettings,
) -> FingerCalibration:
    """Turn measured (relaxed → raised) samples into on/off thresholds.

    Placing the thresholds as a *fraction of the finger's own measured
    range* (rather than at some fixed absolute value) is what makes this
    adapt to a given hand, camera distance, and lighting — a finger that
    only swings from 0.3→1.0 gets thresholds scaled to that range instead
    of the wider range a different hand/camera might produce.
    """
    thresholds: dict[int, tuple[float, float]] = {}
    for i in range(5):
        if i not in down or not up.get(i):
            continue  # this finger was never part of a pose — keep the default
        down_val = down[i]
        up_val = sum(up[i]) / len(up[i])
        span = up_val - down_val
        if span < calib.min_range:
            print(f"[calib] {FINGER_NAMES[i]}: movement too small ({span:.2f}) "
                  f"— keeping the default thresholds.")
            continue
        thresholds[i] = (
            down_val + calib.on_fraction * span,
            down_val + calib.off_fraction * span,
        )
    return FingerCalibration(thresholds)


def run_calibration(
    feed: CameraFeed,
    estimator: HandEstimator,
    config: FingerConfig,
    settings: AppSettings,
    *,
    show_window: bool = True,
    window_title: str = "Hand Cookie Run — Calibration",
) -> FingerCalibration | None:
    """Guided hold-the-pose calibration for the fingers `config` actually uses.

    A fixed global threshold rarely matches every hand size, camera angle,
    and lighting setup — that mismatch is the usual cause of "it doesn't
    detect my finger right." This walks through a relaxed baseline plus each
    pose the mapping needs, measures the real down→up range per finger, and
    derives personal on/off thresholds from it.
    """
    feed.start()
    deadline = time.perf_counter() + settings.camera.warmup_sec
    frame: np.ndarray | None = None
    while frame is None:
        frame, ok = feed.latest()
        if not ok and time.perf_counter() >= deadline:
            print("[FATAL] Camera disconnected.")
            return None
        time.sleep(0.01)

    steps = _calib_steps_for(config)
    down: dict[int, float] = {}
    up: dict[int, list[float]] = {i: [] for i in range(5)}
    disp = [0.0] * 5
    step_i = 0
    fps = 0.0
    prev_t = time.perf_counter()
    frame_failures = 0

    try:
        while step_i < len(steps):
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
            if landmarks is not None:
                draw_skeleton(frame, landmarks)
            for i in range(5):
                disp[i] += settings.setup.smooth_alpha * (raw[i] - disp[i])

            now = time.perf_counter()
            dt = now - prev_t
            if dt > 0:
                fps = fps * (1.0 - settings.fps_alpha) + (1.0 / dt) * settings.fps_alpha
            prev_t = now

            label, hint, active = steps[step_i]
            canvas = make_canvas(frame)
            draw_calibration(canvas, label, hint, active, disp, step_i, len(steps), fps)

            if show_window:
                cv2.imshow(window_title, canvas)
                k = cv2.waitKey(1) & 0xFF
            else:
                k = 0
                time.sleep(0.001)

            if k == ord("q"):
                return None
            elif k == ord("c"):
                step_i = 0
                down.clear()
                up = {i: [] for i in range(5)}
                print("Calibration restarted.")
            elif k == 27 and step_i > 0:  # ESC — undo the previous capture, redo it
                step_i -= 1
                prev_label, _, prev_active = steps[step_i]
                if prev_active:
                    for i in prev_active:
                        if up[i]:
                            up[i].pop()
                else:
                    down.clear()
                print(f"Redo {prev_label}.")
            elif k == ord(" "):
                if active:
                    for i in active:
                        up[i].append(disp[i])
                else:
                    for i in range(5):
                        down[i] = disp[i]
                sample_str = ", ".join(
                    f"{FINGER_NAMES[i]}={disp[i]:.2f}"
                    for i in (sorted(active) if active else range(5))
                )
                print(f"Captured {label}: {sample_str}")
                step_i += 1
    finally:
        if show_window:
            cv2.destroyAllWindows()

    calibration = _derive_calibration(down, up, settings.calibration)
    print("Calibration complete.")
    return calibration
