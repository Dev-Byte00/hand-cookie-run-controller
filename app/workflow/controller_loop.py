"""Main gameplay loop: capture → inference → finger-state → input → HUD."""
from __future__ import annotations

import time

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.automation.input_controller import InputController
from app.config.settings import AppSettings
from app.domain.enums import GestureState, SignalQuality, WorkflowSignal
from app.domain.finger_state import FingerState
from app.domain.gesture_classifier import GestureHold, classify_pose, tracking_confidence
from app.domain.models import FingerCalibration, FingerConfig
from app.ui.canvas import make_canvas
from app.ui.hud_screen import draw_hud
from app.ui.skeleton_overlay import draw_skeleton
from app.ui.theme import (
    C_GREEN,
    C_GREY,
    C_RED,
    C_YELLOW,
    TRACK_BTN_X0,
    TRACK_BTN_X1,
    TRACK_BTN_Y0,
    TRACK_BTN_Y1,
)
from app.vision.camera_feed import CameraFeed
from app.vision.hand_estimator import HandEstimator

#: Maps each :class:`SignalQuality` to the HUD colour used to display it.
_CONFIDENCE_COLOURS: dict[SignalQuality, tuple[int, int, int]] = {
    SignalQuality.GOOD: C_GREEN,
    SignalQuality.WEAK: C_YELLOW,
    SignalQuality.REACQUIRING: C_YELLOW,
    SignalQuality.LOST: C_RED,
}


def run_controller(
    feed: CameraFeed,
    estimator: HandEstimator,
    config: FingerConfig,
    calibration: FingerCalibration | None,
    settings: AppSettings,
    input_controller: InputController,
    *,
    show_window: bool = True,
    window_title: str = "Hand Cookie Run",
) -> WorkflowSignal | None:
    """Run capture → inference → finger-state → input until quit or reconfig.

    Returns :data:`WorkflowSignal.RECONFIGURE` when the user presses 'r' (the
    caller should re-run the setup screen), or None when the session ends
    (quit or camera failure). The feed must already be started
    (``feed.start()`` is idempotent). ``show_window=False`` runs headless,
    e.g. embedded in another script.

    Action semantics with the user's chord mapping:
        * A complete READY chord, or complete JUMP+SLIDE chords together,
          → neutral (no key)
        * A complete JUMP chord → rising edge = tap the jump key
        * A complete SLIDE chord → the slide key is *held* while raised,
          which keeps the game's hold-to-slide mechanic working.
        * A chord must match EXACTLY among this config's active fingers —
          not "at least these fingers" — so a finger reused across actions
          (e.g. SLIDE=MIDDLE nested inside JUMP=INDEX+MIDDLE) still resolves
          to exactly one action instead of firing both at once.
    """
    feed.start()
    detection = settings.detection
    gesture_settings = settings.gesture

    click_toggle = [False]

    def _on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        if (event == cv2.EVENT_LBUTTONDOWN
                and TRACK_BTN_X0 <= x <= TRACK_BTN_X1
                and TRACK_BTN_Y0 <= y <= TRACK_BTN_Y1):
            click_toggle[0] = True

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

    finger_state = FingerState(detection, calibration)
    gesture_hold = GestureHold(gesture_settings.gesture_hold_frames)
    active_fingers = config.active_fingers
    state: str = "READY"
    show_skeleton: bool = True
    hud_minimal: bool = True
    tracking_on: bool = True

    fps = 0.0
    latency_ms = 0.0
    prev_t = time.perf_counter()
    frame_failures = 0
    last_jump: float = 0.0
    slide_holding: bool = False
    prev_raw: GestureState = GestureState.IDLE
    jump_flash: int = 0

    try:
        last_frame: np.ndarray | None = None
        canvas: np.ndarray = make_canvas(frame)
        while True:
            if not tracking_on:
                canvas = make_canvas(None)
                draw_hud(canvas, "TRACKING OFF", "-", finger_state, config, fps,
                         confidence="N/A", confidence_colour=C_GREY,
                         minimal=hud_minimal, jump_flash=0, slide_holding=False,
                         tracking_on=False, latency_ms=0.0)
                if show_window:
                    cv2.imshow(window_title, canvas)
                    k = cv2.waitKey(30) & 0xFF
                else:
                    k = 0
                    time.sleep(0.03)
                if click_toggle[0]:
                    click_toggle[0] = False
                    k = ord("t")
                if k == ord("q"):
                    break
                elif k == ord("t"):
                    if feed.restart():
                        print("Reopening camera...")
                        resume_deadline = time.perf_counter() + settings.camera.warmup_sec
                        got_frame = False
                        while time.perf_counter() < resume_deadline:
                            f, fok = feed.latest()
                            if fok and f is not None:
                                got_frame = True
                                break
                            time.sleep(0.02)
                        if got_frame:
                            tracking_on = True
                            frame_failures = 0
                            last_frame = None
                            print("Tracking ON.")
                        else:
                            print("[WARN] Camera did not respond — still off.")
                            feed.stop()
                    else:
                        print("[WARN] Could not reopen the camera.")
                continue

            frame, ok = feed.latest()
            if not ok or frame is None:
                frame_failures += 1
                if frame_failures >= settings.camera.max_frame_failures:
                    print("[FATAL] Camera disconnected.")
                    break
                time.sleep(0.005)
                continue
            frame_failures = 0

            # Only process *new* frames — if the camera delivers slower than
            # inference, don't waste a pass re-analyzing a frame we've seen.
            if frame is not last_frame:
                last_frame = frame
                now = time.perf_counter()

                _inf_t0 = time.perf_counter()
                landmarks = estimator.process(frame)
                _inf_ms = (time.perf_counter() - _inf_t0) * 1000.0
                latency_ms = latency_ms * (1.0 - settings.fps_alpha) + _inf_ms * settings.fps_alpha

                finger_state.update(landmarks, detection.hand_lost_tolerance)
                ext = finger_state.extended
                if jump_flash > 0:
                    jump_flash -= 1
                confidence = tracking_confidence(finger_state, detection)

                raised = frozenset(i for i in range(5) if ext[i])
                # Only fingers that actually belong to *some* action matter
                # for classification — this keeps noise on an unused finger
                # (e.g. a borderline thumb) from ever blocking a gesture.
                raised_active = raised & active_fingers

                raw = classify_pose(raised_active, config, any_raised=bool(raised))

                # Stabilized gesture — holds JUMP/SLIDE for a short grace
                # period so micro-flickers can't stutter a held slide key.
                gesture = gesture_hold.update(raw)

                # JUMP — fire on the *raw* rising edge so fast double-taps
                # stay responsive even inside the gesture-hold grace window.
                if raw == GestureState.JUMP and prev_raw != GestureState.JUMP:
                    if (now - last_jump) >= gesture_settings.gesture_cooldown_sec:
                        input_controller.tap_async(config.jump_key)
                        last_jump = now
                        state = "JUMP!"
                        jump_flash = gesture_settings.jump_flash_frames
                prev_raw = raw

                # SLIDE — hold the key while the *stabilized* gesture is
                # SLIDE, so a 1-frame finger flicker never stutters the held
                # key.
                if gesture == GestureState.SLIDE:
                    if not slide_holding:
                        input_controller.key_down(config.slide_key)
                        slide_holding = True
                        state = "SLIDE!"
                elif slide_holding:  # finger lowered / hand lost / conflict
                    input_controller.key_up(config.slide_key)
                    slide_holding = False

                if gesture in (GestureState.READY, GestureState.NONE, GestureState.IDLE):
                    state = "READY"

                if show_skeleton and landmarks is not None:
                    draw_skeleton(frame, landmarks)

                dt = now - prev_t
                if dt > 0:
                    fps = fps * (1.0 - settings.fps_alpha) + (1.0 / dt) * settings.fps_alpha
                prev_t = now

                canvas = make_canvas(frame)
                draw_hud(canvas, state, gesture.value, finger_state, config, fps,
                         confidence=confidence.value,
                         confidence_colour=_CONFIDENCE_COLOURS[confidence],
                         minimal=hud_minimal, jump_flash=jump_flash,
                         slide_holding=slide_holding, tracking_on=True,
                         latency_ms=latency_ms)

            if show_window:
                cv2.imshow(window_title, canvas)
                k = cv2.waitKey(1) & 0xFF
            else:
                k = 0
                time.sleep(0.001)  # don't busy-spin in headless mode

            if click_toggle[0]:
                click_toggle[0] = False
                k = ord("t")

            if k == ord("q"):
                break
            elif k == ord("t"):
                if slide_holding:
                    input_controller.key_up(config.slide_key)
                    slide_holding = False
                feed.stop()
                tracking_on = False
                state = "TRACKING OFF"
                last_frame = None
                print("Tracking OFF — camera released.")
            elif k == ord("r"):
                if slide_holding:
                    input_controller.key_up(config.slide_key)
                    slide_holding = False
                print("Reconfiguring fingers...")
                return WorkflowSignal.RECONFIGURE
            elif k == ord("c"):
                if slide_holding:
                    input_controller.key_up(config.slide_key)
                    slide_holding = False
                config.jump_key, config.slide_key = config.slide_key, config.jump_key
                print(f"Keys swapped  |  Jump={config.jump_key}  Slide={config.slide_key}")
            elif k == ord("l"):
                show_skeleton = not show_skeleton
                print(f"Skeleton overlay: {'ON' if show_skeleton else 'OFF'}")
            elif k == ord("h"):
                hud_minimal = not hud_minimal
                print(f"HUD: {'minimal' if hud_minimal else 'full debug'}")

        return None

    finally:
        if slide_holding:
            input_controller.key_up(config.slide_key)
        if show_window:
            cv2.destroyAllWindows()
        print("Controller stopped.")
