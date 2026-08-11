"""
AI Hand-Gesture Cookie Run Game Controller
==========================================
High-performance hand-gesture controller using OpenCV, MediaPipe Tasks API
(HandLandmarker), and PyDirectInput for real-time game input.

Controls:
    Jump  — open palm (fingers extended)
    Slide — closed fist (fingers curled)
    c     — swap jump/slide key bindings
    l     — toggle hand skeleton overlay
    q     — quit

Author : Tanakorn Uamraksa (GitHub Copilot)
Version: 1.0.0 — MediaPipe Tasks API (Hand Landmarker)
"""

from __future__ import annotations

import socket
import time
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import pydirectinput  # type: ignore
from mediapipe.tasks.python import BaseOptions  # type: ignore
from mediapipe.tasks.python.vision import (  # type: ignore
    RunningMode,
    drawing_utils,
    drawing_styles,
)
from mediapipe.tasks.python.vision.hand_landmarker import (  # type: ignore
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmark,
    HandLandmarksConnections,
)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 320
FRAME_HEIGHT: int = 240

GESTURE_COOLDOWN_SEC: float = 0.40
FINGER_EXTENDED_THRESHOLD: float = 0.04  # normalized: tip Y < pip Y by this much = extended
OPEN_HAND_MIN_FINGERS: int = 3  # how many fingers extended to count as "open palm"

KEY_JUMP: str = "space"
KEY_SLIDE: str = "down"

# BGR colours
C_YELLOW = (0, 255, 255)
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_WHITE = (255, 255, 255)
C_ORANGE = (0, 165, 255)
C_CYAN = (255, 255, 0)

# Hand landmark indices (finger tips & PIPs below)
_FINGER_TIPS = [
    HandLandmark.THUMB_TIP,
    HandLandmark.INDEX_FINGER_TIP,
    HandLandmark.MIDDLE_FINGER_TIP,
    HandLandmark.RING_FINGER_TIP,
    HandLandmark.PINKY_TIP,
]
_FINGER_PIPS = [
    HandLandmark.THUMB_IP,
    HandLandmark.INDEX_FINGER_PIP,
    HandLandmark.MIDDLE_FINGER_PIP,
    HandLandmark.RING_FINGER_PIP,
    HandLandmark.PINKY_PIP,
]

# Drawing
_DRAW_STYLE = drawing_styles.get_default_hand_landmarks_style()
_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

# Font cache
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FS_S = 0.45
_FS_M = 0.6
_FS_L = 1.0

# Model download
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)
_MODEL_DIR = Path(__file__).resolve().parent / "model"
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------
def _ensure_model() -> Path | None:
    """Download the hand landmarker model if not already cached."""
    _MODEL_DIR.mkdir(exist_ok=True)
    if not _MODEL_PATH.exists():
        print("Downloading hand model (~5 MB)...")
        try:
            socket.setdefaulttimeout(30)
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print(f"Model saved → {_MODEL_PATH}")
        except (OSError, urllib.request.URLError) as exc:
            print(f"[FATAL] Model download failed: {exc}")
            if _MODEL_PATH.exists():
                _MODEL_PATH.unlink()
            return None
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# Hand estimator
# ---------------------------------------------------------------------------
class HandEstimator:
    """Thin wrapper around MediaPipe HandLandmarker."""

    __slots__ = ("_landmarker", "_t0")

    def __init__(self) -> None:
        model = _ensure_model()
        if model is None:
            raise RuntimeError("Hand model unavailable — check network or model dir.")
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = HandLandmarker.create_from_options(opts)
        self._t0 = time.perf_counter()

    def process(self, bgr: np.ndarray) -> Sequence | None:
        """Run inference. Returns hand landmarks list, or None if no hand detected."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        elapsed_ms = int((time.perf_counter() - self._t0) * 1000)
        res = self._landmarker.detect_for_video(mp_image, elapsed_ms)
        if res.hand_landmarks:
            return res.hand_landmarks[0]
        return None

    def close(self) -> None:
        self._landmarker.close()


# ---------------------------------------------------------------------------
# Gesture detection
# ---------------------------------------------------------------------------
def count_extended_fingers(landmarks: Sequence) -> int:
    """Count how many fingers are extended (tip above PIP in Y)."""
    n = len(landmarks)
    extended = 0
    for tip_idx, pip_idx in zip(_FINGER_TIPS, _FINGER_PIPS):
        tip = landmarks[tip_idx] if tip_idx < n else None
        pip = landmarks[pip_idx] if pip_idx < n else None
        if tip is not None and pip is not None:
            # For thumb, use X distance (thumb moves sideways)
            if tip_idx == HandLandmark.THUMB_TIP:
                if abs(tip.x - pip.x) > FINGER_EXTENDED_THRESHOLD:
                    extended += 1
            else:
                # Tip is ABOVE pip (lower Y = higher on screen) = extended
                if (pip.y - tip.y) > FINGER_EXTENDED_THRESHOLD:
                    extended += 1
    return extended


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    state: str,
    gesture: str,
    extended_count: int,
    fps: float,
) -> None:
    fh, fw = frame.shape[:2]
    h = 20

    # State bar (top)
    cv2.putText(frame, state, (6, h + 2), _FONT, _FS_M, C_WHITE, 2)
    h += 28

    # Gesture label
    if gesture == "OPEN":
        colour = C_GREEN
    elif gesture == "FIST":
        colour = C_RED
    else:
        colour = C_YELLOW
    cv2.putText(frame, f"GESTURE: {gesture}", (6, h + 2), _FONT, _FS_M, colour, 2)
    h += 28

    # Finger count
    cv2.putText(frame, f"FINGERS: {extended_count}/5", (6, h + 2), _FONT, _FS_S, C_CYAN, 1)
    h += 22

    # FPS
    fps_str = f"FPS:{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, _FONT, _FS_S, 1)
    cv2.putText(frame, fps_str, (fw - tw - 6, 16), _FONT, _FS_S, C_GREEN, 1)

    # Key legend (bottom)
    cv2.putText(frame, "c:swap-keys  l:skeleton  q:quit",
                (4, fh - 6), _FONT, _FS_S, C_WHITE, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[FATAL] Camera {CAMERA_INDEX} unavailable.")
        return

    print("Camera ready  |  c=swap-keys  l=toggle-skeleton  q=quit")
    print("Open palm → Jump  |  Fist → Slide")

    try:
        estimator = HandEstimator()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        cap.release()
        return

    last_action: float = 0.0
    state: str = "READY"
    show_skeleton: bool = True
    keys_swapped: bool = False

    fps = 0.0
    fps_alpha = 0.05
    t0 = time.perf_counter()
    fc = 0
    fps_interval = 0.25
    last_fps_update = t0
    frame_failures = 0
    MAX_FRAME_FAILURES = 30

    prev_gesture: Optional[str] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                frame_failures += 1
                if frame_failures >= MAX_FRAME_FAILURES:
                    print("[FATAL] Camera disconnected.")
                    break
                time.sleep(0.01)
                continue
            frame_failures = 0
            frame = cv2.flip(frame, 1)

            # Hand inference
            landmarks = estimator.process(frame)
            extended = 0
            gesture: str = "NONE"

            if landmarks is not None:
                extended = count_extended_fingers(landmarks)
                if extended >= OPEN_HAND_MIN_FINGERS:
                    gesture = "OPEN"
                elif extended <= 1:
                    gesture = "FIST"
                else:
                    gesture = "MID"

                if show_skeleton:
                    drawing_utils.draw_landmarks(
                        frame, landmarks, _CONNECTIONS, _DRAW_STYLE,
                    )

            now = time.perf_counter()

            # Gesture → action
            if gesture != prev_gesture and gesture != "MID" and gesture != "NONE":
                if (now - last_action) >= GESTURE_COOLDOWN_SEC:
                    jump_key = KEY_SLIDE if keys_swapped else KEY_JUMP
                    slide_key = KEY_JUMP if keys_swapped else KEY_SLIDE

                    if gesture == "OPEN":
                        pydirectinput.press(jump_key)
                        last_action = now
                        state = "JUMP!"
                    elif gesture == "FIST":
                        pydirectinput.press(slide_key)
                        last_action = now
                        state = "SLIDE!"
                    prev_gesture = gesture  # only consume on successful action
            elif gesture == "MID" or gesture == "NONE":
                prev_gesture = None  # reset so we can re-trigger same gesture
                state = "READY"

            # FPS
            fc += 1
            if now - last_fps_update >= fps_interval:
                elapsed = now - t0
                if elapsed > 0:
                    instant = fc / elapsed
                    fps = fps * (1 - fps_alpha) + instant * fps_alpha
                t0 = now
                fc = 0
                last_fps_update = now

            # HUD
            draw_hud(frame, state, gesture, extended, fps)

            cv2.imshow("Hand Cookie Run", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("c"):
                keys_swapped = not keys_swapped
                jk = KEY_SLIDE if keys_swapped else KEY_JUMP
                sk = KEY_JUMP if keys_swapped else KEY_SLIDE
                print(f"Keys swapped  |  Open={jk}  Fist={sk}")
            elif k == ord("l"):
                show_skeleton = not show_skeleton
                print(f"Skeleton overlay: {'ON' if show_skeleton else 'OFF'}")

    finally:
        estimator.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown.")


if __name__ == "__main__":
    main()
