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
Version: 1.5.0 — closest-hand selection, per-finger thresholds, latency meter
"""

from __future__ import annotations

import math
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
import pydirectinput  # type: ignore
from mediapipe.tasks.python import BaseOptions  # type: ignore
from mediapipe.tasks.python.vision import (  # type: ignore
    RunningMode,
    drawing_styles,
    drawing_utils,
)
from mediapipe.tasks.python.vision.hand_landmarker import (  # type: ignore
    HandLandmark,
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 320
FRAME_HEIGHT: int = 240
CAMERA_FPS: int = 60
MAX_HANDS: int = 1  # 1 hand is much faster for tracking without hand-switching overhead

GESTURE_COOLDOWN_SEC: float = 0.25
CAMERA_WARMUP_SEC: float = 5.0  # max time to wait for the first camera frame
GESTURE_STABLE_SEC: float = 0.08  # reduced threshold for instantaneous gesture response
MAX_FRAME_FAILURES: int = 30  # consecutive read failures before giving up on the camera

# A hand must be large enough in the frame to be trustworthy — a tiny hand far
# from the camera produces noisy landmarks that misfire gestures.
MIN_PALM_SIZE: float = 0.04  # normalized palm length (wrist → middle MCP)

OPEN_HAND_MIN_FINGERS: int = 3  # how many fingers extended to count as "open palm"
FIST_MAX_FINGERS: int = 1  # at most this many fingers extended to count as "fist"

# Hand-openness vetoes (average fingertip spread, in palm-size units). These
# only override a borderline finger count in clearly extreme poses, so relaxed
# open hands still register while bunched/spread poses can't misfire.
OPENNESS_OPEN_MIN: float = 0.80  # below this, don't trust a high count (claw/grab)
OPENNESS_FIST_MAX: float = 1.20  # above this, don't trust a low count (flat/splayed)

# Rotation-invariant, scale-normalized finger-extension thresholds, expressed
# as fractions of the palm size (wrist → middle-finger MCP distance).
THUMB_EXTEND_RATIO: float = 0.85  # thumb tip ≥85% of palm length from index MCP = extended

KEY_JUMP: str = "space"
KEY_SLIDE: str = "down"

# BGR colours
C_YELLOW = (0, 255, 255)
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_WHITE = (255, 255, 255)
C_ORANGE = (0, 165, 255)
C_CYAN = (255, 255, 0)

# Hand landmark indices, pre-resolved to plain ints so the per-frame hot path
# never touches the enum again. Triples are (tip, pip, per-finger threshold).
# The ring and pinky naturally curl a little even in a relaxed open hand, so
# their thresholds are lower (they need less reach to count as extended).
_FINGER_PAIRS: list[tuple[int, int, float]] = [
    (int(HandLandmark.INDEX_FINGER_TIP), int(HandLandmark.INDEX_FINGER_PIP), 0.22),
    (int(HandLandmark.MIDDLE_FINGER_TIP), int(HandLandmark.MIDDLE_FINGER_PIP), 0.22),
    (int(HandLandmark.RING_FINGER_TIP), int(HandLandmark.RING_FINGER_PIP), 0.20),
    (int(HandLandmark.PINKY_TIP), int(HandLandmark.PINKY_PIP), 0.18),
]
_THUMB_TIP = int(HandLandmark.THUMB_TIP)
_INDEX_MCP = int(HandLandmark.INDEX_FINGER_MCP)
_WRIST = int(HandLandmark.WRIST)
_MIDDLE_MCP = int(HandLandmark.MIDDLE_FINGER_MCP)
_ALL_TIPS = [_THUMB_TIP] + [tip for tip, _pip, _ratio in _FINGER_PAIRS]  # the 5 fingertips

# Drawing
_DRAW_STYLE = drawing_styles.get_default_hand_landmarks_style()
_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

# Font cache
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FS_S = 0.45
_FS_M = 0.6
_FS_L = 1.0

# FPS meter — per-frame exponential smoothing (alpha per frame at ~30 fps)
FPS_ALPHA: float = 0.05

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
        except (OSError, urllib.error.URLError) as exc:
            print(f"[FATAL] Model download failed: {exc}")
            if _MODEL_PATH.exists():
                _MODEL_PATH.unlink()
            return None
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# Camera feed (background thread)
# ---------------------------------------------------------------------------
class CameraFeed:
    """Threaded camera reader.

    ``cap.read()`` can stall (driver buffering, USB hiccups, Windows DirectShow
    quirks). Running it on a dedicated thread keeps the inference loop at full
    speed and always consuming the *newest* frame — stale frames are dropped
    instead of queued, which cuts latency.
    """

    __slots__ = ("_cap", "_latest", "_lock", "_ok", "_running", "_thread")

    def __init__(self, index: int, width: int, height: int, fps: int = 30) -> None:
        self._cap = self._open_capture(index, width, height, fps)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._ok = self._cap.isOpened()
        self._running = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def _open_capture(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
        """Open the camera preferring MJPG, and verify the resolution stuck.

        Some drivers accept MJPG but silently remap it to another resolution
        (e.g. 640×480), which would *hurt* FPS. If the requested resolution
        didn't stick, reopen with the default format instead.
        """
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        # Best effort — ignored silently by drivers that don't support it.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Minimal driver-side buffer → frames are as fresh as possible.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            got_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            got_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            remapped = (got_w and abs(got_w - width) > 2) or \
                (got_h and abs(got_h - height) > 2)
            if remapped:  # usually MJPG's fault → retry with the default format
                cap.release()
                cap = cv2.VideoCapture(index, cv2.CAP_ANY)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @property
    def is_opened(self) -> bool:
        with self._lock:
            return self._ok

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, name="camera-read", daemon=True
        )
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._ok = False
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)  # mirror once, here, not in the hot loop
            with self._lock:
                self._ok = True
                self._latest = frame

    def latest(self) -> tuple[np.ndarray | None, bool]:
        """Return the newest frame and read status. Never blocks."""
        with self._lock:
            return self._latest, self._ok

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._cap.release()


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
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = HandLandmarker.create_from_options(opts)
        self._t0 = time.perf_counter()

    def process(self, bgr: np.ndarray) -> Sequence | None:
        """Run inference.

        Returns the 21 landmarks of the *closest* detected hand (the largest
        palm in the frame), or None if no hand is visible. When two hands are
        in view, the one nearest the camera is almost always the one the user
        is controlling with — the other is ignored instead of fighting it.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        elapsed_ms = int((time.perf_counter() - self._t0) * 1000)
        res = self._landmarker.detect_for_video(mp_image, elapsed_ms)
        if res.hand_landmarks:
            return self._best_hand(res.hand_landmarks)
        return None

    @staticmethod
    def _best_hand(hands: Sequence[Sequence]) -> Sequence | None:
        """Pick the hand with the largest palm span (≈ closest to the camera)."""
        best: Sequence | None = None
        best_span = -1.0
        for hand in hands:
            span = (hand[_MIDDLE_MCP].x - hand[_WRIST].x) ** 2 + \
                (hand[_MIDDLE_MCP].y - hand[_WRIST].y) ** 2
            if span > best_span:
                best_span = span
                best = hand
        return best

    def close(self) -> None:
        self._landmarker.close()


# ---------------------------------------------------------------------------
# Gesture detection
# ---------------------------------------------------------------------------
def count_extended_fingers(landmarks: Sequence, size: float | None = None) -> int:
    """Count extended fingers, invariant to hand rotation and distance.

    Screen-space Y comparisons break as soon as the hand is tilted. Instead,
    each fingertip is projected along the hand's own axis (wrist → middle MCP)
    and normalized by the palm size, so the thresholds hold whether the hand
    is upright, sideways, near or far. The thumb spreads sideways, so it is
    measured by its reach from the index-finger MCP instead.

    ``size`` may be passed in to avoid recomputing the palm size (hot path).
    HandLandmarker always returns 21 landmarks, so plain indexing is safe.
    """
    lm = landmarks

    # Hand axis: from wrist toward the middle-finger MCP.
    size = _palm_size(lm) if size is None else size
    if size <= 1e-6:
        # Degenerate pose (e.g. hand pointed straight at the camera): no
        # reliable reading. Return the "no action" MID count instead of 0,
        # which would be classified as FIST and could fire an unintended key.
        return OPEN_HAND_MIN_FINGERS - 1
    ax = (lm[_MIDDLE_MCP].x - lm[_WRIST].x) / size
    ay = (lm[_MIDDLE_MCP].y - lm[_WRIST].y) / size

    n = 0
    for tip, pip, ratio in _FINGER_PAIRS:
        # How far the fingertip extends past its PIP along the hand axis,
        # compared against that finger's own threshold.
        proj = (lm[tip].x - lm[pip].x) * ax + (lm[tip].y - lm[pip].y) * ay
        if proj / size > ratio:
            n += 1

    # Thumb: an open thumb reaches far from the index-finger MCP.
    tx = lm[_THUMB_TIP].x - lm[_INDEX_MCP].x
    ty = lm[_THUMB_TIP].y - lm[_INDEX_MCP].y
    if math.hypot(tx, ty) / size > THUMB_EXTEND_RATIO:
        n += 1
    return n


def _palm_size(landmarks: Sequence) -> float:
    """Distance wrist → middle-finger MCP, in normalized units (0 = degenerate)."""
    return math.hypot(
        landmarks[_MIDDLE_MCP].x - landmarks[_WRIST].x,
        landmarks[_MIDDLE_MCP].y - landmarks[_WRIST].y,
    )


def hand_openness(landmarks: Sequence, size: float | None = None) -> float:
    """Average fingertip spread from the palm centre, in palm-size units.

    High for an open palm, low for a fist. Scale-normalized and
    rotation-invariant, so it doubles as a confidence signal for the count.
    ``size`` may be passed in to avoid recomputing the palm size (hot path).
    """
    size = _palm_size(landmarks) if size is None else size
    if size <= 1e-6:
        return 0.0
    cx = (landmarks[_WRIST].x + landmarks[_MIDDLE_MCP].x) * 0.5
    cy = (landmarks[_WRIST].y + landmarks[_MIDDLE_MCP].y) * 0.5
    total = sum(
        math.hypot(landmarks[t].x - cx, landmarks[t].y - cy) for t in _ALL_TIPS
    )
    return total / len(_ALL_TIPS) / size


def classify_gesture(landmarks: Sequence) -> tuple[int, float, str]:
    """Finger count + hand openness → (count, openness, raw gesture label).

    Openness acts as a *veto*: a clearly bunched hand (low openness) can't fire
    a jump even with a borderline count, and a clearly spread hand (high
    openness) can't fire a slide. Relaxed open palms still pass. Hands too
    small in the frame (far away) are unreliable and read as "NONE" so they
    can't misfire a jump or slide.
    """
    size = _palm_size(landmarks)
    if size < MIN_PALM_SIZE:
        return 0, 0.0, "NONE"
    extended = count_extended_fingers(landmarks, size)
    openness = hand_openness(landmarks, size)
    if extended >= OPEN_HAND_MIN_FINGERS and openness >= OPENNESS_OPEN_MIN:
        return extended, openness, "OPEN"
    if extended <= FIST_MAX_FINGERS and openness <= OPENNESS_FIST_MAX:
        return extended, openness, "FIST"
    return extended, openness, "MID"


class GestureDebouncer:
    """Smooths raw gestures with near-zero latency on real transitions.

    A gesture held for GESTURE_STABLE_SEC is trusted: the next change is
    accepted on its first frame, so genuine jumps/slides register immediately.
    Rapid toggling needs two consecutive frames, and MID/NONE always need two,
    so a single flickered frame can never leak through and reset the action
    state machine.
    """

    __slots__ = ("_last_change", "_pending", "_pending_raw", "_stable")

    def __init__(self) -> None:
        self._stable: str = "NONE"
        self._pending: int = 0
        self._pending_raw: str = "NONE"  # label the pending streak started with
        self._last_change: float = 0.0

    def update(self, raw: str, now: float) -> str:
        if raw == self._stable:
            self._pending = 0
            return self._stable
        if raw != self._pending_raw:
            # New label mid-streak: restart the count so confirmation truly
            # requires consecutive frames of the *same* new gesture.
            self._pending = 0
            self._pending_raw = raw
        if raw in ("MID", "NONE"):
            self._pending += 1
            if self._pending >= 2:
                self._stable = raw
                self._pending = 0
                self._last_change = now
            return self._stable

        # Real gesture (OPEN / FIST): trust immediately if the previous gesture
        # was stable long enough; otherwise require 2 frames to settle.
        trusted = (now - self._last_change) >= GESTURE_STABLE_SEC
        required = 1 if trusted else 2
        self._pending += 1
        if self._pending >= required:
            self._stable = raw
            self._pending = 0
            self._last_change = now

        return self._stable


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    state: str,
    gesture: str,
    extended_count: int,
    openness: float,
    fps: float,
    lat_ms: float,
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

    # Finger count + hand openness (tuning aid)
    cv2.putText(frame, f"FINGERS: {extended_count}/5  SPREAD: {openness * 100:.0f}%",
                (6, h + 2), _FONT, _FS_S, C_CYAN, 1)
    h += 22

    # FPS + inference latency (top-right)
    fps_str = f"FPS:{fps:.0f}  LAT:{lat_ms:.0f}ms"
    (tw, _), _ = cv2.getTextSize(fps_str, _FONT, _FS_S, 1)
    cv2.putText(frame, fps_str, (fw - tw - 6, 16), _FONT, _FS_S, C_GREEN, 1)

    # Key legend (bottom)
    cv2.putText(frame, "c:swap-keys  l:skeleton  q:quit",
                (4, fh - 6), _FONT, _FS_S, C_WHITE, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    feed = CameraFeed(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CAMERA_FPS)
    if not feed.is_opened:
        print(f"[FATAL] Camera {CAMERA_INDEX} unavailable.")
        feed.stop()
        return

    print("Camera ready  |  c=swap-keys  l=toggle-skeleton  q=quit")
    print("Open palm → Jump  |  Fist → Slide")

    try:
        estimator = HandEstimator()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        feed.stop()
        return

    run_controller(feed, estimator)


def run_controller(
    feed: CameraFeed,
    estimator: HandEstimator,
    *,
    show_window: bool = True,
    window_title: str = "Hand Cookie Run",
) -> None:
    """Run the capture → inference → gesture → input pipeline until quit.

    ``feed`` must expose ``start()``, ``latest() -> (frame, ok)`` and ``stop()``
    (e.g. :class:`CameraFeed`); ``estimator`` must expose
    ``process(frame) -> landmarks | None`` and ``close()``
    (e.g. :class:`HandEstimator`). ``show_window=False`` runs headless, e.g.
    embedded in another script.
    """
    feed.start()

    # Wait for the very first frame. Tolerate transient read failures during
    # camera warm-up (common on Windows right after opening the device).
    deadline = time.perf_counter() + CAMERA_WARMUP_SEC
    frame: np.ndarray | None = None
    while frame is None:
        frame, ok = feed.latest()
        if not ok and time.perf_counter() >= deadline:
            print("[FATAL] Camera disconnected.")
            estimator.close()
            feed.stop()
            return
        time.sleep(0.01)

    last_action: float = 0.0
    state: str = "READY"
    show_skeleton: bool = True
    jump_key: str = KEY_JUMP
    slide_key: str = KEY_SLIDE

    fps = 0.0
    lat_ms = 0.0
    prev_t = time.perf_counter()
    frame_failures = 0

    prev_gesture: str | None = None
    debouncer = GestureDebouncer()

    try:
        last_frame: np.ndarray | None = None
        while True:
            frame, ok = feed.latest()
            if not ok or frame is None:
                frame_failures += 1
                if frame_failures >= MAX_FRAME_FAILURES:
                    print("[FATAL] Camera disconnected.")
                    break
                time.sleep(0.005)
                continue
            frame_failures = 0

            # Only process *new* frames — if the camera delivers slower than
            # inference, don't waste a pass re-analyzing a frame we've seen.
            if frame is not last_frame:
                last_frame = frame

                # Hand inference (timed for the latency meter)
                t0 = time.perf_counter()
                landmarks = estimator.process(frame)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                lat_ms = lat_ms * (1.0 - FPS_ALPHA) + infer_ms * FPS_ALPHA
                now = time.perf_counter()
                extended = 0
                openness = 0.0
                gesture: str = "NONE"

                if landmarks is not None:
                    extended, openness, gesture = classify_gesture(landmarks)

                # Debounce: trust genuine transitions (the previous gesture was
                # held long enough) but require two frames for rapid toggles and
                # MID/NONE, so a single flickered frame can never leak through.
                gesture = debouncer.update(gesture, now)

                if show_skeleton and landmarks is not None:
                    drawing_utils.draw_landmarks(
                        frame, landmarks, _CONNECTIONS, _DRAW_STYLE,
                    )

                # Gesture → action
                if gesture != prev_gesture and gesture != "MID" and gesture != "NONE":
                    if (now - last_action) >= GESTURE_COOLDOWN_SEC:
                        if gesture == "OPEN":
                            threading.Thread(target=pydirectinput.press, args=(jump_key,), daemon=True).start()
                            last_action = now
                            state = "JUMP!"
                        elif gesture == "FIST":
                            threading.Thread(target=pydirectinput.press, args=(slide_key,), daemon=True).start()
                            last_action = now
                            state = "SLIDE!"
                        prev_gesture = gesture  # only consume on successful action
                elif gesture == "MID" or gesture == "NONE":
                    prev_gesture = None  # reset so we can re-trigger same gesture
                    state = "READY"

                # FPS — per-frame exponential smoothing (no interval bookkeeping)
                dt = now - prev_t
                if dt > 0:
                    fps = fps * (1.0 - FPS_ALPHA) + (1.0 / dt) * FPS_ALPHA
                prev_t = now

                # HUD
                draw_hud(frame, state, gesture, extended, openness, fps, lat_ms)

                if show_window:
                    cv2.imshow(window_title, frame)

            if show_window:
                k = cv2.waitKey(1) & 0xFF
            else:
                k = 0
                time.sleep(0.001)  # don't busy-spin in headless mode

            if k == ord("q"):
                break
            elif k == ord("c"):
                jump_key, slide_key = slide_key, jump_key
                print(f"Keys swapped  |  Open={jump_key}  Fist={slide_key}")
            elif k == ord("l"):
                show_skeleton = not show_skeleton
                print(f"Skeleton overlay: {'ON' if show_skeleton else 'OFF'}")

    finally:
        feed.stop()
        estimator.close()
        if show_window:
            cv2.destroyAllWindows()
        print("Shutdown.")


if __name__ == "__main__":
    main()