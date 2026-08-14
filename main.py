"""
AI Hand-Gesture Cookie Run Game Controller
==========================================
High-performance hand-gesture controller using OpenCV, MediaPipe Tasks API
(HandLandmarker), and PyDirectInput for real-time game input.

FINGER-CONFIGURABLE CONTROLS
----------------------------
On first launch a SETUP SCREEN lets you assign ANY finger (or CHORD of
fingers) to each action. For each step, raise ONE finger or a POSE of several
fingers together (e.g. index+middle) and press SPACE — every finger currently
raised is captured at once. Raise more poses and press SPACE again, then press
ENTER when done:

    Step 1  JUMP   — the finger(s) that make Cookie Run jump
    Step 2  SLIDE  — the finger(s) that make Cookie Run slide (held while raised)
    Step 3  READY  — the finger(s) that represent a neutral / resting pose

Bind ONE finger or a CHORD of several fingers to each action (e.g. JUMP =
INDEX+MIDDLE together, SLIDE = RING alone, READY = PINKY). After setup the
controller runs with your mapping:

    * JUMP  fires on the *rising edge* of a complete JUMP chord (tap → double-jump)
    * SLIDE holds the slide key while the complete SLIDE chord stays raised (hold)
    * READY wins over everything: a complete READY chord forces neutral
    * A complete JUMP and SLIDE chord at the same time = ambiguous → neutral
    * Raising only PART of a chord fires nothing — the whole chord must be up

Smoothness & accuracy:
    * Per-finger EMA smoothing + wide hysteresis + 4-frame stability confirmation
    * Gesture-hold grace prevents a 1-frame flicker from stuttering a slide
      or re-firing a jump, while fast double-taps stay responsive
    * Tiny / far-away hands are ignored so noisy landmarks can't misfire
    * The thumb uses a dedicated distance-based reach metric (it abducts
      across the palm instead of extending along one bone), so a resting
      thumb can't be misread as raised
    * Lower MediaPipe confidence thresholds (0.3) make the hand easier to find
      even in unusual poses, and the setup register threshold (0.80) means a
      comfortably-raised finger is captured without needing to be fully rigid

On-screen / keyboard controls:
    setup:  space = capture pose    enter = finish action    esc = undo
            c = restart    q = quit
    play:   r = reconfigure    c = swap keys  l = skeleton   q = quit

Author : Tanakorn Uamraksa (GitHub Copilot)
Version: 3.5.0 — finger-configurable controls (accurate pose capture, thumb fix)
"""

from __future__ import annotations

import math
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
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
FRAME_WIDTH: int = 480  # higher resolution → more accurate landmarks
FRAME_HEIGHT: int = 360
CAMERA_FPS: int = 60
MAX_HANDS: int = 1  # 1 hand is much faster for tracking without hand-switching overhead

CAMERA_WARMUP_SEC: float = 5.0  # max time to wait for the first camera frame
GESTURE_COOLDOWN_SEC: float = 0.25  # minimum gap between two jump taps
MAX_FRAME_FAILURES: int = 30  # consecutive read failures before giving up

# Per-finger extension smoothing & debounce (orientation-invariant, scale-invariant)
FINGER_SMOOTH_ALPHA: float = 0.35  # EMA alpha per finger — lower = smoother, less jitter
EXTEND_ON_RATIO: float = 0.70     # extension ratio above this = finger "up"
EXTEND_OFF_RATIO: float = 0.35    # below this = finger "down" (wider hysteresis)
CONFIRM_FRAMES: int = 4           # frames a finger must hold a state before commit
HAND_LOST_TOLERANCE: int = 5      # lost-hand frames before finger states reset

# Gesture hold — once JUMP/SLIDE fires, keep it sticky for this many frames so
# micro-flickers can't stutter a held slide key or re-fire a jump.
GESTURE_HOLD_FRAMES: int = 6

# A hand must be large enough in the frame to be trustworthy — a tiny hand far
# from the camera produces noisy landmarks that misfire finger states.
MIN_PALM_SIZE: float = 0.05  # normalized palm size (wrist → middle MCP)

# The thumb abducts across the palm instead of extending along one bone, so a
# bone-axis projection is unreliable. These thresholds use the dedicated
# distance-based thumb metric (see `thumb_extension`): a resting thumb is
# ≈0.5–0.9 palm-units from the index MCP, an extended thumb ≈1.3–2.0.
THUMB_EXTEND_ON_RATIO: float = 1.10  # thumb reach ≥ this = thumb "up"
THUMB_EXTEND_OFF_RATIO: float = 0.85 # below this = thumb "down" (hysteresis)

# Setup screen — pose capture registers comfortably-raised fingers. A finger
# counts as raised past SELECT_RATIO and stays raised until it drops below
# SELECT_OFF_RATIO (hysteresis). A resting/crawled finger reads ≈0.2–0.6,
# while a raised finger reads ≈0.8+ — so the 0.80 threshold captures the pose
# easily without letting resting fingers sneak in.
SETUP_SMOOTH_ALPHA: float = 0.35   # display smoothing for the live bar read-out
SETUP_SELECT_RATIO: float = 0.80   # min extension to count a finger as raised
SETUP_SELECT_OFF_RATIO: float = 0.55  # below this a raised finger is released
SETUP_CONFIRM_FRAMES: int = 4      # frames the same pose must persist to be "confirmed"

KEY_JUMP: str = "space"
KEY_SLIDE: str = "down"

# BGR colours
C_YELLOW = (0, 255, 255)
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_WHITE = (255, 255, 255)
C_CYAN = (255, 255, 0)
C_GREY = (128, 128, 128)

# Finger identity -----------------------------------------------------------
FINGER_THUMB = 0
FINGER_INDEX = 1
FINGER_MIDDLE = 2
FINGER_RING = 3
FINGER_PINKY = 4
_FINGER_NAMES: tuple[str, ...] = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")

# (tip, pivot, base) landmark indices per finger. The pivot/base form the bone
# axis used to measure extension. The thumb has no PIP → use IP instead.
_FINGER_LANDMARKS: list[tuple[int, int, int]] = [
    (int(HandLandmark.THUMB_TIP), int(HandLandmark.THUMB_IP), int(HandLandmark.THUMB_MCP)),
    (int(HandLandmark.INDEX_FINGER_TIP), int(HandLandmark.INDEX_FINGER_PIP), int(HandLandmark.INDEX_FINGER_MCP)),
    (int(HandLandmark.MIDDLE_FINGER_TIP), int(HandLandmark.MIDDLE_FINGER_PIP), int(HandLandmark.MIDDLE_FINGER_MCP)),
    (int(HandLandmark.RING_FINGER_TIP), int(HandLandmark.RING_FINGER_PIP), int(HandLandmark.RING_FINGER_MCP)),
    (int(HandLandmark.PINKY_TIP), int(HandLandmark.PINKY_PIP), int(HandLandmark.PINKY_MCP)),
]

# Landmarks used by the closest-hand picker + thumb metric
_WRIST = int(HandLandmark.WRIST)
_MIDDLE_MCP = int(HandLandmark.MIDDLE_FINGER_MCP)
_THUMB_TIP = int(HandLandmark.THUMB_TIP)
_INDEX_MCP = int(HandLandmark.INDEX_FINGER_MCP)

# Drawing
_DRAW_STYLE = drawing_styles.get_default_hand_landmarks_style()
_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
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

# Sentinel returned by run_controller() when the user asks to reconfigure
RECONFIGURE = object()


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
        """Start the reader thread. Idempotent — safe to call repeatedly."""
        if self._running:
            return
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
            # Lower confidence values make hand detection easier in unusual
            # poses (fists, peace signs, angled hands) or slightly out of frame.
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
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
# Per-finger extension measurement
# ---------------------------------------------------------------------------
def finger_extension(
    landmarks: Sequence,
    tip_idx: int,
    pivot_idx: int,
    base_idx: int,
) -> float:
    """How far the fingertip extends past its pivot, along the base→pivot axis,
    normalized by the base→pivot bone length.

    ≈1.4–2.2 when the finger is straight, ≈0 when relaxed, negative when
    curled. Because it is normalized by the bone length it is robust to hand
    distance, hand size, and hand rotation. Only used for the four long
    fingers (index→pinky) — the thumb abducts across the palm and uses the
    dedicated :func:`thumb_extension` metric instead.
    """
    if len(landmarks) <= base_idx:
        return 0.0
    tip = landmarks[tip_idx]
    pivot = landmarks[pivot_idx]
    base = landmarks[base_idx]
    vx = pivot.x - base.x
    vy = pivot.y - base.y
    seg = math.hypot(vx, vy)
    if seg < 1e-5:
        return 0.0
    ux, uy = vx / seg, vy / seg
    tx = tip.x - pivot.x
    ty = tip.y - pivot.y
    return (tx * ux + ty * uy) / seg


def thumb_extension(landmarks: Sequence, size: float | None = None) -> float:
    """Dedicated thumb reach metric, normalized by palm size.

    The thumb abducts *across* the palm instead of extending along one bone,
    so a bone-axis projection is unreliable (a resting thumb can score as if
    extended). Instead we measure the distance from the thumb tip to the
    index-finger MCP, normalized by the palm size (wrist → middle MCP):

        resting thumb  → ≈0.5–0.9
        extended thumb → ≈1.3–2.0

    ``size`` may be passed in to avoid recomputing the palm size.
    """
    if len(landmarks) <= _INDEX_MCP:
        return 0.0
    size = _palm_size(landmarks) if size is None else size
    if size <= 1e-6:
        return 0.0
    return math.hypot(
        landmarks[_THUMB_TIP].x - landmarks[_INDEX_MCP].x,
        landmarks[_THUMB_TIP].y - landmarks[_INDEX_MCP].y,
    ) / size


class FingerDebouncer:
    """Smooths + debounces a single finger's extension signal.

    Pipeline: EMA smoothing → hysteresis → stability confirmation. A finger
    only flips state after holding the new state for `confirm` consecutive
    frames, which eliminates jitter-triggered false actions while keeping real
    movements snappy.
    """

    __slots__ = ("on_ratio", "off_ratio", "confirm", "extended", "smooth", "streak")

    def __init__(
        self,
        on_ratio: float = EXTEND_ON_RATIO,
        off_ratio: float = EXTEND_OFF_RATIO,
        confirm: int = CONFIRM_FRAMES,
    ) -> None:
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self.confirm = max(1, confirm)
        self.extended = False
        self.smooth = 0.0
        self.streak = 0

    def update(self, raw: float) -> bool:
        # 1) EMA smoothing
        self.smooth += FINGER_SMOOTH_ALPHA * (raw - self.smooth)

        # 2) hysteresis target
        target = self.extended
        if self.extended:
            if self.smooth < self.off_ratio:
                target = False
        elif self.smooth > self.on_ratio:
            target = True

        # 3) stability confirmation
        if target == self.extended:
            self.streak = 0
        else:
            self.streak += 1
            if self.streak >= self.confirm:
                self.extended = target
                self.streak = 0
        return self.extended

    def force(self, value: bool) -> None:
        """Immediately set the state (used when the hand is lost)."""
        self.extended = value
        self.streak = 0


def _palm_size(landmarks: Sequence) -> float:
    """Normalized palm size (wrist → middle MCP distance); 0 = degenerate."""
    return math.hypot(
        landmarks[_MIDDLE_MCP].x - landmarks[_WRIST].x,
        landmarks[_MIDDLE_MCP].y - landmarks[_WRIST].y,
    )


class FingerState:
    """Tracks all 5 fingers with per-finger smoothing/debounce + lost-hand reset.

    The thumb uses its own distance-based metric (:func:`thumb_extension`)
    with dedicated thresholds, because a bone-axis projection is unreliable
    for the abduction-moving thumb. The four long fingers use the standard
    bone-axis projection.
    """

    __slots__ = ("_db", "extension", "extended", "lost")

    def __init__(self) -> None:
        self._db = [
            FingerDebouncer(on_ratio=THUMB_EXTEND_ON_RATIO,
                            off_ratio=THUMB_EXTEND_OFF_RATIO),
            FingerDebouncer(), FingerDebouncer(), FingerDebouncer(), FingerDebouncer(),
        ]
        self.extension = [0.0] * 5   # smoothed raw values (for the HUD)
        self.extended = [False] * 5  # committed up/down states
        self.lost = 0

    def update(self, landmarks: Sequence | None, lost_tolerance: int) -> None:
        # A hand too small in the frame (far from the camera) produces noisy
        # landmarks — treat it as absent so it can't misfire a finger state.
        if landmarks is not None and _palm_size(landmarks) >= MIN_PALM_SIZE:
            self.lost = 0
            size = _palm_size(landmarks)
            self.extension[0] = thumb_extension(landmarks, size)
            self.extended[0] = self._db[0].update(self.extension[0])
            for i in range(1, 5):
                tip, pivot, base = _FINGER_LANDMARKS[i]
                self.extension[i] = finger_extension(landmarks, tip, pivot, base)
                self.extended[i] = self._db[i].update(self.extension[i])
        else:
            self.lost += 1
            if self.lost >= lost_tolerance:
                for db in self._db:
                    db.force(False)
                self.extension = [0.0] * 5
                self.extended = [False] * 5


class GestureHold:
    """Sticky gesture grace period to eliminate micro-stutter.

    Once JUMP/SLIDE fires it is held for ``hold_frames`` even if the raw
    classification flickers to "IDLE" (finger dipped below threshold) — so a
    held slide key never stutters off/on. An explicit READY finger raise, an
    ambiguous NONE, or a different action gesture overrides the hold
    immediately; a sustained IDLE releases it after the grace expires.
    """

    __slots__ = ("hold_frames", "_sticky", "_remaining")

    def __init__(self, hold_frames: int = GESTURE_HOLD_FRAMES) -> None:
        self.hold_frames = max(1, hold_frames)
        self._sticky: str = "IDLE"
        self._remaining = 0

    def update(self, raw: str) -> str:
        if raw == self._sticky:
            self._remaining = self.hold_frames  # refresh while still held
            return raw
        if raw in ("JUMP", "SLIDE", "READY", "NONE"):
            self._sticky = raw
            self._remaining = self.hold_frames
            return raw
        # raw is "IDLE" — grace period before releasing a held gesture
        if self._remaining > 0:
            self._remaining -= 1
            return self._sticky
        self._sticky = raw
        return raw


# ---------------------------------------------------------------------------
# Finger → action configuration
# ---------------------------------------------------------------------------
class FingerConfig:
    """The user-selected finger→action chord bindings plus key bindings.

    Each action binds a *set* of fingers (a chord). A 1-finger chord is just
    the old single-finger mapping; several fingers form a combo that must all
    be raised together to trigger the action.
    """

    __slots__ = ("jump_fingers", "slide_fingers", "ready_fingers", "jump_key", "slide_key")

    def __init__(
        self,
        jump_fingers: Sequence[int],
        slide_fingers: Sequence[int],
        ready_fingers: Sequence[int],
        jump_key: str = KEY_JUMP,
        slide_key: str = KEY_SLIDE,
    ) -> None:
        self.jump_fingers = frozenset(jump_fingers)
        self.slide_fingers = frozenset(slide_fingers)
        self.ready_fingers = frozenset(ready_fingers)
        self.jump_key = jump_key
        self.slide_key = slide_key

    @property
    def label(self) -> str:
        def fmt(fingers: frozenset[int]) -> str:
            return "+".join(_FINGER_NAMES[i] for i in sorted(fingers))
        return (f"J:{fmt(self.jump_fingers)}  "
                f"S:{fmt(self.slide_fingers)}  "
                f"R:{fmt(self.ready_fingers)}")


class PoseTracker:
    """Hysteresis-based raised-finger tracker for the setup screen.

    Each finger is marked "raised" when its extension crosses above
    ``SETUP_SELECT_RATIO`` and stays raised until it drops below
    ``SETUP_SELECT_OFF_RATIO``. This prevents a pose hovering around the
    on-threshold from flickering on/off — the same multi-finger pose can then
    persist for the required confirmation frames and register reliably.
    """

    __slots__ = ("on_ratio", "off_ratio", "_raised")

    def __init__(
        self,
        on_ratio: float = SETUP_SELECT_RATIO,
        off_ratio: float = SETUP_SELECT_OFF_RATIO,
    ) -> None:
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self._raised = [False] * 5

    def update(
        self,
        extensions: Sequence[float],
        unavailable: set[int] = frozenset(),
    ) -> tuple[int, ...]:
        """Return the current raised-finger tuple (sorted), with hysteresis."""
        if len(extensions) != 5:
            return ()
        for i in range(5):
            if i in unavailable:
                self._raised[i] = False
                continue
            if self._raised[i]:
                if extensions[i] < self.off_ratio:
                    self._raised[i] = False
            elif extensions[i] >= self.on_ratio:
                self._raised[i] = True
        return tuple(i for i in range(5) if self._raised[i])


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------
def _role_of(finger: int, chosen: Mapping[str, frozenset[int]]) -> str:
    for role, fingers in chosen.items():
        if finger in fingers:
            return role
    return ""


def draw_setup(
    frame: np.ndarray,
    step_name: str,
    step_hint: str,
    detected: Sequence[int],
    chosen: Mapping[str, frozenset[int]],
    ext: Sequence[float],
    fps: float,
) -> None:
    fh, fw = frame.shape[:2]

    cv2.putText(frame, "=== FINGER CONTROL SETUP ===",
                (6, 26), _FONT, _FS_L * 0.8, C_WHITE, 2)
    cv2.putText(frame, f"STEP: {step_name}", (6, 52), _FONT, _FS_M, C_CYAN, 2)
    cv2.putText(frame, step_hint, (6, 72), _FONT, _FS_S, C_WHITE, 1)

    if detected:
        names = "+".join(_FINGER_NAMES[i] for i in detected)
        text = f"DETECTED: {names}   [space] to add"
        colour = C_GREEN
    else:
        text = "DETECTED: ---   (raise the pose, then press space)"
        colour = C_YELLOW
    cv2.putText(frame, text, (6, 106), _FONT, _FS_M * 1.4, colour, 2)

    y = 134
    for i, name in enumerate(_FINGER_NAMES):
        role = _role_of(i, chosen)
        if role:
            label = f"{name} → {role}"
            col = C_GREEN
        elif i in detected:
            label = f"{name}"
            col = C_CYAN
        else:
            label = f"{name}"
            col = C_GREY
        cv2.putText(frame, f"{label:<10}{ext[i]:+.2f}", (6, y), _FONT, _FS_S, col, 1)
        y += 22

    # FPS (top-right)
    fps_str = f"FPS:{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, _FONT, _FS_S, 1)
    cv2.putText(frame, fps_str, (fw - tw - 6, 16), _FONT, _FS_S, C_GREEN, 1)

    # Legend (bottom)
    cv2.putText(frame, "space:capture-pose  enter:done  esc:undo  c:restart  q:quit",
                (4, fh - 6), _FONT, _FS_S, C_WHITE, 1)


def draw_hud(
    frame: np.ndarray,
    state: str,
    gesture: str,
    finger_state: FingerState,
    config: FingerConfig,
    fps: float,
) -> None:
    fh, fw = frame.shape[:2]
    h = 20

    # State bar (top)
    cv2.putText(frame, state, (6, h + 2), _FONT, _FS_M, C_WHITE, 2)
    h += 28

    # Gesture label
    if gesture == "JUMP":
        colour = C_GREEN
    elif gesture == "SLIDE":
        colour = C_RED
    else:
        colour = C_YELLOW
    cv2.putText(frame, f"GESTURE: {gesture}", (6, h + 2), _FONT, _FS_M, colour, 2)
    h += 28

    # Active finger mapping
    cv2.putText(frame, config.label, (6, h + 2), _FONT, _FS_S, C_CYAN, 1)
    h += 22

    # Per-finger live read-out (tuning aid)
    for i, name in enumerate(_FINGER_NAMES):
        up = finger_state.extended[i]
        role = ""
        if i in config.jump_fingers:
            role = " [JUMP]"
        elif i in config.slide_fingers:
            role = " [SLIDE]"
        elif i in config.ready_fingers:
            role = " [READY]"
        col = C_GREEN if up else C_GREY
        cv2.putText(frame,
                    f"{name:<6}{'UP' if up else 'down'} {finger_state.extension[i]:4.2f}{role}",
                    (6, h + 2), _FONT, _FS_S, col, 1)
        h += 20

    # FPS + latency tag (top-right)
    fps_str = f"FPS:{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, _FONT, _FS_S, 1)
    cv2.putText(frame, fps_str, (fw - tw - 6, 16), _FONT, _FS_S, C_GREEN, 1)

    # Key legend (bottom)
    cv2.putText(frame, "r:setup  c:swap-keys  l:skeleton  q:quit",
                (4, fh - 6), _FONT, _FS_S, C_WHITE, 1)


# ---------------------------------------------------------------------------
# Setup screen (finger-assignment wizard)
# ---------------------------------------------------------------------------
_SETUP_STEPS: tuple[tuple[str, str], ...] = (
    ("JUMP", "Raise the JUMP pose (1+ fingers) — SPACE captures it, ENTER when done"),
    ("SLIDE", "Raise the SLIDE pose (held) — SPACE captures it, ENTER when done"),
    ("READY", "Raise the READY pose (neutral) — SPACE captures it, ENTER when done"),
)


def run_setup(
    feed: CameraFeed,
    estimator: HandEstimator,
    *,
    current: FingerConfig | None = None,
    show_window: bool = True,
    window_title: str = "Hand Cookie Run — Setup",
) -> FingerConfig | None:
    """Interactive finger-assignment screen.

    Walks the user through JUMP → SLIDE → READY and returns the resulting
    :class:`FingerConfig`, or None if the user quits. The ``current`` config
    (if any) is used to carry over the user's key bindings across reconfigures.
    The feed must already be started (``feed.start()`` is idempotent).
    """
    feed.start()

    # Wait for the first frame (tolerate transient failures during warm-up)
    deadline = time.perf_counter() + CAMERA_WARMUP_SEC
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
    pose_tracker = PoseTracker()
    last_detected: frozenset[int] = frozenset()
    detected_streak = 0
    confirmed: frozenset[int] = frozenset()

    # Lightly smoothed raw extensions for the live "detected finger" read-out
    disp = [0.0] * 5
    fps = 0.0
    prev_t = time.perf_counter()
    frame_failures = 0

    try:
        while True:
            frame, ok = feed.latest()
            if not ok or frame is None:
                frame_failures += 1
                if frame_failures >= MAX_FRAME_FAILURES:
                    print("[FATAL] Camera disconnected.")
                    return None
                time.sleep(0.005)
                continue
            frame_failures = 0

            landmarks = estimator.process(frame)
            if landmarks is not None:
                size = _palm_size(landmarks)
                raw = (
                    [thumb_extension(landmarks, size)]
                    + [finger_extension(landmarks, *_FINGER_LANDMARKS[i])
                       for i in range(1, 5)]
                )
            else:
                raw = [0.0] * 5
            for i in range(5):
                disp[i] += SETUP_SMOOTH_ALPHA * (raw[i] - disp[i])

            # Fingers already assigned (completed actions or already added to
            # the current action) can't be re-picked.
            unavailable = {f for fs in chosen.values() for f in fs}
            unavailable.update(current_fingers)
            detected = pose_tracker.update(disp, unavailable)

            # Require the SAME pose (same finger set) to persist for
            # SETUP_CONFIRM_FRAMES before it is "confirmed" — keeps the setup
            # read-out stable and accurate.
            if not detected:
                last_detected = frozenset()
                detected_streak = 0
                confirmed = frozenset()
            elif detected == last_detected:
                detected_streak += 1
                if detected_streak >= SETUP_CONFIRM_FRAMES:
                    confirmed = frozenset(detected)
            else:
                last_detected = frozenset(detected)
                detected_streak = 1
                confirmed = frozenset()

            if show_skeleton and landmarks is not None:
                drawing_utils.draw_landmarks(
                    frame, landmarks, _CONNECTIONS, _DRAW_STYLE,
                )

            now = time.perf_counter()
            dt = now - prev_t
            if dt > 0:
                fps = fps * (1.0 - FPS_ALPHA) + (1.0 / dt) * FPS_ALPHA
            prev_t = now

            step_name, step_hint = _SETUP_STEPS[step]
            draw_setup(frame, step_name, step_hint, confirmed, chosen, disp, fps)

            if show_window:
                cv2.imshow(window_title, frame)

            if show_window:
                k = cv2.waitKey(1) & 0xFF
            else:
                k = 0
                time.sleep(0.001)  # don't busy-spin in headless mode

            if k == ord("q"):
                return None
            elif k == ord("c"):
                step = 0
                chosen.clear()
                current_fingers.clear()
                print("Setup restarted.")
            elif k == 27:  # ESC → remove the last finger added, or step back
                if current_fingers:
                    removed = current_fingers.pop()
                    last_detected = frozenset()
                    detected_streak = 0
                    confirmed = frozenset()
                    print(f"Removed {_FINGER_NAMES[removed]} from {step_name} — "
                          f"{len(current_fingers)} finger(s) in this action.")
                elif step > 0:
                    step -= 1
                    current_fingers = list(chosen.popitem()[1])
                    print(f"Editing {_SETUP_STEPS[step][0]} again — "
                          f"{len(current_fingers)} finger(s) already added.")
            elif k == ord(" ") and confirmed:  # SPACE → capture the confirmed pose
                added: list[int] = []
                for f in sorted(confirmed):
                    if f not in current_fingers:
                        current_fingers.append(f)
                        added.append(f)
                if added:
                    print(f"{step_name} += "
                          f"{'+'.join(_FINGER_NAMES[f] for f in added)} "
                          f"({len(current_fingers)} finger(s))")
                else:
                    print(f"All detected fingers already added to {step_name}.")
                # Re-arm confirmation so the user must hold the pose again
                last_detected = frozenset()
                detected_streak = 0
                confirmed = frozenset()
            elif k == 13 and current_fingers:  # ENTER → finish this action
                chosen[step_name] = frozenset(current_fingers)
                print(f"{step_name} = "
                      f"{'+'.join(_FINGER_NAMES[i] for i in sorted(current_fingers))}")
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
            elif k == 13:
                print("No fingers added yet — raise a finger and press SPACE.")

    finally:
        if show_window:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
def run_controller(
    feed: CameraFeed,
    estimator: HandEstimator,
    config: FingerConfig,
    *,
    show_window: bool = True,
    window_title: str = "Hand Cookie Run",
) -> object:
    """Run capture → inference → finger-state → input until quit or reconfig.

    Returns :data:`RECONFIGURE` when the user presses 'r' (the caller should
    re-run the setup screen), or None when the session ends (quit or camera
    failure). The feed must already be started (``feed.start()`` is
    idempotent). ``show_window=False`` runs headless, e.g. embedded in
    another script.

    Action semantics with the user's chord mapping:
        * A complete READY chord, or complete JUMP+SLIDE chords together,
          → neutral (no key)
        * A complete JUMP chord → rising edge = tap the jump key
        * A complete SLIDE chord → the slide key is *held* while raised,
          which keeps the game's hold-to-slide mechanic working.
        * Raising only PART of a chord fires nothing — all fingers assigned
          to an action must be raised together.
    """
    feed.start()

    # Wait for the first frame
    deadline = time.perf_counter() + CAMERA_WARMUP_SEC
    frame: np.ndarray | None = None
    while frame is None:
        frame, ok = feed.latest()
        if not ok and time.perf_counter() >= deadline:
            print("[FATAL] Camera disconnected.")
            return None
        time.sleep(0.01)

    finger_state = FingerState()
    gesture_hold = GestureHold()
    state: str = "READY"
    show_skeleton: bool = True

    fps = 0.0
    prev_t = time.perf_counter()
    frame_failures = 0
    last_jump: float = 0.0
    slide_holding: bool = False
    prev_raw: str = "IDLE"

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
                now = time.perf_counter()

                landmarks = estimator.process(frame)
                finger_state.update(landmarks, HAND_LOST_TOLERANCE)
                ext = finger_state.extended

                raised = {i for i in range(5) if ext[i]}

                # An action fires only when ALL of its assigned fingers are
                # raised (a complete chord). A single finger is still a
                # 1-finger chord, so the old single-finger mappings work
                # exactly as before.
                r_up = bool(config.ready_fingers and config.ready_fingers <= raised)
                j_up = bool(config.jump_fingers and config.jump_fingers <= raised)
                s_up = bool(config.slide_fingers and config.slide_fingers <= raised)

                # Classify the pose against the user's mapping. "IDLE" means no
                # complete action chord is raised (the resting default) and is
                # kept distinct from an explicit "READY" finger raise.
                if r_up:
                    raw = "READY"          # explicit resting pose → immediate neutral
                elif j_up and s_up:
                    raw = "NONE"           # two complete chords → ambiguous
                elif j_up:
                    raw = "JUMP"
                elif s_up:
                    raw = "SLIDE"
                else:
                    raw = "IDLE"           # no complete action chord raised

                # Stabilized gesture — holds JUMP/SLIDE for a short grace period
                # so micro-flickers can't stutter a held slide key.
                gesture = gesture_hold.update(raw)

                # JUMP — fire on the *raw* rising edge so fast double-taps stay
                # responsive even inside the gesture-hold grace window.
                if raw == "JUMP" and prev_raw != "JUMP":
                    if (now - last_jump) >= GESTURE_COOLDOWN_SEC:
                        threading.Thread(
                            target=pydirectinput.press,
                            args=(config.jump_key,),
                            daemon=True,
                        ).start()
                        last_jump = now
                        state = "JUMP!"
                prev_raw = raw

                # SLIDE — hold the key while the *stabilized* gesture is SLIDE,
                # so a 1-frame finger flicker never stutters the held key.
                if gesture == "SLIDE":
                    if not slide_holding:
                        pydirectinput.keyDown(config.slide_key)
                        slide_holding = True
                        state = "SLIDE!"
                elif slide_holding:  # finger lowered / hand lost / conflict
                    pydirectinput.keyUp(config.slide_key)
                    slide_holding = False

                if gesture in ("READY", "NONE", "IDLE"):
                    state = "READY"

                if show_skeleton and landmarks is not None:
                    drawing_utils.draw_landmarks(
                        frame, landmarks, _CONNECTIONS, _DRAW_STYLE,
                    )

                # FPS — per-frame exponential smoothing
                dt = now - prev_t
                if dt > 0:
                    fps = fps * (1.0 - FPS_ALPHA) + (1.0 / dt) * FPS_ALPHA
                prev_t = now

                # HUD
                draw_hud(frame, state, gesture, finger_state, config, fps)

            if show_window:
                cv2.imshow(window_title, frame)

            if show_window:
                k = cv2.waitKey(1) & 0xFF
            else:
                k = 0
                time.sleep(0.001)  # don't busy-spin in headless mode

            if k == ord("q"):
                break
            elif k == ord("r"):
                if slide_holding:
                    pydirectinput.keyUp(config.slide_key)
                    slide_holding = False
                print("Reconfiguring fingers...")
                return RECONFIGURE
            elif k == ord("c"):
                if slide_holding:
                    pydirectinput.keyUp(config.slide_key)
                    slide_holding = False
                config.jump_key, config.slide_key = config.slide_key, config.jump_key
                print(f"Keys swapped  |  Jump={config.jump_key}  Slide={config.slide_key}")
            elif k == ord("l"):
                show_skeleton = not show_skeleton
                print(f"Skeleton overlay: {'ON' if show_skeleton else 'OFF'}")

        return None

    finally:
        if slide_holding:
            pydirectinput.keyUp(config.slide_key)
            slide_holding = False
        if show_window:
            cv2.destroyAllWindows()
        print("Controller stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    feed = CameraFeed(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CAMERA_FPS)
    if not feed.is_opened:
        print(f"[FATAL] Camera {CAMERA_INDEX} unavailable.")
        feed.stop()
        return

    print("Camera ready  |  Assign your fingers on the setup screen  |  q=quit")

    try:
        estimator = HandEstimator()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        feed.stop()
        return

    feed.start()
    try:
        config = run_setup(feed, estimator)
        while config is not None:
            result = run_controller(feed, estimator, config)
            if result is RECONFIGURE:
                config = run_setup(feed, estimator, current=config)
                continue
            break
    finally:
        feed.stop()
        estimator.close()


if __name__ == "__main__":
    main()