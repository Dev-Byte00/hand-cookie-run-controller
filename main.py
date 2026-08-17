"""
AI Hand-Gesture Cookie Run Game Controller
==========================================
High-performance hand-gesture controller using OpenCV, MediaPipe Tasks API
(HandLandmarker), and PyDirectInput for real-time game input.

QUICK START (recommended default)
----------------------------------
On first launch you're offered a QUICK START preset — no wizard needed:

    INDEX only          → JUMP
    MIDDLE only         → SLIDE (held)
    INDEX + MIDDLE both → READY / neutral

Right after picking it, a short CALIBRATION walks you through holding your
relaxed hand, then each pose, once each — this measures YOUR actual raised vs.
rested finger values (instead of assuming a fixed threshold tuned on someone
else's hand/camera/lighting) and is what actually fixes most "it doesn't
detect my finger right" misfires. The result is saved to ``hand_config.json``
next to this script, so future launches skip straight to play.

CUSTOM FINGER MAPPING (advanced)
---------------------------------
If you'd rather assign your own finger(s)/chord per action, choose "Custom" at
the startup menu. For each step, raise ONE finger or a POSE of several fingers
together (e.g. index+middle) and press SPACE — every finger currently raised
is captured at once. Raise more poses and press SPACE again, then press ENTER
when done:

    Step 1  JUMP   — the finger(s) that make Cookie Run jump
    Step 2  SLIDE  — the finger(s) that make Cookie Run slide (held while raised)
    Step 3  READY  — the finger(s) that represent a neutral / resting pose
    Step 4  FIST   — optional fist/punch alternate for slide

Bind ONE finger or a CHORD of several fingers to each action. A finger MAY be
reused across different actions (e.g. JUMP = INDEX+MIDDLE, SLIDE = MIDDLE
alone) — the only rule is the exact same full set can't be used twice; if
your new chord would exactly duplicate one already assigned, add or remove a
finger until it's distinguishable. After setup you calibrate the same way as
Quick Start, then the controller runs with your mapping:

    * JUMP  fires on the *rising edge* of a complete JUMP chord (tap → double-jump)
    * SLIDE holds the slide key while the complete SLIDE chord stays raised (hold)
    * READY wins over everything: a complete READY chord forces neutral
    * Matching is EXACT among the fingers your mapping actually uses — a
      chord must be raised precisely, not "at least these fingers" — which is
      what keeps a reused finger (e.g. MIDDLE alone vs. INDEX+MIDDLE) resolving
      to exactly one action instead of firing both

Smoothness & accuracy:
    * Per-finger EMA smoothing + wide hysteresis + 4-frame stability confirmation
    * Per-user CALIBRATION derives on/off thresholds from your own measured
      down→up range per finger, instead of one fixed global guess
    * Gesture-hold grace prevents a 1-frame flicker from stuttering a slide
      or re-firing a jump, while fast double-taps stay responsive
    * Tiny / far-away hands are ignored so noisy landmarks can't misfire
    * The thumb uses a dedicated distance-based reach metric (it abducts
      across the palm instead of extending along one bone), so a resting
      thumb can't be misread as raised
    * A live SIGNAL indicator (GOOD / WEAK / LOST) in the HUD tells you when
      a misfire is really a camera/visibility problem, not a bad gesture

Window & controls:
    The window is a big black-theme layout: an enlarged camera preview on the
    left, and all HUD text in a dedicated sidebar on the right — text never
    overlaps the video. A TRACKING: ON/OFF button sits at the top of the
    sidebar (click it, or press 't'); turning tracking off fully releases the
    camera, not just pauses the display.

On-screen / keyboard controls:
    calibration: space = capture pose    esc = redo step   c = restart   q = quit
    setup:       space = capture pose    enter = finish action   esc = undo
                 c = restart    q = quit
    play:        r = reconfigure   c = swap keys   l = skeleton
                 h = toggle HUD detail   t = tracking on/off (click also works)
                 q = quit

Author : Tanakorn Uamraksa (GitHub Copilot)
Version: 5.1.0 — manual click-to-select fingers in setup (camera detection now optional there)
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
# while a raised finger reads ≈0.8+ — so a lower threshold captures the pose
# on the first attempt without letting resting fingers sneak in.
SETUP_SMOOTH_ALPHA: float = 0.35   # display smoothing for the live bar read-out
SETUP_SELECT_RATIO: float = 0.40   # min extension to count a finger as raised (lower = easier capture)
SETUP_SELECT_OFF_RATIO: float = 0.25  # below this a raised finger is released
# The thumb uses a hybrid metric (max of abduction distance + bone-axis
# projection) so it is detected whether it points forward or sideways. Resting
# thumb ≈0.5–0.9, extended ≈1.3+, forward-extended ≈1.0+.
SETUP_THUMB_SELECT_RATIO: float = 0.90  # min hybrid thumb score to count as raised
SETUP_THUMB_SELECT_OFF_RATIO: float = 0.70  # below this a raised thumb is released
# Long-finger tendons are coupled: raising only middle/ring/pinky also lifts
# its neighbours a little (~0.4–0.6). A raised finger must be within this
# margin of the highest-raised long finger, so partial lifts are ignored while
# a deliberate chord (peace sign, thumb+index+middle) is still captured.
SETUP_SELECT_DOMINANCE_MARGIN: float = 0.25

# FIST / punch pose — all fingers must be genuinely curled (not just relaxed).
# A relaxed hand reads ~0.4–0.6 on the long fingers, while a clenched fist
# drops them below this. The thumb hybrid metric sits ~0.2–0.5 when wrapped.
FIST_LONG_CURL_RATIO: float = 0.25   # long finger extension below this = curled
FIST_THUMB_CURL_RATIO: float = 0.60  # thumb hybrid below this = not extended

KEY_JUMP: str = "space"
KEY_SLIDE: str = "down"

# Quick-start preset — an accurate, easy one-hand default so you don't have to
# run the full chord-assignment wizard just to get going: point the index
# finger alone to JUMP, the middle finger alone to SLIDE, and hold both up
# together for the neutral/READY pose.
PRESET_JUMP_FINGERS: tuple[int, ...] = (1,)       # INDEX
PRESET_SLIDE_FINGERS: tuple[int, ...] = (2,)      # MIDDLE
PRESET_READY_FINGERS: tuple[int, ...] = (1, 2)    # INDEX + MIDDLE

# Calibration — a finger's on/off thresholds are placed this fraction of the
# way up its own *measured* down→up range, rather than assuming one global
# range fits every hand/camera/lighting setup.
CALIB_ON_FRACTION: float = 0.62
CALIB_OFF_FRACTION: float = 0.32
CALIB_MIN_RANGE: float = 0.15  # below this measured range, distrust the sample and keep the default

# Saved finger-mapping + calibration, next to this script.
CONFIG_PATH: Path = Path(__file__).resolve().parent / "hand_config.json"

# How many frames the HUD border flashes green after a JUMP fires.
JUMP_FLASH_FRAMES: int = 8

# BGR colours
C_YELLOW = (0, 255, 255)
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_WHITE = (255, 255, 255)
C_CYAN = (255, 255, 0)
C_GREY = (128, 128, 128)

# ---------------------------------------------------------------------------
# Window layout — black theme, camera + a dedicated sidebar
# ---------------------------------------------------------------------------
# The camera preview and all HUD text live in separate regions of one big
# canvas, so text can never overlap the video the way it used to when it was
# drawn directly on top of the camera frame.
CAM_DISPLAY_SCALE: float = 1.5                          # camera preview enlarged for visibility
CAM_DISPLAY_W: int = int(FRAME_WIDTH * CAM_DISPLAY_SCALE)   # 720
CAM_DISPLAY_H: int = int(FRAME_HEIGHT * CAM_DISPLAY_SCALE)  # 540
UI_PADDING: int = 20
SIDEBAR_WIDTH: int = 380
CAM_X: int = UI_PADDING
CAM_Y: int = UI_PADDING
SIDEBAR_X: int = UI_PADDING + CAM_DISPLAY_W + UI_PADDING
CANVAS_W: int = SIDEBAR_X + SIDEBAR_WIDTH + UI_PADDING
CANVAS_H: int = CAM_DISPLAY_H + UI_PADDING * 2

# Black theme palette (BGR)
BG_CANVAS = (16, 16, 16)     # window background
BG_PANEL = (28, 28, 30)      # sidebar card background
BG_CAM_OFF = (30, 30, 30)    # placeholder when the camera is turned off
BORDER = (70, 70, 74)        # subtle card/preview border

# "TRACKING: ON/OFF" button — centered at the top of the sidebar, click or
# press 't' to toggle. Turning tracking off releases the camera entirely.
TRACK_BTN_W: int = 220
TRACK_BTN_H: int = 44
TRACK_BTN_X0: int = SIDEBAR_X + (SIDEBAR_WIDTH - TRACK_BTN_W) // 2
TRACK_BTN_Y0: int = UI_PADDING + 14
TRACK_BTN_X1: int = TRACK_BTN_X0 + TRACK_BTN_W
TRACK_BTN_Y1: int = TRACK_BTN_Y0 + TRACK_BTN_H

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

    __slots__ = ("_cap", "_latest", "_lock", "_ok", "_running", "_thread",
                 "_index", "_width", "_height", "_fps")

    def __init__(self, index: int, width: int, height: int, fps: int = 30) -> None:
        self._index, self._width, self._height, self._fps = index, width, height, fps
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
        with self._lock:
            self._ok = False
            self._latest = None

    def restart(self) -> bool:
        """Re-open the camera device after a stop() and resume streaming.

        Used by the in-program tracking toggle: turning tracking off fully
        releases the camera (not just pausing reads), so this has to reopen
        the actual device, not merely restart the reader thread.
        """
        if self._running:
            return True
        self._cap = self._open_capture(self._index, self._width, self._height, self._fps)
        ok = self._cap.isOpened()
        with self._lock:
            self._ok = ok
        if ok:
            self.start()
        return ok


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
    """Hybrid thumb extension metric, normalized by palm size.

    The thumb can be raised in two ways: abducted across the palm (sideways),
    or extended forward along its own bone (pointing up). Neither metric alone
    catches both poses reliably:

    * Abduction distance (thumb tip → index MCP) catches a sideways thumb well
      (resting ≈0.5–0.9, extended ≈1.3–2.0), but a forward-pointing thumb stays
      near the index MCP and scores low.
    * Bone-axis projection (thumb tip along IP→MCP) catches a forward-extending
      thumb (≈1.0–2.0), but a sideways-resting thumb can still read high.

    This function returns the *maximum* of the two metrics, so the thumb is
    recognised whichever direction it points. ``size`` may be passed in to
    avoid recomputing the palm size.
    """
    if len(landmarks) <= _THUMB_TIP:
        return 0.0
    size = _palm_size(landmarks) if size is None else size
    if size <= 1e-6:
        return 0.0

    # 1) Abduction distance (thumb tip → index MCP) — good for a sideways thumb
    abduction = math.hypot(
        landmarks[_THUMB_TIP].x - landmarks[_INDEX_MCP].x,
        landmarks[_THUMB_TIP].y - landmarks[_INDEX_MCP].y,
    ) / size

    # 2) Bone-axis projection (thumb tip along IP→MCP) — good for a forward thumb
    thumb_tip, thumb_ip, thumb_mcp = _FINGER_LANDMARKS[FINGER_THUMB]
    vx = landmarks[thumb_ip].x - landmarks[thumb_mcp].x
    vy = landmarks[thumb_ip].y - landmarks[thumb_mcp].y
    seg = math.hypot(vx, vy)
    if seg > 1e-5:
        ux, uy = vx / seg, vy / seg
        tx = landmarks[thumb_tip].x - landmarks[thumb_ip].x
        ty = landmarks[thumb_tip].y - landmarks[thumb_ip].y
        bone_ext = (tx * ux + ty * uy) / seg
    else:
        bone_ext = 0.0

    return max(abduction, bone_ext)


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


@dataclass
class FingerCalibration:
    """Per-finger (on_ratio, off_ratio) pairs measured from a real hold of
    each pose during :func:`run_calibration`.

    A finger with no entry here falls back to the tuned global default for
    its type (thumb vs. long finger) — calibration only overrides the
    fingers actually used by a given :class:`FingerConfig`, since those are
    the only ones ever exercised during the guided capture.
    """

    thresholds: dict[int, tuple[float, float]]

    def on_off(self, finger: int) -> tuple[float, float]:
        if finger in self.thresholds:
            return self.thresholds[finger]
        if finger == FINGER_THUMB:
            return THUMB_EXTEND_ON_RATIO, THUMB_EXTEND_OFF_RATIO
        return EXTEND_ON_RATIO, EXTEND_OFF_RATIO

    @staticmethod
    def empty() -> "FingerCalibration":
        return FingerCalibration({})


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
    bone-axis projection. Thresholds default to the tuned globals, but a
    per-user :class:`FingerCalibration` can override any finger's on/off
    ratio — that's what makes detection match an individual hand and camera
    instead of a fixed guess.
    """

    __slots__ = ("_db", "extension", "extended", "lost", "palm_size")

    def __init__(self, calibration: FingerCalibration | None = None) -> None:
        calib = calibration or FingerCalibration.empty()
        self._db = []
        for i in range(5):
            on_ratio, off_ratio = calib.on_off(i)
            self._db.append(FingerDebouncer(on_ratio=on_ratio, off_ratio=off_ratio))
        self.extension = [0.0] * 5   # smoothed raw values (for the HUD)
        self.extended = [False] * 5  # committed up/down states
        self.lost = 0
        self.palm_size = 0.0

    def update(self, landmarks: Sequence | None, lost_tolerance: int) -> None:
        # A hand too small in the frame (far from the camera) produces noisy
        # landmarks — treat it as absent so it can't misfire a finger state.
        if landmarks is not None and _palm_size(landmarks) >= MIN_PALM_SIZE:
            self.lost = 0
            size = _palm_size(landmarks)
            self.palm_size = size
            self.extension[0] = thumb_extension(landmarks, size)
            self.extended[0] = self._db[0].update(self.extension[0])
            for i in range(1, 5):
                tip, pivot, base = _FINGER_LANDMARKS[i]
                self.extension[i] = finger_extension(landmarks, tip, pivot, base)
                self.extended[i] = self._db[i].update(self.extension[i])
        else:
            self.lost += 1
            self.palm_size = 0.0
            if self.lost >= lost_tolerance:
                for db in self._db:
                    db.force(False)
                self.extension = [0.0] * 5
                self.extended = [False] * 5

    def threshold(self, finger: int) -> tuple[float, float]:
        """The (on_ratio, off_ratio) actually in effect for this finger —
        reflects calibration if one was applied."""
        db = self._db[finger]
        return db.on_ratio, db.off_ratio


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
    be raised together to trigger the action. A finger MAY appear in more
    than one action's chord (e.g. JUMP = INDEX+MIDDLE, SLIDE = MIDDLE alone)
    — matching at runtime is exact-set-equality among the fingers actually
    used by this config, not "at least these fingers", so overlapping chords
    stay distinguishable as long as the full sets aren't identical (enforced
    at setup time — see :func:`run_setup`). An empty frozenset for
    ``slide_fingers`` means FIST mode: slide fires when NO fingers are raised
    (a fist/punch), matching the game's built-in gesture mapping.
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

    @classmethod
    def preset(cls, current: "FingerConfig | None" = None) -> "FingerConfig":
        """The quick-start mapping: INDEX=JUMP, MIDDLE=SLIDE, both=READY."""
        cfg = cls(PRESET_JUMP_FINGERS, PRESET_SLIDE_FINGERS, PRESET_READY_FINGERS)
        if current is not None:  # carry over key bindings across a reconfigure
            cfg.jump_key = current.jump_key
            cfg.slide_key = current.slide_key
        return cfg

    @property
    def label(self) -> str:
        def fmt(fingers: frozenset[int]) -> str:
            if not fingers:
                return "(fist)"
            return "+".join(_FINGER_NAMES[i] for i in sorted(fingers))
        return (f"J:{fmt(self.jump_fingers)}  "
                f"S:{fmt(self.slide_fingers)}  "
                f"R:{fmt(self.ready_fingers)}")


def detect_pose(
    extensions: Sequence[float],
    unavailable: set[int] = frozenset(),
) -> tuple[int, ...]:
    """Return the clearly-raised fingers for the current pose (dominance-aware).

    The setup capture must behave for realistic human hands:

    * The thumb is judged on its own hybrid scale (see :func:`thumb_extension`),
      independent of the long fingers, so a raised thumb is always seen no
      matter which way it points.
    * The four long fingers share one scale, but their tendons are coupled:
      raising only the middle/ring/pinky also lifts its neighbours a little
      (typically to ≈0.4–0.6). An absolute threshold alone either misses
      gently-extending ring/pinky or falsely grabs their neighbours.

    So each long finger must BOTH clear an absolute floor
    (``SETUP_SELECT_RATIO``) and lie within ``SETUP_SELECT_DOMINANCE_MARGIN``
    of the highest-raised long finger. This captures the finger(s) the user
    deliberately raised while ignoring the involuntary partial lift of
    neighbours, yet still allows a deliberate chord (peace sign, or
    thumb+index+middle) where several fingers are raised close together.
    """
    if len(extensions) != 5:
        return ()

    raised: list[int] = []

    # Thumb — independent hybrid scale.
    if (FINGER_THUMB not in unavailable
            and extensions[FINGER_THUMB] >= SETUP_THUMB_SELECT_RATIO):
        raised.append(FINGER_THUMB)

    # Long fingers — absolute floor + relative dominance among themselves.
    available = [i for i in range(1, 5) if i not in unavailable]
    if available:
        peak = max(extensions[i] for i in available)
        floor = max(SETUP_SELECT_RATIO, peak - SETUP_SELECT_DOMINANCE_MARGIN)
        raised.extend(i for i in available if extensions[i] >= floor)

    return tuple(sorted(raised))


def is_fist(extensions: Sequence[float]) -> bool:
    """True when every finger is curled into a fist (not merely relaxed).

    A relaxed, open-but-not-raised hand still reads ≈0.4–0.6 on the long
    fingers; a genuine fist/punch curls them below ``FIST_LONG_CURL_RATIO``
    and drops the thumb below ``FIST_THUMB_CURL_RATIO``. This distinguishes a
    deliberate punch from the default resting hand.
    """
    if len(extensions) != 5:
        return False
    return (extensions[FINGER_THUMB] < FIST_THUMB_CURL_RATIO
            and all(extensions[i] < FIST_LONG_CURL_RATIO for i in range(1, 5)))


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------
SIDEBAR_MARGIN: int = 18
SIDEBAR_TEXT_W: int = SIDEBAR_WIDTH - 2 * SIDEBAR_MARGIN


def _wrap_text(text: str, max_width: int, font_scale: float, thickness: int = 1) -> list[str]:
    """Greedy word-wrap so instruction text fits the sidebar's fixed width."""
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        (tw, _), _ = cv2.getTextSize(trial, _FONT, font_scale, thickness)
        if tw > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _draw_wrapped(
    frame: np.ndarray, text: str, x: int, y: int, max_width: int,
    font_scale: float, colour: tuple[int, int, int], thickness: int = 1, line_h: int = 18,
) -> int:
    """Draw word-wrapped text starting at (x, y); returns the y just below it."""
    for line in _wrap_text(text, max_width, font_scale, thickness):
        cv2.putText(frame, line, (x, y), _FONT, font_scale, colour, thickness)
        y += line_h
    return y


def _make_canvas(cam_frame: np.ndarray | None) -> np.ndarray:
    """Build the black-theme window: an enlarged camera preview on the left,
    a solid sidebar card on the right for all HUD text.

    Text is drawn ONLY inside the sidebar region, so it can never overlap the
    camera image — that's a layout guarantee, not something each drawing
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
        (tw, th), _ = cv2.getTextSize(msg, _FONT, _FS_L, 2)
        tx = CAM_X + (CAM_DISPLAY_W - tw) // 2
        ty = CAM_Y + (CAM_DISPLAY_H + th) // 2
        cv2.putText(canvas, msg, (tx, ty), _FONT, _FS_L, C_GREY, 2)
        hint = "click TRACKING or press 't' to resume"
        (hw, _), _ = cv2.getTextSize(hint, _FONT, _FS_S, 1)
        cv2.putText(canvas, hint, (CAM_X + (CAM_DISPLAY_W - hw) // 2, ty + 32),
                    _FONT, _FS_S, C_GREY, 1)

    cv2.rectangle(canvas, (CAM_X, CAM_Y),
                  (CAM_X + CAM_DISPLAY_W, CAM_Y + CAM_DISPLAY_H), BORDER, 2)
    cv2.rectangle(canvas, (SIDEBAR_X, UI_PADDING),
                  (SIDEBAR_X + SIDEBAR_WIDTH, CANVAS_H - UI_PADDING), BG_PANEL, -1)
    cv2.rectangle(canvas, (SIDEBAR_X, UI_PADDING),
                  (SIDEBAR_X + SIDEBAR_WIDTH, CANVAS_H - UI_PADDING), BORDER, 1)
    return canvas


def _draw_bar(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    value: float, on_ratio: float, off_ratio: float, up: bool,
) -> None:
    """A small meter showing the raw value against its on/off thresholds —
    at a glance this shows *why* a finger is or isn't counted as raised,
    instead of making you read and compare two raw numbers."""
    lo = off_ratio - 0.15
    hi = on_ratio + 0.30
    span = max(hi - lo, 0.3)
    frac = max(0.0, min(1.0, (value - lo) / span))
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_GREY, 1)
    fill_w = int(w * frac)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), C_GREEN if up else C_GREY, -1)
    tick_x = x + int(w * max(0.0, min(1.0, (on_ratio - lo) / span)))
    cv2.line(frame, (tick_x, y), (tick_x, y + h), C_CYAN, 2)


def _role_of(finger: int, chosen: Mapping[str, frozenset[int]]) -> str:
    for role, fingers in chosen.items():
        if finger in fingers:
            return role
    return ""


def draw_setup(
    canvas: np.ndarray,
    step_name: str,
    step_hint: str,
    detected: Sequence[int],
    chosen: Mapping[str, frozenset[int]],
    ext: Sequence[float],
    fps: float,
    current_fingers: Sequence[int],
    click_rects: dict[object, tuple[int, int, int, int]],
) -> None:
    """Draw the setup screen and populate ``click_rects`` (cleared and
    refilled every call) with this frame's clickable regions, so the mouse
    callback in :func:`run_setup` always hit-tests against exactly what's
    currently on screen.
    """
    click_rects.clear()
    x0 = SIDEBAR_X + SIDEBAR_MARGIN
    y = UI_PADDING + 34

    cv2.putText(canvas, "FINGER SETUP", (x0, y), _FONT, _FS_L * 0.8, C_WHITE, 2)
    y += 34
    cv2.putText(canvas, f"STEP: {step_name}", (x0, y), _FONT, _FS_M, C_CYAN, 2)
    y += 26
    y = _draw_wrapped(canvas, step_hint, x0, y, SIDEBAR_TEXT_W, _FS_S, C_WHITE, line_h=18) + 14

    if step_name == "FIST":
        # No fingers to pick here — either make a real fist for the camera,
        # or just confirm the empty set manually (a fist is the pose the
        # camera struggles with most, so this needs a reliable fallback).
        fist_now = is_fist(ext)
        status = "Camera sees a fist" if fist_now else "Camera: not curled enough"
        cv2.putText(canvas, status, (x0, y), _FONT, _FS_S, C_GREEN if fist_now else C_YELLOW, 1)
        y += 30

        btn_w, btn_h = SIDEBAR_TEXT_W, 42
        rect = (x0, y, x0 + btn_w, y + btn_h)
        click_rects["FIST"] = rect
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (24, 50, 24), -1)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), C_GREEN, 2)
        label = "CONFIRM: NO FINGERS (click)"
        (lw, lth), _ = cv2.getTextSize(label, _FONT, _FS_S, 1)
        cv2.putText(canvas, label, (rect[0] + (btn_w - lw) // 2, rect[1] + (btn_h + lth) // 2),
                    _FONT, _FS_S, C_WHITE, 1)
        y += btn_h + 16
    else:
        if detected:
            names = "+".join(_FINGER_NAMES[i] for i in detected)
            colour = C_GREEN
            y = _draw_wrapped(canvas, f"DETECTED: {names}", x0, y, SIDEBAR_TEXT_W,
                               _FS_M, colour, thickness=2, line_h=24)
            cv2.putText(canvas, "[space] to add", (x0, y), _FONT, _FS_S, colour, 1)
            y += 26
        else:
            colour = C_YELLOW
            cv2.putText(canvas, "DETECTED: ---", (x0, y), _FONT, _FS_M, colour, 2)
            y += 24
            cv2.putText(canvas, "(raise the pose, then press space)", (x0, y), _FONT, _FS_S, colour, 1)
            y += 26

        cv2.putText(canvas, "or click a finger to toggle it:", (x0, y), _FONT, _FS_S, C_WHITE, 1)
        y += 24

        row_h = 26
        for i, name in enumerate(_FINGER_NAMES):
            checked = i in current_fingers
            role = _role_of(i, chosen)
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
            cv2.putText(canvas, label, (x0, y), _FONT, _FS_S, col, 1)
            ext_str = f"{ext[i]:+.2f}"
            (ew, _), _ = cv2.getTextSize(ext_str, _FONT, _FS_S, 1)
            cv2.putText(canvas, ext_str, (x0 + SIDEBAR_TEXT_W - ew, y), _FONT, _FS_S, col, 1)
            y += row_h

    # FPS (top-right of the sidebar)
    fps_str = f"FPS:{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, _FONT, _FS_S, 1)
    cv2.putText(canvas, fps_str, (SIDEBAR_X + SIDEBAR_WIDTH - tw - 12, UI_PADDING + 16),
                _FONT, _FS_S, C_GREEN, 1)

    # Key legend (bottom of the sidebar)
    cv2.putText(canvas, "space:capture  enter:done", (x0, CANVAS_H - UI_PADDING - 32),
                _FONT, _FS_S, C_WHITE, 1)
    cv2.putText(canvas, "esc:undo  c:restart  q:quit", (x0, CANVAS_H - UI_PADDING - 12),
                _FONT, _FS_S, C_WHITE, 1)


def draw_calibration(
    canvas: np.ndarray,
    step_name: str,
    step_hint: str,
    active: frozenset[int],
    ext: Sequence[float],
    step_index: int,
    step_count: int,
    fps: float,
) -> None:
    x0 = SIDEBAR_X + SIDEBAR_MARGIN
    y = UI_PADDING + 34

    cv2.putText(canvas, "CALIBRATION", (x0, y), _FONT, _FS_L * 0.8, C_WHITE, 2)
    y += 34
    cv2.putText(canvas, f"Step {step_index + 1}/{step_count}: {step_name}",
                (x0, y), _FONT, _FS_M, C_CYAN, 2)
    y += 26
    y = _draw_wrapped(canvas, step_hint, x0, y, SIDEBAR_TEXT_W, _FS_S, C_WHITE, line_h=18) + 16

    targets = active if active else set(range(5))
    for i in sorted(targets):
        col = C_GREEN if (not active or i in active) else C_GREY
        cv2.putText(canvas, f"{_FINGER_NAMES[i]:<8}{ext[i]:+.2f}", (x0, y), _FONT, _FS_M, col, 2)
        y += 26

    fps_str = f"FPS:{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, _FONT, _FS_S, 1)
    cv2.putText(canvas, fps_str, (SIDEBAR_X + SIDEBAR_WIDTH - tw - 12, UI_PADDING + 16),
                _FONT, _FS_S, C_GREEN, 1)

    cv2.putText(canvas, "space:capture  esc:redo step", (x0, CANVAS_H - UI_PADDING - 32),
                _FONT, _FS_S, C_WHITE, 1)
    cv2.putText(canvas, "c:restart  q:quit", (x0, CANVAS_H - UI_PADDING - 12),
                _FONT, _FS_S, C_WHITE, 1)


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
    (ltw, lth), _ = cv2.getTextSize(label, _FONT, _FS_M, 2)
    cv2.putText(canvas, label,
                (TRACK_BTN_X0 + (TRACK_BTN_W - ltw) // 2, TRACK_BTN_Y0 + (TRACK_BTN_H + lth) // 2),
                _FONT, _FS_M, C_WHITE, 2)

    x0 = SIDEBAR_X + SIDEBAR_MARGIN
    y = TRACK_BTN_Y1 + 34

    cv2.putText(canvas, state, (x0, y), _FONT, _FS_L * 0.85, C_WHITE, 2)
    y += 30

    colour = C_GREEN if gesture == "JUMP" else C_RED if gesture == "SLIDE" else C_YELLOW
    cv2.putText(canvas, f"GESTURE: {gesture}", (x0, y), _FONT, _FS_M, colour, 2)
    y += 26

    cv2.putText(canvas, f"SIGNAL: {confidence}", (x0, y), _FONT, _FS_S, confidence_colour, 1)
    y += 24

    if not minimal:
        cv2.putText(canvas, config.label, (x0, y), _FONT, _FS_S, C_CYAN, 1)
        y += 24
        for i, name in enumerate(_FINGER_NAMES):
            role = ""
            if i in config.jump_fingers:
                role = "JUMP"
            elif i in config.slide_fingers:
                role = "SLIDE"
            elif i in config.ready_fingers:
                role = "READY"
            up = finger_state.extended[i]
            col = C_GREEN if up else C_GREY
            cv2.putText(canvas, f"{name:<7}{role}", (x0, y + 5), _FONT, _FS_S, col, 1)
            on_r, off_r = finger_state.threshold(i)
            _draw_bar(canvas, x0 + 130, y - 9, SIDEBAR_TEXT_W - 130, 14,
                      finger_state.extension[i], on_r, off_r, up)
            y += 24

    # FPS, right-aligned, above the key legend
    fps_str = f"FPS:{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, _FONT, _FS_S, 1)
    cv2.putText(canvas, fps_str, (SIDEBAR_X + SIDEBAR_WIDTH - tw - 12, CANVAS_H - UI_PADDING - 54),
                _FONT, _FS_S, C_GREEN, 1)

    # Key legend (bottom of the sidebar)
    cv2.putText(canvas, "r:setup  c:swap-keys  l:skeleton", (x0, CANVAS_H - UI_PADDING - 32),
                _FONT, _FS_S, C_WHITE, 1)
    cv2.putText(canvas, "h:toggle-hud  t:tracking  q:quit", (x0, CANVAS_H - UI_PADDING - 12),
                _FONT, _FS_S, C_WHITE, 1)


# ---------------------------------------------------------------------------
# Setup screen (finger-assignment wizard)
# ---------------------------------------------------------------------------
_SETUP_STEPS: tuple[tuple[str, str], ...] = (
    ("JUMP", "Raise the JUMP pose (1+ fingers) — SPACE captures it, ENTER when done"),
    ("SLIDE", "Raise the SLIDE pose (held) — SPACE captures it, ENTER when done"),
    ("READY", "Raise the READY pose (neutral) — SPACE captures it, ENTER when done"),
    ("FIST", "Make a FIST (no fingers raised) — SPACE captures it, ENTER when done"),
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

    A finger CAN be reused across different actions (e.g. JUMP = INDEX+MIDDLE,
    SLIDE = MIDDLE alone) — but two actions can't end up with the exact same
    finger set, since that would make them indistinguishable at runtime.
    Pressing ENTER on an exact duplicate is rejected; add or remove a finger
    and try again.

    Every finger can also be toggled by CLICKING it in the sidebar instead of
    raising it for the camera — useful since the camera can be flaky mid-pose,
    and the actual detection thresholds get tuned properly in the calibration
    step afterward anyway, so the assignment itself doesn't need a perfect
    camera read.
    """
    feed.start()

    click_rects: dict[object, tuple[int, int, int, int]] = {}
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
    fist_assigned = False  # True once the FIST/punch pose is captured
    step = 0
    show_skeleton = True

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

            # Fingers already added to the action currently being built can't
            # be re-picked mid-build (that's just duplicate-add protection).
            # Fingers used by OTHER, already-finished actions are fine to
            # reuse here — chords are told apart by ENTER's duplicate-chord
            # check below, not by reserving whole fingers per action.
            unavailable = set(current_fingers)
            # Use the RAW extension for detection (not the smoothed ``disp``) so
            # a pose is captured immediately on the first frame it's raised,
            # instead of waiting for the display smoothing to catch up.
            detected = detect_pose(raw, unavailable)

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
            canvas = _make_canvas(frame)
            draw_setup(canvas, step_name, step_hint, detected, chosen, disp, fps,
                       current_fingers, click_rects)

            if show_window:
                cv2.imshow(window_title, canvas)

            if show_window:
                k = cv2.waitKey(1) & 0xFF
            else:
                k = 0
                time.sleep(0.001)  # don't busy-spin in headless mode

            while click_events:
                ev = click_events.pop(0)
                if ev == "FIST" and step == 3:
                    current_fingers = []
                    fist_assigned = True
                    print(f"{step_name} = (fist/punch) [manual]")
                elif isinstance(ev, int) and step != 3:
                    if ev in current_fingers:
                        current_fingers.remove(ev)
                        print(f"Removed {_FINGER_NAMES[ev]} from {step_name} (click) — "
                              f"{len(current_fingers)} finger(s) in this action.")
                    else:
                        current_fingers.append(ev)
                        print(f"{step_name} += {_FINGER_NAMES[ev]} (click) — "
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
                    print(f"Removed {_FINGER_NAMES[removed]} from {step_name} — "
                          f"{len(current_fingers)} finger(s) in this action.")
                elif step > 0:
                    step -= 1
                    current_fingers = list(chosen.popitem()[1])
                    print(f"Editing {_SETUP_STEPS[step][0]} again — "
                          f"{len(current_fingers)} finger(s) already added.")
            elif k == ord(" "):  # SPACE → capture the raised pose now
                if step == 3:  # FIST step — capture a genuinely curled fist
                    if is_fist(raw):
                        current_fingers = []
                        fist_assigned = True
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
                              f"{'+'.join(_FINGER_NAMES[f] for f in added)} "
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
                    chord_names = "+".join(_FINGER_NAMES[i] for i in sorted(new_chord))
                    print(f"{step_name} can't use the exact same fingers as {clash} "
                          f"({chord_names}) — add one more finger (or remove one) "
                          f"so the two poses are distinguishable.")
                    continue
                chosen[step_name] = new_chord
                names = ("+".join(_FINGER_NAMES[i] for i in sorted(current_fingers))
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


# ---------------------------------------------------------------------------
# Calibration (per-user threshold measurement)
# ---------------------------------------------------------------------------
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
            names = "+".join(_FINGER_NAMES[i] for i in sorted(fingers))
            steps.append((role, f"Hold {names} up ({role} pose) — then press SPACE", fingers))
            seen.add(fingers)
    return steps


def _derive_calibration(
    down: Mapping[int, float],
    up: Mapping[int, Sequence[float]],
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
        if span < CALIB_MIN_RANGE:
            print(f"[calib] {_FINGER_NAMES[i]}: movement too small ({span:.2f}) "
                  f"— keeping the default thresholds.")
            continue
        thresholds[i] = (
            down_val + CALIB_ON_FRACTION * span,
            down_val + CALIB_OFF_FRACTION * span,
        )
    return FingerCalibration(thresholds)


def run_calibration(
    feed: CameraFeed,
    estimator: HandEstimator,
    config: FingerConfig,
    *,
    show_window: bool = True,
    window_title: str = "Hand Cookie Run — Calibration",
) -> FingerCalibration | None:
    """Guided hold-the-pose calibration for the fingers `config` actually uses.

    A fixed global threshold rarely matches every hand size, camera angle,
    and lighting setup — that mismatch is the usual cause of "it doesn't
    detect my finger right." This walks through a relaxed baseline plus each
    pose the mapping needs, measures the real down→up range per finger, and
    derives personal on/off thresholds from it (see :func:`_derive_calibration`).
    """
    feed.start()
    deadline = time.perf_counter() + CAMERA_WARMUP_SEC
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
                drawing_utils.draw_landmarks(frame, landmarks, _CONNECTIONS, _DRAW_STYLE)
            else:
                raw = [0.0] * 5
            for i in range(5):
                disp[i] += SETUP_SMOOTH_ALPHA * (raw[i] - disp[i])

            now = time.perf_counter()
            dt = now - prev_t
            if dt > 0:
                fps = fps * (1.0 - FPS_ALPHA) + (1.0 / dt) * FPS_ALPHA
            prev_t = now

            label, hint, active = steps[step_i]
            canvas = _make_canvas(frame)
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
                    f"{_FINGER_NAMES[i]}={disp[i]:.2f}"
                    for i in (sorted(active) if active else range(5))
                )
                print(f"Captured {label}: {sample_str}")
                step_i += 1
    finally:
        if show_window:
            cv2.destroyAllWindows()

    calibration = _derive_calibration(down, up)
    print("Calibration complete.")
    return calibration


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
def tracking_confidence(finger_state: FingerState) -> tuple[str, tuple[int, int, int]]:
    """A quick, honest readout of whether the *camera* can currently see the
    hand well — so a misfire during a bad angle/lighting moment reads as a
    visibility problem instead of a mysterious bug."""
    if finger_state.lost >= HAND_LOST_TOLERANCE:
        return "LOST", C_RED
    if finger_state.lost > 0:
        return "REACQUIRING", C_YELLOW  # brief miss — old finger states still held
    if finger_state.palm_size < MIN_PALM_SIZE * 1.5:
        return "WEAK (move closer)", C_YELLOW
    return "GOOD", C_GREEN


def run_controller(
    feed: CameraFeed,
    estimator: HandEstimator,
    config: FingerConfig,
    calibration: FingerCalibration | None = None,
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
        * A chord must match EXACTLY among this config's active fingers — not
          "at least these fingers" — so a finger reused across actions (e.g.
          SLIDE=MIDDLE nested inside JUMP=INDEX+MIDDLE) still resolves to
          exactly one action instead of firing both at once.
    """
    feed.start()

    click_toggle = [False]

    def _on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        if (event == cv2.EVENT_LBUTTONDOWN
                and TRACK_BTN_X0 <= x <= TRACK_BTN_X1
                and TRACK_BTN_Y0 <= y <= TRACK_BTN_Y1):
            click_toggle[0] = True

    if show_window:
        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, _on_click)

    # Wait for the first frame
    deadline = time.perf_counter() + CAMERA_WARMUP_SEC
    frame: np.ndarray | None = None
    while frame is None:
        frame, ok = feed.latest()
        if not ok and time.perf_counter() >= deadline:
            print("[FATAL] Camera disconnected.")
            return None
        time.sleep(0.01)

    finger_state = FingerState(calibration)
    gesture_hold = GestureHold()
    active_fingers = config.jump_fingers | config.slide_fingers | config.ready_fingers
    state: str = "READY"
    show_skeleton: bool = True
    hud_minimal: bool = True
    tracking_on: bool = True

    fps = 0.0
    prev_t = time.perf_counter()
    frame_failures = 0
    last_jump: float = 0.0
    slide_holding: bool = False
    prev_raw: str = "IDLE"
    jump_flash: int = 0

    try:
        last_frame: np.ndarray | None = None
        canvas: np.ndarray = _make_canvas(frame)
        while True:
            if not tracking_on:
                canvas = _make_canvas(None)
                draw_hud(canvas, "TRACKING OFF", "-", finger_state, config, fps,
                         confidence="N/A", confidence_colour=C_GREY,
                         minimal=hud_minimal, jump_flash=0, slide_holding=False,
                         tracking_on=False)
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
                        resume_deadline = time.perf_counter() + CAMERA_WARMUP_SEC
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
                if jump_flash > 0:
                    jump_flash -= 1
                confidence, confidence_colour = tracking_confidence(finger_state)

                raised = {i for i in range(5) if ext[i]}
                # Only fingers that actually belong to *some* action matter for
                # classification — this keeps noise on an unused finger (e.g. a
                # borderline thumb) from ever blocking a gesture.
                raised_active = raised & active_fingers

                # A chord must match EXACTLY (not just "at least these fingers"),
                # so that one action's chord being a subset of another's (e.g.
                # SLIDE=MIDDLE inside JUMP=INDEX+MIDDLE) can't make both read as
                # "up" at once — reused fingers stay distinguishable by the
                # complete raised set, not by fingers reserved per action.
                r_up = bool(config.ready_fingers and config.ready_fingers == raised_active)
                j_up = bool(config.jump_fingers and config.jump_fingers == raised_active)
                if config.slide_fingers:
                    s_up = bool(config.slide_fingers == raised_active)
                else:
                    # FIST mode: slide fires when no fingers are raised at all
                    s_up = not raised

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
                        jump_flash = JUMP_FLASH_FRAMES
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
                canvas = _make_canvas(frame)
                draw_hud(canvas, state, gesture, finger_state, config, fps,
                         confidence=confidence, confidence_colour=confidence_colour,
                         minimal=hud_minimal, jump_flash=jump_flash,
                         slide_holding=slide_holding, tracking_on=True)

            if show_window:
                cv2.imshow(window_title, canvas)

            if show_window:
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
                    pydirectinput.keyUp(config.slide_key)
                    slide_holding = False
                feed.stop()
                tracking_on = False
                state = "TRACKING OFF"
                last_frame = None
                print("Tracking OFF — camera released.")
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
            elif k == ord("h"):
                hud_minimal = not hud_minimal
                print(f"HUD: {'minimal' if hud_minimal else 'full debug'}")

        return None

    finally:
        if slide_holding:
            pydirectinput.keyUp(config.slide_key)
            slide_holding = False
        if show_window:
            cv2.destroyAllWindows()
        print("Controller stopped.")


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------
def save_config(config: FingerConfig, calibration: FingerCalibration, path: Path = CONFIG_PATH) -> None:
    """Persist the finger mapping + calibrated thresholds so the next launch
    can skip straight to play."""
    data = {
        "jump_fingers": sorted(config.jump_fingers),
        "slide_fingers": sorted(config.slide_fingers),
        "ready_fingers": sorted(config.ready_fingers),
        "jump_key": config.jump_key,
        "slide_key": config.slide_key,
        "calibration": {str(i): list(v) for i, v in calibration.thresholds.items()},
    }
    try:
        path.write_text(json.dumps(data, indent=2))
        print(f"Config saved → {path}")
    except OSError as exc:
        print(f"[WARN] Could not save config: {exc}")


def load_config(path: Path = CONFIG_PATH) -> tuple[FingerConfig, FingerCalibration] | None:
    """Load a previously saved mapping + calibration, or None if unavailable."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        config = FingerConfig(
            data["jump_fingers"], data["slide_fingers"], data["ready_fingers"],
            jump_key=data.get("jump_key", KEY_JUMP),
            slide_key=data.get("slide_key", KEY_SLIDE),
        )
        calibration = FingerCalibration(
            {int(k): tuple(v) for k, v in data.get("calibration", {}).items()}
        )
        return config, calibration
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"[WARN] Could not load saved config ({exc}) — starting fresh.")
        return None


def _choose_startup_mode() -> str:
    """Console menu: quick-start preset (recommended) vs. custom mapping."""
    print("=" * 56)
    print(" Hand Cookie Run Controller")
    print("=" * 56)
    print(" 1) Quick Start — INDEX=JUMP, MIDDLE=SLIDE, both=READY  (recommended)")
    print(" 2) Custom      — assign your own finger chord(s)")
    print("=" * 56)
    while True:
        choice = input("Choose [1/2, default 1]: ").strip()
        if choice in ("", "1"):
            return "preset"
        if choice == "2":
            return "custom"
        print("Please enter 1 or 2.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    feed = CameraFeed(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CAMERA_FPS)
    if not feed.is_opened:
        print(f"[FATAL] Camera {CAMERA_INDEX} unavailable.")
        feed.stop()
        return

    try:
        estimator = HandEstimator()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        feed.stop()
        return

    feed.start()
    try:
        loaded = load_config()
        if loaded is not None:
            config, calibration = loaded
            print(f"Loaded saved config  |  {config.label}")
            print("(press 'r' during play to recalibrate or remap)")
        else:
            mode = _choose_startup_mode()
            config = FingerConfig.preset() if mode == "preset" else run_setup(feed, estimator)
            if config is None:
                return
            calibration = run_calibration(feed, estimator, config)
            if calibration is None:
                return
            save_config(config, calibration)

        while config is not None:
            result = run_controller(feed, estimator, config, calibration)
            if result is RECONFIGURE:
                mode = _choose_startup_mode()
                new_config = (
                    FingerConfig.preset(current=config) if mode == "preset"
                    else run_setup(feed, estimator, current=config)
                )
                if new_config is None:
                    break
                new_calibration = run_calibration(feed, estimator, new_config)
                if new_calibration is None:
                    break
                config, calibration = new_config, new_calibration
                save_config(config, calibration)
                continue
            break
    finally:
        feed.stop()
        estimator.close()


if __name__ == "__main__":
    main()