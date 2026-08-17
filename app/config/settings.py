"""Central tunable configuration for the whole application.

Every tunable constant from the original script lives here, grouped into
small, typed, immutable dataclasses by concern (camera, detection, gesture,
setup, calibration, keys, presets) instead of one flat list of module-level
globals. Passing a concrete dataclass instance around gives every module an
explicit, typed contract for exactly the parameters it needs — no hidden
reliance on global state.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraSettings:
    """Camera device selection + capture-loop resilience tuning."""

    index: int = 0
    width: int = 480          # higher resolution → more accurate landmarks
    height: int = 360
    fps: int = 60
    warmup_sec: float = 5.0        # max time to wait for the first camera frame
    max_frame_failures: int = 30   # consecutive read failures before giving up


@dataclass(frozen=True)
class DetectionSettings:
    """Per-finger extension smoothing/debounce + hand-quality gating.

    Orientation-invariant, scale-invariant thresholds for the bone-axis
    projection (long fingers) and the hybrid distance metric (thumb).
    """

    max_hands: int = 1  # 1 hand is much faster without hand-switching overhead
    finger_smooth_alpha: float = 0.35   # EMA alpha — lower = smoother, less jitter
    extend_on_ratio: float = 0.70       # extension ratio above this = finger "up"
    extend_off_ratio: float = 0.35      # below this = finger "down" (hysteresis)
    confirm_frames: int = 4             # frames a finger must hold a state before commit
    hand_lost_tolerance: int = 5        # lost-hand frames before finger states reset
    min_palm_size: float = 0.05         # normalized palm size (wrist → middle MCP)
    thumb_extend_on_ratio: float = 1.10   # thumb reach ≥ this = thumb "up"
    thumb_extend_off_ratio: float = 0.85  # below this = thumb "down" (hysteresis)


@dataclass(frozen=True)
class GestureSettings:
    """Gesture stability (sticky-hold) + jump timing."""

    gesture_hold_frames: int = 6     # sticky frames so micro-flicker can't stutter a hold
    gesture_cooldown_sec: float = 0.25  # minimum gap between two jump taps
    jump_flash_frames: int = 8       # HUD border flash duration after a jump


@dataclass(frozen=True)
class SetupSettings:
    """Setup-wizard pose-capture thresholds (also reused for the live
    display smoothing during calibration)."""

    smooth_alpha: float = 0.35            # display smoothing for the live bar read-out
    select_ratio: float = 0.40            # min extension to count a finger as raised
    select_off_ratio: float = 0.25        # below this a raised finger is released
    thumb_select_ratio: float = 0.90      # min hybrid thumb score to count as raised
    thumb_select_off_ratio: float = 0.70  # below this a raised thumb is released
    select_dominance_margin: float = 0.25  # margin vs. the peak coupled long finger
    fist_long_curl_ratio: float = 0.25    # long finger extension below this = curled
    fist_thumb_curl_ratio: float = 0.60   # thumb hybrid below this = not extended


@dataclass(frozen=True)
class CalibrationSettings:
    """Per-user threshold derivation from measured relaxed→raised ranges."""

    on_fraction: float = 0.62   # on-threshold placed this far up the measured range
    off_fraction: float = 0.32  # off-threshold placed this far up the measured range
    min_range: float = 0.15     # below this measured range, distrust the sample


@dataclass(frozen=True)
class KeyBindings:
    """Default jump/slide key bindings (swappable in-game with 'c')."""

    jump_key: str = "space"
    slide_key: str = "down"


@dataclass(frozen=True)
class PresetFingers:
    """Quick-start finger→action chord mapping.

    An accurate, easy one-hand default so the user doesn't have to run the
    full chord-assignment wizard just to get going.
    """

    jump: tuple[int, ...] = (1,)      # INDEX
    slide: tuple[int, ...] = (2,)     # MIDDLE
    ready: tuple[int, ...] = (1, 2)   # INDEX + MIDDLE


@dataclass(frozen=True)
class AppSettings:
    """Aggregate root for every tunable setting in the application."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    gesture: GestureSettings = field(default_factory=GestureSettings)
    setup: SetupSettings = field(default_factory=SetupSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    keys: KeyBindings = field(default_factory=KeyBindings)
    preset: PresetFingers = field(default_factory=PresetFingers)
    fps_alpha: float = 0.05  # FPS meter — per-frame exponential smoothing


#: The single, ready-to-use settings instance the application runs with.
DEFAULT_SETTINGS: AppSettings = AppSettings()
