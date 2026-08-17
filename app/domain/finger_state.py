"""Per-finger smoothing, hysteresis, and lost-hand state tracking.

Pipeline for each of the 5 fingers: EMA smoothing → hysteresis → N-frame
stability confirmation — the core performance logic preserved exactly from
the original monolithic script, just parameterized by typed settings instead
of module globals.
"""
from __future__ import annotations

from app.config.settings import DetectionSettings
from app.domain.finger_metrics import extract_extensions, palm_size
from app.domain.models import FingerCalibration, HandLandmarks


class FingerDebouncer:
    """Smooths + debounces a single finger's extension signal.

    Pipeline: EMA smoothing → hysteresis → stability confirmation. A finger
    only flips state after holding the new state for ``confirm`` consecutive
    frames, which eliminates jitter-triggered false actions while keeping
    real movements snappy.
    """

    __slots__ = ("on_ratio", "off_ratio", "confirm", "smooth_alpha", "extended", "smooth", "streak")

    def __init__(self, on_ratio: float, off_ratio: float, confirm: int, smooth_alpha: float) -> None:
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self.confirm = max(1, confirm)
        self.smooth_alpha = smooth_alpha
        self.extended = False
        self.smooth = 0.0
        self.streak = 0

    def update(self, raw: float) -> bool:
        """Feed one frame's raw extension value, return the committed state."""
        self.smooth += self.smooth_alpha * (raw - self.smooth)

        target = self.extended
        if self.extended:
            if self.smooth < self.off_ratio:
                target = False
        elif self.smooth > self.on_ratio:
            target = True

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


class FingerState:
    """Tracks all 5 fingers with per-finger smoothing/debounce + lost-hand reset.

    The thumb uses its own distance-based metric with dedicated thresholds,
    because a bone-axis projection is unreliable for the abduction-moving
    thumb. The four long fingers use the standard bone-axis projection.
    Thresholds default to the tuned globals, but a per-user
    :class:`FingerCalibration` can override any finger's on/off ratio —
    that's what makes detection match an individual hand and camera instead
    of a fixed guess.
    """

    __slots__ = ("_settings", "_db", "extension", "extended", "lost", "palm_size_value")

    def __init__(self, settings: DetectionSettings, calibration: FingerCalibration | None = None) -> None:
        self._settings = settings
        calib = calibration or FingerCalibration.empty()
        self._db: list[FingerDebouncer] = []
        for i in range(5):
            on_ratio, off_ratio = calib.on_off(i, settings)
            self._db.append(FingerDebouncer(
                on_ratio=on_ratio,
                off_ratio=off_ratio,
                confirm=settings.confirm_frames,
                smooth_alpha=settings.finger_smooth_alpha,
            ))
        self.extension = [0.0] * 5    # smoothed raw values (for the HUD)
        self.extended = [False] * 5   # committed up/down states
        self.lost = 0
        self.palm_size_value = 0.0

    def update(self, landmarks: HandLandmarks | None, lost_tolerance: int) -> None:
        """Feed one frame's landmarks (or None if the hand isn't visible)."""
        if landmarks is not None:
            size = palm_size(landmarks)
            # A hand too small in the frame (far from the camera) produces
            # noisy landmarks — treat it as absent so it can't misfire a
            # finger state.
            if size >= self._settings.min_palm_size:
                self.lost = 0
                self.palm_size_value = size
                values = extract_extensions(landmarks, size)
                for i in range(5):
                    self.extension[i] = values[i]
                    self.extended[i] = self._db[i].update(values[i])
                return

        self.lost += 1
        self.palm_size_value = 0.0
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
