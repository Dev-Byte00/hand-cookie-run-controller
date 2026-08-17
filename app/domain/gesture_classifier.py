"""Gesture stabilization + classification, and tracking-confidence scoring."""
from __future__ import annotations

from app.config.settings import DetectionSettings
from app.domain.enums import GestureState, SignalQuality
from app.domain.finger_state import FingerState
from app.domain.models import FingerConfig

_STABLE_STATES = (GestureState.JUMP, GestureState.SLIDE, GestureState.READY, GestureState.NONE)


class GestureHold:
    """Sticky gesture grace period to eliminate micro-stutter.

    Once JUMP/SLIDE fires it is held for ``hold_frames`` even if the raw
    classification flickers to IDLE (finger dipped below threshold) — so a
    held slide key never stutters off/on. An explicit READY finger raise, an
    ambiguous NONE, or a different action gesture overrides the hold
    immediately; a sustained IDLE releases it after the grace expires.
    """

    __slots__ = ("hold_frames", "_sticky", "_remaining")

    def __init__(self, hold_frames: int) -> None:
        self.hold_frames = max(1, hold_frames)
        self._sticky: GestureState = GestureState.IDLE
        self._remaining = 0

    def update(self, raw: GestureState) -> GestureState:
        """Feed one frame's raw classification, return the stabilized gesture."""
        if raw == self._sticky:
            self._remaining = self.hold_frames  # refresh while still held
            return raw
        if raw in _STABLE_STATES:
            self._sticky = raw
            self._remaining = self.hold_frames
            return raw
        # raw is IDLE — grace period before releasing a held gesture
        if self._remaining > 0:
            self._remaining -= 1
            return self._sticky
        self._sticky = raw
        return raw


def classify_pose(raised_active: frozenset[int], config: FingerConfig, any_raised: bool) -> GestureState:
    """Classify the current raised-finger set against the user's chord mapping.

    A chord matches EXACTLY (not "at least these fingers") among the config's
    active fingers, so a finger reused across actions (e.g. SLIDE=MIDDLE
    nested inside JUMP=INDEX+MIDDLE) still resolves to exactly one action
    instead of firing both at once. ``any_raised`` is whether *any* finger at
    all is raised — used by FIST mode, which is independent of which fingers
    are "active" in this config.
    """
    r_up = bool(config.ready_fingers and config.ready_fingers == raised_active)
    j_up = bool(config.jump_fingers and config.jump_fingers == raised_active)
    if config.slide_fingers:
        s_up = config.slide_fingers == raised_active
    else:
        s_up = not any_raised  # FIST mode: slide fires when no fingers are raised

    if r_up:
        return GestureState.READY          # explicit resting pose → immediate neutral
    if j_up and s_up:
        return GestureState.NONE           # two complete chords → ambiguous
    if j_up:
        return GestureState.JUMP
    if s_up:
        return GestureState.SLIDE
    return GestureState.IDLE               # no complete action chord raised


def tracking_confidence(finger_state: FingerState, detection: DetectionSettings) -> SignalQuality:
    """A quick, honest readout of whether the *camera* can currently see the
    hand well — so a misfire during a bad angle/lighting moment reads as a
    visibility problem instead of a mysterious bug."""
    if finger_state.lost >= detection.hand_lost_tolerance:
        return SignalQuality.LOST
    if finger_state.lost > 0:
        return SignalQuality.REACQUIRING  # brief miss — old finger states still held
    if finger_state.palm_size_value < detection.min_palm_size * 1.5:
        return SignalQuality.WEAK
    return SignalQuality.GOOD
