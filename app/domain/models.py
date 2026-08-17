"""Typed domain models shared by the workflow, UI, and vision layers."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.config.settings import DetectionSettings
from app.domain.enums import Finger

#: Display names for the five fingers, indexed by :class:`Finger` value.
FINGER_NAMES: tuple[str, ...] = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")


class LandmarkPoint(Protocol):
    """Structural type for a single hand landmark — only x/y are used."""

    x: float
    y: float


#: One detected hand: exactly 21 index-addressable landmark points.
HandLandmarks = Sequence[LandmarkPoint]


@dataclass(frozen=True)
class FingerCalibration:
    """Per-finger ``(on_ratio, off_ratio)`` pairs measured from a real hold
    of each pose during guided calibration.

    A finger with no entry here falls back to the tuned global default for
    its type (thumb vs. long finger) — calibration only overrides the
    fingers actually used by a given :class:`FingerConfig`, since those are
    the only ones ever exercised during the guided capture.
    """

    thresholds: dict[int, tuple[float, float]] = field(default_factory=dict)

    def on_off(self, finger: int, detection: DetectionSettings) -> tuple[float, float]:
        """Resolve the effective ``(on, off)`` thresholds for ``finger``."""
        if finger in self.thresholds:
            return self.thresholds[finger]
        if finger == Finger.THUMB:
            return detection.thumb_extend_on_ratio, detection.thumb_extend_off_ratio
        return detection.extend_on_ratio, detection.extend_off_ratio

    @staticmethod
    def empty() -> "FingerCalibration":
        """An empty calibration — every finger uses tuned global defaults."""
        return FingerCalibration({})


@dataclass
class FingerConfig:
    """The user-selected finger→action chord bindings plus key bindings.

    Each action binds a *set* of fingers (a chord). A 1-finger chord is just
    a single-finger mapping; several fingers form a combo that must all be
    raised together to trigger the action. A finger MAY appear in more than
    one action's chord (e.g. JUMP = INDEX+MIDDLE, SLIDE = MIDDLE alone) —
    matching at runtime is exact-set-equality among the fingers actually used
    by this config, not "at least these fingers", so overlapping chords stay
    distinguishable as long as the full sets aren't identical (enforced at
    setup time). An empty ``slide_fingers`` means FIST mode: slide fires when
    NO fingers are raised at all (a fist/punch).
    """

    jump_fingers: frozenset[int]
    slide_fingers: frozenset[int]
    ready_fingers: frozenset[int]
    jump_key: str = "space"
    slide_key: str = "down"

    def __post_init__(self) -> None:
        self.jump_fingers = frozenset(self.jump_fingers)
        self.slide_fingers = frozenset(self.slide_fingers)
        self.ready_fingers = frozenset(self.ready_fingers)

    @property
    def active_fingers(self) -> frozenset[int]:
        """Every finger used by any action — noise on an unused finger can
        never influence gesture classification."""
        return self.jump_fingers | self.slide_fingers | self.ready_fingers

    @property
    def label(self) -> str:
        """Short one-line summary shown in logs and the HUD."""

        def fmt(fingers: frozenset[int]) -> str:
            if not fingers:
                return "(fist)"
            return "+".join(FINGER_NAMES[i] for i in sorted(fingers))

        return (f"J:{fmt(self.jump_fingers)}  "
                f"S:{fmt(self.slide_fingers)}  "
                f"R:{fmt(self.ready_fingers)}")

    @classmethod
    def from_preset(
        cls,
        jump_fingers: Sequence[int],
        slide_fingers: Sequence[int],
        ready_fingers: Sequence[int],
        current: "FingerConfig | None" = None,
    ) -> "FingerConfig":
        """Build the quick-start mapping, carrying over key bindings from
        ``current`` (if reconfiguring) instead of resetting them."""
        cfg = cls(frozenset(jump_fingers), frozenset(slide_fingers), frozenset(ready_fingers))
        if current is not None:
            cfg.jump_key = current.jump_key
            cfg.slide_key = current.slide_key
        return cfg
