"""Domain-level enumerations shared across every layer.

Using enums instead of raw strings/ints gives each module an explicit,
typed contract for finger identity, gesture state, and workflow signals,
removing "magic string" coupling between the workflow and UI layers.
"""
from __future__ import annotations

from enum import Enum, IntEnum, auto


class Finger(IntEnum):
    """Identity of one of the five tracked fingers.

    Values match the index into every 5-element per-finger array used
    throughout the domain and UI layers.
    """

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


class GestureState(str, Enum):
    """Classified hand gesture for the current frame.

    Inherits from :class:`str` so it compares equal to, and can be printed
    as, its underlying label — a convenient bridge to the original script's
    string-based classification without losing type-safety.
    """

    JUMP = "JUMP"
    SLIDE = "SLIDE"
    READY = "READY"
    NONE = "NONE"
    IDLE = "IDLE"


class SignalQuality(str, Enum):
    """Live camera/tracking confidence indicator shown in the HUD."""

    GOOD = "GOOD"
    WEAK = "WEAK (move closer)"
    REACQUIRING = "REACQUIRING"
    LOST = "LOST"


class StartupMode(str, Enum):
    """The user's choice at the startup menu."""

    PRESET = "preset"
    CUSTOM = "custom"


class WorkflowSignal(Enum):
    """Out-of-band signal returned by a workflow loop to its caller."""

    RECONFIGURE = auto()
