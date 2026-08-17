"""
AI Hand-Gesture Cookie Run Game Controller
==========================================
High-performance hand-gesture controller using OpenCV, MediaPipe Tasks API
(HandLandmarker), and PyDirectInput for real-time game input.

This file is now a thin entry point — all application logic lives in the
``app`` package, organized by responsibility (config, domain, vision,
automation, ui, workflow). See ``app/__init__.py`` for the package layout.

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

Window & controls:
    calibration: space = capture pose    esc = redo step   c = restart   q = quit
    setup:       space = capture pose    enter = finish action   esc = undo
                 c = restart    q = quit
    play:        r = reconfigure   c = swap keys   l = skeleton
                 h = toggle HUD detail   t = tracking on/off (click also works)
                 q = quit

Author : Tanakorn Uamraksa (GitHub Copilot)
Version: 6.0.0 — refactored into a modular package architecture
"""
from __future__ import annotations

from app.config.settings import DEFAULT_SETTINGS
from app.workflow.application import Application


def main() -> None:
    """Application entry point."""
    Application(DEFAULT_SETTINGS).run()


if __name__ == "__main__":
    main()
