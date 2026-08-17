"""Console-based startup menu: quick-start preset vs. custom mapping."""
from __future__ import annotations

from app.domain.enums import StartupMode


def choose_startup_mode() -> StartupMode:
    """Prompt the console for the user's startup choice."""
    print("=" * 56)
    print(" Hand Cookie Run Controller")
    print("=" * 56)
    print(" 1) Quick Start — INDEX=JUMP, MIDDLE=SLIDE, both=READY  (recommended)")
    print(" 2) Custom      — assign your own finger chord(s)")
    print("=" * 56)
    while True:
        choice = input("Choose [1/2, default 1]: ").strip()
        if choice in ("", "1"):
            return StartupMode.PRESET
        if choice == "2":
            return StartupMode.CUSTOM
        print("Please enter 1 or 2.")
