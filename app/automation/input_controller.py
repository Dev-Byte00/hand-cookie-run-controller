"""Keyboard input dispatch, isolated behind a small typed interface.

Keeping PyDirectInput calls behind :class:`InputController` means the
workflow layer never imports ``pydirectinput`` directly, and a future
platform/backend swap (or a headless test double) only needs to satisfy
this one interface.
"""
from __future__ import annotations

import threading

import pydirectinput  # type: ignore


class InputController:
    """Thin wrapper around PyDirectInput key press / hold operations."""

    def tap_async(self, key: str) -> None:
        """Press-and-release ``key`` on a background thread (non-blocking),
        so a slow input backend can never stall the capture/render loop."""
        threading.Thread(target=pydirectinput.press, args=(key,), daemon=True).start()

    def key_down(self, key: str) -> None:
        """Hold ``key`` down."""
        pydirectinput.keyDown(key)

    def key_up(self, key: str) -> None:
        """Release ``key``."""
        pydirectinput.keyUp(key)
