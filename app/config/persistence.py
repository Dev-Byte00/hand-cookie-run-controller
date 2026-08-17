"""Saved finger-mapping + calibration persistence (JSON on disk)."""
from __future__ import annotations

import json
from pathlib import Path

from app.config.paths import CONFIG_PATH
from app.config.settings import KeyBindings
from app.domain.models import FingerCalibration, FingerConfig


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


def load_config(
    path: Path = CONFIG_PATH,
    defaults: KeyBindings = KeyBindings(),
) -> tuple[FingerConfig, FingerCalibration] | None:
    """Load a previously saved mapping + calibration, or None if unavailable."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        config = FingerConfig(
            frozenset(data["jump_fingers"]),
            frozenset(data["slide_fingers"]),
            frozenset(data["ready_fingers"]),
            jump_key=data.get("jump_key", defaults.jump_key),
            slide_key=data.get("slide_key", defaults.slide_key),
        )
        calibration = FingerCalibration(
            {int(k): tuple(v) for k, v in data.get("calibration", {}).items()}
        )
        return config, calibration
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"[WARN] Could not load saved config ({exc}) — starting fresh.")
        return None
