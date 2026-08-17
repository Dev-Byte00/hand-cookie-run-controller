"""Filesystem path configuration, using :mod:`pathlib` exclusively.

Centralizes every on-disk location the application touches (saved user
config, downloaded ML model) so no other module has to compute a path
itself or hard-code a relative string.
"""
from __future__ import annotations

from pathlib import Path

#: Root directory of the project (parent of the ``app`` package).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Directory the MediaPipe hand-landmarker model is cached in.
MODEL_DIR: Path = PROJECT_ROOT / "model"

#: Full path to the cached hand-landmarker model file.
MODEL_PATH: Path = MODEL_DIR / "hand_landmarker.task"

#: Download URL for the hand-landmarker model (float16, 1-hand variant).
MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

#: Saved finger-mapping + calibration, next to the project root.
CONFIG_PATH: Path = PROJECT_ROOT / "hand_config.json"
