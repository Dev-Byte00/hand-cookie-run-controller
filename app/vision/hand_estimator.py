"""MediaPipe HandLandmarker wrapper — the sole module that imports MediaPipe
for inference (drawing helpers live in :mod:`app.ui.skeleton_overlay`)."""
from __future__ import annotations

import time
from collections.abc import Sequence

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
from mediapipe.tasks.python import BaseOptions  # type: ignore
from mediapipe.tasks.python.vision import RunningMode  # type: ignore
from mediapipe.tasks.python.vision.hand_landmarker import (  # type: ignore
    HandLandmark,
    HandLandmarker,
    HandLandmarkerOptions,
)

from app.config.settings import DetectionSettings
from app.domain.models import HandLandmarks
from app.vision.model_provider import ensure_model

_WRIST = int(HandLandmark.WRIST)
_MIDDLE_MCP = int(HandLandmark.MIDDLE_FINGER_MCP)


class HandEstimator:
    """Thin wrapper around MediaPipe HandLandmarker.

    Runs in ``VIDEO`` mode (frame-timestamp based) and, when more than one
    hand is detected, returns only the closest one (the largest palm span) —
    the other hand is ignored instead of fighting it for control.
    """

    __slots__ = ("_landmarker", "_t0")

    def __init__(self, detection: DetectionSettings) -> None:
        model = ensure_model()
        if model is None:
            raise RuntimeError("Hand model unavailable — check network or model dir.")
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.VIDEO,
            num_hands=detection.max_hands,
            # Lower confidence values make hand detection easier in unusual
            # poses (fists, peace signs, angled hands) or slightly out of frame.
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = HandLandmarker.create_from_options(opts)
        self._t0 = time.perf_counter()

    def process(self, bgr) -> HandLandmarks | None:
        """Run inference on one BGR frame.

        Returns the 21 landmarks of the *closest* detected hand (the largest
        palm in the frame), or None if no hand is visible.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        elapsed_ms = int((time.perf_counter() - self._t0) * 1000)
        res = self._landmarker.detect_for_video(mp_image, elapsed_ms)
        if res.hand_landmarks:
            return self._best_hand(res.hand_landmarks)
        return None

    @staticmethod
    def _best_hand(hands: Sequence[Sequence]) -> Sequence | None:
        """Pick the hand with the largest palm span (≈ closest to the camera)."""
        best: Sequence | None = None
        best_span = -1.0
        for hand in hands:
            span = (hand[_MIDDLE_MCP].x - hand[_WRIST].x) ** 2 + \
                (hand[_MIDDLE_MCP].y - hand[_WRIST].y) ** 2
            if span > best_span:
                best_span = span
                best = hand
        return best

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._landmarker.close()
