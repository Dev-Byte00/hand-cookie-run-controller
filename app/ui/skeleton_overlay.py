"""Optional hand-skeleton landmark overlay, drawn onto the raw camera frame."""
from __future__ import annotations

import numpy as np  # type: ignore
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils  # type: ignore
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections  # type: ignore

from app.domain.models import HandLandmarks

_DRAW_STYLE = drawing_styles.get_default_hand_landmarks_style()
_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


def draw_skeleton(frame: np.ndarray, landmarks: HandLandmarks) -> None:
    """Draw the 21-point hand skeleton + connections directly onto ``frame``."""
    drawing_utils.draw_landmarks(frame, landmarks, _CONNECTIONS, _DRAW_STYLE)
