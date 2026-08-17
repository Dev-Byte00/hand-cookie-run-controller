"""Landmark → per-finger feature extraction.

Pure math over a single hand's 21 landmarks: bone-axis projections for the
four long fingers, a hybrid distance metric for the thumb, and the
setup-time pose/fist classifiers built on top of them. Nothing here touches
MediaPipe, OpenCV, or any I/O — it only consumes/produces plain floats and
sequences, so it's trivially unit-testable.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from app.config.settings import SetupSettings
from app.domain.enums import Finger
from app.domain.landmark_indices import (
    FINGER_LANDMARKS,
    INDEX_MCP,
    MIDDLE_MCP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)
from app.domain.models import HandLandmarks


def palm_size(landmarks: HandLandmarks) -> float:
    """Normalized palm size (wrist → middle MCP distance); 0 = degenerate."""
    return math.hypot(
        landmarks[MIDDLE_MCP].x - landmarks[WRIST].x,
        landmarks[MIDDLE_MCP].y - landmarks[WRIST].y,
    )


def finger_extension(landmarks: HandLandmarks, tip_idx: int, pivot_idx: int, base_idx: int) -> float:
    """How far the fingertip extends past its pivot, along the base→pivot
    axis, normalized by the base→pivot bone length.

    ≈1.4–2.2 when the finger is straight, ≈0 when relaxed, negative when
    curled. Because it is normalized by the bone length it is robust to hand
    distance, hand size, and hand rotation. Only used for the four long
    fingers (index→pinky) — the thumb abducts across the palm and uses the
    dedicated :func:`thumb_extension` metric instead.
    """
    if len(landmarks) <= base_idx:
        return 0.0
    tip = landmarks[tip_idx]
    pivot = landmarks[pivot_idx]
    base = landmarks[base_idx]
    vx = pivot.x - base.x
    vy = pivot.y - base.y
    seg = math.hypot(vx, vy)
    if seg < 1e-5:
        return 0.0
    ux, uy = vx / seg, vy / seg
    tx = tip.x - pivot.x
    ty = tip.y - pivot.y
    return (tx * ux + ty * uy) / seg


def thumb_extension(landmarks: HandLandmarks, size: float | None = None) -> float:
    """Hybrid thumb extension metric, normalized by palm size.

    The thumb can be raised in two ways: abducted across the palm (sideways),
    or extended forward along its own bone (pointing up). Neither metric
    alone catches both poses reliably:

    * Abduction distance (thumb tip → index MCP) catches a sideways thumb
      well (resting ≈0.5–0.9, extended ≈1.3–2.0), but a forward-pointing
      thumb stays near the index MCP and scores low.
    * Bone-axis projection (thumb tip along IP→MCP) catches a
      forward-extending thumb (≈1.0–2.0), but a sideways-resting thumb can
      still read high.

    This returns the *maximum* of the two metrics, so the thumb is
    recognised whichever direction it points. ``size`` may be passed in to
    avoid recomputing the palm size.
    """
    if len(landmarks) <= THUMB_TIP:
        return 0.0
    size = palm_size(landmarks) if size is None else size
    if size <= 1e-6:
        return 0.0

    abduction = math.hypot(
        landmarks[THUMB_TIP].x - landmarks[INDEX_MCP].x,
        landmarks[THUMB_TIP].y - landmarks[INDEX_MCP].y,
    ) / size

    vx = landmarks[THUMB_IP].x - landmarks[THUMB_MCP].x
    vy = landmarks[THUMB_IP].y - landmarks[THUMB_MCP].y
    seg = math.hypot(vx, vy)
    if seg > 1e-5:
        ux, uy = vx / seg, vy / seg
        tx = landmarks[THUMB_TIP].x - landmarks[THUMB_IP].x
        ty = landmarks[THUMB_TIP].y - landmarks[THUMB_IP].y
        bone_ext = (tx * ux + ty * uy) / seg
    else:
        bone_ext = 0.0

    return max(abduction, bone_ext)


def extract_extensions(landmarks: HandLandmarks | None, size: float | None = None) -> list[float]:
    """Compute all 5 per-finger extension values for one frame, or zeros if
    no hand is present.

    Centralizes the "thumb hybrid metric + 4x bone-axis projection" pattern
    used identically by setup, calibration, and the live control loop.
    """
    if landmarks is None:
        return [0.0] * 5
    size = palm_size(landmarks) if size is None else size
    values = [thumb_extension(landmarks, size)]
    for i in range(1, 5):
        values.append(finger_extension(landmarks, *FINGER_LANDMARKS[i]))
    return values


def detect_pose(
    extensions: Sequence[float],
    settings: SetupSettings,
    unavailable: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    """Return the clearly-raised fingers for the current pose (dominance-aware).

    The setup capture must behave for realistic human hands:

    * The thumb is judged on its own hybrid scale (see :func:`thumb_extension`),
      independent of the long fingers, so a raised thumb is always seen no
      matter which way it points.
    * The four long fingers share one scale, but their tendons are coupled:
      raising only the middle/ring/pinky also lifts its neighbours a little
      (typically to ≈0.4–0.6). An absolute threshold alone either misses
      gently-extending ring/pinky or falsely grabs their neighbours.

    So each long finger must BOTH clear an absolute floor
    (``settings.select_ratio``) and lie within ``settings.select_dominance_margin``
    of the highest-raised long finger. This captures the finger(s) the user
    deliberately raised while ignoring the involuntary partial lift of
    neighbours, yet still allows a deliberate chord where several fingers are
    raised close together.
    """
    if len(extensions) != 5:
        return ()

    raised: list[int] = []

    if Finger.THUMB not in unavailable and extensions[Finger.THUMB] >= settings.thumb_select_ratio:
        raised.append(Finger.THUMB)

    available = [i for i in range(1, 5) if i not in unavailable]
    if available:
        peak = max(extensions[i] for i in available)
        floor = max(settings.select_ratio, peak - settings.select_dominance_margin)
        raised.extend(i for i in available if extensions[i] >= floor)

    return tuple(sorted(raised))


def is_fist(extensions: Sequence[float], settings: SetupSettings) -> bool:
    """True when every finger is curled into a fist (not merely relaxed).

    A relaxed, open-but-not-raised hand still reads ≈0.4–0.6 on the long
    fingers; a genuine fist/punch curls them below
    ``settings.fist_long_curl_ratio`` and drops the thumb below
    ``settings.fist_thumb_curl_ratio``. This distinguishes a deliberate punch
    from the default resting hand.
    """
    if len(extensions) != 5:
        return False
    return (extensions[Finger.THUMB] < settings.fist_thumb_curl_ratio
            and all(extensions[i] < settings.fist_long_curl_ratio for i in range(1, 5)))
