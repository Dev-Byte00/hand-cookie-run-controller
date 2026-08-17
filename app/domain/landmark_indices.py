"""Raw MediaPipe hand-landmark index constants.

Hard-coded to the standard 21-point hand-landmark topology so the domain
layer's feature-extraction math has zero import-time dependency on the
``mediapipe`` package — only the ``vision`` layer needs to know about
MediaPipe itself.
"""
from __future__ import annotations

WRIST: int = 0

THUMB_MCP: int = 2
THUMB_IP: int = 3
THUMB_TIP: int = 4

INDEX_MCP: int = 5
INDEX_PIP: int = 6
INDEX_TIP: int = 8

MIDDLE_MCP: int = 9
MIDDLE_PIP: int = 10
MIDDLE_TIP: int = 12

RING_MCP: int = 13
RING_PIP: int = 14
RING_TIP: int = 16

PINKY_MCP: int = 17
PINKY_PIP: int = 18
PINKY_TIP: int = 20

#: (tip, pivot, base) landmark index triples, ordered by :class:`Finger` value.
#: The pivot/base form the bone axis used to measure extension. The thumb has
#: no PIP joint, so IP is used as its pivot instead.
FINGER_LANDMARKS: tuple[tuple[int, int, int], ...] = (
    (THUMB_TIP, THUMB_IP, THUMB_MCP),
    (INDEX_TIP, INDEX_PIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    (RING_TIP, RING_PIP, RING_MCP),
    (PINKY_TIP, PINKY_PIP, PINKY_MCP),
)
