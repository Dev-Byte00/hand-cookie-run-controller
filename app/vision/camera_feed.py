"""Threaded camera capture that always exposes the newest frame."""
from __future__ import annotations

import threading
import time

import cv2  # type: ignore
import numpy as np  # type: ignore

from app.config.settings import CameraSettings


class CameraFeed:
    """Threaded camera reader.

    ``cap.read()`` can stall (driver buffering, USB hiccups, Windows
    DirectShow quirks). Running it on a dedicated thread keeps the inference
    loop at full speed and always consuming the *newest* frame — stale
    frames are dropped instead of queued, which cuts latency.
    """

    __slots__ = ("_cap", "_latest", "_lock", "_ok", "_running", "_thread", "_settings")

    def __init__(self, settings: CameraSettings) -> None:
        self._settings = settings
        self._cap = self._open_capture(settings)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._ok = self._cap.isOpened()
        self._running = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def _open_capture(settings: CameraSettings) -> cv2.VideoCapture:
        """Open the camera preferring MJPG, and verify the resolution stuck.

        Some drivers accept MJPG but silently remap it to another resolution
        (e.g. 640×480), which would *hurt* FPS. If the requested resolution
        didn't stick, reopen with the default format instead.
        """
        index, width, height, fps = settings.index, settings.width, settings.height, settings.fps
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        # Best effort — ignored silently by drivers that don't support it.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Minimal driver-side buffer → frames are as fresh as possible.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            got_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            got_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            remapped = (got_w and abs(got_w - width) > 2) or \
                (got_h and abs(got_h - height) > 2)
            if remapped:  # usually MJPG's fault → retry with the default format
                cap.release()
                cap = cv2.VideoCapture(index, cv2.CAP_ANY)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @property
    def is_opened(self) -> bool:
        """Whether the underlying device is currently open and healthy."""
        with self._lock:
            return self._ok

    def start(self) -> None:
        """Start the reader thread. Idempotent — safe to call repeatedly."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, name="camera-read", daemon=True
        )
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._ok = False
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)  # mirror once, here, not in the hot loop
            with self._lock:
                self._ok = True
                self._latest = frame

    def latest(self) -> tuple[np.ndarray | None, bool]:
        """Return the newest frame and read status. Never blocks."""
        with self._lock:
            return self._latest, self._ok

    def stop(self) -> None:
        """Stop the reader thread and fully release the camera device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._cap.release()
        with self._lock:
            self._ok = False
            self._latest = None

    def restart(self) -> bool:
        """Re-open the camera device after a stop() and resume streaming.

        Used by the in-program tracking toggle: turning tracking off fully
        releases the camera (not just pausing reads), so this has to reopen
        the actual device, not merely restart the reader thread.
        """
        if self._running:
            return True
        self._cap = self._open_capture(self._settings)
        ok = self._cap.isOpened()
        with self._lock:
            self._ok = ok
        if ok:
            self.start()
        return ok
