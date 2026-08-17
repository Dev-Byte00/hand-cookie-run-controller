"""Ensures the MediaPipe hand-landmarker model is available on disk."""
from __future__ import annotations

import socket
import urllib.error
import urllib.request
from pathlib import Path

from app.config.paths import MODEL_DIR, MODEL_PATH, MODEL_URL


def ensure_model(
    model_dir: Path = MODEL_DIR,
    model_path: Path = MODEL_PATH,
    model_url: str = MODEL_URL,
    download_timeout_sec: float = 30.0,
) -> Path | None:
    """Download the hand-landmarker model if not already cached.

    Returns the local model path, or ``None`` if the download failed (and
    prints a ``[FATAL]`` diagnostic in that case).
    """
    model_dir.mkdir(exist_ok=True)
    if not model_path.exists():
        print("Downloading hand model (~5 MB)...")
        try:
            socket.setdefaulttimeout(download_timeout_sec)
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Model saved → {model_path}")
        except (OSError, urllib.error.URLError) as exc:
            print(f"[FATAL] Model download failed: {exc}")
            if model_path.exists():
                model_path.unlink()
            return None
    return model_path
