# Hand Gesture Cookie Run Controller

Control Cookie Run (or any game) with hand gestures via webcam.

## Controls

| Gesture | Action | Key |
|---------|--------|-----|
| Open palm  | Jump | `space` |
| Closed fist  | Slide | `down` |
| `c` | Swap jump/slide keys | |
| `l` | Toggle hand skeleton overlay | |
| `q` | Quit | |

## How It Works

- **Open palm**: 3+ fingers extended → triggers jump
- **Closed fist**: 0-1 fingers extended → triggers slide
- Finger extension is detected by projecting each fingertip along the hand's
  own axis (wrist → middle MCP), normalized by palm size — so it stays accurate
  when the hand is tilted, rotated, or held far from the camera
- Each finger has its own threshold (the ring and pinky naturally curl a bit
  even in a relaxed open hand), which reduces both missed opens and misfires
- Up to two hands are tracked; the one closest to the camera (largest palm)
  controls the game, so a second hand in the background can't hijack it
- Hands too small in the frame (far from the camera) are treated as absent, so
  noisy far-away detections can't misfire a jump or slide
- A hand-openness score (average fingertip spread vs palm size) must agree with
  the finger count, so partial or ambiguous poses are ignored instead of
  misfiring a jump or slide
- An adaptive debounce accepts genuine gesture changes instantly while blocking
  single-frame flicker
- The HUD shows a live FPS and inference-latency meter so you can verify
  performance while tuning

## Install & Run

```powershell
pip install -r requirements.txt
python main.py
```

Model auto-downloads on first run (~5 MB).

## Performance Notes

- Camera capture runs on a background thread, so the inference loop never blocks
  on camera I/O and always processes the newest frame (stale frames are dropped).
- The camera driver buffer is minimized and MJPG capture is requested
  (best effort) to lower input latency.
- Inference latency and FPS are shown live in the HUD's top-right corner.

## Tuning

All recognition thresholds live at the top of `main.py` as named constants:

| Constant | Meaning |
|----------|---------|
| `OPEN_HAND_MIN_FINGERS` | min extended fingers to count as open palm (jump) |
| `FIST_MAX_FINGERS` | max extended fingers to count as fist (slide) |
| `OPENNESS_OPEN_MIN` | below this spread, a high finger count is vetoed |
| `OPENNESS_FIST_MAX` | above this spread, a low finger count is vetoed |
| `MIN_PALM_SIZE` | smallest hand size (normalized) that is trusted |
| `GESTURE_COOLDOWN_SEC` | minimum time between key presses |
| `GESTURE_STABLE_SEC` | how long a gesture is held before it's "trusted" |
| `MAX_HANDS` | max hands tracked (nearest one wins control) |
| `CAMERA_INDEX` / `FRAME_WIDTH` / `FRAME_HEIGHT` / `CAMERA_FPS` | camera setup |
