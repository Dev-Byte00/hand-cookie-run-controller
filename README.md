# Hand Gesture Cookie Run Controller

Control Cookie Run (or any game) with your hand via webcam. Uses MediaPipe hand tracking to detect finger poses and translates them into keyboard input in real time.

**Version 5.0.0** — reusable finger chords, black-theme window with camera/sidebar split, in-program tracking toggle.

## Features

- **Quick Start preset** — INDEX = JUMP, MIDDLE = SLIDE, both together = READY. No configuration needed.
- **Custom finger mapping** — assign any finger or chord (multiple fingers together) to each action.
- **Reusable fingers** — a finger may be used in more than one action (e.g. JUMP = INDEX+MIDDLE, SLIDE = MIDDLE alone). Matching is exact-set-equality, so overlapping chords stay distinguishable.
- **Per-user calibration** — measures *your* raised vs. rested finger values instead of assuming a fixed threshold, which fixes most "it doesn't detect my finger right" misfires.
- **Saved config** — your mapping and calibration are stored in `hand_config.json` next to the script, so future launches skip straight to play.
- **FIST / punch mode** — optional alternate for slide that fires when no fingers are raised.
- **Chord support** — an action only fires when *all* of its assigned fingers are raised together.
- **Black-theme window** — enlarged camera preview on the left, dedicated sidebar on the right for all HUD text (text never overlaps the video).
- **In-program tracking toggle** — click the `TRACKING: ON/OFF` button (or press `t`) to fully release the camera, not just pause the display.
- **Live HUD** — color-coded border, gesture state, signal quality (GOOD / WEAK / REACQUIRING / LOST), per-finger progress bars, and FPS meter.
- **Hand skeleton overlay** — optional landmark visualization.
- **Threaded camera** — background thread always consumes the newest frame, cutting latency.
- **Auto-downloading model** — the MediaPipe hand model (~5 MB) downloads on first run.

## How It Works

### Finger Detection

- **Long fingers** (index → pinky) are measured by bone-axis projection, normalized by bone length — robust to hand distance, size, and rotation.
- **Thumb** uses a dedicated hybrid metric (max of abduction distance + bone-axis projection) so it is recognized whether it points sideways or forward.
- **Per-finger EMA smoothing** (alpha 0.35) for fast yet stable response.
- **Wide hysteresis** — separate on/off thresholds (on 0.70, off 0.35) prevent flickering.
- **Stability confirmation** — a finger must hold a state for 4 consecutive frames before it commits.
- **Palm-size gate** — tiny or far-away hands are ignored so noisy landmarks can't misfire.

### Gesture Recognition

- **Gesture-hold grace** (6 frames) — prevents a 1-frame flicker from stuttering a held slide key or re-firing a jump.
- **Rising-edge jump** — jump fires instantly on the finger raise, so fast double-taps stay responsive.
- **READY wins** — a complete READY chord forces neutral.
- **Ambiguous poses** — a complete JUMP and SLIDE chord at the same time = neutral.
- **Partial chords fire nothing** — all fingers assigned to an action must be raised together.
- **FIST mode** — if slide is bound to an empty set, slide fires when *no* fingers are raised (a fist/punch).

## Getting Started

### Requirements

- **Python 3.10+**
- **Webcam** (720p or higher recommended)
- **Windows** (optimized for Windows with DirectShow)
- **Good lighting** for best hand detection

### Install & Run

```powershell
pip install -r requirements.txt
python main.py
```

The hand model auto-downloads on first run (~5 MB) into the `model/` folder.

### First Launch

1. **Startup menu** — choose:
   - **1) Quick Start** (recommended) — INDEX=JUMP, MIDDLE=SLIDE, both=READY
   - **2) Custom** — assign your own finger chord(s)
2. **Calibration** — a short guided walkthrough. Hold your relaxed hand, then each pose once. This measures your actual raised vs. rested finger values and is what actually fixes most detection misfires.
3. **Play** — the controller runs with your mapping. Your config is saved to `hand_config.json`, so next launch skips straight to play.

## Custom Finger Mapping

If you choose **Custom** at the startup menu, you're walked through assigning fingers to each action. For each step, raise one finger or a pose of several fingers together (e.g. index+middle) and press `SPACE` — every finger currently raised is captured at once. Raise more poses and press `SPACE` again, then press `ENTER` when done.

| Step | Action | Description |
|------|--------|-------------|
| 1 | **JUMP** | Finger(s) that make Cookie Run jump (tap for double-jump) |
| 2 | **SLIDE** | Finger(s) that make Cookie Run slide (held while raised) |
| 3 | **READY** | Finger(s) that represent a neutral / resting pose |
| 4 | **FIST** | Optional — a fist/punch alternate for slide |

You can bind **one finger** or a **chord** of several fingers to each action. A finger may be reused across different actions, but the exact same full set can't be used twice — if your new chord would exactly duplicate one already assigned, add or remove a finger until it's distinguishable.

### Setup Screen Controls

| Key | Action |
|-----|--------|
| `SPACE` | Capture the currently raised pose (all fingers detected at once) |
| `ENTER` | Finish this action and move to the next step |
| `ESC` | Remove the last finger added (or go back to previous step) |
| `C` | Restart the setup from scratch |
| `Q` | Quit |

### Calibration Controls

| Key | Action |
|-----|--------|
| `SPACE` | Capture the current pose (relaxed hand, then each action pose) |
| `ESC` | Redo the previous step |
| `C` | Restart calibration from scratch |
| `Q` | Quit |

### In-Game Controls

| Key | Action |
|-----|--------|
| `R` | Re-run setup + calibration (reconfigure fingers) |
| `C` | Swap jump/slide key bindings |
| `L` | Toggle hand skeleton overlay |
| `H` | Toggle HUD detail (minimal / full debug) |
| `T` | Toggle tracking on/off (click the button also works) |
| `Q` | Quit |

## HUD

- **Color border** — flashes green on jump, red while sliding.
- **GESTURE** — current recognized action (JUMP / SLIDE / READY / IDLE).
- **SIGNAL** — live quality indicator: `GOOD` / `WEAK (move closer)` / `REACQUIRING` / `LOST`. A misfire during a bad angle or lighting moment reads as a visibility problem, not a bug.
- **Per-finger progress bars** (full HUD) — show the raw extension value against its on/off thresholds, so you can see *why* a finger is or isn't counted as raised.
- **FPS meter** — top-right, color-coded.

## Tuning Parameters

All thresholds are at the top of `main.py` as constants.

### Accuracy Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `FINGER_SMOOTH_ALPHA` | 0.35 | EMA smoothing (lower = smoother, higher = faster) |
| `EXTEND_ON_RATIO` | 0.70 | Long finger "up" threshold |
| `EXTEND_OFF_RATIO` | 0.35 | Long finger "down" threshold (hysteresis) |
| `CONFIRM_FRAMES` | 4 | Frames a finger must hold a state before commit |
| `HAND_LOST_TOLERANCE` | 5 | Lost-hand frames before finger states reset |
| `THUMB_EXTEND_ON_RATIO` | 1.10 | Thumb "up" threshold (hybrid metric) |
| `THUMB_EXTEND_OFF_RATIO` | 0.85 | Thumb "down" threshold (hysteresis) |
| `MIN_PALM_SIZE` | 0.05 | Minimum trusted hand size (normalized) |

### Gesture Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `GESTURE_HOLD_FRAMES` | 6 | Sticky frames for gesture stability |
| `GESTURE_COOLDOWN_SEC` | 0.25 | Minimum gap between two jump taps |
| `JUMP_FLASH_FRAMES` | 8 | Frames the HUD border flashes green after a jump |

### Setup Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `SETUP_SELECT_RATIO` | 0.40 | Min extension to count a finger as raised in setup |
| `SETUP_SELECT_OFF_RATIO` | 0.25 | Release threshold (hysteresis) |
| `SETUP_THUMB_SELECT_RATIO` | 0.90 | Thumb capture threshold (hybrid) |
| `SETUP_THUMB_SELECT_OFF_RATIO` | 0.70 | Thumb release threshold |
| `SETUP_SELECT_DOMINANCE_MARGIN` | 0.25 | Dominance margin for coupled fingers |
| `FIST_LONG_CURL_RATIO` | 0.25 | Long finger extension below this = curled (fist) |
| `FIST_THUMB_CURL_RATIO` | 0.60 | Thumb hybrid below this = not extended (fist) |

### Calibration Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `CALIB_ON_FRACTION` | 0.62 | On-threshold placed this fraction up the measured range |
| `CALIB_OFF_FRACTION` | 0.32 | Off-threshold placed this fraction up the measured range |
| `CALIB_MIN_RANGE` | 0.15 | Below this measured range, distrust the sample and keep the default |

### Performance Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `FRAME_WIDTH` | 480 | Camera resolution width |
| `FRAME_HEIGHT` | 360 | Camera resolution height |
| `CAMERA_FPS` | 60 | Target camera FPS |
| `CAMERA_INDEX` | 0 | Camera device index |
| `MAX_HANDS` | 1 | Hands tracked (closest one controls) |

### General Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `KEY_JUMP` | "space" | Default jump key |
| `KEY_SLIDE` | "down" | Default slide key |
| `PRESET_JUMP_FINGERS` | (1,) | Quick-start jump fingers (INDEX) |
| `PRESET_SLIDE_FINGERS` | (2,) | Quick-start slide fingers (MIDDLE) |
| `PRESET_READY_FINGERS` | (1, 2) | Quick-start ready fingers (INDEX + MIDDLE) |

## Performance Tips

- Position your hand 30–60 cm from the camera.
- Use a well-lit environment (avoid backlighting).
- Keep your hand clear of background clutter.
- Check the SIGNAL indicator — aim for `GOOD`.
- FPS is shown top-right (target: >25 fps).

## Troubleshooting

### Camera Issues
- Check camera permissions in Windows Settings.
- Try different `CAMERA_INDEX` values (0, 1, 2...).
- Ensure no other app is using the camera.

### Detection Issues
- Improve lighting (face a window, avoid strong backlight).
- Move your hand closer to the camera.
- Check the SIGNAL indicator — it should read `GOOD`.
- Re-run calibration (`R` during play) — a fresh calibration often fixes misfires.
- Lower `MIN_PALM_SIZE` if your hand reads too small.
- Adjust `EXTEND_ON_RATIO` if fingers aren't detected.

### Performance Issues
- Lower `FRAME_WIDTH` and `FRAME_HEIGHT`.
- Close other applications.
- Check the FPS meter — target >25 fps.

### False Triggers
- Increase `CONFIRM_FRAMES` for more stability.
- Increase `MIN_PALM_SIZE` to ignore small hands.
- Widen the hysteresis gap (lower `EXTEND_OFF_RATIO`).

## Project Structure

```
hand-cookie-run-controller/
├── main.py              # Controller (camera, detection, input, HUD)
├── requirements.txt     # Python dependencies
├── hand_config.json     # Saved finger mapping + calibration (auto-generated)
├── model/
│   └── hand_landmarker.task   # MediaPipe hand model (auto-downloaded)
└── README.md
```

## Credits

Built with:
- **MediaPipe** — Google's ML hand tracking (Tasks API, HandLandmarker)
- **OpenCV** — computer vision and display
- **PyDirectInput** — game input simulation
- **NumPy** — array math

Author: Tanakorn Uamraksa (GitHub Copilot)

## License

MIT License — feel free to use and modify!