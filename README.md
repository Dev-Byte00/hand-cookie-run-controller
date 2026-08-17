# Hand Gesture Cookie Run Controller

Control Cookie Run (or any game) with your hand via webcam. Uses MediaPipe hand tracking to detect finger poses and translates them into keyboard input in real time.

**Version 6.0.0** — refactored into a modular package architecture (`app/`), reusable finger chords, black-theme window with camera/sidebar split, in-program tracking toggle.

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

All thresholds live in `app/config/settings.py` as typed, immutable dataclasses
grouped by concern — no more scanning one flat list of module-level globals.

### `CameraSettings`

| Field | Default | Description |
|-------|---------|-------------|
| `width` / `height` | 480 / 360 | Camera resolution |
| `fps` | 60 | Target camera FPS |
| `index` | 0 | Camera device index |
| `warmup_sec` | 5.0 | Max time to wait for the first camera frame |
| `max_frame_failures` | 30 | Consecutive read failures before giving up |

### `DetectionSettings`

| Field | Default | Description |
|-------|---------|-------------|
| `max_hands` | 1 | Hands tracked (closest one controls) |
| `finger_smooth_alpha` | 0.35 | EMA smoothing (lower = smoother, higher = faster) |
| `extend_on_ratio` | 0.70 | Long finger "up" threshold |
| `extend_off_ratio` | 0.35 | Long finger "down" threshold (hysteresis) |
| `confirm_frames` | 4 | Frames a finger must hold a state before commit |
| `hand_lost_tolerance` | 5 | Lost-hand frames before finger states reset |
| `min_palm_size` | 0.05 | Minimum trusted hand size (normalized) |
| `thumb_extend_on_ratio` | 1.10 | Thumb "up" threshold (hybrid metric) |
| `thumb_extend_off_ratio` | 0.85 | Thumb "down" threshold (hysteresis) |

### `GestureSettings`

| Field | Default | Description |
|-------|---------|-------------|
| `gesture_hold_frames` | 6 | Sticky frames for gesture stability |
| `gesture_cooldown_sec` | 0.25 | Minimum gap between two jump taps |
| `jump_flash_frames` | 8 | Frames the HUD border flashes green after a jump |

### `SetupSettings`

| Field | Default | Description |
|-------|---------|-------------|
| `select_ratio` | 0.40 | Min extension to count a finger as raised in setup |
| `select_off_ratio` | 0.25 | Release threshold (hysteresis) |
| `thumb_select_ratio` | 0.90 | Thumb capture threshold (hybrid) |
| `thumb_select_off_ratio` | 0.70 | Thumb release threshold |
| `select_dominance_margin` | 0.25 | Dominance margin for coupled fingers |
| `fist_long_curl_ratio` | 0.25 | Long finger extension below this = curled (fist) |
| `fist_thumb_curl_ratio` | 0.60 | Thumb hybrid below this = not extended (fist) |

### `CalibrationSettings`

| Field | Default | Description |
|-------|---------|-------------|
| `on_fraction` | 0.62 | On-threshold placed this fraction up the measured range |
| `off_fraction` | 0.32 | Off-threshold placed this fraction up the measured range |
| `min_range` | 0.15 | Below this measured range, distrust the sample and keep the default |

### `KeyBindings` / `PresetFingers`

| Field | Default | Description |
|-------|---------|-------------|
| `jump_key` | "space" | Default jump key |
| `slide_key` | "down" | Default slide key |
| `preset.jump` | (1,) | Quick-start jump fingers (INDEX) |
| `preset.slide` | (2,) | Quick-start slide fingers (MIDDLE) |
| `preset.ready` | (1, 2) | Quick-start ready fingers (INDEX + MIDDLE) |

## Performance Tips

- Position your hand 30–60 cm from the camera.
- Use a well-lit environment (avoid backlighting).
- Keep your hand clear of background clutter.
- Check the SIGNAL indicator — aim for `GOOD`.
- FPS is shown top-right (target: >25 fps).

## Troubleshooting

### Camera Issues
- Check camera permissions in Windows Settings.
- Try different `CameraSettings.index` values (0, 1, 2...).
- Ensure no other app is using the camera.

### Detection Issues
- Improve lighting (face a window, avoid strong backlight).
- Move your hand closer to the camera.
- Check the SIGNAL indicator — it should read `GOOD`.
- Re-run calibration (`R` during play) — a fresh calibration often fixes misfires.
- Lower `DetectionSettings.min_palm_size` if your hand reads too small.
- Adjust `DetectionSettings.extend_on_ratio` if fingers aren't detected.

### Performance Issues
- Lower `CameraSettings.width` / `height`.
- Close other applications.
- Check the FPS meter — target >25 fps.

### False Triggers
- Increase `DetectionSettings.confirm_frames` for more stability.
- Increase `DetectionSettings.min_palm_size` to ignore small hands.
- Widen the hysteresis gap (lower `DetectionSettings.extend_off_ratio`).

## Project Structure

The project is organized as a modular package (`app/`) split by
responsibility, with `main.py` as a thin entry point:

```
hand-cookie-run-controller/
├── main.py                       # Thin entry point — delegates to app.workflow.application
├── requirements.txt               # Python dependencies
├── hand_config.json                # Saved finger mapping + calibration (auto-generated)
├── model/
│   └── hand_landmarker.task        # MediaPipe hand model (auto-downloaded)
├── README.md
└── app/
    ├── config/                     # Configuration management
    │   ├── settings.py               # Typed dataclasses for every tunable constant
    │   ├── paths.py                  # Pathlib-based filesystem locations
    │   └── persistence.py            # Save/load hand_config.json
    ├── domain/                      # Pure gesture/finger domain logic (no I/O)
    │   ├── enums.py                   # Finger, GestureState, SignalQuality, StartupMode...
    │   ├── landmark_indices.py        # Raw MediaPipe landmark index constants
    │   ├── models.py                  # FingerConfig, FingerCalibration data classes
    │   ├── finger_metrics.py          # Landmark → extension math, pose/fist detection
    │   ├── finger_state.py            # EMA smoothing + hysteresis + debounce
    │   └── gesture_classifier.py      # Chord matching, GestureHold, signal quality
    ├── vision/                      # Video capture + MediaPipe inference
    │   ├── camera_feed.py              # Threaded camera reader
    │   ├── hand_estimator.py           # HandLandmarker wrapper
    │   └── model_provider.py           # Model download/cache
    ├── automation/                  # Keyboard input dispatch
    │   └── input_controller.py         # PyDirectInput wrapper behind a typed interface
    ├── ui/                          # OpenCV canvas/HUD rendering
    │   ├── theme.py                    # Layout geometry + black-theme colour palette
    │   ├── text.py                     # Word-wrap text helpers
    │   ├── canvas.py                   # Base camera+sidebar canvas builder
    │   ├── widgets.py                  # Small reusable drawing widgets (bars, role lookup)
    │   ├── skeleton_overlay.py         # Hand-landmark skeleton drawing
    │   ├── setup_screen.py             # Finger-assignment wizard screen
    │   ├── calibration_screen.py       # Calibration wizard screen
    │   └── hud_screen.py               # In-game HUD screen
    └── workflow/                    # Application orchestration
        ├── setup_wizard.py             # run_setup() interactive loop
        ├── calibration_wizard.py       # run_calibration() interactive loop
        ├── controller_loop.py          # run_controller() main gameplay loop
        ├── startup_menu.py             # Console quick-start/custom menu
        └── application.py              # Application class — wires everything together
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
