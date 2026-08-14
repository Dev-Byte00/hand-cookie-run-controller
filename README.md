# Hand Gesture Cookie Run Controller

Control Cookie Run (or any game) with your fingers via webcam.

## Finger-Configurable Controls

On first launch a **setup screen** appears where you assign **one or more
fingers at once** to each action. Raise **any pose** — a single finger, or
several fingers together like a peace sign (index+middle) — and press **SPACE**
to capture every finger currently raised in one shot. Add more poses if you
like, then press **ENTER** when the action is complete:

| Step | Action | Meaning |
|------|--------|---------|
| 1 | **JUMP** | the finger(s) that make your character jump |
| 2 | **SLIDE** | the finger(s) that make your character slide (held while raised) |
| 3 | **READY** | the finger(s) for a resting / neutral pose |

Anything can be bound to anything — e.g. JUMP = **INDEX+MIDDLE** together
(the whole chord must be raised), SLIDE = **RING** alone, READY = **PINKY**.
A single finger is just a 1-finger chord.

During play your mapping behaves like this:

- A **complete JUMP chord** up → taps the jump key (tap twice for a double jump)
- A **complete SLIDE chord** up → *holds* the slide key while raised (hold-to-slide)
- A **complete READY chord** up → neutral; overrides everything else
- Raising only **part** of a chord → nothing fires — the whole chord must be up
- A complete JUMP chord and SLIDE chord at the same time → neutral (ambiguous)

### Setup screen keys

| Key | Action |
|-----|--------|
| `space` | Capture the currently raised pose (all fingers at once) |
| `enter` | Finish this action and move to the next step |
| `esc` | Remove the last finger added (or edit the previous action) |
| `c` | Restart the setup from scratch |
| `q` | Quit |

### In-game keys

| Key | Action |
|-----|--------|
| `r` | Re-run the setup screen (reconfigure fingers) |
| `c` | Swap jump/slide key bindings |
| `l` | Toggle hand skeleton overlay |
| `q` | Quit |

## How It Works

- Each of the four long fingers (index→pinky) is measured by projecting the
  fingertip along its base→PIP bone axis, normalized by bone length — robust
  to hand size, distance, and rotation
- **Dedicated thumb metric** — the thumb abducts *across* the palm instead of
  extending along one bone, so it uses its own distance-based reach metric
  (thumb tip → index MCP, normalized by palm size) with separate thresholds.
  A resting thumb (≈0.5–0.9) is never misread as raised
- Per-finger **EMA smoothing** (0.35), **wide hysteresis** (separate on/off
  thresholds), and **stability confirmation** (4 frames) remove landmark
  jitter and flicker
- **Gesture-hold grace** (6 frames) — once JUMP/SLIDE fires, the gesture stays
  sticky, so a 1-frame flicker can't stutter a held slide key or re-fire a
  jump, while fast double-taps stay responsive (jump fires on the raw edge)
- **Palm-size gate** — hands too small in the frame (far from the camera) are
  treated as absent so their noisy landmarks can't misfire finger states
- Brief hand-tracking dropouts don't reset your finger states
- **Easy, accurate pose capture** — the setup screen registers
  comfortably-raised fingers (extension ≥ 0.80), so you don't need to hold a
  perfectly rigid pose; hysteresis (stays raised until below 0.55) plus
  4-frame persistence prevents flicker, and a resting finger (≈0.2–0.6) never
  sneaks into the pose. MediaPipe's model confidence is also lowered (0.3) so
  the hand is found even in unusual poses (fists, peace signs, angled hands)
- The closest hand (largest palm) controls the game, so a second hand in the
  background can't hijack it
- **Pose capture** — the setup screen reads ALL fingers raised at once, so a
  multi-finger pose like index+middle is captured as one combo in a single
  SPACE press
- **Chord matching** — an action fires only when all of its assigned fingers
  are raised together (extra non-assigned fingers are fine), so combos like
  index+middle for jump work naturally
- The HUD shows each finger's live extension value and up/down state, plus FPS

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
- FPS is shown live in the HUD's top-right corner.

## Tuning

All recognition thresholds live at the top of `main.py` as named constants:

| Constant | Meaning |
|----------|---------|
| `FINGER_SMOOTH_ALPHA` | EMA smoothing per finger (lower = smoother) |
| `EXTEND_ON_RATIO` | extension ratio above this = long finger "up" |
| `EXTEND_OFF_RATIO` | below this = long finger "down" (hysteresis gap) |
| `THUMB_EXTEND_ON_RATIO` | thumb reach (palm-units) above this = thumb "up" |
| `THUMB_EXTEND_OFF_RATIO` | below this = thumb "down" (hysteresis gap) |
| `CONFIRM_FRAMES` | frames a finger must hold a state before committing |
| `HAND_LOST_TOLERANCE` | lost-hand frames before finger states reset |
| `GESTURE_HOLD_FRAMES` | sticky grace frames that hold JUMP/SLIDE through flicker |
| `MIN_PALM_SIZE` | smallest hand size (normalized) that is trusted |
| `SETUP_SELECT_RATIO` | min extension for a finger to count as raised in setup (0.80) |
| `SETUP_SELECT_OFF_RATIO` | below this a raised finger is released (hysteresis, 0.55) |
| `SETUP_CONFIRM_FRAMES` | frames the same pose must persist before setup confirmation |
| `GESTURE_COOLDOWN_SEC` | minimum time between jump taps |
| `KEY_JUMP` / `KEY_SLIDE` | default key bindings |
| `MAX_HANDS` | max hands tracked (nearest one wins control) |
| `CAMERA_INDEX` / `FRAME_WIDTH` / `FRAME_HEIGHT` / `CAMERA_FPS` | camera setup |
