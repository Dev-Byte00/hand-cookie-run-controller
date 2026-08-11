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
- Finger extension detected by comparing fingertip Y vs PIP joint Y

## Install & Run

```powershell
pip install -r requirements.txt
python main.py
```

Model auto-downloads on first run (~5 MB).
