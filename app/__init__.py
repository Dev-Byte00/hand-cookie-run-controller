"""Application package for the AI Hand-Gesture Cookie Run Game Controller.

Sub-packages are organized by responsibility:

* ``config``   — tunable settings, filesystem paths, saved-config persistence
* ``domain``   — pure gesture/finger domain logic (no I/O, no OpenCV/MediaPipe)
* ``vision``   — camera capture + MediaPipe hand-landmark detection
* ``automation`` — keyboard input dispatch (PyDirectInput)
* ``ui``       — OpenCV canvas/HUD rendering
* ``workflow`` — the setup / calibration / gameplay orchestration loops
"""
