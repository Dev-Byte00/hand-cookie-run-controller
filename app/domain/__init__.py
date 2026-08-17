"""Pure domain logic: gesture/finger models and math.

Nothing in this package touches OpenCV, MediaPipe, or the filesystem — it
only consumes/produces plain data (floats, sequences, dataclasses), which
keeps it trivially unit-testable and decoupled from the vision/UI layers.
"""
