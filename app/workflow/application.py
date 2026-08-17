"""Top-level application orchestration: wires config, vision, workflow
stages together and drives the load → setup → calibrate → play lifecycle.
"""
from __future__ import annotations

from app.automation.input_controller import InputController
from app.config.persistence import load_config, save_config
from app.config.settings import AppSettings, DEFAULT_SETTINGS
from app.domain.enums import StartupMode, WorkflowSignal
from app.domain.models import FingerConfig
from app.vision.camera_feed import CameraFeed
from app.vision.hand_estimator import HandEstimator
from app.workflow.calibration_wizard import run_calibration
from app.workflow.controller_loop import run_controller
from app.workflow.setup_wizard import run_setup
from app.workflow.startup_menu import choose_startup_mode


class Application:
    """Owns the camera/estimator lifecycle and drives the overall workflow:
    load a saved config, or run setup + calibration, then loop the live
    controller until the user quits."""

    def __init__(self, settings: AppSettings = DEFAULT_SETTINGS) -> None:
        self.settings = settings
        self.feed = CameraFeed(settings.camera)
        self.estimator: HandEstimator | None = None
        self.input_controller = InputController()

    def run(self) -> None:
        """Run the full application lifecycle until the user quits."""
        if not self.feed.is_opened:
            print(f"[FATAL] Camera {self.settings.camera.index} unavailable.")
            self.feed.stop()
            return

        try:
            self.estimator = HandEstimator(self.settings.detection)
        except RuntimeError as exc:
            print(f"[FATAL] {exc}")
            self.feed.stop()
            return

        self.feed.start()
        try:
            self._run_lifecycle()
        finally:
            self.feed.stop()
            if self.estimator is not None:
                self.estimator.close()

    def _run_lifecycle(self) -> None:
        assert self.estimator is not None
        loaded = load_config(defaults=self.settings.keys)
        if loaded is not None:
            config, calibration = loaded
            print(f"Loaded saved config  |  {config.label}")
            print("(press 'r' during play to recalibrate or remap)")
        else:
            config, calibration = self._configure(current=None)
            if config is None or calibration is None:
                return
            save_config(config, calibration)

        while config is not None:
            result = run_controller(
                self.feed, self.estimator, config, calibration,
                self.settings, self.input_controller,
            )
            if result is WorkflowSignal.RECONFIGURE:
                new_config, new_calibration = self._configure(current=config)
                if new_config is None or new_calibration is None:
                    break
                config, calibration = new_config, new_calibration
                save_config(config, calibration)
                continue
            break

    def _configure(self, current: FingerConfig | None):
        """Run the startup menu → setup wizard → calibration wizard chain."""
        assert self.estimator is not None
        mode = choose_startup_mode()
        if mode == StartupMode.PRESET:
            config = FingerConfig.from_preset(
                self.settings.preset.jump, self.settings.preset.slide,
                self.settings.preset.ready, current=current,
            )
        else:
            config = run_setup(self.feed, self.estimator, self.settings, current=current)
        if config is None:
            return None, None
        calibration = run_calibration(self.feed, self.estimator, config, self.settings)
        if calibration is None:
            return None, None
        return config, calibration
