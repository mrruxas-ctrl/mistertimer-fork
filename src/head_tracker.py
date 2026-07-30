import base64
import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.coordinate_mapper import CoordinateMapper
from src.face_detector import FaceDetector, ForeheadData
from src.obs_connector import OBSConnector, SceneItemTransform
from src.smoother import Smoother

log = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    scene_name: str = ""
    timer_source_name: str = ""
    camera_source_name: str = ""
    smoothing_alpha: float = 0.3
    rotation_enabled: bool = True
    offset_y: int = 20


class HeadTracker:
    def __init__(self, obs: OBSConnector):
        self.obs = obs
        self.config = TrackerConfig()
        self._detector: FaceDetector | None = None
        self._mapper: CoordinateMapper | None = None
        self._smoother: Smoother | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._face_detected = False
        self._latest_frame: np.ndarray | None = None
        self._latest_forehead: ForeheadData | None = None
        self._fps = 0.0
        self._lock = threading.Lock()
        self._obs_lock = threading.Lock()
        self._timer_item_id: int | None = None
        self._webcam_item_id: int | None = None
        self._webcam_transform_cache: SceneItemTransform | None = None
        self._current_rotation = 0.0
        self._saved_transform: SceneItemTransform | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def face_detected(self) -> bool:
        return self._face_detected

    @property
    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_frame

    @property
    def latest_forehead(self) -> ForeheadData | None:
        with self._lock:
            return self._latest_forehead

    @property
    def fps(self) -> float:
        return self._fps

    def configure(self, config: TrackerConfig):
        self.config = config

    def start(self):
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Previous tracking thread is still stopping")
        if self._running:
            return

        log.info("Starting head tracker")
        self._stop_event.clear()
        self._resolve_source_ids()

        if self._timer_item_id is None:
            raise RuntimeError(f"Source '{self.config.timer_source_name}' not found")

        if self._webcam_item_id is None:
            log.warning(
                "Webcam source '%s' not found in scene '%s' — screenshot fetch may fail",
                self.config.camera_source_name,
                self.config.scene_name,
            )

        self._detector = FaceDetector()
        self._mapper = CoordinateMapper(
            canvas_width=self.obs.canvas_width,
            canvas_height=self.obs.canvas_height,
        )
        self._smoother = Smoother(alpha=self.config.smoothing_alpha)
        self._webcam_transform_cache = None
        self._current_rotation = 0.0

        with self._obs_lock:
            self._saved_transform = self.obs.get_scene_item_transform(
                self.config.scene_name,
                self._timer_item_id,
            )
        if self._saved_transform is None:
            raise RuntimeError("Could not read timer source transform from OBS")

        self._running = True
        self._thread = threading.Thread(
            target=self._track_loop,
            name="mistertimer-head-tracker",
            daemon=True,
        )
        self._thread.start()
        log.info("Head tracker started")

    def stop(self, timeout: float = 3.0):
        thread = self._thread
        if thread is None and not self._running:
            return

        log.info("Stopping head tracker")
        self._stop_event.set()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.warning("Tracking thread did not stop within %.1f seconds", timeout)
                return

        self._running = False
        self._thread = None
        self._restore_timer_transform()
        self._webcam_transform_cache = None
        log.info("Head tracker stopped")

    def _restore_timer_transform(self):
        if self._saved_transform is None or self._timer_item_id is None:
            return
        transform = self._saved_transform
        with self._obs_lock:
            self.obs.set_scene_item_transform(
                self.config.scene_name,
                self._timer_item_id,
                pos_x=transform.pos_x,
                pos_y=transform.pos_y,
                rotation=transform.rotation,
                scale_x=transform.scale_x,
                scale_y=transform.scale_y,
                base_transform=transform,
            )

    def _resolve_source_ids(self):
        with self._obs_lock:
            self._timer_item_id = self.obs.get_scene_item_id_by_name(
                self.config.scene_name,
                self.config.timer_source_name,
            )
            self._webcam_item_id = self.obs.get_scene_item_id_by_name(
                self.config.scene_name,
                self.config.camera_source_name,
            )

    def _refresh_webcam_transform(self) -> SceneItemTransform | None:
        if self._webcam_item_id is None:
            return None
        with self._obs_lock:
            transform = self.obs.get_scene_item_transform(
                self.config.scene_name,
                self._webcam_item_id,
            )
        if transform is not None:
            self._webcam_transform_cache = transform
        return self._webcam_transform_cache

    def _fetch_frame(self) -> np.ndarray | None:
        with self._obs_lock:
            image_data = self.obs.get_source_screenshot(
                self.config.camera_source_name
            )
        if not image_data:
            return None
        try:
            payload = image_data.partition(",")[-1]
            encoded = base64.b64decode(payload, validate=True)
            frame = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                log.debug("cv2.imdecode returned None")
            return frame
        except Exception:
            log.error("Failed to decode screenshot frame", exc_info=True)
            return None

    def _track_loop(self):
        frame_count = 0
        last_fps_time = time.monotonic()
        last_transform_refresh = 0.0
        frame_fail_logged = False
        no_face_logged = False

        try:
            while not self._stop_event.is_set():
                detector = self._detector
                mapper = self._mapper
                smoother = self._smoother
                if detector is None or mapper is None or smoother is None:
                    break

                frame = self._fetch_frame()
                if frame is None:
                    if not frame_fail_logged:
                        log.warning(
                            "Cannot fetch screenshot from source '%s'",
                            self.config.camera_source_name,
                        )
                        frame_fail_logged = True
                    self._stop_event.wait(0.5)
                    continue
                frame_fail_logged = False

                forehead = detector.detect(frame)
                if forehead is None:
                    if not no_face_logged:
                        log.info("No face detected in frame — waiting for face")
                        no_face_logged = True
                    with self._lock:
                        self._face_detected = False
                    frame_count += 1
                    elapsed = time.monotonic() - last_fps_time
                    if elapsed >= 1.0:
                        self._fps = frame_count / elapsed
                        frame_count = 0
                        last_fps_time = time.monotonic()
                    self._stop_event.wait(0.002)
                    continue

                if no_face_logged:
                    log.info("Face detected")
                no_face_logged = False

                now = time.monotonic()
                if (
                    self._webcam_transform_cache is None
                    or now - last_transform_refresh >= 5.0
                ):
                    transform = self._refresh_webcam_transform()
                    last_transform_refresh = now
                    if transform:
                        mapper.update_webcam_transform(transform)

                canvas_coords = mapper.map_to_canvas(
                    forehead.x,
                    forehead.y,
                    forehead.frame_width,
                    forehead.frame_height,
                )

                with self._lock:
                    self._latest_frame = forehead.frame.copy()
                    self._latest_forehead = forehead
                    self._face_detected = True

                if canvas_coords and self._timer_item_id is not None:
                    canvas_x, canvas_y = canvas_coords
                    canvas_y -= self.config.offset_y
                    canvas_x, canvas_y = smoother.smooth_position(canvas_x, canvas_y)

                    if self.config.rotation_enabled:
                        self._current_rotation = smoother.smooth_angle(forehead.roll)

                    base = self._saved_transform
                    if base is not None:
                        perspective_sx = max(0.1, 1.0 - abs(forehead.yaw) * 0.005)
                        perspective_sy = max(0.1, 1.0 - abs(forehead.pitch) * 0.005)
                        with self._obs_lock:
                            self.obs.set_scene_item_transform(
                                self.config.scene_name,
                                self._timer_item_id,
                                canvas_x,
                                canvas_y,
                                rotation=self._current_rotation,
                                scale_x=base.scale_x * perspective_sx,
                                scale_y=base.scale_y * perspective_sy,
                                base_transform=base,
                            )

                frame_count += 1
                elapsed = now - last_fps_time
                if elapsed >= 1.0:
                    self._fps = frame_count / elapsed
                    frame_count = 0
                    last_fps_time = now

                self._stop_event.wait(0.002)
        except Exception:
            log.exception("Tracking loop crashed")
        finally:
            detector = self._detector
            self._detector = None
            if detector is not None:
                detector.release()
            self._running = False
            log.debug("Tracking loop ended")
