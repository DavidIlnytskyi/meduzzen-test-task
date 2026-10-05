"""Parameters used by the notebook's final video pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Hold detector, tracker, and homography settings."""

    target_class_id: int = 0
    detector_threshold: float = 0.15
    nms_threshold: float = 0.5
    tile_size: int = 640
    tile_overlap: float = 0.25
    base_distance: float = 80.0
    max_tracking_distance: float = 300.0
    velocity_distance_factor: float = 2.5
    reacquire_after: int = 6
    reacquire_confidence: float = 0.30
    reacquire_frames: int = 2
    reacquire_distance: float = 120.0
    velocity_smoothing: float = 0.4
    max_homography_missing_frames: int = 10
