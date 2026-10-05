"""Process a video using the notebook's final tracking and pitch overlay flow."""

import csv
import math
from pathlib import Path

import cv2
import supervision as sv
from tqdm import tqdm

from .balltracker import BallTracker
from .config import Config
from .models import load_models
from .pitch import (
    add_pitch_overlay,
    estimate_homography,
    image_to_pitch,
    is_valid_pitch_position,
)
from .utils import tiled_predict


CSV_COLUMNS = (
    "frame",
    "timestamp",
    "image_x",
    "image_y",
    "pitch_x",
    "pitch_y",
    "confidence",
    "homography_inliers",
)


def process_video(input_path: Path, output_path: Path, csv_path: Path) -> None:
    """Write an annotated MP4 and per-frame ball positions CSV."""
    config = Config()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {input_path}")

    writer = None
    progress = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Cannot read video properties: {input_path}")

        print("Model loading...", flush=True)
        ball_model, field_model = load_models()
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_path}")

        tracker = BallTracker(
            base_distance=config.base_distance,
            max_distance=config.max_tracking_distance,
            velocity_factor=config.velocity_distance_factor,
            reacquire_after=config.reacquire_after,
            reacquire_confidence=config.reacquire_confidence,
            reacquire_frames=config.reacquire_frames,
            reacquire_distance=config.reacquire_distance,
            velocity_smoothing=config.velocity_smoothing,
        )
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        frame_num = 0
        current_homography = None
        homography_missing_frames = 0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else None
        progress = tqdm(total=total_frames, desc="Processing video", unit="frame")

        with csv_path.open("w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_COLUMNS)

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                new_homography, homography_inliers = estimate_homography(frame, field_model)
                if new_homography is not None:
                    current_homography = new_homography
                    homography_missing_frames = 0
                else:
                    homography_missing_frames += 1
                    if homography_missing_frames > config.max_homography_missing_frames:
                        current_homography = None

                detections = tiled_predict(
                    frame=frame,
                    model=ball_model,
                    tile_size=config.tile_size,
                    overlap=config.tile_overlap,
                    threshold=config.detector_threshold,
                    nms_threshold=config.nms_threshold,
                )
                detections = detections[detections.class_id == config.target_class_id]
                ball_idx = tracker.update(detections)
                timestamp = frame_num / fps
                annotated_frame = frame.copy()
                pitch_x = None
                pitch_y = None

                if ball_idx is None:
                    csv_writer.writerow(
                        [frame_num, timestamp, "", "", "", "", "", homography_inliers]
                    )
                else:
                    ball = detections[[ball_idx]]
                    x1, y1, x2, y2 = ball.xyxy[0]
                    ball_x = float((x1 + x2) / 2)
                    ball_y = float((y1 + y2) / 2)
                    confidence = float(ball.confidence[0])

                    if current_homography is not None:
                        transformed_x, transformed_y = image_to_pitch(
                            ball_x, ball_y, current_homography
                        )
                        if is_valid_pitch_position(transformed_x, transformed_y):
                            pitch_x, pitch_y = transformed_x, transformed_y

                    annotated_frame = box_annotator.annotate(annotated_frame, ball)
                    label = f"Ball {confidence:.2f}"
                    if pitch_x is not None:
                        label += f" | ({pitch_x:.1f}, {pitch_y:.1f}) m"
                    annotated_frame = label_annotator.annotate(
                        annotated_frame, ball, labels=[label]
                    )
                    cv2.circle(
                        annotated_frame, (int(ball_x), int(ball_y)), 6, (0, 0, 255), -1
                    )
                    csv_writer.writerow(
                        [
                            frame_num,
                            timestamp,
                            ball_x,
                            ball_y,
                            "" if pitch_x is None else pitch_x,
                            "" if pitch_y is None else pitch_y,
                            confidence,
                            homography_inliers,
                        ]
                    )

                annotated_frame = add_pitch_overlay(
                    annotated_frame, pitch_x=pitch_x, pitch_y=pitch_y
                )
                writer.write(annotated_frame)
                frame_num += 1
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        capture.release()
        if writer is not None:
            writer.release()
