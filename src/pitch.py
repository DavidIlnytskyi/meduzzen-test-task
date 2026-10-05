"""Pitch calibration and minimap rendering."""

import cv2
import numpy as np
import supervision as sv


PITCH_WIDTH = 120.0


PITCH_HEIGHT = 70.0


PITCH_VERTICES = np.array(
    [
        [0.0, 0.0],
        [0.0, 14.5],
        [0.0, 25.84],
        [0.0, 44.16],
        [0.0, 55.5],
        [0.0, 70.0],
        [5.5, 25.84],
        [5.5, 44.16],
        [11.0, 35.0],
        [20.15, 14.5],
        [20.15, 25.84],
        [20.15, 44.16],
        [20.15, 55.5],
        [60.0, 0.0],
        [60.0, 25.85],
        [60.0, 44.15],
        [60.0, 70.0],
        [99.85, 14.5],
        [99.85, 25.84],
        [99.85, 44.16],
        [99.85, 55.5],
        [109.0, 35.0],
        [114.5, 25.84],
        [114.5, 44.16],
        [120.0, 0.0],
        [120.0, 14.5],
        [120.0, 25.84],
        [120.0, 44.16],
        [120.0, 55.5],
        [120.0, 70.0],
        [50.85, 35.0],
        [69.15, 35.0],
    ],
    dtype=np.float32,
)


def estimate_homography(
    frame,
    field_model,
    confidence_threshold=0.5,
    ransac_threshold=0.75,
):
    result = field_model.infer(
        frame,
        confidence=0.3,
    )[0]

    keypoints = sv.KeyPoints.from_inference(result)

    if len(keypoints.xy) == 0:
        return None, 0

    if keypoints.keypoint_confidence is None:
        return None, 0

    xy = keypoints.xy[0]
    confidence = keypoints.keypoint_confidence[0]

    if len(xy) != len(PITCH_VERTICES):
        return None, 0

    valid = (
        (confidence >= confidence_threshold)
        & np.isfinite(xy[:, 0])
        & np.isfinite(xy[:, 1])
        & (xy[:, 0] > 0)
        & (xy[:, 1] > 0)
    )

    source_points = xy[valid].astype(np.float32)
    target_points = PITCH_VERTICES[valid].astype(np.float32)

    if len(source_points) < 4:
        return None, 0

    homography, inlier_mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        ransac_threshold,
    )

    if homography is None or inlier_mask is None:
        return None, 0

    inlier_count = int(inlier_mask.sum())

    if inlier_count < 4:
        return None, inlier_count

    return homography, inlier_count


def image_to_pitch(
    x,
    y,
    homography,
):
    point = np.array(
        [[[x, y]]],
        dtype=np.float32,
    )

    transformed = cv2.perspectiveTransform(
        point,
        homography,
    )[0, 0]

    return float(transformed[0]), float(transformed[1])


def is_valid_pitch_position(
    x,
    y,
    margin=5.0,
):
    return (
        -margin <= x <= PITCH_WIDTH + margin
        and -margin <= y <= PITCH_HEIGHT + margin
    )


def create_pitch_map(
    width=360,
    height=210,
):
    pitch = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    pitch[:] = (50, 120, 50)

    margin = 8
    line_color = (255, 255, 255)
    thickness = 2

    left = margin
    right = width - margin
    top = margin
    bottom = height - margin

    usable_width = right - left
    usable_height = bottom - top

    cv2.rectangle(
        pitch,
        (left, top),
        (right, bottom),
        line_color,
        thickness,
    )

    center_x = left + usable_width // 2
    center_y = top + usable_height // 2

    cv2.line(
        pitch,
        (center_x, top),
        (center_x, bottom),
        line_color,
        thickness,
    )

    center_circle_radius = int(
        usable_height * 9.15 / PITCH_HEIGHT
    )

    cv2.circle(
        pitch,
        (center_x, center_y),
        center_circle_radius,
        line_color,
        thickness,
    )

    cv2.circle(
        pitch,
        (center_x, center_y),
        3,
        line_color,
        -1,
    )

    penalty_depth = int(
        usable_width * 16.5 / PITCH_WIDTH
    )

    penalty_height = int(
        usable_height * 40.32 / PITCH_HEIGHT
    )

    penalty_top = center_y - penalty_height // 2
    penalty_bottom = center_y + penalty_height // 2

    cv2.rectangle(
        pitch,
        (left, penalty_top),
        (left + penalty_depth, penalty_bottom),
        line_color,
        thickness,
    )

    cv2.rectangle(
        pitch,
        (right - penalty_depth, penalty_top),
        (right, penalty_bottom),
        line_color,
        thickness,
    )

    goal_area_depth = int(
        usable_width * 5.5 / PITCH_WIDTH
    )

    goal_area_height = int(
        usable_height * 18.32 / PITCH_HEIGHT
    )

    goal_area_top = center_y - goal_area_height // 2
    goal_area_bottom = center_y + goal_area_height // 2

    cv2.rectangle(
        pitch,
        (left, goal_area_top),
        (left + goal_area_depth, goal_area_bottom),
        line_color,
        thickness,
    )

    cv2.rectangle(
        pitch,
        (right - goal_area_depth, goal_area_top),
        (right, goal_area_bottom),
        line_color,
        thickness,
    )

    left_penalty_x = left + int(
        usable_width * 11.0 / PITCH_WIDTH
    )

    right_penalty_x = right - int(
        usable_width * 11.0 / PITCH_WIDTH
    )

    cv2.circle(
        pitch,
        (left_penalty_x, center_y),
        3,
        line_color,
        -1,
    )

    cv2.circle(
        pitch,
        (right_penalty_x, center_y),
        3,
        line_color,
        -1,
    )

    return pitch


def pitch_position_to_map(
    pitch_x,
    pitch_y,
    map_width,
    map_height,
):
    margin = 8

    usable_width = map_width - 2 * margin
    usable_height = map_height - 2 * margin

    normalized_x = np.clip(
        pitch_x / PITCH_WIDTH,
        0.0,
        1.0,
    )

    normalized_y = np.clip(
        pitch_y / PITCH_HEIGHT,
        0.0,
        1.0,
    )

    map_x = int(
        margin + normalized_x * usable_width
    )

    map_y = int(
        margin + normalized_y * usable_height
    )

    return map_x, map_y


def add_pitch_overlay(
    frame,
    pitch_x=None,
    pitch_y=None,
    map_width=360,
    map_height=210,
    margin=20,
    opacity=0.85,
):
    frame_height, frame_width = frame.shape[:2]

    max_width = frame_width - 2 * margin
    max_height = frame_height - 2 * margin

    if max_width <= 0 or max_height <= 0:
        return frame

    if map_width > max_width:
        scale = max_width / map_width
        map_width = int(map_width * scale)
        map_height = int(map_height * scale)

    if map_height > max_height:
        scale = max_height / map_height
        map_width = int(map_width * scale)
        map_height = int(map_height * scale)

    if map_width <= 0 or map_height <= 0:
        return frame

    pitch_map = create_pitch_map(
        width=map_width,
        height=map_height,
    )

    if pitch_x is not None and pitch_y is not None:
        ball_x, ball_y = pitch_position_to_map(
            pitch_x,
            pitch_y,
            map_width,
            map_height,
        )

        cv2.circle(
            pitch_map,
            (ball_x, ball_y),
            8,
            (255, 255, 255),
            -1,
        )

        cv2.circle(
            pitch_map,
            (ball_x, ball_y),
            6,
            (0, 0, 255),
            -1,
        )

    x1 = margin
    y1 = frame_height - map_height - margin

    x2 = x1 + map_width
    y2 = y1 + map_height

    roi = frame[y1:y2, x1:x2]

    blended = cv2.addWeighted(
        roi,
        1.0 - opacity,
        pitch_map,
        opacity,
        0,
    )

    frame[y1:y2, x1:x2] = blended

    cv2.rectangle(
        frame,
        (x1 - 2, y1 - 2),
        (x2 + 2, y2 + 2),
        (255, 255, 255),
        2,
    )

    return frame
