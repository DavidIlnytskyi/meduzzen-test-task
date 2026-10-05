"""Tiled object detection helpers."""

import cv2
import numpy as np
import supervision as sv
from PIL import Image


def get_tile_positions(
    length: int,
    tile_size: int,
    overlap: float,
) -> list[int]:
    if length <= tile_size:
        return [0]

    stride = int(tile_size * (1 - overlap))
    positions = list(range(0, length - tile_size + 1, stride))

    last_position = length - tile_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


def tiled_predict(
    frame: np.ndarray,
    model,
    tile_size: int,
    overlap: float,
    threshold: float,
    nms_threshold: float,
) -> sv.Detections:
    height, width = frame.shape[:2]

    x_positions = get_tile_positions(width, tile_size, overlap)
    y_positions = get_tile_positions(height, tile_size, overlap)

    boxes = []
    confidences = []
    class_ids = []

    for y in y_positions:
        for x in x_positions:
            tile = frame[
                y:min(y + tile_size, height),
                x:min(x + tile_size, width),
            ]

            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile = Image.fromarray(tile)

            detections = model.predict(
                tile,
                threshold=threshold,
            )

            if len(detections) == 0:
                continue

            tile_boxes = detections.xyxy.copy()
            tile_boxes[:, [0, 2]] += x
            tile_boxes[:, [1, 3]] += y

            boxes.append(tile_boxes)
            confidences.append(detections.confidence)
            class_ids.append(detections.class_id)

    if not boxes:
        return sv.Detections.empty()

    detections = sv.Detections(
        xyxy=np.concatenate(boxes),
        confidence=np.concatenate(confidences),
        class_id=np.concatenate(class_ids),
    )

    return detections.with_nms(
        threshold=nms_threshold,
        class_agnostic=False,
    )


def filter_pitch_detections(
    frame: np.ndarray,
    detections: sv.Detections,
    min_green_ratio: float,
    context_scale: float,
) -> sv.Detections:
    if len(detections) == 0:
        return detections

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 25, 20])
    upper_green = np.array([95, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green,
    )

    height, width = frame.shape[:2]
    keep = []

    for x1, y1, x2, y2 in detections.xyxy:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        box_width = max(x2 - x1, 10) * context_scale
        box_height = max(y2 - y1, 10) * context_scale

        px1 = max(0, int(center_x - box_width / 2))
        py1 = max(0, int(center_y - box_height / 2))
        px2 = min(width, int(center_x + box_width / 2))
        py2 = min(height, int(center_y + box_height / 2))

        patch = green_mask[py1:py2, px1:px2]

        if patch.size == 0:
            keep.append(False)
            continue

        green_ratio = np.mean(patch > 0)

        keep.append(
            green_ratio >= min_green_ratio
        )

    return detections[np.asarray(keep, dtype=bool)]
