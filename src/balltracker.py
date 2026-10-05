"""Track a ball across detections with motion and recovery gates."""

import numpy as np
import supervision as sv


class BallTracker:
    def __init__(
        self,
        base_distance: float,
        max_distance: float,
        velocity_factor: float,
        reacquire_after: int,
        reacquire_confidence: float,
        reacquire_frames: int,
        reacquire_distance: float,
        velocity_smoothing: float,
    ):
        self.base_distance = base_distance
        self.max_distance = max_distance
        self.velocity_factor = velocity_factor

        self.reacquire_after = reacquire_after
        self.reacquire_confidence = reacquire_confidence
        self.reacquire_frames = reacquire_frames
        self.reacquire_distance = reacquire_distance

        self.velocity_smoothing = velocity_smoothing

        self.position = None
        self.velocity = np.zeros(2, dtype=np.float32)

        self.missed_frames = 0

        self.candidate_position = None
        self.candidate_count = 0

    def update(self, detections: sv.Detections):
        if len(detections) == 0:
            self.missed_frames += 1
            return None

        centers = np.column_stack(
            (
                (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2,
                (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2,
            )
        )

        if self.position is None:
            return self._initialize(
                detections,
                centers,
            )

        frame_gap = self.missed_frames + 1

        predicted_position = (
            self.position
            + self.velocity * frame_gap
        )

        speed = np.linalg.norm(self.velocity)

        allowed_distance = min(
            self.max_distance,
            self.base_distance
            + self.velocity_factor * speed
            + 20 * self.missed_frames,
        )

        distances = np.linalg.norm(
            centers - predicted_position,
            axis=1,
        )

        valid_indices = np.where(
            distances <= allowed_distance
        )[0]

        if len(valid_indices) > 0:
            best_idx = valid_indices[
                np.argmin(distances[valid_indices])
            ]

            if self.missed_frames == 0:
                self._update_position(
                    centers[best_idx]
                )

                self._reset_candidate()

                return int(best_idx)

            return self._confirm_recovery(
                detections,
                centers,
                best_idx,
            )

        self.missed_frames += 1

        if self.missed_frames >= self.reacquire_after:
            return self._global_reacquire(
                detections,
                centers,
            )

        return None

    def _initialize(
        self,
        detections,
        centers,
    ):
        best_idx = np.argmax(
            detections.confidence
        )

        candidate = centers[best_idx]

        if self.candidate_position is None:
            self.candidate_position = candidate
            self.candidate_count = 1

            return None

        distance = np.linalg.norm(
            candidate
            - self.candidate_position
        )

        if distance <= self.reacquire_distance:
            self.candidate_count += 1
            self.candidate_position = candidate
        else:
            self.candidate_position = candidate
            self.candidate_count = 1

        if self.candidate_count < 2:
            return None

        self.position = candidate
        self.velocity[:] = 0
        self.missed_frames = 0

        self._reset_candidate()

        return int(best_idx)

    def _confirm_recovery(
        self,
        detections,
        centers,
        best_idx,
    ):
        candidate = centers[best_idx]

        if self.candidate_position is None:
            self.candidate_position = candidate
            self.candidate_count = 1
            self.missed_frames += 1

            return None

        distance = np.linalg.norm(
            candidate
            - self.candidate_position
        )

        if distance <= self.reacquire_distance:
            self.candidate_position = candidate
            self.candidate_count += 1
        else:
            self.candidate_position = candidate
            self.candidate_count = 1

        if self.candidate_count < self.reacquire_frames:
            self.missed_frames += 1
            return None

        self._update_position(
            candidate
        )

        self._reset_candidate()

        return int(best_idx)

    def _global_reacquire(
        self,
        detections,
        centers,
    ):
        valid_indices = np.where(
            detections.confidence
            >= self.reacquire_confidence
        )[0]

        if len(valid_indices) == 0:
            self._reset_candidate()
            return None

        if self.candidate_position is None:
            best_idx = valid_indices[
                np.argmax(
                    detections.confidence[
                        valid_indices
                    ]
                )
            ]

            self.candidate_position = (
                centers[best_idx]
            )

            self.candidate_count = 1

            return None

        distances = np.linalg.norm(
            centers[valid_indices]
            - self.candidate_position,
            axis=1,
        )

        best_local_idx = np.argmin(
            distances
        )

        best_idx = valid_indices[
            best_local_idx
        ]

        if (
            distances[best_local_idx]
            > self.reacquire_distance
        ):
            best_idx = valid_indices[
                np.argmax(
                    detections.confidence[
                        valid_indices
                    ]
                )
            ]

            self.candidate_position = (
                centers[best_idx]
            )

            self.candidate_count = 1

            return None

        self.candidate_position = (
            centers[best_idx]
        )

        self.candidate_count += 1

        if self.candidate_count < self.reacquire_frames:
            return None

        self.position = (
            self.candidate_position.copy()
        )

        self.velocity[:] = 0
        self.missed_frames = 0

        self._reset_candidate()

        return int(best_idx)

    def _update_position(
        self,
        new_position,
    ):
        frame_gap = self.missed_frames + 1

        measured_velocity = (
            new_position
            - self.position
        ) / frame_gap

        self.velocity = (
            self.velocity_smoothing
            * self.velocity
            + (1 - self.velocity_smoothing)
            * measured_velocity
        )

        self.position = (
            new_position.copy()
        )

        self.missed_frames = 0

    def _reset_candidate(self):
        self.candidate_position = None
        self.candidate_count = 0
