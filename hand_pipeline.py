"""MediaPipe Hands: landmark extraction and optional drawing on BGR frames."""

from __future__ import annotations

import cv2
import mediapipe as mp


class HandPipeline:
    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.5,
        )
        self._draw = mp.solutions.drawing_utils

    def close(self):
        self._hands.close()

    def process(self, frame_bgr):
        """
        Draws landmarks on frame_bgr in place.
        Returns (landmark_list_or_None, frame_bgr).
        landmark_list: list of 21 landmarks with .x .y .z normalized to image.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not result.multi_hand_landmarks:
            return None, frame_bgr

        hand_landmarks = result.multi_hand_landmarks[0]
        self._draw.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
        )
        h, w = frame_bgr.shape[:2]
        cv2.rectangle(frame_bgr, (8, 8), (w - 8, h - 8), (40, 180, 99), 2)
        return list(hand_landmarks.landmark), frame_bgr
