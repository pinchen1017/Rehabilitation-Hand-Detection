"""
手部復健偵測程式 - 手部偵測器模組
封裝 Mediapipe 手部偵測功能
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import mediapipe as mp

from config import (
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_MAX_HANDS
)


@dataclass
class HandResult:
    """手部偵測結果"""
    landmarks: np.ndarray  # shape: (63,) - 21點 x (x, y, z)
    confidence: float
    raw_landmarks: List  # 原始 Mediapipe 關鍵點，用於視覺化繪製


class HandDetector:
    """手部偵測器 - 封裝 Mediapipe 手部偵測"""

    def __init__(
        self,
        max_hands: int = DEFAULT_MAX_HANDS,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, frame: np.ndarray) -> Optional[HandResult]:
        """
        偵測手部並轉換為 63 維骨架向量

        Args:
            frame: BGR 格式的影像幀

        Returns:
            HandResult 或 None（如未偵測到手部）
        """
        # 轉換為 RGB
        rgb_frame = frame[:, :, ::-1]

        # 執行偵測
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        # 取得第一隻手的關鍵點
        hand_landmarks = results.multi_hand_landmarks[0]

        # 轉換為 63 維向量 [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        landmarks_array = []
        for landmark in hand_landmarks.landmark:
            landmarks_array.extend([landmark.x, landmark.y, landmark.z])

        landmarks = np.array(landmarks_array, dtype=np.float32)

        # 計算信心度（使用所有關鍵點的平均可見度）
        confidence = np.mean([lm.visibility if hasattr(lm, 'visibility') else 1.0
                            for lm in hand_landmarks.landmark])

        return HandResult(
            landmarks=landmarks,
            confidence=float(confidence),
            raw_landmarks=hand_landmarks
        )

    def close(self):
        """釋放資源"""
        self.hands.close()
