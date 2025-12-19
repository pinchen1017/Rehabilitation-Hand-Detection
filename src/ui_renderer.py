"""
手部復健偵測程式 - UI 渲染器模組
在 OpenCV 視窗上疊加顯示資訊
"""

from typing import Optional, Dict
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from config import (
    GESTURE_NAMES_ZH,
    FONT_SCALE,
    FONT_THICKNESS,
    TEXT_COLOR,
    LANDMARK_COLOR,
    CONNECTION_COLOR,
    VALID_STRETCH_TYPES,
    CHINESE_FONT_PATH
)
from hand_detector import HandResult
from gesture_classifier import GesturePrediction
from stretch_tracker import StretchStats, TrackerState


class UIRenderer:
    """UI 渲染器 - 在畫面上疊加顯示資訊"""

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        # 使用支援中文的字體（如果可用）
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _put_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        font_scale: float = FONT_SCALE,
        color: tuple = TEXT_COLOR,
        thickness: int = FONT_THICKNESS,
        bg_color: tuple = (0, 0, 0),
        padding: int = 5
    ):
        """在文字後方加上背景，支援中文"""
        # 嘗試判斷是否有中文
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            # 使用 Pillow 畫中文
            # OpenCV frame → PIL Image
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 字型大小依 cv2 FONT_SCALE 調整
            font_size = max(int(font_scale * 30), 12)
            font = ImageFont.truetype(CHINESE_FONT_PATH, font_size)

            # 計算文字尺寸
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            x, y = position

            # 背景矩形
            draw.rectangle(
                (x - padding, y - padding, x + text_w + padding, y + text_h + padding),
                fill=bg_color
            )
            # 文字
            draw.text((x, y), text, font=font, fill=color)

            # PIL → OpenCV
            frame[:, :, :] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # 原本 OpenCV 畫英文
            (text_width, text_height), baseline = cv2.getTextSize(
                text, self.font, font_scale, thickness
            )
            x, y = position
            # 繪製背景矩形
            cv2.rectangle(
                frame,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + baseline + padding),
                bg_color,
                -1
            )
            # 繪製文字
            cv2.putText(frame, text, position, self.font, font_scale, color, thickness)

    def _draw_hand_landmarks(self, frame: np.ndarray, hand_result: HandResult):
        """繪製手部骨架點與連線"""
        if hand_result.raw_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_result.raw_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=LANDMARK_COLOR, thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=2)
            )

    def _draw_gesture_info(
        self,
        frame: np.ndarray,
        gesture: Optional[GesturePrediction],
        no_hand_detected: bool = False
    ):
        """繪製手勢資訊（左上角）"""
        h, w = frame.shape[:2]

        if no_hand_detected:
            text = "No Hand Detected"
            text_zh = "未偵測到手部"
        elif gesture:
            text = f"Gesture: {gesture.class_name}"
            text_zh = f"手勢: {GESTURE_NAMES_ZH.get(gesture.class_id, '未知')}"
        else:
            text = "Detecting..."
            text_zh = "偵測中..."

        # 顯示英文
        self._put_text_with_background(frame, text, (10, 30))
        # 顯示中文
        self._put_text_with_background(frame, text_zh, (10, 65), font_scale=0.6)

        # 顯示信心度
        if gesture and not no_hand_detected:
            conf_text = f"Confidence: {gesture.confidence:.2f}"
            self._put_text_with_background(frame, conf_text, (10, 110), font_scale=0.5)

    def _draw_total_count(self, frame: np.ndarray, stats: StretchStats):
        """繪製伸展總次數（右上角）"""
        h, w = frame.shape[:2]
        text = f"Total: {stats.total_count}"
        # 計算文字寬度以靠右對齊
        (text_width, _), _ = cv2.getTextSize(text, self.font, FONT_SCALE, FONT_THICKNESS)
        self._put_text_with_background(frame, text, (w - text_width - 20, 30))

    def _draw_stats_by_type(self, frame: np.ndarray, stats: StretchStats):
        """繪製各類型伸展統計（底部）"""
        h, w = frame.shape[:2]

        # 建立統計文字
        stats_parts = []
        for stretch_type in VALID_STRETCH_TYPES:
            count = stats.counts_by_type.get(stretch_type, 0)
            # 簡化名稱顯示
            short_name = stretch_type[:4] if len(stretch_type) > 4 else stretch_type
            stats_parts.append(f"{short_name}:{count}")

        stats_text = " | ".join(stats_parts)

        # 顯示在底部
        self._put_text_with_background(
            frame, stats_text, (10, h - 20),
            font_scale=0.5, padding=3
        )

    def _draw_state_info(self, frame: np.ndarray, state_info: Dict):
        """繪製狀態機資訊"""
        h, w = frame.shape[:2]

        state_name = state_info.get("state", "")
        progress = state_info.get("progress", 0.0)
        hold_duration = state_info.get("hold_duration", 2.0)
        elapsed = state_info.get("elapsed", 0.0)
        start_gesture = state_info.get("start_gesture", "")

        # 狀態文字
        if state_name != "閒置":
            state_text = f"State: {state_name}"
            if start_gesture:
                state_text += f" ({start_gesture})"
            self._put_text_with_background(
                frame, state_text, (10, h - 80),
                font_scale=0.5, padding=3
            )

            # 進度條
            bar_width = 200
            bar_height = 15
            bar_x = 10
            bar_y = h - 55

            # 背景
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )
            # 進度
            fill_width = int(bar_width * progress)
            color = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                color,
                -1
            )
            # 時間文字
            time_text = f"{elapsed:.1f}s / {hold_duration:.1f}s"
            cv2.putText(
                frame, time_text,
                (bar_x + bar_width + 10, bar_y + 12),
                self.font, 0.4, TEXT_COLOR, 1
            )

    def render(
        self,
        frame: np.ndarray,
        hand_result: Optional[HandResult],
        gesture: Optional[GesturePrediction],
        stats: StretchStats,
        state_info: Dict
    ) -> np.ndarray:
        """
        渲染完整 UI

        Args:
            frame: 原始影像幀
            hand_result: 手部偵測結果
            gesture: 手勢預測結果
            stats: 伸展統計
            state_info: 狀態機資訊

        Returns:
            渲染後的影像幀
        """
        # 複製一份避免修改原始影像
        display_frame = frame.copy()

        # 繪製手部骨架
        if hand_result:
            self._draw_hand_landmarks(display_frame, hand_result)

        # 繪製手勢資訊
        self._draw_gesture_info(
            display_frame, gesture,
            no_hand_detected=(hand_result is None)
        )

        # 繪製伸展總次數
        self._draw_total_count(display_frame, stats)

        # 繪製各類型統計
        self._draw_stats_by_type(display_frame, stats)

        # 繪製狀態機資訊
        self._draw_state_info(display_frame, state_info)

        return display_frame
