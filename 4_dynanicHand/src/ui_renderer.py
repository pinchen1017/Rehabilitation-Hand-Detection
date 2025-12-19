"""
手部復健偵測程式 - UI 渲染器模組
在 OpenCV 視窗上疊加顯示資訊
"""

from typing import Optional, Dict
import cv2
import numpy as np
import mediapipe as mp

from config import (
    GESTURE_NAMES_ZH,
    FONT_SCALE,
    FONT_THICKNESS,
    TEXT_COLOR,
    LANDMARK_COLOR,
    CONNECTION_COLOR,
    VALID_STRETCH_TYPES
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
        """在文字後方加上背景"""
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
        # 顯示中文（用英文代替，因為 OpenCV 預設不支援中文）
        self._put_text_with_background(frame, text_zh, (10, 65), font_scale=0.6)

        # 顯示信心度
        if gesture and not no_hand_detected:
            conf_text = f"Confidence: {gesture.confidence:.2f}"
            self._put_text_with_background(frame, conf_text, (10, 95), font_scale=0.5)

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
                frame, state_text, (10, h - 60),
                font_scale=0.5, padding=3
            )

            # 進度條
            bar_width = 200
            bar_height = 15
            bar_x = 10
            bar_y = h - 50

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
    
    # 新增凸包繪製功能
    def draw_convex_hull(self, image, hand_result):
        """
        [醫療分析版] 繪製凸包並計算復健指數
        """
        if not hand_result or hand_result.landmarks is None:
            return image

        h, w, c = image.shape
        points = []

        # 1. 取得 landmarks 並確保格式正確
        lms = hand_result.landmarks
        import numpy as np
        
        # 處理 NumPy 陣列格式
        if isinstance(lms, np.ndarray) or isinstance(lms, list):
            lms = np.array(lms)
            if lms.ndim == 1:
                lms = lms.reshape(-1, 3) # 轉回 (21, 3)
            
            # 轉換成像素座標
            for lm in lms:
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                points.append((cx, cy))
                
            # 取得關鍵點做為「正規化基準」
            # 點 0: 手腕 (Wrist)
            # 點 9: 中指根部 (Middle Finger MCP)
            p0 = lms[0]
            p9 = lms[9]
            # 計算基準長度 (像素距離)
            ref_len = np.linalg.norm(np.array([p0[0]*w, p0[1]*h]) - np.array([p9[0]*w, p9[1]*h]))
            
        else:
            # 處理原始 MediaPipe 物件格式 (備用)
            if hasattr(lms, 'landmark'): lms = lms.landmark
            elif hasattr(lms, 'landmarks'): lms = lms.landmarks
            
            for lm in lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
            
            p0 = lms[0]
            p9 = lms[9]
            ref_len = np.linalg.norm(np.array([p0.x*w, p0.y*h]) - np.array([p9.x*w, p9.y*h]))

        if not points:
            return image

        # 2. 計算凸包 (Convex Hull)
        points_np = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_np)
        
        # 3. 畫出黃色框框
        cv2.drawContours(image, [hull], -1, (0, 255, 255), 2)
        
        # ==========================================
        # 🏥 醫療分析核心 (Quantitative Analysis)
        # ==========================================
        
        # A. 計算凸包面積 (Pixel Area)
        hull_area = cv2.contourArea(hull)
        
        
        # B. 計算正規化指數 (修正版)
        # 試著把 2 改成 2.1, 2.2 或 2.3，直到你覺得遠近數值差不多為止
        # 乘數也可以調整 (例如 * 10 改成 * 15)，讓數值落在好看的區間 (0~100)
        
        exponent = 2.15  # <---【在這裡微調】建議從 2.1 開始試
        scale_factor = 15 # <---【在這裡微調】讓數字大小剛好落在 0~100 之間
        
        if ref_len > 0:
            rehab_index = (hull_area / (ref_len ** exponent)) * scale_factor
        else:
            rehab_index = 0
        

        # C. 設定顯示顏色 (視覺化回饋)
        # 假設: 指數 > 25 代表張很開 (綠色)，指數 < 15 代表握很緊 (紅色)
        # 這些數值你可以自己測試微調
        if rehab_index > 25:
            status_color = (0, 255, 0)   # 綠色 (Good Stretch)
            status_text = "Status: OPEN"
        elif rehab_index < 15:
            status_color = (0, 0, 255)   # 紅色 (Closed/Tight)
            status_text = "Status: CLOSED"
        else:
            status_color = (0, 255, 255) # 黃色 (Normal)
            status_text = "Status: RELAX"

        # D. 顯示數據在畫面上
        # 顯示指數數值
        cv2.putText(image, f"Rehab Index: {rehab_index:.1f}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4) # 黑邊
        cv2.putText(image, f"Rehab Index: {rehab_index:.1f}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)    # 彩字
        
        # 顯示面積 (Debug用，或是給老師看原始數據)
        cv2.putText(image, f"Area: {int(hull_area)}", (20, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return image