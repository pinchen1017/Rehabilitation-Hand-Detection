"""
手部復健偵測程式 - 資料蒐集模組
提供簡易 OpenCV 前端，蒐集手勢特徵點資料
"""

import os
import sys
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from config import (
    DATA_RAW_DIR,
    RAW_DATA_FILE,
    TARGET_SAMPLES_PER_CLASS,
    NUM_CLASSES,
    GESTURE_NAMES,
    GESTURE_NAMES_ZH,
    FONT_SCALE,
    FONT_THICKNESS,
    TEXT_COLOR,
    LANDMARK_COLOR,
    CONNECTION_COLOR,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_MAX_HANDS
)
from hand_detector import HandDetector


class DataCollector:
    """資料蒐集器 - 含簡易 OpenCV 前端"""

    def __init__(self, output_path: Optional[str] = None):
        """
        初始化資料蒐集器

        Args:
            output_path: 輸出 CSV 檔案路徑，預設為 data/raw/gesture_data.csv
        """
        # 重用既有的 HandDetector
        self.detector = HandDetector(
            max_hands=DEFAULT_MAX_HANDS,
            min_detection_confidence=DEFAULT_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=DEFAULT_MIN_TRACKING_CONFIDENCE
        )

        # 設定輸出路徑
        if output_path is None:
            os.makedirs(DATA_RAW_DIR, exist_ok=True)
            self.output_path = os.path.join(DATA_RAW_DIR, RAW_DATA_FILE)
        else:
            self.output_path = output_path

        # 初始化狀態
        self.current_class: int = 0
        self.samples: List[Dict] = []
        self.counts_per_class: Dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}
        self.is_collecting: bool = False

        # MediaPipe 繪圖工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # UI 字體
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _collect_sample(self, landmarks: np.ndarray, label: int):
        """
        蒐集一筆樣本

        Args:
            landmarks: 63 維特徵點向量
            label: 類別標籤 (0-7)
        """
        sample = {"label": label}
        for i in range(21):
            sample[f"x{i}"] = float(landmarks[i * 3])
            sample[f"y{i}"] = float(landmarks[i * 3 + 1])
            sample[f"z{i}"] = float(landmarks[i * 3 + 2])
        self.samples.append(sample)
        self.counts_per_class[label] += 1
        
        # 檢查當前類別是否達標
        if self.counts_per_class[self.current_class] >= TARGET_SAMPLES_PER_CLASS:
            print(f"  類別 {self.current_class} 已達標 ({TARGET_SAMPLES_PER_CLASS} 筆)")
            self.is_collecting = False

    def save_data(self) -> str:
        """
        儲存資料至 CSV 檔案

        Returns:
            儲存的檔案路徑
        """
        if not self.samples:
            print("警告：沒有資料可儲存")
            return ""

        df = pd.DataFrame(self.samples)

        # 確保目錄存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # 如果檔案已存在，追加資料
        if os.path.exists(self.output_path):
            existing_df = pd.read_csv(self.output_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(self.output_path, index=False)
        print(f"資料已儲存至: {self.output_path}")
        print(f"總樣本數: {len(df)}")
        return self.output_path

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
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1
        )
        cv2.putText(frame, text, position, self.font, font_scale, color, thickness)

    def _render_ui(
        self,
        frame: np.ndarray,
        hand_result,
        collecting: bool,
        no_hand: bool
    ) -> np.ndarray:
        """
        繪製 UI 介面

        Args:
            frame: 原始影像幀
            hand_result: 手部偵測結果
            collecting: 是否正在蒐集資料
            no_hand: 是否未偵測到手部

        Returns:
            渲染後的影像幀
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # 蒐集中顯示綠色邊框
        if collecting and not no_hand:
            cv2.rectangle(display_frame, (0, 0), (w-1, h-1), (0, 255, 0), 5)

        # 繪製手部骨架
        if hand_result and hand_result.raw_landmarks:
            self.mp_drawing.draw_landmarks(
                display_frame,
                hand_result.raw_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=LANDMARK_COLOR, thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=2)
            )

        # 左上角：目前選擇的類別
        class_name = GESTURE_NAMES.get(self.current_class, "unknown")
        class_name_zh = GESTURE_NAMES_ZH.get(self.current_class, "未知")
        class_text = f"Class: {self.current_class} - {class_name}"
        class_text_zh = f"類別: {class_name_zh}"
        self._put_text_with_background(display_frame, class_text, (10, 30))
        self._put_text_with_background(display_frame, class_text_zh, (10, 65), font_scale=0.6)

        # 右上角：各類別已蒐集數量
        y_offset = 30
        self._put_text_with_background(
            display_frame, "Samples:", (w - 150, y_offset), font_scale=0.5
        )
        y_offset += 25
        for i in range(NUM_CLASSES):
            count = self.counts_per_class[i]
            target = TARGET_SAMPLES_PER_CLASS
            color = (0, 255, 0) if count >= target else TEXT_COLOR
            count_text = f"{i}: {count}/{target}"
            self._put_text_with_background(
                display_frame, count_text, (w - 150, y_offset),
                font_scale=0.4, color=color, padding=2
            )
            y_offset += 20

        # 中央：狀態提示
        if no_hand:
            status_text = "No Hand Detected"
            status_color = (0, 0, 255)  # 紅色
            self._put_text_with_background(
                display_frame, status_text, (w // 2 - 100, h // 2),
                color=status_color, font_scale=0.8
            )
        elif collecting:
            status_text = "Collecting..."
            status_color = (0, 255, 0)  # 綠色
            self._put_text_with_background(
                display_frame, status_text, (w // 2 - 80, h // 2),
                color=status_color, font_scale=0.8
            )

        # 底部：操作說明
        instructions = [
            "0-7: Select class | SPACE: Collect | S: Save | Q: Quit"
        ]
        for i, instruction in enumerate(instructions):
            self._put_text_with_background(
                display_frame, instruction, (10, h - 20 - i * 25),
                font_scale=0.5, padding=3
            )

        # 顯示本次蒐集的總數
        total_text = f"Total this session: {len(self.samples)}"
        self._put_text_with_background(
            display_frame, total_text, (10, 100), font_scale=0.5
        )

        return display_frame

    def run(self):
        """執行蒐集主迴圈"""
        print("=" * 50)
        print("手勢資料蒐集程式")
        print("=" * 50)
        print("操作說明:")
        print("  數字鍵 0-7: 切換目標類別")
        print("  空白鍵 (按住): 蒐集資料")
        print("  S: 儲存資料")
        print("  Q: 退出程式")
        print("=" * 50)

        # 開啟攝影機
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("錯誤：無法開啟攝影機")
            return

        print("\n攝影機已開啟，開始蒐集...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("警告：無法讀取攝影機畫面")
                    break

                # 水平翻轉（鏡像效果）
                frame = cv2.flip(frame, 1)

                # 偵測手部
                hand_result = self.detector.detect(frame)
                no_hand = hand_result is None

                # 檢查按鍵狀態
                key = cv2.waitKey(1) & 0xFF

                # 數字鍵 0-7 切換類別
                if ord('0') <= key <= ord('7'):
                    self.current_class = key - ord('0')
                    print(f"切換至類別 {self.current_class}: {GESTURE_NAMES[self.current_class]}")

                # 空白鍵蒐集資料
                self.is_collecting = (key == ord(' '))
                if self.is_collecting:
                    if no_hand:
                        # 無手部偵測時不蒐集
                        pass
                    else:
                        # 蒐集資料
                        self._collect_sample(hand_result.landmarks, self.current_class)

                # S 儲存資料
                if key == ord('s') or key == ord('S'):
                    self.save_data()

                # Q 退出
                if key == ord('q') or key == ord('Q'):
                    print("\n使用者退出程式")
                    break

                # 渲染 UI
                display_frame = self._render_ui(
                    frame, hand_result, self.is_collecting, no_hand
                )

                # 顯示畫面
                cv2.imshow("Gesture Data Collector", display_frame)

        except KeyboardInterrupt:
            print("\n程式被中斷")

        finally:
            # 釋放資源
            cap.release()
            self.detector.close()
            cv2.destroyAllWindows()

            # 顯示蒐集摘要
            self._print_summary()

    def _print_summary(self):
        """顯示蒐集摘要"""
        print("\n" + "=" * 50)
        print("蒐集摘要")
        print("=" * 50)
        print(f"本次蒐集總數: {len(self.samples)}")
        print("\n各類別數量:")
        for i in range(NUM_CLASSES):
            count = self.counts_per_class[i]
            name = GESTURE_NAMES.get(i, "unknown")
            name_zh = GESTURE_NAMES_ZH.get(i, "未知")
            status = "✓" if count >= TARGET_SAMPLES_PER_CLASS else " "
            print(f"  [{status}] {i}: {name} ({name_zh}): {count}")
        print("=" * 50)

        # 提示未儲存的資料
        if self.samples:
            print("\n警告：有未儲存的資料！")
            save_input = input("是否儲存？(y/n): ")
            if save_input.lower() == 'y':
                self.save_data()


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
