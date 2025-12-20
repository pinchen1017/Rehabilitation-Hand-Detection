"""
手部復健偵測系統 - 最終精確版 (ROI 鎖定 + 視覺化驗證)
改進點:
1. 利用 DNN 座標鎖定手部區域 (ROI)，解決 Convex Hull 亂框問題。
2. 新增 Debug 視窗顯示 HSV 處理結果。
"""

import sys
import cv2
import numpy as np
import os
import math
from ultralytics import YOLO

# 引入原專案模組
from config import DEFAULT_MODEL_PATH, DEFAULT_YOLO_MODEL_PATH
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier, GestureSmoother
from stretch_tracker import StretchTracker
from ui_renderer import UIRenderer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", DEFAULT_MODEL_PATH) 
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "..", DEFAULT_YOLO_MODEL_PATH)

class CVAnalyzer:
    def __init__(self):
        print("  [系統] 初始化 YOLOv8-Seg 與 CV 分析模組...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # 膚色閥值 (HSV) - 可根據現場光線微調
        self.lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    def get_person_frame(self, frame):
        """YOLO: 鎖定最大的人並去背"""
        results = self.yolo_model(frame, verbose=False, stream=True)
        mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        person_found = False

        for r in results:
            if r.masks is None: continue
            max_area = 0
            best_mask = None
            boxes = r.boxes
            masks = r.masks
            for i, box in enumerate(boxes):
                if int(box.cls[0]) == 0: # Person
                    mask_raw = masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
                    area = np.sum(mask_resized)
                    if area > max_area:
                        max_area = area
                        best_mask = (mask_resized * 255).astype(np.uint8)
                        person_found = True
            if best_mask is not None:
                mask_combined = best_mask

        if person_found:
            clean_frame = cv2.bitwise_and(frame, frame, mask=mask_combined)
            return True, clean_frame, mask_combined
        else:
            return False, frame, None

    def analyze_hand_geometry(self, clean_frame, hand_result):
        """
        改良版幾何分析：
        只針對 MediaPipe 抓到的手部區域 (ROI) 做 Convex Hull
        """
        h, w, _ = clean_frame.shape
        
        # 1. 計算 ROI (Region of Interest)
        # 從 normalized landmarks 轉換回像素座標
        x_list = [int(lm * w) for lm in hand_result.landmarks[0::3]] # x coordinates
        y_list = [int(lm * h) for lm in hand_result.landmarks[1::3]] # y coordinates
        
        x_min, x_max = max(0, min(x_list)), min(w, max(x_list))
        y_min, y_max = max(0, min(y_list)), min(h, max(y_list))
        
        # 往外擴張一點範圍 (Padding)，以免切太貼
        padding = 40
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # 2. 切割出 ROI 圖片
        roi_img = clean_frame[y_min:y_max, x_min:x_max]
        
        if roi_img.size == 0:
            return None, 0, 0, None

        # 3. 在 ROI 內做 HSV 膚色偵測
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 形態學優化
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 回傳這個 mask 給主程式顯示 (視覺化驗證)
        debug_mask = mask.copy()

        # 4. 找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0, 0, debug_mask

        # 找最大輪廓
        cnt = max(contours, key=cv2.contourArea)
        area_cnt = cv2.contourArea(cnt)
        
        if area_cnt < 500: # ROI 裡的雜訊過濾
            return None, 0, 0, debug_mask

        # 5. 計算幾何特徵
        hull_local = cv2.convexHull(cnt) # 這是相對於 ROI 的座標
        area_hull = cv2.contourArea(hull_local)
        perimeter = cv2.arcLength(cnt, True)
        
        if area_hull == 0 or perimeter == 0: 
            return None, 0, 0, debug_mask

        solidity = area_cnt / area_hull
        circularity = (4 * math.pi * area_cnt) / (perimeter ** 2)
        
        # 6. 將 Hull 座標轉回全域座標 (Global Coordinates) 才能畫在原圖上
        hull_global = hull_local.copy()
        for point in hull_global:
            point[0][0] += x_min # Shift X
            point[0][1] += y_min # Shift Y

        return hull_global, solidity, circularity, debug_mask

def main():
    print("=" * 50)
    print("手部復健偵測系統 (ROI 精確版)")
    print("=" * 50)

    try:
        classifier = GestureClassifier(MODEL_PATH)
        print("  [DNN] 模型載入成功!")
    except Exception as e:
        print(f"  [錯誤] 模型載入失敗: {e}")
        sys.exit(1)

    detector = HandDetector()
    smoother = GestureSmoother()
    tracker = StretchTracker()
    renderer = UIRenderer()
    cv_analyzer = CVAnalyzer()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Rehab System", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: ROI Mask", cv2.WINDOW_NORMAL) # 新增 Debug 視窗

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. YOLO 去背
        has_person, clean_frame, _ = cv_analyzer.get_person_frame(frame)
        
        display_frame = frame.copy()
        warning_msg = ""
        cv_feedback = ""
        
        if has_person:
            cv2.putText(display_frame, "[YOLO] Locked", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 2. DNN 手部偵測 (取得座標)
            hand_result = detector.detect(clean_frame)
            
            gesture_obj = None
            gesture_int = 0
            hull = None
            solidity = 0
            circularity = 0
            
            if hand_result:
                # DNN 預測
                raw_prediction = classifier.predict(hand_result.landmarks)
                gesture_obj = smoother.smooth(raw_prediction)
                
                if hasattr(gesture_obj, 'class_id'):
                    gesture_int = gesture_obj.class_id
                else:
                    gesture_int = 0

                tracker.update(gesture_obj)

                # 3. CV 幾何驗證 (改用 ROI 鎖定法)
                # 傳入 hand_result 以便切出小框框
                hull, solidity, circularity, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)
                
                # 顯示 Debug 畫面
                if debug_mask is not None:
                    cv2.imshow("Debug: ROI Mask", debug_mask)
                else:
                    # 如果沒切出東西，顯示全黑
                    cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))

                if hull is not None:
                    # 畫出藍色凸包
                    
                    cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)
                    
                    # === 針對七種手勢的判斷邏輯 ===
                    # 0:idle, 1:hook, 2:angry, 3:thumb, 4:straight, 5:duck, 6:fist, 7:spend
                    
                    # 定義「緊握類」: 這些手勢理論上應該比較實心
                    closed_group = [1, 2, 3, 4, 5, 6] 
                    
                    # 定義「完全張開」
                    open_group = [7] 
                    
                    # 判斷物理狀態
                    # 只要比較實心 (Solidity高) 或是 形狀很圓 (Circularity高)，都視為有抓握
                    is_physically_gripped = (solidity > 0.70) or (circularity > 0.60)
                    
                    # 驗證 1: DNN 說是「握拳類」，但 CV 算出來很鬆散
                    if (gesture_int in closed_group) and (not is_physically_gripped):
                        warning_msg = "Warning: Grip Not Tight!"
                        cv_feedback = "Force: Low"
                    
                    # 驗證 2: DNN 說是「張開」，但 CV 算出來很實心
                    if (gesture_int in open_group) and (is_physically_gripped):
                        warning_msg = "Warning: Hand Not Open!"
                        cv_feedback = "Relax Hand"

                    # 繪製進度條
                    bar_h = 200
                    # 調整顯示範圍：0.6 (鬆) ~ 0.85 (緊)
                    progress = np.clip((solidity - 0.60) / (0.85 - 0.60), 0, 1)
                    
                    # 顏色隨力度變化 (綠->紅)
                    bar_color = (0, int(255 * (1-progress)), int(255 * progress))
                    
                    cv2.rectangle(display_frame, (50, 150), (80, 150+bar_h), (255, 255, 255), 2)
                    cv2.rectangle(display_frame, (50, 150 + bar_h - int(bar_h*progress)), 
                                 (80, 150+bar_h), bar_color, -1)
                    cv2.putText(display_frame, f"{int(progress*100)}%", (45, 145), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
                    
                    # 顯示數值以便觀察
                    cv2.putText(display_frame, f"Sol: {solidity:.2f}", (90, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                 cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))

        # 4. 渲染 UI
        if has_person and hand_result and gesture_obj is not None:
            final_frame = renderer.render(
                display_frame, hand_result, gesture_obj, 
                tracker.get_stats(), tracker.get_state_info()
            )
            if warning_msg:
                cv2.putText(final_frame, warning_msg, (200, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(final_frame, cv_feedback, (200, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            final_frame = display_frame

        cv2.imshow("Rehab System", final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            tracker.reset()
            smoother.reset()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()