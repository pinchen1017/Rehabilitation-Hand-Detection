"""
手部復健偵測系統 - v29 最終流暢修正版
修正內容:
1. 修復 NameError: 修正 analyze_hand_geometry 函式中迴圈變數名稱錯誤 (p -> point)。
2. 維持 v28 邏輯:
   - Tracker 使用即時 DNN 結果 (gesture)，確保動態手勢計數流暢。
   - UI 提示使用 CV 結果 (Convex Hull)，在 Mask 視窗顯示動作建議。
"""

import sys
import cv2
import numpy as np
import os
import math
import time
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
    """
    核心分析器 (維持 convex16 的指縫優先邏輯參數)
    線寬 10 / 半徑 6，確保指縫計算準確。
    """
    def __init__(self):
        print("  [系統] 初始化 YOLOv8-Seg 與 CV 分析模組...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    def get_person_frame(self, frame):
        results = self.yolo_model(frame, verbose=False, stream=True)
        mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        person_found = False
        for r in results:
            if r.masks is None: continue
            max_area = 0
            best_mask = None
            for i, box in enumerate(r.boxes):
                if int(box.cls[0]) == 0:
                    mask_raw = r.masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
                    if np.sum(mask_resized) > max_area:
                        max_area = np.sum(mask_resized)
                        best_mask = (mask_resized * 255).astype(np.uint8)
                        person_found = True
            if best_mask is not None: mask_combined = best_mask

        if person_found:
            clean_frame = cv2.bitwise_and(frame, frame, mask=mask_combined)
            return True, clean_frame, mask_combined
        return False, frame, None

    def analyze_hand_geometry(self, clean_frame, hand_result):
        """計算幾何特徵"""
        h, w, _ = clean_frame.shape
        
        # ROI 切割
        x_list = [int(lm * w) for lm in hand_result.landmarks[0::3]]
        y_list = [int(lm * h) for lm in hand_result.landmarks[1::3]]
        x_min, x_max = max(0, min(x_list)), min(w, max(x_list))
        y_min, y_max = max(0, min(y_list)), min(h, max(y_list))
        
        padding = 80
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        roi_img = clean_frame[y_min:y_max, x_min:x_max]
        if roi_img.size == 0: return None, 0, 0, 0, 0, None

        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_skin = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask_skeleton = np.zeros_like(mask_skin)
        
        roi_landmarks = []
        for i in range(21):
            lx = int(hand_result.landmarks[i*3] * w) - x_min
            ly = int(hand_result.landmarks[i*3+1] * h) - y_min
            roi_landmarks.append((lx, ly))
            
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_points = np.array([roi_landmarks[i] for i in palm_indices], dtype=np.int32)
        cv2.fillConvexPoly(mask_skeleton, palm_points, 255)
        
        finger_connections = [(2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20), (0,5,9,13,17)]
        for conn in finger_connections:
            for i in range(len(conn)-1):
                pt1, pt2 = roi_landmarks[conn[i]], roi_landmarks[conn[i+1]]
                if 0 <= pt1[0] < roi_img.shape[1] and 0 <= pt1[1] < roi_img.shape[0]:
                    cv2.line(mask_skeleton, pt1, pt2, 255, 10)
                    cv2.circle(mask_skeleton, pt1, 6, 255, -1)
                    cv2.circle(mask_skeleton, pt2, 6, 255, -1)

        mask_final = cv2.bitwise_or(mask_skin, mask_skeleton)
        kernel = np.ones((3, 3), np.uint8)
        mask_final = cv2.dilate(mask_final, kernel, iterations=1)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=1)
        debug_mask = mask_final.copy()

        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0, 0, 0, 0, debug_mask
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return None, 0, 0, 0, 0, debug_mask

        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        hull_points = cv2.convexHull(cnt, returnPoints=True)
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull_points)
        perimeter = cv2.arcLength(cnt, True)
        
        solidity = area_cnt / area_hull if area_hull > 0 else 0
        circularity = (4 * math.pi * area_cnt) / (perimeter ** 2) if perimeter > 0 else 0
        
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        aspect_ratio = float(rect_h) / rect_w if rect_w > 0 else 0

        defects_count = 0
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(cnt, hull_indices)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0
                        far = tuple(cnt[f][0])
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        if 2*b*c > 0:
                            angle = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c)))) * 57.2958
                            if angle < 100 and depth > 8:
                                defects_count += 1
                                cv2.circle(debug_mask, far, 5, 128, -1)
            except: pass

        hull_global = hull_points.copy()
        # 【修正】這裡原本寫 for p in ... 但下面用 point，已修正為 for point in ...
        for point in hull_global:
            point[0][0] += x_min
            point[0][1] += y_min

        return hull_global, solidity, circularity, defects_count, aspect_ratio, debug_mask

def main():
    print("=" * 50)
    print("手部復健偵測系統 - v29 流暢修正版")
    print("特色: 恢復動態手勢流暢度，CV 提示僅供參考")
    print("=" * 50)

    try:
        classifier = GestureClassifier(MODEL_PATH)
        print("  [DNN] 模型載入成功!")
    except Exception as e:
        print(f"  [錯誤] 模型載入失敗: {e}")
        sys.exit(1)

    detector = HandDetector()
    # Smoother 本身就有微小的 buffer，足夠過濾雜訊，不需要額外加長時間的 debounce
    smoother = GestureSmoother(window_size=5) 
    tracker = StretchTracker()
    renderer = UIRenderer()
    cv_analyzer = CVAnalyzer()
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow("Rehab System", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: ROI Mask", cv2.WINDOW_NORMAL)

    print("\n系統啟動完成 - 按 'q' 退出, 'r' 重置")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        has_person, clean_frame, _ = cv_analyzer.get_person_frame(frame)
        display_frame = frame.copy()
        
        hand_result = None
        gesture = None
        
        # 狀態變數初始化
        cv_group_name = "Wait..."
        hint_msg = "Focusing..."
        hint_color = (200, 200, 200) 
        debug_mask_display = np.zeros((300, 300, 3), dtype=np.uint8)
        progress_bar_val = 0.0

        if has_person:
            cv2.putText(display_frame, "[YOLO] Locked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            hand_result = detector.detect(frame)

            if hand_result is not None:
                # 1. DNN 預測 (權威結果)
                raw_prediction = classifier.predict(hand_result.landmarks)
                # 使用 Smoother 進行極短時間的平滑 (約 0.1~0.2秒)，保證 ID 不亂跳但反應夠快
                gesture = smoother.smooth(raw_prediction)
                dnn_id = gesture.class_id if hasattr(gesture, 'class_id') else 0

                # 2. CV 幾何分析 (輔助分析)
                hull, solidity, circularity, defects, aspect_ratio, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)
                
                # 處理 Mask 顯示
                if debug_mask is not None:
                    mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
                    debug_mask_display = cv2.resize(mask_bgr, (300, 300))

                if hull is not None:
                    cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)

                    # === A. 幾何分類 (依據 convex16 指縫優先規則) ===
                    cv_group = "UNKNOWN"
                    if defects >= 4: cv_group = "OPEN"
                    elif defects >= 3: cv_group = "IDLE"
                    elif defects == 0 and 0.9 < aspect_ratio < 1.34 and solidity > 0.82: cv_group = "ROUND"
                    elif defects == 0 or (aspect_ratio > 1.35 and solidity > 0.68): cv_group = "LONG"
                    elif aspect_ratio < 1.0 and solidity > 0.65: cv_group = "WIDE"
                    elif 0.65 < solidity <= 0.82: cv_group = "C-SHAPE"
                    else: cv_group = "IDLE"
                    
                    cv_group_name = cv_group

                    # === B. 教練提示邏輯 (不影響計數) ===
                    # 邏輯: 你的 ID 是對的，但我建議你姿勢可以更好
                    
                    hint_msg = "Perfect!"
                    hint_color = (0, 255, 0) # Green

                    # 根據 ID 檢查形狀特徵
                    if dnn_id == 7: # Spread
                        if cv_group != "OPEN":
                            hint_msg = "Open Fingers!"
                            hint_color = (0, 255, 255)
                    elif dnn_id in [2, 6]: # Fist
                        if cv_group != "ROUND":
                            if defects > 0: hint_msg = "Close Gaps!"
                            else: hint_msg = "Grip Tighter!"
                            hint_color = (0, 255, 255)
                    elif dnn_id == 4: # Straight
                        if cv_group != "WIDE":
                            hint_msg = "Flatten Hand!"
                            hint_color = (0, 255, 255)
                    elif dnn_id == 1: # Hook
                        if cv_group != "C-SHAPE":
                            hint_msg = "Bend Fingers!"
                            hint_color = (0, 255, 255)
                    elif dnn_id == 3: # Thumb Flexion
                        if cv_group != "LONG":
                            hint_msg = "Four fingers together!"
                            hint_color = (0, 255, 255)
                    elif dnn_id == 5: # The Duck
                        if cv_group != "LONG":
                            hint_msg = "Form a Beak!"
                            hint_color = (0, 255, 255)
                    elif dnn_id == 0: # Idle
                        if cv_group != "IDLE":
                            hint_msg = "Relax!"
                            hint_color = (0, 255, 0)

                    # 進度條 (視覺效果)
                    if dnn_id == 7: progress_bar_val = np.clip((0.75 - solidity) / (0.75 - 0.55), 0, 1)
                    elif dnn_id in [3, 5]: progress_bar_val = np.clip((aspect_ratio - 1.3) / (1.6 - 1.3), 0, 1)
                    else: progress_bar_val = np.clip((solidity - 0.60) / (0.85 - 0.60), 0, 1)

                # === C. 更新 Tracker (移除人為延遲) ===
                # 直接使用經過 Smooth 的 gesture 更新 Tracker
                # Tracker 內部有狀態機邏輯，可以處理動態變化，不需要外部 Debounce 阻擋
                if gesture:
                    stretch_record = tracker.update(gesture)
                    if stretch_record:
                        print(f"完成伸展! {stretch_record.stretch_type} (Total: {tracker.get_stats().total_count})")

            else:
                pass # 沒偵測到手
        else:
             cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # === D. 繪製 UI ===
        # 1. Debug Mask 視窗 (顯示詳細提示)
        cv2.putText(debug_mask_display, f"CV: {cv_group_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(debug_mask_display, f"Hint: {hint_msg}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hint_color, 2)
        
        if has_person and hand_result and hull is not None:
             cv2.putText(debug_mask_display, f"D:{defects} S:{solidity:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 2. 主畫面 (只顯示進度條與 Tracker 結果)
        bar_color = (0, 255, 0)
        cv2.rectangle(display_frame, (50, 150), (80, 350), (255, 255, 255), 2)
        cv2.rectangle(display_frame, (50, 350 - int(200*progress_bar_val)), (80, 350), bar_color, -1)

        if has_person and hand_result and gesture:
            final_frame = renderer.render(display_frame, hand_result, gesture, tracker.get_stats(), tracker.get_state_info())
        else:
            final_frame = display_frame

        cv2.imshow("Rehab System", final_frame)
        cv2.imshow("Debug: ROI Mask", debug_mask_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            tracker.reset()
            smoother.reset()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()