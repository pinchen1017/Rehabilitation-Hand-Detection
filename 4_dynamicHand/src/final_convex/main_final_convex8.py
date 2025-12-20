"""
手部復健偵測系統 - v25 最終 UI 調整版
整合內容:
1. 系統架構: main_final3.py (DNN + Tracker + Renderer)。
2. 幾何核心: main_convex15-1.py (瘦身骨架 + AR回傳)。
3. 邏輯規則: main_convex15-1.py 的幾何判斷。
4. UI 調整: 
   - 移除主畫面的紅色錯誤框。
   - 將 CV 判斷結果 (文字) 移至 Debug Mask 視窗。
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
    """
    核心分析器 (來自 main_convex15-1.py)
    使用瘦身骨架參數 (Line=10, Circle=6)，確保 Defects 與 AR 計算準確。
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
        """
        計算幾何特徵 (維持瘦身版參數)
        """
        h, w, _ = clean_frame.shape
        
        # 1. ROI 切割
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

        # 2. Mask
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_skin = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask_skeleton = np.zeros_like(mask_skin)
        
        roi_landmarks = []
        for i in range(21):
            lx = int(hand_result.landmarks[i*3] * w) - x_min
            ly = int(hand_result.landmarks[i*3+1] * h) - y_min
            roi_landmarks.append((lx, ly))
            
        # 3. 骨架回填 (瘦身版: 線寬10, 圓半徑6)
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
        
        # 輕微形態學處理
        kernel = np.ones((3, 3), np.uint8)
        mask_final = cv2.dilate(mask_final, kernel, iterations=1)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=1)
        debug_mask = mask_final.copy()

        # 4. 輪廓
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0, 0, 0, 0, debug_mask
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return None, 0, 0, 0, 0, debug_mask

        # 5. 特徵計算
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        hull_points = cv2.convexHull(cnt, returnPoints=True)
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull_points)
        perimeter = cv2.arcLength(cnt, True)
        
        solidity = area_cnt / area_hull if area_hull > 0 else 0
        circularity = (4 * math.pi * area_cnt) / (perimeter ** 2) if perimeter > 0 else 0
        
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        aspect_ratio = float(rect_h) / rect_w if rect_w > 0 else 0

        # Defects Counting
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
        for point in hull_global:
            point[0][0] += x_min
            point[0][1] += y_min

        return hull_global, solidity, circularity, defects_count, aspect_ratio, debug_mask

def main():
    print("=" * 50)
    print("手部復健偵測系統 - v25 UI 調整版")
    print("邏輯來源: main_convex15-1.py")
    print("UI調整: 移除紅框，CV資訊移至 Debug 視窗")
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
        
        cv_group_name = "Wait..."
        
        # 準備 Debug Mask 顯示 (300x300, 黑色背景)
        debug_mask_display = np.zeros((300, 300, 3), dtype=np.uint8)

        if has_person:
            cv2.putText(display_frame, "[YOLO] Locked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            hand_result = detector.detect(frame)

            if hand_result is not None:
                # 1. DNN 預測
                raw_prediction = classifier.predict(hand_result.landmarks)
                gesture = smoother.smooth(raw_prediction)
                dnn_id = gesture.class_id if hasattr(gesture, 'class_id') else 0

                # 2. CV 幾何驗證
                hull, solidity, circularity, defects, aspect_ratio, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)
                
                # 處理 Mask 顯示 (轉成彩色以便寫字)
                if debug_mask is not None:
                    mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
                    debug_mask_display = cv2.resize(mask_bgr, (300, 300))

                if hull is not None:
                    cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)

                    # ==========================================================
                    # A. 幾何分類 (嚴格遵守 main_convex15-1.py 規則)
                    # ==========================================================
                    cv_group = "UNKNOWN"
                    
                    if defects >= 4:
                        cv_group = "OPEN" # Spread
                    
                    elif defects >= 3:
                        cv_group = "IDLE_PARTIAL" # 3指算 Idle
                    
                    # convex15-1 特有規則
                    elif defects == 0 and 0.9 < aspect_ratio < 1.34 and solidity > 0.82:
                        cv_group = "ROUND" # Fist/Angry
                    
                    elif defects == 0 or (aspect_ratio > 1.35 and solidity > 0.68):
                        cv_group = "LONG" # Thumb/Duck
                        
                    elif aspect_ratio < 1.0 and solidity > 0.65:
                        cv_group = "WIDE" # Straight
                        
                    elif 0.65 < solidity <= 0.82:
                        cv_group = "C-SHAPE" # Hook
                    
                    else:
                        cv_group = "IDLE"
                    
                    cv_group_name = cv_group

                    # ==========================================================
                    # B. 雙重驗證 (Double Verification)
                    # ==========================================================
                    is_match = False
                    
                    # 定義匹配規則
                    if dnn_id == 7 and cv_group == "OPEN": is_match = True
                    elif dnn_id == 3 and cv_group == "LONG": is_match = True
                    elif dnn_id == 5 and cv_group in ["LONG", "C-SHAPE", "ROUND"]: is_match = True
                    elif dnn_id in [2, 6] and cv_group == "ROUND": is_match = True
                    elif dnn_id == 4 and cv_group == "WIDE": is_match = True
                    elif dnn_id == 1 and cv_group == "C-SHAPE": is_match = True

                    # C. 更新 Tracker
                    gesture_to_send = None
                    if is_match:
                        # 驗證通過，正常計數
                        gesture_to_send = gesture
                        # [可選] 顯示綠色 MATCH 提示，如果想要更乾淨可以連這行也註解掉
                        cv2.putText(display_frame, f"MATCH: {gesture.class_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # 驗證失敗，送入 IDLE，但不顯示紅色框
                        if gesture:
                            gesture.class_id = 0
                            gesture_to_send = gesture
                        # 這裡移除了紅色的 MISMATCH 顯示

                    if gesture_to_send:
                        stretch_record = tracker.update(gesture_to_send)
                        if stretch_record:
                             print(f"完成伸展! {stretch_record.stretch_type}")

        else:
             cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # === E. 繪製 UI ===
        # 1. 【關鍵變更】將 CV 類別顯示在 Debug Mask 視窗上方 (藍綠色)
        cv2.putText(debug_mask_display, f"CV Logic: {cv_group_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 2. 呼叫 Renderer 畫最後的 UI (次數統計等)
        if has_person and hand_result and gesture:
            # 為了避免畫面閃爍，我們傳入 gesture_to_send (過濾後的)
            # 如果是 None (例如手剛伸出來還沒判斷好)，就傳原始 gesture
            render_gesture = gesture_to_send if gesture_to_send else gesture
            final_frame = renderer.render(display_frame, hand_result, render_gesture, tracker.get_stats(), tracker.get_state_info())
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