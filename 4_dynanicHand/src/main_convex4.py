"""
手部復健偵測系統 - 客製化規則版 (修正 NameError)
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
        # 稍微放寬膚色範圍
        self.lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    def get_person_frame(self, frame):
        # YOLO 去背 (維持不變)
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
            return True, clean_frame
        return False, frame

    def analyze_hand_geometry(self, clean_frame, hand_result):
        """
        計算幾何特徵 (Solidity, Circularity, Defects, AspectRatio)
        """
        h, w, _ = clean_frame.shape
        
        # 1. 計算 ROI (Padding 80)
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

        # 2. 建立 Mask (HSV + 骨架回填)
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_skin = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask_skeleton = np.zeros_like(mask_skin)
        
        roi_landmarks = []
        for i in range(21):
            lx = int(hand_result.landmarks[i*3] * w) - x_min
            ly = int(hand_result.landmarks[i*3+1] * h) - y_min
            roi_landmarks.append((lx, ly))
            
        # 填滿手掌
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_points = np.array([roi_landmarks[i] for i in palm_indices], dtype=np.int32)
        cv2.fillConvexPoly(mask_skeleton, palm_points, 255)
        
        # 畫骨架 (瘦身版：線寬10，圓半徑6) -> 為了保留指縫
        # 
        finger_connections = [(2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20), (0,5,9,13,17)]
        
        # === 【修正點】 變數名稱修正 ===
        for conn in finger_connections: # 這裡原本寫錯成 connections
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

        # 3. 找輪廓
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0, 0, 0, 0, debug_mask
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return None, 0, 0, 0, 0, debug_mask

        # 4. 計算特徵
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        hull_points = cv2.convexHull(cnt, returnPoints=True)
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull_points)
        perimeter = cv2.arcLength(cnt, True)
        
        # [特徵 1] 實心度
        solidity = area_cnt / area_hull if area_hull > 0 else 0
        
        # [特徵 2] 圓形度
        circularity = (4 * math.pi * area_cnt) / (perimeter ** 2) if perimeter > 0 else 0
        
        # [特徵 3] 長寬比 (Aspect Ratio)
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        aspect_ratio = float(rect_h) / rect_w if rect_w > 0 else 0

        # [特徵 4] 指縫 (Defects)
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
                            angle = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c)))) * 57.3
                            if angle < 100 and depth > 8:
                                defects_count += 1
                                cv2.circle(debug_mask, far, 5, 128, -1)
            except: pass

        hull_global = hull_points.copy()
        for p in hull_global:
            p[0][0] += x_min
            p[0][1] += y_min

        return hull_global, solidity, circularity, defects_count, aspect_ratio, debug_mask

def main():
    print("="*50)
    print("手部復健 - 客製化規則版 (修正變數錯誤)")
    print("="*50)

    try:
        classifier = GestureClassifier(MODEL_PATH)
        print("  [DNN] 模型載入成功")
    except: sys.exit(1)

    detector = HandDetector()
    smoother = GestureSmoother()
    tracker = StretchTracker()
    renderer = UIRenderer()
    cv_analyzer = CVAnalyzer()
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow("Rehab System", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: ROI Mask", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        has_person, clean_frame = cv_analyzer.get_person_frame(frame)
        display_frame = frame.copy()
        
        hand_result = detector.detect(frame) # DNN 讀原圖
        gesture = None
        warning_msg = ""

        if has_person and hand_result:
            raw_pred = classifier.predict(hand_result.landmarks)
            gesture = smoother.smooth(raw_pred)
            dnn_id = gesture.class_id if hasattr(gesture, 'class_id') else 0
            
            # CV 分析 (讀去背圖)
            hull, solidity, circularity, defects, aspect_ratio, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)

            if debug_mask is not None:
                cv2.imshow("Debug: ROI Mask", debug_mask)
                # 顯示 AR 方便調試
                cv2.putText(display_frame, f"AR:{aspect_ratio:.2f} Sol:{solidity:.2f}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.imshow("Debug: ROI Mask", np.zeros((200,200), np.uint8))

            if hull is not None:
                cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)
                
                # ========================================================
                # 針對您的 7 點定義，客製化驗證規則
                # ========================================================
                
                # 1. Angry Fist & 6. Fist (小面積、實心、圓)
                if dnn_id in [2, 6]:
                    if solidity < 0.70:
                        warning_msg = "Grip Tighter!"
                
                # 3. Hook (大拇指凸出、非實心、偏長)
                elif dnn_id == 1:
                    # 允許實心度低 (因為 C 型有缺口)，但形狀要偏長
                    if aspect_ratio < 1.1: 
                        warning_msg = "Extend Thumb!"
                
                # 4. Straight Hand (大拇指凸出、實心、圓) -> Best of Luck
                elif dnn_id == 4:
                    if solidity < 0.70:
                        warning_msg = "Close Fingers!"
                
                # 5. The Duck (三角形)
                elif dnn_id == 5:
                    if defects > 2: 
                        warning_msg = "Form a Beak!"
                
                # 6. Thumb Flexion (大拇指折進去、四指併攏伸直 -> 長條形)
                elif dnn_id == 3:
                    if aspect_ratio < 1.25: 
                        warning_msg = "Straighten Fingers!"
                    if solidity < 0.70:     
                        warning_msg = "Close Gaps!"
                
                # 7. Spread Hand (五指伸展)
                elif dnn_id == 7:
                    if defects < 2 and solidity > 0.70:
                        warning_msg = "Spread Fingers!"

                # 更新 Tracker
                stretch_record = tracker.update(gesture)
                if stretch_record: print(f"完成: {stretch_record.stretch_type}")

                # 繪製進度條
                progress = np.clip((solidity - 0.60) / (0.85 - 0.60), 0, 1)
                bar_color = (0, 255, 0) if defects >= 3 else (0, int(255*(1-progress)), int(255*progress))
                cv2.rectangle(display_frame, (50, 150), (80, 350), (255, 255, 255), 2)
                cv2.rectangle(display_frame, (50, 350 - int(200*progress)), (80, 350), bar_color, -1)

        else:
             cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if has_person and hand_result and gesture:
            final_frame = renderer.render(display_frame, hand_result, gesture, tracker.get_stats(), tracker.get_state_info())
            if warning_msg: cv2.putText(final_frame, warning_msg, (180, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            final_frame = display_frame

        cv2.imshow("Rehab System", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()