"""
純幾何特徵測試儀 - v10 修正版 (Pure CV Fixed)
修正內容:
1. 解決左手 Round 旋轉誤判問題 (優化 Solidity 與 AR 的權重)。
2. Spread Hand 嚴格判定 (需 5 指)。
3. 修復 Python 縮排錯誤。
"""

import sys
import cv2
import numpy as np
import os
import math
from ultralytics import YOLO
from hand_detector import HandDetector

# 設定路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from config import DEFAULT_YOLO_MODEL_PATH
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "..", DEFAULT_YOLO_MODEL_PATH)

class CVAnalyzer:
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
            return True, clean_frame
        return False, frame

    def analyze_hand_geometry(self, clean_frame, hand_result):
        """
        CV 分析核心 (維持骨架瘦身版)
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

        # 2. 建立 Mask
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
        
        # 畫骨架 (細線模式)
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
        
        solidity = area_cnt / area_hull if area_hull > 0 else 0
        circularity = (4 * math.pi * area_cnt) / (perimeter ** 2) if perimeter > 0 else 0
        
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        aspect_ratio = float(rect_h) / rect_w if rect_w > 0 else 0

        # 5. 指縫計數
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
    print("純幾何特徵測試儀 - v10 修正版 (Pure CV Fixed)")
    print("邏輯修正: 左手 Round 旋轉優化 + 嚴格 Spread Hand")
    print("="*50)

    detector = HandDetector()
    cv_analyzer = CVAnalyzer()
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Pure CV Analysis", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: ROI Mask", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        has_person, clean_frame = cv_analyzer.get_person_frame(frame)
        display_frame = frame.copy()
        
        hand_result = detector.detect(frame) 

        if has_person and hand_result:
            cv2.putText(display_frame, "Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            hull, solidity, circularity, defects, ar, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)
            
            if debug_mask is not None:
                cv2.imshow("Debug: ROI Mask", debug_mask)
            else:
                cv2.imshow("Debug: ROI Mask", np.zeros((200,200), np.uint8))

            if hull is not None:
                cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)
                
                # ========================================================
                # 嚴格化群組分類邏輯 (v10 Update)
                # ========================================================
                
                group_name = "Unknown"
                color = (200, 200, 200)

                # 1. 優先級: 嚴格的 Spread Hand
                if defects >= 4:
                    group_name = "Group: OPEN (5 Fingers)"
                    color = (0, 255, 0) # Green
                
                elif defects >= 2:
                    # 2~3 指不算 Open，也不進入後面的形狀判斷，直接 IDLE
                    group_name = "IDLE (Not 5 Fingers)"
                    color = (128, 128, 128) # Gray

                # 2. 優先級: Round (Fist) - 針對左手旋轉優化
                # 條件 A: 極度實心 (Sol > 0.80)，允許 AR 稍大 (直到 1.5)，解決 angry_fist10 問題
                # 條件 B: 中度實心 (Sol > 0.77) 且 形狀方正 (AR < 1.3)，解決 angry_fist10-2 問題
                elif (solidity > 0.80 and ar < 1.5) or (solidity > 0.77 and ar < 1.3):
                    group_name = "Group: ROUND (Fist/Angry)"
                    color = (0, 0, 255) # Red

                # 3. 優先級: 形狀特殊的實心手勢
                # Long Group (Thumb/Duck)
                elif ar > 1.35 and solidity > 0.68:
                    group_name = "Group: LONG (Thumb/Duck)"
                    color = (255, 255, 0) # Cyan
                
                # Wide Group (Straight)
                elif ar < 0.95 and solidity > 0.65:
                    group_name = "Group: WIDE (Straight)"
                    color = (0, 255, 255) # Yellow

                # 4. 優先級: Hook (剩下且實心度還行的)
                elif 0.65 < solidity <= 0.80:
                    group_name = "Group: C-SHAPE (Hook)"
                    color = (0, 165, 255) # Orange
                
                else:
                    group_name = "IDLE / UNKNOWN"

                # 顯示數據
                info_text = [
                    f"{group_name}",
                    f"Defects: {defects} (Need >=4 for Open)",
                    f"Solidity: {solidity:.2f}",
                    f"Aspect Ratio: {ar:.2f}"
                ]
                
                for i, line in enumerate(info_text):
                    text_color = color
                    if "IDLE" in group_name and i == 0: text_color = (150, 150, 150)
                    cv2.putText(display_frame, line, (10, 80 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        else:
             cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
             cv2.imshow("Debug: ROI Mask", np.zeros((200,200), np.uint8))

        cv2.imshow("Pure CV Analysis", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()