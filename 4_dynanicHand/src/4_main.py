"""
純幾何特徵測試儀 - v14 最終嚴謹版 (Pure CV Final v14)
邏輯核心：
1. Image 1 (Fist/Round): Sol=0.88, AR=1.44 -> 必須判為 ROUND
2. Image 2 (Long/Thumb): Sol=0.87, AR=1.64 -> 必須判為 LONG
解決方案: 設定 AR 1.60 為絕對長條門檻，優先於 Solidity 判定。
"""
# 步驟 A： 引入模組
# A-1 引入標準模組
import sys
import cv2
import numpy as np
import os
import math
from ultralytics import YOLO
from hand_detector import HandDetector

# A-2 設定路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from config import DEFAULT_YOLO_MODEL_PATH
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "..", DEFAULT_YOLO_MODEL_PATH)

# 步驟 B： 定義 CVAnalyzer 類別
class CVAnalyzer:
    # B-1 YOLO 去背
    # B-1-1 初始化 YOLOv8-Seg 與 CV 分析模組
    def __init__(self):
        print("  [系統] 初始化 YOLOv8-Seg 與 CV 分析模組...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    # B-1-2 鎖定最大的人並去背
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

    # B-2 凸包幾何驗證
    def analyze_hand_geometry(self, clean_frame, hand_result):
        """CV 分析核心 (維持 v9 骨架瘦身版)"""
        h, w, _ = clean_frame.shape
        
        # B-2-1 計算 ROI (Padding 維持 80)
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

        # B-2-2 ROI 內 HSV 偵測
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_skin = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask_skeleton = np.zeros_like(mask_skin)
        
        # B-2-2.1 取得 ROI 內的 landmarks
        roi_landmarks = []
        for i in range(21):
            lx = int(hand_result.landmarks[i*3] * w) - x_min
            ly = int(hand_result.landmarks[i*3+1] * h) - y_min
            roi_landmarks.append((lx, ly))
            
        # B-2-2.2 填滿手掌
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_points = np.array([roi_landmarks[i] for i in palm_indices], dtype=np.int32)
        cv2.fillConvexPoly(mask_skeleton, palm_points, 255)
        
        # B-2-2.3 畫骨架
        finger_connections = [(2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20), (0,5,9,13,17)]
        for conn in finger_connections:
            for i in range(len(conn)-1):
                pt1, pt2 = roi_landmarks[conn[i]], roi_landmarks[conn[i+1]]
                if 0 <= pt1[0] < roi_img.shape[1] and 0 <= pt1[1] < roi_img.shape[0]:
                    cv2.line(mask_skeleton, pt1, pt2, 255, 10)
                    cv2.circle(mask_skeleton, pt1, 6, 255, -1)
                    cv2.circle(mask_skeleton, pt2, 6, 255, -1)

        # B-2-2.4 合併 mask
        mask_final = cv2.bitwise_or(mask_skin, mask_skeleton)

        # B-3 形態學運算 (膨脹 & 閉合)
        # B-3-1 定義 kernel
        kernel = np.ones((3, 3), np.uint8)
        # B-3-2 膨脹
        mask_final = cv2.dilate(mask_final, kernel, iterations=1)
        # B-3-3 閉合
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=1)
        # B-3-4 取得 debug mask
        debug_mask = mask_final.copy()

        # B-4 找輪廓
        # B-4-1 找輪廓
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0, 0, 0, 0, debug_mask
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return None, 0, 0, 0, 0, debug_mask

        # B-4-2 計算特徵
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        hull_points = cv2.convexHull(cnt, returnPoints=True)
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull_points)
        
        solidity = area_cnt / area_hull if area_hull > 0 else 0
        circularity = 0 
        
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        aspect_ratio = float(rect_h) / rect_w if rect_w > 0 else 0

        # B-5 指縫計數 (Defects)
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

        # B-6 轉回全域座標
        hull_global = hull_points.copy()
        for point in hull_global:
            point[0][0] += x_min
            point[0][1] += y_min

        return hull_global, solidity, circularity, defects_count, aspect_ratio, debug_mask

# 步驟 C: 主程式
def main():
    print("="*50)
    print("純幾何特徵測試儀 - v14 最終嚴謹版")
    print("邏輯修正: 嚴格利用 AR=1.60 作為 Fist/Long 的絕對分水嶺")
    print("="*50)

    # 2. 初始化 HandDetector 與 CVAnalyzer
    detector = HandDetector()
    cv_analyzer = CVAnalyzer()
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Pure CV Analysis", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: ROI Mask", cv2.WINDOW_NORMAL)

    while True:
        # 3-1 讀取影像
        ret, frame = cap.read()
        if not ret: break

        # 3-2 水平翻轉 (保持鏡像習慣)
        frame = cv2.flip(frame, 1)

        # 3-3 YOLO 去背 (取得 clean_frame 備用)
        has_person, clean_frame = cv_analyzer.get_person_frame(frame)
        display_frame = frame.copy()
        
        # 3-4 偵測手部
        hand_result = detector.detect(frame) 

        # 3-5 如果有人且有手部，顯示手部偵測成功
        if has_person and hand_result:
            cv2.putText(display_frame, "Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 4-1 凸包幾何驗證
            hull, solidity, circularity, defects, ar, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)
            
            # 4-2 如果 debug_mask 存在，顯示 debug_mask
            if debug_mask is not None:
                cv2.imshow("Debug: ROI Mask", debug_mask)
            else:
                cv2.imshow("Debug: ROI Mask", np.zeros((200,200), np.uint8))

            # 4-3 如果 hull 存在，畫出 hull
            if hull is not None:
                cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)
                
                # 4-4 嚴謹判斷邏輯
                group_name = "Unknown"
                color = (200, 200, 200)
                # 4-4.1 嚴格 Spread Hand (維持不變)
                if defects >= 4:
                    group_name = "Group: OPEN (5 Fingers)"
                    color = (0, 255, 0)

                elif defects >= 3:
                    group_name = "Group: IDLE"
                
                # 4-4.2 高實心握拳組 (High Solidity Round)
                # 專門抓 Image 1 (Sol 0.88, AR 1.44)
                # 如果沒被上面抓走 (代表 AR < 1.60)，且實心度極高 (>0.82)，判定為 Round
                elif defects == 0 and 0.9 < ar < 1.34 and solidity > 0.82:
                    group_name = "Group: ROUND (Fist/Angry)"
                    color = (0, 0, 255) # Red

                # 4-4.3 一般長條組 (Intermediate Long)
                # 處理 AR 介於 1.35 ~ 1.60 且 Solidity < 0.82 的 Duck/Thumb
                elif defects == 0 or (ar > 1.35 and solidity > 0.68):
                    group_name = "Group: LONG (Thumb/Duck)"
                    color = (255, 255, 0)

                # 4-4.4 寬扁組 (Straight Hand)
                # 維持 AR < 1.0 的嚴格限制
                elif ar < 1.0 and solidity > 0.65:
                    group_name = "Group: WIDE (Straight)"
                    color = (0, 255, 255) # Yellow

                # 4-4.5 Hook (C-Shape)
                # 剩下的中間地帶 (0.65 < Sol <= 0.82)
                elif 0.65 < solidity <= 0.82:
                    group_name = "Group: C-SHAPE (Hook/Beak)"
                    color = (0, 165, 255) # Orange
                
                # 4-4.6 其他情況
                else:
                    group_name = "IDLE"

                # 4-5 顯示數據
                info_text = [
                    f"{group_name}",
                    f"Defects: {defects}",
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