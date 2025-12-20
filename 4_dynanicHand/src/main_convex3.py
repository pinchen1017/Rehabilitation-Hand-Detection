"""
手部復健偵測系統 - 論文實作版 (Paper Implementation)
參考文獻: "Hand Gesture Recognition Using Convex Hull-Based Approach" (P.161)
核心邏輯:
1. 膚色偵測 (HSV) -> 取得二值化 Mask
2. 骨架回填 (Skeleton Injection) -> 修復光線不足導致的左手破洞
3. 凸包 (Convex Hull) & 凸缺陷 (Convexity Defects)
4. 手指計數 (Finger Counting): Count = Defects + 1
5. 面積比率 (Solidity): Area_Hand / Area_Hull
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

class PaperMethodAnalyzer:
    def __init__(self):
        print("  [系統] 初始化論文演算法模組 (Convex Hull)...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        # HSV 膚色範圍 (論文 P.162 提及需轉 HSV)
        self.lower_skin = np.array([0, 20, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    def get_person_frame(self, frame):
        """YOLO 去背 (排除背景干擾，讓 Convex Hull 更準)"""
        results = self.yolo_model(frame, verbose=False, stream=True)
        mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        person_found = False
        for r in results:
            if r.masks is None: continue
            max_area = 0
            best_mask = None
            for i, box in enumerate(r.boxes):
                if int(box.cls[0]) == 0: # Person
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

    def process_hand_geometry(self, clean_frame, hand_result):
        """
        論文核心演算法: Convex Hull & Defects
        """
        h, w, _ = clean_frame.shape
        
        # 1. ROI 切割 (只專注在手部區域)
        x_list = [int(lm * w) for lm in hand_result.landmarks[0::3]]
        y_list = [int(lm * h) for lm in hand_result.landmarks[1::3]]
        padding = 80
        x_min, x_max = max(0, min(x_list)-padding), min(w, max(x_list)+padding)
        y_min, y_max = max(0, min(y_list)-padding), min(h, max(y_list)+padding)
        
        roi_img = clean_frame[y_min:y_max, x_min:x_max]
        if roi_img.size == 0: return None, 0, 0, 0, None

        # 2. 影像前處理 (Preprocessing)
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_skin = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # === [優化] 骨架回填 (解決左手/右手不對稱問題) ===
        # 論文假設膚色偵測完美，但現實中左手可能有陰影。
        # 我們用 MediaPipe 的座標強制把手指「畫」出來。
        mask_skeleton = np.zeros_like(mask_skin)
        roi_landmarks = []
        for i in range(21):
            lx = int(hand_result.landmarks[i*3] * w) - x_min
            ly = int(hand_result.landmarks[i*3+1] * h) - y_min
            roi_landmarks.append((lx, ly))
            
        # 畫粗骨架 (模擬手指)
        connections = [(2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20), (0,5,9,13,17)]
        for conn in connections:
            for i in range(len(conn)-1):
                pt1, pt2 = roi_landmarks[conn[i]], roi_landmarks[conn[i+1]]
                # 線條粗細 12，確保手指不會斷
                cv2.line(mask_skeleton, pt1, pt2, 255, 12)
                # 關節處畫圓 (焊接)
                cv2.circle(mask_skeleton, pt1, 7, 255, -1)

        # 合併 Mask (HSV + Skeleton)
        mask_final = cv2.bitwise_or(mask_skin, mask_skeleton)
        # 形態學閉運算 (Close) 填補小洞
        kernel = np.ones((5,5), np.uint8)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)
        
        debug_mask = mask_final.copy() # 這是給您看 Debug 用的

        # 3. 尋找輪廓 (Find Contours)
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0, 0, 0, debug_mask
        
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return None, 0, 0, 0, debug_mask

        # 4. 凸包與缺陷 (Convex Hull & Defects)
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        hull_points = cv2.convexHull(cnt, returnPoints=True)
        
        # 計算面積比率 (Solidity)
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull_points)
        solidity = area_cnt / area_hull if area_hull > 0 else 0
        
        # 計算指縫 (Defects)
        defects_count = 0
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(cnt, hull_indices)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        
                        # 利用餘弦定理計算角度 (論文方法)
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        
                        if 2*b*c > 0:
                            angle = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c)))) * 57.3
                            # 論文建議：角度 < 90度 且 深度足夠 才算指縫
                            if angle < 90 and d > 1000: # d 是距離*256
                                defects_count += 1
                                cv2.circle(debug_mask, far, 5, 128, -1) # 畫出指縫點
            except: pass

        # 5. 轉換座標回原圖
        hull_global = hull_points.copy()
        for p in hull_global:
            p[0][0] += x_min
            p[0][1] += y_min

        # 回傳：凸包點, 實心度, 指縫數, Debug圖
        return hull_global, solidity, defects_count, debug_mask

def main():
    print("="*50)
    print("手部復健 - 論文演算法實作版 (P.161)")
    print("解決左右手不對稱問題")
    print("="*50)

    try:
        classifier = GestureClassifier(MODEL_PATH)
        print("  [DNN] 模型載入成功")
    except: sys.exit(1)

    detector = HandDetector()
    smoother = GestureSmoother()
    tracker = StretchTracker()
    renderer = UIRenderer()
    analyzer = PaperMethodAnalyzer()
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Main System", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: Hull Analysis", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. 鏡像翻轉 (讓左右手符合直覺)
        frame = cv2.flip(frame, 1)
        
        # 2. YOLO 去背
        has_person, clean_frame = analyzer.get_person_frame(frame)
        display_frame = frame.copy()
        
        hand_result = detector.detect(frame) # 用原圖偵測骨架 (最穩)
        gesture = None
        
        if has_person and hand_result:
            # DNN 預測
            raw_pred = classifier.predict(hand_result.landmarks)
            gesture = smoother.smooth(raw_pred)
            dnn_id = gesture.class_id if hasattr(gesture, 'class_id') else 0

            # === 3. 論文演算法分析 (使用 clean_frame) ===
            hull, solidity, defects, debug_mask = analyzer.process_hand_geometry(clean_frame, hand_result)
            
            if debug_mask is not None:
                cv2.imshow("Debug: Hull Analysis", debug_mask)
            else:
                cv2.imshow("Debug: Hull Analysis", np.zeros((200,200), np.uint8))

            if hull is not None:
                
                cv2.drawContours(display_frame, [hull], -1, (0, 255, 0), 2)
                
                # === 4. 論文手勢判定 (根據指縫數) ===
                # 論文 P.163: "Number of fingers = Defects + 1"
                # 手指數 5 -> 張開 (Open)
                # 手指數 1 -> 握拳 (Fist)
                
                est_fingers = defects + 1
                paper_state = "Unknown"
                
                if est_fingers >= 4 or (est_fingers >= 3 and solidity < 0.7):
                    paper_state = "OPEN (Spread)"
                    cv2.putText(display_frame, "Paper: OPEN", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif est_fingers <= 2 and solidity > 0.8:
                    paper_state = "CLOSED (Fist)"
                    cv2.putText(display_frame, "Paper: CLOSED", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 顯示數據
                info = f"Defects:{defects} Fingers:~{est_fingers} Sol:{solidity:.2f}"
                cv2.putText(display_frame, info, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 雙重驗證警告
                if dnn_id == 7 and paper_state != "OPEN (Spread)":
                    cv2.putText(display_frame, "Warning: Spread Fingers!", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                if dnn_id in [1,2,3,4,5,6] and paper_state == "OPEN (Spread)":
                    cv2.putText(display_frame, "Warning: Loose Grip!", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # 更新 Tracker (不中斷計時)
                stretch_record = tracker.update(gesture)
                if stretch_record: print(f"完成: {stretch_record.stretch_type}")

        # 5. 渲染 UI
        if has_person and hand_result and gesture:
            final_frame = renderer.render(display_frame, hand_result, gesture, tracker.get_stats(), tracker.get_state_info())
        else:
            final_frame = display_frame
            if not has_person: cv2.putText(final_frame, "Searching Person...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Main System", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()