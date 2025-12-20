"""
手部復健偵測系統 - v5 最終平衡版
修正重點:
1. 移除「強制 IDLE」機制：讓 StretchTracker 能正常讀秒，不會因為微小誤差而重置。
2. DNN 權威性提升：手勢類別以 DNN 為主，CV 只負責給予「握力/張開程度」的建議。
3. 解決跳針：DNN 讀取原圖 (Frame)，CV 讀取去背圖 (Clean Frame)。
4. 包含骨架回填技術：解決無名指斷裂問題。
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
    """處理 YOLO 去背與 Convex Hull / 指縫分析的類別"""
    def __init__(self):
        print("  [系統] 初始化 YOLOv8-Seg 與 CV 分析模組...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        # 稍微放寬膚色範圍，避免手指斷裂
        self.lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    def get_person_frame(self, frame):
        # YOLO 去背邏輯 (維持不變)
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
        CV 分析核心 (包含 v6 骨架回填技術)
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
        if roi_img.size == 0: return None, 0, 0, 0, None

        # 2. 建立 Mask (HSV + Skeleton Injection)
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_skin = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask_skeleton = np.zeros_like(mask_skin)
        
        # 取得 ROI 相對座標
        roi_landmarks = []
        for i in range(21):
            lx = int(hand_result.landmarks[i*3] * w) - x_min
            ly = int(hand_result.landmarks[i*3+1] * h) - y_min
            roi_landmarks.append((lx, ly))
            
        # 填滿手掌 (解決掌心破洞)
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_points = np.array([roi_landmarks[i] for i in palm_indices], dtype=np.int32)
        cv2.fillConvexPoly(mask_skeleton, palm_points, 255)
        
        # 畫粗骨架 + 關節點焊 (解決手指斷裂)
        finger_connections = [
            (2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), 
            (13, 14, 15, 16), (17, 18, 19, 20), (0, 5, 9, 13, 17)
        ]
        for connection in finger_connections:
            for i in range(len(connection) - 1):
                pt1 = roi_landmarks[connection[i]]
                pt2 = roi_landmarks[connection[i+1]]
                if 0 <= pt1[0] < roi_img.shape[1] and 0 <= pt1[1] < roi_img.shape[0]:
                    cv2.line(mask_skeleton, pt1, pt2, 255, 20)
                    cv2.circle(mask_skeleton, pt1, 12, 255, -1) # 焊點
                    cv2.circle(mask_skeleton, pt2, 12, 255, -1) # 焊點

        # 合併 Mask
        mask_final = cv2.bitwise_or(mask_skin, mask_skeleton)
        kernel = np.ones((5, 5), np.uint8)
        mask_final = cv2.dilate(mask_final, kernel, iterations=1) # 膨脹讓手指變粗
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)
        debug_mask = mask_final.copy()

        # 3. 找輪廓
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0, 0, 0, debug_mask

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return None, 0, 0, 0, debug_mask

        # 4. 計算特徵
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        hull_points = cv2.convexHull(cnt, returnPoints=True)
        
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull_points)
        perimeter = cv2.arcLength(cnt, True)
        
        solidity = 0
        circularity = 0
        if area_hull > 0: solidity = area_cnt / area_hull
        if perimeter > 0: circularity = (4 * math.pi * area_cnt) / (perimeter ** 2)

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
                        
                        if 2 * b * c > 0:
                            angle = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2 * b * c)))) * 57.2958
                            # 角度放寬到 100，深度 > 8
                            if angle < 100 and depth > 8:
                                defects_count += 1
                                cv2.circle(debug_mask, far, 5, 128, -1)
            except:
                pass

        # 6. 轉回全域座標
        hull_global = hull_points.copy()
        for point in hull_global:
            point[0][0] += x_min
            point[0][1] += y_min

        return hull_global, solidity, circularity, defects_count, debug_mask

def main():
    print("=" * 50)
    print("手部復健偵測系統 (平衡版 - 修復計時器)")
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
        
        # 1. 鏡像翻轉
        frame = cv2.flip(frame, 1)

        # 2. YOLO 去背 (取得 Clean Frame 供 CV 使用)
        has_person, clean_frame, _ = cv_analyzer.get_person_frame(frame)
        display_frame = frame.copy()
        
        hand_result = None
        gesture = None
        warning_msg = ""
        
        if has_person:
            cv2.putText(display_frame, "[YOLO] Locked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 3. DNN 偵測 (關鍵：使用原圖 frame，避免關節跳針)
            hand_result = detector.detect(frame)

            if hand_result is not None:
                # DNN 預測
                raw_prediction = classifier.predict(hand_result.landmarks)
                gesture = smoother.smooth(raw_prediction)
                
                # 取得 DNN 判斷的 ID
                dnn_id = gesture.class_id if hasattr(gesture, 'class_id') else 0

                # 4. CV 幾何驗證 (使用去背圖 clean_frame)
                hull, solidity, circularity, defects, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)
                
                if debug_mask is not None:
                    cv2.imshow("Debug: ROI Mask", debug_mask)
                else:
                    cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))

                if hull is not None:
                    cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)

                    # ============================================
                    # 雙重驗證邏輯 (Soft Check - 不強制 IDLE)
                    # ============================================
                    
                    # 物理狀態定義
                    # is_gripped: 實心度高 OR 圓形度高 OR 指縫少
                    is_gripped = (solidity > 0.65) or (circularity > 0.60) or (defects <= 1)
                    
                    closed_group = [1, 2, 3, 4, 5, 6] # 握拳類
                    open_group = [7]                  # 伸展類

                    # --- 情境 A: 握拳類 ---
                    if dnn_id in closed_group:
                        # 只有當「非常明顯」沒握緊時才警告 (指縫 >= 3 且 實心度低)
                        if defects >= 3 and solidity < 0.65:
                            warning_msg = "Loose Grip (Tighten!)"
                            # 這裡不強制 IDLE，讓 DNN 的判斷通過，但顯示警告
                        
                        # 針對 Thumb Flexion (ID=3) 的長寬比檢查太不穩，先移除
                        # 讓 DNN 決定它是 Fist 還是 Thumb Flexion
                        
                    # --- 情境 B: 伸展類 ---
                    elif dnn_id in open_group:
                        # 只有當「非常明顯」沒張開時才警告 (指縫 <= 1 且 實心度高)
                        if defects <= 1 and solidity > 0.70:
                            warning_msg = "Spread Fingers!"
                    
                    # 繪製 CV 數據 (Debug 用)
                    # cv2.putText(display_frame, f"S:{solidity:.2f} D:{defects}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # 繪製握力進度條
                    bar_h = 200
                    progress = np.clip((solidity - 0.60) / (0.85 - 0.60), 0, 1)
                    if defects >= 3: 
                        bar_color = (0, 255, 0) # 綠色 (放鬆)
                        progress = 0.0
                    else:
                        bar_color = (0, int(255 * (1-progress)), int(255 * progress))

                    cv2.rectangle(display_frame, (50, 150), (80, 150+bar_h), (255, 255, 255), 2)
                    cv2.rectangle(display_frame, (50, 150 + bar_h - int(bar_h*progress)), 
                                 (80, 150+bar_h), bar_color, -1)

                # ============================================
                # 關鍵：讓 Tracker 正常運作
                # ============================================
                # 我們不再強制修改 gesture.class_id 為 0
                # 這樣即使 CV 覺得不完美，計時器也會繼續跑
                # 警告文字 (warning_msg) 會在 UI 上提醒使用者調整，但不中斷流程
                
                stretch_record = tracker.update(gesture)
                if stretch_record:
                     print(f"完成伸展! {stretch_record.stretch_type} Count: {tracker.get_stats().total_count}")

            else:
                cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))
        else:
             cv2.putText(display_frame, "Searching for Person...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 渲染 UI
        if has_person and hand_result and gesture is not None:
            final_frame = renderer.render(display_frame, hand_result, gesture, tracker.get_stats(), tracker.get_state_info())
            if warning_msg:
                 # 警告文字顯示在明顯處
                 cv2.putText(final_frame, warning_msg, (180, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
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