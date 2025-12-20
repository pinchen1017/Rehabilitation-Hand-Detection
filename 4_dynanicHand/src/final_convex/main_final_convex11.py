"""
手部復健偵測系統 - v27 教練模式 (Coach Mode)
邏輯核心:
1. 權威判定: DNN 模型說是什麼手勢，就是什麼手勢 (計數器累加)。
2. 教練提示: 根據 DNN 的結果，去檢查 CV (Convex Hull) 的特徵。
   - 如果 DNN=Fist 但 CV=Long -> 提示 "Grip Tighter" (握緊)
   - 如果 DNN=Straight 但 CV=Round -> 提示 "Rotate Hand" (旋轉)
   - 如果 DNN=Spread 但 CV!=Open -> 提示 "Open Fingers" (張開)
3. 視覺呈現: 提示文字顯示於 Debug Mask 視窗，不干擾主畫面。
"""
# Step A: 引入模組
# A-1 引入標準模組
import sys
import cv2
import numpy as np
import os
import time
import math
from ultralytics import YOLO

# A-2 引入原專案模組
from config import DEFAULT_MODEL_PATH, DEFAULT_YOLO_MODEL_PATH
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier, GestureSmoother
from stretch_tracker import StretchTracker
from ui_renderer import UIRenderer

# A-3 確保路徑指向正確
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", DEFAULT_MODEL_PATH) 
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "..", DEFAULT_YOLO_MODEL_PATH)

# Step B: 定義 CVAnalyzer 類別
class CVAnalyzer:
    """
    核心分析器 (使用 convex16 的指縫優先邏輯參數)
    線寬 10 / 半徑 6，確保指縫計算準確。
    """
    # B-1 初始化 YOLOv8-Seg 與 CV 分析模組
    def __init__(self):
        print("  [系統] 初始化 YOLOv8-Seg 與 CV 分析模組...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    # B-1-1 鎖定最大的人並去背
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

    # B-2 計算幾何特徵
    def analyze_hand_geometry(self, clean_frame, hand_result):
        """計算幾何特徵"""
        h, w, _ = clean_frame.shape
        
        # B-2-1 ROI 切割
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

        # B-2-2 建立 Mask
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_skin = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask_skeleton = np.zeros_like(mask_skin)
        
        # B-2-3 取得 ROI 內的 landmarks
        roi_landmarks = []
        for i in range(21):
            lx = int(hand_result.landmarks[i*3] * w) - x_min
            ly = int(hand_result.landmarks[i*3+1] * h) - y_min
            roi_landmarks.append((lx, ly))
            
        # B-2-4 填滿手掌
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_points = np.array([roi_landmarks[i] for i in palm_indices], dtype=np.int32)
        cv2.fillConvexPoly(mask_skeleton, palm_points, 255)
        
        # B-2-5 畫骨架
        finger_connections = [(2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20), (0,5,9,13,17)]
        for conn in finger_connections:
            for i in range(len(conn)-1):
                pt1, pt2 = roi_landmarks[conn[i]], roi_landmarks[conn[i+1]]
                if 0 <= pt1[0] < roi_img.shape[1] and 0 <= pt1[1] < roi_img.shape[0]:
                    cv2.line(mask_skeleton, pt1, pt2, 255, 10)
                    cv2.circle(mask_skeleton, pt1, 6, 255, -1)
                    cv2.circle(mask_skeleton, pt2, 6, 255, -1)

        # B-2-6 合併 Mask
        mask_final = cv2.bitwise_or(mask_skin, mask_skeleton)
        # B-2-6.1 定義 kernel
        kernel = np.ones((3, 3), np.uint8)
        # B-2-6.2 膨脹
        mask_final = cv2.dilate(mask_final, kernel, iterations=1)
        # B-2-6.3 閉合
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=1)
        # B-2-6.4 取得 debug mask
        debug_mask = mask_final.copy()

        # B-2-7 計算幾何特徵：找輪廓
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0, 0, 0, 0, debug_mask
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return None, 0, 0, 0, 0, debug_mask

        # B-2-8 計算凸包：計算凸包的索引與點
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        hull_points = cv2.convexHull(cnt, returnPoints=True)
        # B-2-8.1 計算凸包的面積
        area_cnt = cv2.contourArea(cnt)
        # B-2-8.2 計算凸包的面積
        area_hull = cv2.contourArea(hull_points)
        # B-2-8.2 計算凸包的周長
        perimeter = cv2.arcLength(cnt, True)
        
        # B-2-9 計算 Solidity 與 Circularit: 計算凸包的密度與圓度
        # B-2-9.1 計算凸包的密度
        solidity = area_cnt / area_hull if area_hull > 0 else 0
        # B-2-9.2 計算凸包的圓度
        circularity = (4 * math.pi * area_cnt) / (perimeter ** 2) if perimeter > 0 else 0
        
        # B-2-10 計算 Aspect Ratio: 計算凸包的長寬比
        # B-2-10.1 計算凸包的長寬比
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        # B-2-10.2 計算凸包的長寬比
        aspect_ratio = float(rect_h) / rect_w if rect_w > 0 else 0

        # B-2-11 計算 Defects: 計算凸包的缺陷
        # B-2-11.1 計算凸包的缺陷
        defects_count = 0
        # B-2-11.2 計算凸包的缺陷
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

        # B-2-12 計算凸包
        hull_global = hull_points.copy()
        for point in hull_global:
            point[0][0] += x_min
            point[0][1] += y_min

        # B-2-13 返回結果
        return hull_global, solidity, circularity, defects_count, aspect_ratio, debug_mask

# Step C: 主程式
def main():
    print("=" * 50)
    print("手部復健偵測系統 - v27 教練模式")
    print("策略: 相信模型計數，利用 CV 給予動作改進提示")
    print("=" * 50)

    # C-1 初始化
    # C-1-1 載入模型
    try:
        classifier = GestureClassifier(MODEL_PATH)
        print("  [DNN] 模型載入成功!")
    except Exception as e:
        print(f"  [錯誤] 模型載入失敗: {e}")
        sys.exit(1)

    # C-1-2 初始化 HandDetector, GestureSmoother, StretchTracker, UIRenderer, CVAnalyzer
    detector = HandDetector()
    smoother = GestureSmoother()
    tracker = StretchTracker()
    renderer = UIRenderer()
    cv_analyzer = CVAnalyzer()

    # C-1-3 初始化 VideoCapture
    cap = cv2.VideoCapture(0)
    
    # C-1-4 初始化視窗
    cv2.namedWindow("Rehab System", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: ROI Mask", cv2.WINDOW_NORMAL)

    print("\n系統啟動完成 - 按 'q' 退出, 'r' 重置")

    # C-1-5 防抖動參數 (讓計數更穩定)
    stable_id = 0
    state_change_start = None
    STABILITY_THRESHOLD = 1.5 # 稍微縮短緩衝，反應快一點
    last_valid_gesture = None

    # C-2 主迴圈
    while True:
        # C-2-1 讀取影像
        ret, frame = cap.read()
        if not ret: break
        
        # C-2-2 翻轉影像
        frame = cv2.flip(frame, 1)
        # C-2-3 取得人像
        has_person, clean_frame, _ = cv_analyzer.get_person_frame(frame)
        # C-2-4 複製影像
        display_frame = frame.copy()
        
        # C-2-5 初始化手勢
        hand_result = None
        gesture = None
        
        # C-2-6 狀態變數
        cv_group_name = "Wait..."
        hint_msg = "Focusing..."
        hint_color = (200, 200, 200) 
        
        debug_mask_display = np.zeros((300, 300, 3), dtype=np.uint8)

        # C-3 鎖定單人
        if has_person:
            cv2.putText(display_frame, "[YOLO] Locked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            hand_result = detector.detect(frame)

            # C-3-1 偵測手勢
            if hand_result is not None:
                # C-3-2 DNN 預測 (權威結果)
                raw_prediction = classifier.predict(hand_result.landmarks)
                gesture = smoother.smooth(raw_prediction)
                dnn_id = gesture.class_id if hasattr(gesture, 'class_id') else 0

                # C-3-3 CV 幾何分析 (輔助分析)
                hull, solidity, circularity, defects, aspect_ratio, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)
                
                # C-3-4 處理 Mask 顯示
                if debug_mask is not None:
                    mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
                    debug_mask_display = cv2.resize(mask_bgr, (300, 300))

                # C-3-5 畫凸包
                if hull is not None:
                    cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)

                    # C-3-5-1 幾何分類 (Defects Priority from convex16)
                    cv_group = "UNKNOWN"
                    # if dnn_id == 0: cv_group = "IDLE"
                    # elif dnn_id == 1: cv_group = "C-SHAPE"
                    # elif dnn_id == 2: cv_group = "ROUND"
                    # elif dnn_id == 3: cv_group = "LONG"
                    # elif dnn_id == 4: cv_group = "WIDE"
                    # elif dnn_id == 5: cv_group = "LONG"
                    # elif dnn_id == 6: cv_group = "ROUND"
                    # elif dnn_id == 7: cv_group = "OPEN"

                    if defects >= 4: cv_group = "OPEN"
                    elif defects >= 3: cv_group = "IDLE"
                    elif defects == 0 and 0.84 < aspect_ratio < 1.34 and solidity > 0.82:
                        cv_group = "ROUND"
                    elif defects == 0 or (aspect_ratio > 1.35 and solidity > 0.68):
                        cv_group = "LONG"
                    elif aspect_ratio < 1.0 and solidity > 0.65:
                        cv_group = "WIDE"
                    elif 0.65 < solidity <= 0.82:
                        cv_group = "C-SHAPE"
                    else:
                        cv_group = "IDLE"
                    cv_group_name = cv_group

                    # C-3-5-2 教練提示邏輯 (Coach Logic)
                    # 邏輯: 根據「模型說你是誰」，去檢查「你做得標準嗎？」
                    
                    hint_msg = "Success !!"
                    hint_color = (0, 255, 0) # Green (Default OK)

                    # C-3-5-2.1 Spread Hand (7)
                    if dnn_id == 7:
                        if cv_group != "OPEN":
                            hint_msg = "Open Fingers!"
                            hint_color = (0, 255, 255) # Yellow

                    # C-3-5-2.2 Fist (2, 6)
                    elif dnn_id in [2, 6]:
                        if cv_group != "ROUND":
                            # 如果不是圓形，看看出了什麼問題
                            if defects > 0: hint_msg = "Close Gaps!"
                            else: hint_msg = "Grip Tighter!"
                            hint_color = (0, 255, 255)

                    # C-3-5-2.3 Straight Hand (4)
                    elif dnn_id == 4:
                        if cv_group != "WIDE":
                            if aspect_ratio > 1.2: hint_msg = "Rotate Hand!"
                            else: hint_msg = "Flatten Hand!"
                            hint_color = (0, 255, 255)

                    # C-3-5-2.4 Hook (1)
                    elif dnn_id == 1:
                        if cv_group != "C-SHAPE":
                            hint_msg = "Bend Fingers!"
                            hint_color = (0, 255, 255)

                    # C-3-5-2.5 Thumb Flexion (3) -> 應該是 LONG
                    elif dnn_id == 3:
                        if cv_group != "LONG":
                            hint_msg = "Keep Long!"
                            hint_color = (0, 255, 255)

                    # C-3-5-3 輸出給 Tracker (完全信任 DNN)
                    instant_id = dnn_id
                    last_valid_gesture = gesture 

                    # C-3-5-4 進度條 (視覺效果)
                    progress_bar_val = 0
                    if dnn_id == 7: progress_bar_val = np.clip((0.75 - solidity) / (0.75 - 0.55), 0, 1)
                    elif dnn_id in [3, 5]: progress_bar_val = np.clip((aspect_ratio - 1.3) / (1.6 - 1.3), 0, 1)
                    else: progress_bar_val = np.clip((solidity - 0.60) / (0.85 - 0.60), 0, 1)

                else:
                    # C-3-5-5 輸出給 Tracker (完全信任 DNN)
                    instant_id = dnn_id
            else:
                # C-3-5-6 輸出給 Tracker (完全信任 DNN)
                instant_id = 0
        else:
             cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             instant_id = 0

        # C-3-6 防抖動更新 (只為了讓 Tracker 收到的訊號穩定)
        current_time = time.time()
        final_id_to_tracker = 0
        
        if instant_id == stable_id:
            state_change_start = None 
            final_id_to_tracker = stable_id
        else:
            if state_change_start is None: state_change_start = current_time
            elapsed = current_time - state_change_start
            
            if elapsed < STABILITY_THRESHOLD:
                final_id_to_tracker = stable_id 
            else:
                stable_id = instant_id 
                final_id_to_tracker = stable_id
                state_change_start = None

        # C-3-7 更新 Tracker
        gesture_to_send = None
        if final_id_to_tracker != 0:
            if last_valid_gesture:
                gesture_to_send = last_valid_gesture
                gesture_to_send.class_id = final_id_to_tracker
            elif gesture:
                gesture_to_send = gesture
                gesture_to_send.class_id = final_id_to_tracker
        else:
            if gesture:
                gesture.class_id = 0
                gesture_to_send = gesture
        
        if gesture_to_send:
            stretch_record = tracker.update(gesture_to_send)
            if stretch_record:
                 print(f"完成伸展! {stretch_record.stretch_type}")

        # C-3-8 繪製 UI (集中在 Mask 視窗)
        # C-3-8.1 提示訊息 (最重要)
        cv2.putText(debug_mask_display, f"Hint: {hint_msg}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hint_color, 2)
        # C-3-8.2 目前 CV 狀態
        cv2.putText(debug_mask_display, f"Shape: {cv_group_name}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        # C-3-8.3 數據
        if has_person and hand_result and hull is not None:
             cv2.putText(debug_mask_display, f"D:{defects} S:{solidity:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # C-3-8.4 主畫面進度條
        if stable_id != 0:
            bar_color = (0, 255, 0)
            cv2.rectangle(display_frame, (50, 150), (80, 350), (255, 255, 255), 2)
            cv2.rectangle(display_frame, (50, 350 - int(200*progress_bar_val)), (80, 350), bar_color, -1)

        # C-3-8.5 渲染手勢
        if has_person and hand_result and gesture:
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
            stable_id = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()