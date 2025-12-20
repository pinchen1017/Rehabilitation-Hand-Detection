"""
手部復健偵測系統 - v17 最終整合版 (Final Integration)
核心架構:
1. 幾何核心: 使用 v14 黃金參數 (AR 1.6 分水嶺, 左手旋轉優化)。
2. 驗證邏輯: DNN 與 CV 必須「類別一致」才算有效，否則視為 IDLE。
3. 防抖機制: 引入 2 秒緩衝區 (State Buffering)，防止誤判導致計數中斷。
4. 修正: 補上 math 模組，修復所有縮排與語法錯誤。
"""

import sys
import cv2
import numpy as np
import os
import time
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
    """處理 YOLO 去背與 Convex Hull 幾何分析 (v14 核心)"""
    def __init__(self):
        print("  [系統] 初始化 YOLOv8-Seg 與 CV 分析模組...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        # v14 參數: 寬容膚色，避免斷裂
        self.lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    def get_person_frame(self, frame):
        """YOLO 去背"""
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
            return True, clean_frame, mask_combined
        return False, frame, None

    def analyze_hand_geometry(self, clean_frame, hand_result):
        """
        計算幾何特徵 (Solidity, Defects, AR)
        【v14 特色】：使用瘦身版骨架 (Line=10, Circle=6)，確保指縫不被填滿
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
        
        # 畫骨架 (v14 瘦身參數)
        finger_connections = [(2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20), (0,5,9,13,17)]
        for conn in finger_connections:
            for i in range(len(conn)-1):
                pt1, pt2 = roi_landmarks[conn[i]], roi_landmarks[conn[i+1]]
                if 0 <= pt1[0] < roi_img.shape[1] and 0 <= pt1[1] < roi_img.shape[0]:
                    cv2.line(mask_skeleton, pt1, pt2, 255, 10) # 線寬 10
                    cv2.circle(mask_skeleton, pt1, 6, 255, -1) # 半徑 6
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
                            angle = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c)))) * 57.2958
                            if angle < 100 and depth > 8:
                                defects_count += 1
                                cv2.circle(debug_mask, far, 5, 128, -1)
            except: pass

        # 6. 轉回全域座標
        hull_global = hull_points.copy()
        for point in hull_global:
            point[0][0] += x_min
            point[0][1] += y_min

        return hull_global, solidity, circularity, defects_count, aspect_ratio, debug_mask

def main():
    print("=" * 50)
    print("手部復健偵測系統 - v17 最終整合版")
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

    # ==========================================
    # 防抖動參數 (Debouncing Vars)
    # ==========================================
    stable_id = 0           # 目前穩定的手勢 ID
    state_change_start = None # 狀態改變開始的時間
    STABILITY_THRESHOLD = 2.0 # 緩衝時間 2 秒
    last_valid_gesture = None # 暫存最後一個正確的手勢物件

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        has_person, clean_frame, _ = cv_analyzer.get_person_frame(frame)
        display_frame = frame.copy()
        
        hand_result = None
        gesture = None
        warning_msg = ""
        
        # 這一幀的即時判定結果
        instant_id = 0 
        is_match = False
        
        # UI 顯示用
        debug_status = "Scanning..."
        progress_bar_val = 0.0
        cv_group_name = "Wait..."

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
                
                if debug_mask is not None:
                    cv2.imshow("Debug: ROI Mask", debug_mask)
                else:
                    cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))

                if hull is not None:
                    cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)

                    # ==========================================================
                    # v14 嚴謹幾何分類 (Geometric Classification)
                    # ==========================================================
                    cv_group = "UNKNOWN"
                    
                    # 優先級 1: 嚴格 Spread Hand (需 5 指)
                    if defects >= 4: cv_group = "OPEN"
                    elif defects >= 2: cv_group = "IDLE_PARTIAL" # 2-3指不算
                    
                    # 優先級 2: 絕對長條 (Thumb Flexion)
                    elif aspect_ratio >= 1.60: cv_group = "LONG"
                    
                    # 優先級 3: 高實心握拳 (Round) - 左手旋轉優化
                    elif solidity > 0.82: cv_group = "ROUND"
                    
                    # 優先級 4: 寬扁 (Straight)
                    elif aspect_ratio < 1.0 and solidity > 0.65: cv_group = "WIDE"
                    
                    # 優先級 5: 一般長條 (Intermediate Duck)
                    elif aspect_ratio > 1.35 and solidity > 0.68: cv_group = "LONG"
                    
                    # 優先級 6: C型 (Hook / Beak)
                    elif 0.65 < solidity <= 0.82: cv_group = "C-SHAPE"
                    
                    cv_group_name = cv_group

                    # ==========================================================
                    # 一致性檢查 (Matching Logic)
                    # ==========================================================
                    is_match = False
                    
                    # 規則: DNN 與 CV 必須在同一個「邏輯群組」
                    if dnn_id == 7 and cv_group == "OPEN": is_match = True
                    elif dnn_id == 3 and cv_group == "LONG": is_match = True
                    # Duck 比較特別，容許 Long, C-Shape, Round (因為變形大)
                    elif dnn_id == 5 and cv_group in ["LONG", "C-SHAPE", "ROUND"]: is_match = True
                    elif dnn_id in [2, 6] and cv_group == "ROUND": is_match = True
                    elif dnn_id == 4 and cv_group == "WIDE": is_match = True
                    elif dnn_id == 1 and cv_group == "C-SHAPE": is_match = True

                    if is_match:
                        instant_id = dnn_id
                        last_valid_gesture = gesture # 備份正確的 gesture 物件
                        
                        # 計算進度條 (視覺化)
                        if dnn_id == 7: progress_bar_val = np.clip((0.75 - solidity) / (0.75 - 0.55), 0, 1)
                        elif dnn_id in [3, 5]: progress_bar_val = np.clip((aspect_ratio - 1.3) / (1.6 - 1.3), 0, 1)
                        else: progress_bar_val = np.clip((solidity - 0.60) / (0.85 - 0.60), 0, 1)
                    else:
                        instant_id = 0 # 驗證失敗 -> 視為 IDLE
                        warning_msg = f"Shape mismatch! ({cv_group})"
            else:
                cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))
        else:
             cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ==========================================
        # 核心：防抖動緩衝 (Debounce Buffer)
        # ==========================================
        current_time = time.time()
        final_id_to_tracker = 0
        
        # 情境 A: 偵測結果與穩定狀態一致 -> 重置計時器，維持穩定
        if instant_id == stable_id:
            state_change_start = None 
            final_id_to_tracker = stable_id
            debug_status = "Stable"
            
        # 情境 B: 偵測結果改變 (或錯誤) -> 進入緩衝區
        else:
            if state_change_start is None:
                state_change_start = current_time
            
            elapsed = current_time - state_change_start
            remaining = STABILITY_THRESHOLD - elapsed
            
            if elapsed < STABILITY_THRESHOLD:
                # [緩衝期]: 雖然現在做錯了，但我們「假裝」還在做原本的動作
                # 這樣計數器就不會斷掉！
                final_id_to_tracker = stable_id 
                debug_status = f"Holding... {remaining:.1f}s"
                
                # 顯示倒數條
                bar_w = 200
                fill_w = int(bar_w * (remaining / STABILITY_THRESHOLD))
                cv2.rectangle(display_frame, (10, 400), (10 + 200, 420), (100, 100, 100), -1)
                cv2.rectangle(display_frame, (10, 400), (10 + fill_w, 420), (0, 165, 255), -1)
                cv2.putText(display_frame, "Keep Going!", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                # [超時]: 錯誤持續太久，確認切換
                stable_id = instant_id
                final_id_to_tracker = stable_id
                state_change_start = None
                debug_status = "Switched"

        # ==========================================
        # 更新 Tracker
        # ==========================================
        
        # 為了讓 Tracker 運作，我們需要構造一個 gesture 物件
        gesture_to_send = None
        
        if final_id_to_tracker != 0:
            # 如果判定有效，我們優先使用備份的 gesture 物件 (防止 hand_result 消失時崩潰)
            # 並強制覆寫它的 class_id 為我們判定後的 final_id
            if last_valid_gesture:
                gesture_to_send = last_valid_gesture
                gesture_to_send.class_id = final_id_to_tracker
            elif gesture:
                gesture_to_send = gesture
                gesture_to_send.class_id = final_id_to_tracker
        else:
            # 如果判定為 Idle，送一個 ID=0 的物件讓 Tracker 暫停
            if gesture:
                gesture.class_id = 0
                gesture_to_send = gesture
        
        if gesture_to_send:
            stretch_record = tracker.update(gesture_to_send)
            if stretch_record:
                 print(f"完成伸展! {stretch_record.stretch_type} (Count: {tracker.get_stats().total_count})")

        # ==========================================
        # 繪製 UI
        # ==========================================
        
        # 1. 繪製進度條 (如果穩定 ID 不是 Idle)
        if stable_id != 0:
            bar_color = (0, 255, 0) # 綠色
            if "Holding" in debug_status: bar_color = (0, 165, 255) # 緩衝期變橘色
            
            cv2.rectangle(display_frame, (50, 150), (80, 350), (255, 255, 255), 2)
            cv2.rectangle(display_frame, (50, 350 - int(200*progress_bar_val)), (80, 350), bar_color, -1)
        
        # 2. 顯示診斷資訊
        cv2.putText(display_frame, f"CV Group: {cv_group_name}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, f"State: {debug_status}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 3. 呼叫 Renderer 畫最後的 UI
        if has_person and hand_result and gesture:
            final_frame = renderer.render(display_frame, hand_result, gesture_to_send if gesture_to_send else gesture, tracker.get_stats(), tracker.get_state_info())
        else:
            final_frame = display_frame

        cv2.imshow("Rehab System", final_frame)

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