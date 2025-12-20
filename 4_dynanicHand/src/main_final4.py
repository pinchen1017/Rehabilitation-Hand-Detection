"""
手部復健偵測系統 - 最終優化版 v4 (DNN 骨架回填修復版)
特色:
1. 解決第四指/中指 HSV 偵測斷裂問題。
2. 使用「DNN 骨架回填 (Skeleton Injection)」技術，強制修補 Mask。
3. 優化形態學運算，保留細微手指。
"""
# 步驟 A： 引入模組
# A-1 引入標準模組
import sys
import cv2
import numpy as np
import os
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

# 步驟 B： 定義 CVAnalyzer 類別
class CVAnalyzer:
    """處理 YOLO 去背與 Convex Hull / 指縫分析的類別"""
    # B-1: 初始化
    def __init__(self):
        print("  [系統] 初始化 YOLOv8-Seg 與 CV 分析模組...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # 膚色閥值 (HSV) - 稍微放寬範圍
        self.lower_skin = np.array([0, 15, 60], dtype=np.uint8) # Saturation 降到 15
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
        計算幾何特徵 (Solidity, Circularity, Defects)
        【關鍵修改】：加入 DNN 骨架回填，修復手指斷裂
        """
        h, w, _ = clean_frame.shape
        
        # B-2-1: 計算 ROI (Padding 加大到 80，確保手指不被切斷)
        x_list = [int(lm * w) for lm in hand_result.landmarks[0::3]]
        y_list = [int(lm * h) for lm in hand_result.landmarks[1::3]]
        
        x_min, x_max = max(0, min(x_list)), min(w, max(x_list))
        y_min, y_max = max(0, min(y_list)), min(h, max(y_list))
        
        padding = 80 # 加大 Padding
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        roi_img = clean_frame[y_min:y_max, x_min:x_max]
        if roi_img.size == 0: return None, 0, 0, 0, None

        # B-2-2: ROI 內 HSV 偵測
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 【關鍵修復步骤】：DNN 骨架回填 (Skeleton Injection)
        # 如果 HSV 沒抓到手指，我們用 DNN 的座標手動把手指「畫」在 Mask 上
        # 這樣 Convex Hull 就絕對不會切過手指了
        
        # 取得 ROI 相對座標的 landmarks
        roi_landmarks = []
        for i in range(21):
            lx = int(hand_result.landmarks[i*3] * w) - x_min
            ly = int(hand_result.landmarks[i*3+1] * h) - y_min
            roi_landmarks.append((lx, ly))
            
        # 定義手指連接順序 (大拇指到小指)
        finger_connections = [
            (2, 3, 4),         # Thumb
            (5, 6, 7, 8),      # Index
            (9, 10, 11, 12),   # Middle
            (13, 14, 15, 16),  # Ring (這隻最容易斷)
            (17, 18, 19, 20),  # Pinky
            (0, 5, 9, 13, 17)  # Palm
        ]
        
        # 在 Mask 上畫白線 (修補斷裂)
        # Thickness 設為 15~20 (模擬手指粗細)
        for connection in finger_connections:
            for i in range(len(connection) - 1):
                pt1 = roi_landmarks[connection[i]]
                pt2 = roi_landmarks[connection[i+1]]
                # 只有當點在 ROI 範圍內才畫
                if 0 <= pt1[0] < roi_img.shape[1] and 0 <= pt1[1] < roi_img.shape[0]:
                    cv2.line(mask, pt1, pt2, (255), 15) 

        # B-2-3: 形態學運算 (現在 Mask 已經很完整了，只需要去噪)
        # 使用較小的 Kernel (3x3) 做 Open，避免吃掉指尖
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        # 使用較大的 Kernel (7x7) 做 Close，填補內部空隙
        kernel_large = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
        debug_mask = mask.copy()

        # B-3: 找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0, 0, 0, debug_mask

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return None, 0, 0, 0, debug_mask

        # B-4: 計算特徵 (Convex Hull & Defects)
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        hull_points = cv2.convexHull(cnt, returnPoints=True)
        
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull_points)
        perimeter = cv2.arcLength(cnt, True)
        
        solidity = 0
        circularity = 0
        if area_hull > 0: solidity = area_cnt / area_hull
        if perimeter > 0: circularity = (4 * math.pi * area_cnt) / (perimeter ** 2)

        # B-5: 指縫計數 (Defects)
        defects_count = 0
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(cnt, hull_indices)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        
                        if 2 * b * c > 0:
                            angle = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2 * b * c)))) * 57.2958
                            # 判定: 角度銳利且深度夠深
                            if angle < 90 and depth > 10:
                                defects_count += 1
                                # 畫出指縫位置 (除錯用)
                                cv2.circle(debug_mask, far, 5, 128, -1)
            except:
                pass

        # B-6: 轉回全域座標
        hull_global = hull_points.copy()
        for point in hull_global:
            point[0][0] += x_min
            point[0][1] += y_min

        return hull_global, solidity, circularity, defects_count, debug_mask

def main():
    print("=" * 50)
    print("手部復健偵測系統 (骨架回填修正版)")
    print("=" * 50)

    # Step 1: 初始化 DNN 模組
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
    
    # Step 2: 初始化 CV 分析器
    cv_analyzer = CVAnalyzer()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Rehab System", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: ROI Mask", cv2.WINDOW_NORMAL)

    print("\n系統啟動完成 - 按 'q' 退出, 'r' 重置")

    while True:
        # Step 3: 主迴圈邏輯
        # 3-1 讀取影像
        ret, frame = cap.read()
        if not ret:
            print("警告: 無法讀取攝影機畫面")
            break

        # 3-2 水平翻轉
        frame = cv2.flip(frame, 1)

        # 3-3 YOLO 去背
        has_person, clean_frame, _ = cv_analyzer.get_person_frame(frame)
        
        display_frame = frame.copy()
        
        hand_result = None
        gesture = None
        warning_msg = ""
        cv_feedback = ""

        if has_person:
            cv2.putText(display_frame, "[YOLO] Person Locked", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 3-4 DNN 手勢辨識
            hand_result = detector.detect(clean_frame)

            if hand_result is not None:
                # 預測與平滑
                raw_prediction = classifier.predict(hand_result.landmarks)
                gesture = smoother.smooth(raw_prediction)

                # 更新狀態機
                stretch_record = tracker.update(gesture)
                if stretch_record:
                    print(f"完成伸展! 類型: {stretch_record.stretch_type}, "
                          f"總次數: {tracker.get_stats().total_count}")

                # 3-5 CV 幾何驗證 (含骨架回填)
                hull, solidity, circularity, defects, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)
                
                # 3-6 顯示 Debug 視窗
                if debug_mask is not None:
                    cv2.imshow("Debug: ROI Mask", debug_mask)
                    cv2.putText(display_frame, f"Gaps:{defects}", (10, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))

                if hull is not None:
                    # 畫出 Convex Hull
                    cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)
                    
                    # 雙重驗證邏輯
                    gesture_int = gesture.class_id if hasattr(gesture, 'class_id') else 0
                    
                    closed_group = [1, 2, 3, 4, 5, 6]
                    open_group = [7]

                    is_physically_gripped = (solidity > 0.70) or (circularity > 0.60) or (defects <= 1)
                    
                    if (gesture_int in closed_group) and (defects >= 3):
                        warning_msg = "Warning: Loose Grip!"
                        cv_feedback = "Close Gaps"
                    
                    if (gesture_int in open_group) and (defects <= 1):
                        warning_msg = "Warning: Fingers Not Spread!"
                        cv_feedback = "Spread Fingers"

                    # 繪製進度條
                    bar_h = 200
                    progress = np.clip((solidity - 0.60) / (0.85 - 0.60), 0, 1)
                    
                    if defects >= 3: 
                        bar_color = (0, 255, 0)
                        progress = 0.0
                    else:
                        bar_color = (0, int(255 * (1-progress)), int(255 * progress))

                    cv2.rectangle(display_frame, (50, 150), (80, 150+bar_h), (255, 255, 255), 2)
                    cv2.rectangle(display_frame, (50, 150 + bar_h - int(bar_h*progress)), 
                                 (80, 150+bar_h), bar_color, -1)
                    cv2.putText(display_frame, f"{int(progress*100)}%", (45, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

            else:
                cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))
        else:
             cv2.putText(display_frame, "Searching for Person...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Step 4: 渲染 UI
        if has_person and hand_result and gesture is not None:
            final_frame = renderer.render(
                display_frame,
                hand_result,
                gesture,
                tracker.get_stats(),
                tracker.get_state_info()
            )
            if warning_msg:
                cv2.putText(final_frame, warning_msg, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(final_frame, cv_feedback, (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            final_frame = display_frame

        cv2.imshow("Rehab System", final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n使用者退出程式")
            break
        elif key == ord('r'):
            tracker.reset()
            smoother.reset()
            print("\n統計已重置!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()