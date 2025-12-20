"""
手部復健偵測系統 - 最終優化版
特色:
1. 嚴格遵守原始 DNN 處理流程 (Flip -> Detect -> Predict -> Smooth -> Tracker)。
2. 整合 YOLOv8 去背，提升 DNN 準確度。
3. 整合 CV 指縫計數 (Defects)，解決手指沾黏誤判問題。
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
    # B-1: YOLO 去背
    # B-1-1 初始化 YOLOv8-Seg 與 CV 分析模組
    def __init__(self):
        print("  [系統] 初始化 YOLOv8-Seg 與 CV 分析模組...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # 膚色閥值 (HSV)
        self.lower_skin = np.array([0, 20, 60], dtype=np.uint8) # Saturation 降到 20 避免手指斷掉
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    # B-1-2 鎖定最大的人並去背
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

    # B-2: 凸包幾何驗證
    def analyze_hand_geometry(self, clean_frame, hand_result):
        """
        計算 ROI (Padding 加大到 60)
        """
        h, w, _ = clean_frame.shape
        
        # B-2-1 計算 ROI (Padding 加大到 60)
        x_list = [int(lm * w) for lm in hand_result.landmarks[0::3]]
        y_list = [int(lm * h) for lm in hand_result.landmarks[1::3]]
        
        x_min, x_max = max(0, min(x_list)), min(w, max(x_list))
        y_min, y_max = max(0, min(y_list)), min(h, max(y_list))
        
        padding = 60
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        roi_img = clean_frame[y_min:y_max, x_min:x_max]
        if roi_img.size == 0: return None, 0, 0, 0, None

        # B-2-2 ROI 內 RGB 轉 HSV
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        debug_mask = mask.copy()

        # B-2-3 找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, 0, 0, 0, debug_mask

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1000: return None, 0, 0, 0, debug_mask

        # B-2-4 計算特徵
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        hull_points = cv2.convexHull(cnt, returnPoints=True)
        
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull_points)
        perimeter = cv2.arcLength(cnt, True)
        
        solidity = 0
        circularity = 0
        if area_hull > 0: solidity = area_cnt / area_hull
        if perimeter > 0: circularity = (4 * math.pi * area_cnt) / (perimeter ** 2)

        # B-2-5 指縫計數 (Defects)
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
                                cv2.circle(debug_mask, far, 5, 128, -1)
            except:
                pass

        # B-2-6 轉回全域座標
        hull_global = hull_points.copy()
        for point in hull_global:
            point[0][0] += x_min
            point[0][1] += y_min

        return hull_global, solidity, circularity, defects_count, debug_mask

# 步驟 C： 主程式
def main():
    # Step 0: 初始化系統
    print("=" * 50)
    print("手部復健偵測系統 (最終整合版)")
    print("=" * 50)

    # 0-1 初始化 DNN 模組
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
    
    # 0-2 初始化 CV 分析器
    cv_analyzer = CVAnalyzer()
    cap = cv2.VideoCapture(0)

    # 0-3 初始化視窗
    cv2.namedWindow("Rehab System", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: ROI Mask", cv2.WINDOW_NORMAL)
    print("\n系統啟動完成 - 按 'q' 退出, 'r' 重置")

    while True:
        # Step 1: 讀取影像
        ret, frame = cap.read()
        if not ret:
            print("警告: 無法讀取攝影機畫面")
            break

        # 1-1 水平翻轉（鏡像效果）
        frame = cv2.flip(frame, 1)

        # 1-2 YOLO 去背 (Hybrid 核心：將 frame 傳進去，得到去背後的 clean_frame，可以大幅提高準確度)
        has_person, clean_frame, _ = cv_analyzer.get_person_frame(frame)
        
        # 1-3 準備 CV 線條的畫布 (使用原始 frame 複製)
        display_frame = frame.copy()
        
        # 1-4 初始化變數
        hand_result = None
        gesture = None
        warning_msg = ""
        cv_feedback = ""

        # 1-5 有人就顯示 YOLO 鎖定提示
        if has_person:
            cv2.putText(display_frame, "[YOLO] Person Locked", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Step 2: DNN 分類手勢
            # 注意：這裡我們傳入 clean_frame 給偵測器，效果最好
            # 如果您堅持要傳原始 frame，改成 detector.detect(frame) 即可，但強烈建議用 clean_frame
            hand_result = detector.detect(clean_frame)

            if hand_result is not None:
                # 2-1 預測
                raw_prediction = classifier.predict(hand_result.landmarks)
                # 2-2 平滑化 (得到 gesture 物件)
                gesture = smoother.smooth(raw_prediction)

                # 2-3 更新狀態機 (傳入 gesture 物件)
                stretch_record = tracker.update(gesture)
                if stretch_record:
                    print(f"完成伸展! 類型: {stretch_record.stretch_type}, "
                          f"總次數: {tracker.get_stats().total_count}")

                # 2-4 CV 幾何驗證 (平行處理)
                # 利用剛剛抓到的 hand_result 進行 ROI 切割與幾何分析
                hull, solidity, circularity, defects, debug_mask = cv_analyzer.analyze_hand_geometry(clean_frame, hand_result)
                
                # 2-5 顯示 Debug 視窗
                if debug_mask is not None:
                    cv2.imshow("Debug: ROI Mask", debug_mask)
                else:
                    cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))

                # Step 3: Convex hull 雙重驗證
                if hull is not None:
                    # 3-1 畫出 Convex Hull
                    cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)
                    
                    # 3-2 雙重驗證邏輯
                    gesture_int = gesture.class_id if hasattr(gesture, 'class_id') else 0
                    
                    closed_group = [1, 2, 3, 4, 5, 6]
                    open_group = [7]

                    # 3-3 物理判定: 實心度高 OR 圓度高 OR 指縫少 -> 視為握緊
                    is_physically_gripped = (solidity > 0.70) or (circularity > 0.60) or (defects <= 1)
                    
                    # 3-4.1 握緊警告邏輯
                    if (gesture_int in closed_group) and (defects >= 3):
                        warning_msg = "Warning: Loose Grip!"
                        cv_feedback = "Close Gaps"
                    
                    # 3-4.2 放鬆警告邏輯
                    if (gesture_int in open_group) and (defects <= 1):
                        warning_msg = "Warning: Fingers Not Spread!"
                        cv_feedback = "Spread Fingers"

                    # 3-5 繪製進度條 (加入指縫判斷變色)
                    bar_h = 200
                    progress = np.clip((solidity - 0.60) / (0.85 - 0.60), 0, 1)
                    
                    if defects >= 3: # 如果有很多指縫，強制顯示綠色(放鬆)
                        bar_color = (0, 255, 0)
                        progress = 0.0
                    else:
                        bar_color = (0, int(255 * (1-progress)), int(255 * progress))

                    cv2.rectangle(display_frame, (50, 150), (80, 150+bar_h), (255, 255, 255), 2)
                    cv2.rectangle(display_frame, (50, 150 + bar_h - int(bar_h*progress)), 
                                 (80, 150+bar_h), bar_color, -1)
                    cv2.putText(display_frame, f"{int(progress*100)}%", (45, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
                    
                    # 3-6 顯示數據 Debug
                    cv2.putText(display_frame, f"Gaps:{defects} Sol:{solidity:.2f}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            else:
                # 3-7 沒偵測到手時，Debug視窗黑屏
                cv2.imshow("Debug: ROI Mask", np.zeros((200, 200), dtype=np.uint8))
        else:
             cv2.putText(display_frame, "Searching for Person...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Step 4: 渲染 UI (依照您的原始邏輯)
        # 4-1 使用 renderer 渲染 UI (將 DNN 資訊與 CV 畫布結合)
        if has_person and hand_result and gesture is not None:
            final_frame = renderer.render(
                display_frame, # 這裡傳入已經畫了 CV 線條的圖
                hand_result,
                gesture,       # 傳入 gesture 物件
                tracker.get_stats(),
                tracker.get_state_info()
            )
            # 4-2 加上雙重驗證警告文字
            if warning_msg:
                cv2.putText(final_frame, warning_msg, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(final_frame, cv_feedback, (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            final_frame = display_frame

        # 4-3 顯示畫面
        cv2.imshow("Rehab System", final_frame)

        # 4-4 處理按鍵
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