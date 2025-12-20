"""
手部復健偵測系統 - 最終整合版 (YOLOv8 + DNN + Convex Hull)
整合了:
1. YOLOv8 去背與人物鎖定
2. DNN 手勢分類 (7種手勢)
3. Convex Hull 幾何分析 (握緊度與雙重驗證)
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
# 確保路徑指向你的模型
MODEL_PATH = os.path.join(BASE_DIR, "..", DEFAULT_MODEL_PATH) 
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "..", DEFAULT_YOLO_MODEL_PATH)

class CVAnalyzer:
    """處理 YOLO 去背與 Convex Hull 幾何分析的類別"""
    # Step 1: 初始化 YOLOv8-Seg 與 CV 分析模組
    def __init__(self):
        print("  [系統] 初始化 YOLOv8-Seg 與 CV 分析模組...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # 人臉偵測 (用於移除臉部干擾)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 膚色閥值 (HSV)
        self.lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    # Step 2: 鎖定最大的人並去背
    def get_person_frame(self, frame):
        """YOLO: 鎖定最大的人並去背"""
        results = self.yolo_model(frame, verbose=False, stream=True)
        mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        person_found = False

        for r in results:
            if r.masks is None: continue
            
            # 找出最大的 person (class_id=0)
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

        # 產生去背圖 (背景變黑)
        if person_found:
            clean_frame = cv2.bitwise_and(frame, frame, mask=mask_combined)
            return True, clean_frame, mask_combined
        else:
            return False, frame, None

    # Step 3: 移除臉部像素，避免膚色偵測抓錯
    def remove_face(self, img):
        """移除臉部像素，避免膚色偵測抓錯"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        img_no_face = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_no_face, (x-20, y-50), (x+w+20, y+h+50), (0, 0, 0), -1)
        return img_no_face

    # Step 4: Convex Hull 分析
    def analyze_geometry(self, clean_frame):
        """
        Convex Hull 分析
        回傳: (hull_points, solidity, circularity, feedback_text)
        """
        # 1. 移除臉部
        img_proc = self.remove_face(clean_frame)
        
        # 2. HSV 膚色遮罩
        hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 3. 形態學優化 (切斷手腕干擾)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 4. 輪廓搜尋
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0, 0, "No Hand"

        cnt = max(contours, key=cv2.contourArea)
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < 3000: 
            return None, 0, 0, "Too Small"

        # 5. 計算幾何特徵
        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull)
        perimeter = cv2.arcLength(cnt, True)
        
        if area_hull == 0 or perimeter == 0: return None, 0, 0, "Error"

        solidity = area_cnt / area_hull
        # 圓形度公式: 4 * pi * Area / Perimeter^2 (完美圓=1.0)
        circularity = (4 * math.pi * area_cnt) / (perimeter ** 2)

        return hull, solidity, circularity, "OK"

def main():
    print("=" * 50)
    print("手部復健偵測系統 (整合版)")
    print("YOLO 去背 -> DNN 辨識 -> CV 驗證")
    print("=" * 50)

    # 1. 初始化 DNN 相關模組
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
    
    # 2. 初始化 CV/YOLO 分析器
    cv_analyzer = CVAnalyzer()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Rehab System (Integrated)", cv2.WINDOW_NORMAL)

    print("\n系統啟動完成 - 按 'q' 退出, 'r' 重置")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # === 第一步: YOLO 去背 ===
        has_person, clean_frame, person_mask = cv_analyzer.get_person_frame(frame)
        
        display_frame = frame.copy()
        warning_msg = ""
        cv_feedback = "Wait..."
        solidity = 0
        circularity = 0 # 確保變數有初始化
        
        if has_person:
            # 顯示 YOLO 鎖定提示
            cv2.putText(display_frame, "[YOLO] Person Locked", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # === 第二步: DNN 手勢辨識 (使用去背後的圖 clean_frame) ===
            # MediaPipe 在乾淨的圖上跑，準確度會提高
            # 使用 .landmarks 屬性傳入預測
            hand_result = detector.detect(clean_frame) 
            
            gesture_obj = None # 這是物件 (給 tracker 與 renderer 用)
            gesture_int = 0    # 這是整數 (給邏輯判斷用)
            
            if hand_result:
                # 取得 DNN 預測
                # 從 hand_result 取出 landmarks 陣列
                raw_prediction = classifier.predict(hand_result.landmarks)
                
                # 平滑化處理 (回傳的 gesture_obj 是一個物件)
                gesture_obj = smoother.smooth(raw_prediction)
                
                # 從物件中取出整數 ID (用於雙重驗證邏輯)
                if hasattr(gesture_obj, 'class_id'):
                    gesture_int = gesture_obj.class_id
                else:
                    # Fallback 機制
                    gesture_int = 0

                # 更新復健計數器 (Tracker 吃物件)
                stretch_record = tracker.update(gesture_obj)
                if stretch_record:
                    print(f"動作完成: {stretch_record.stretch_type}, 次數: {tracker.get_stats().total_count}")

            # === 第三步: CV 幾何驗證 (雙重確認) ===
            hull, solidity, circularity, cv_msg = cv_analyzer.analyze_geometry(clean_frame)
            
            if hull is not None:
                # 畫出 Convex Hull (藍色多邊形) 
                cv2.drawContours(display_frame, [hull], -1, (255, 0, 0), 2)
                
                # 判定物理狀態
                is_physically_closed = (solidity > 0.76) or (circularity > 0.65)
                
                # --- 雙重驗證邏輯 ---
                # 使用整數 gesture_int 來做判斷
                dnn_says_closed = gesture_int in [1, 2, 3, 6] # hook, angry, thumb, fist
                if dnn_says_closed and not is_physically_closed:
                    warning_msg = "Warning: Grip not tight!"
                    cv_feedback = "Tighten Grip!"
                
                dnn_says_open = (gesture_int == 7) # spend_hand
                if dnn_says_open and is_physically_closed:
                    warning_msg = "Warning: Hand not open!"
                    cv_feedback = "Open Hand!"

                # 繪製握緊進度條
                bar_h = 200
                progress = np.clip((solidity - 0.60) / (0.85 - 0.60), 0, 1)
                cv2.rectangle(display_frame, (50, 150), (80, 150+bar_h), (255, 255, 255), 2)
                cv2.rectangle(display_frame, (50, 150 + bar_h - int(bar_h*progress)), 
                             (80, 150+bar_h), (0, 0, 255), -1)
                cv2.putText(display_frame, f"{int(progress*100)}%", (45, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        else:
            cv2.putText(display_frame, "Searching for Person...", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # === 第四步: 整合渲染 UI ===
        # 使用原本 renderer 的功能，再加上我們的警告訊息
        if has_person and hand_result and gesture_obj is not None:
            # 這裡必須傳 gesture_obj (物件)，不能傳 gesture_int
            final_frame = renderer.render(
                display_frame, 
                hand_result,
                gesture_obj, 
                tracker.get_stats(),
                tracker.get_state_info()
            )
            # 疊加雙重驗證警告 (如果有)
            if warning_msg:
                cv2.putText(final_frame, warning_msg, (200, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(final_frame, f"CV Check: {cv_feedback}", (200, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                # 顯示幾何數值 debug
                cv2.putText(final_frame, f"Sol:{solidity:.2f} Cir:{circularity:.2f}", (200, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            final_frame = display_frame

        cv2.imshow("Rehab System (Integrated)", final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
            smoother.reset()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()