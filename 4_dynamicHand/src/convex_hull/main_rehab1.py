import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

class HandRehabSystem:
    def __init__(self):
        # 1. 初始化 YOLOv8 分割模型
        print("正在載入 YOLOv8-Seg 模型...")
        self.model = YOLO('yolov8n-seg.pt')
        
        # 2. 初始化人臉偵測器 (用來把臉塗黑，避免誤判)
        # 這是 OpenCV 內建的特徵檔，通常在 cv2.data.haarcascades 路徑下
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 3. 設定膚色偵測範圍 (HSV) - 針對亞洲膚色微調
        # 如果背景還是干擾，可以試著把 Saturation (30) 調高到 40 或 50
        self.lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    def remove_face(self, img):
        """
        使用 Haar Cascade 偵測臉部並塗黑
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        img_no_face = img.copy()
        for (x, y, w, h) in faces:
            # 把臉部區域畫黑色矩形遮掉 (稍微畫大一點確保遮乾淨)
            # 這樣膚色偵測就不會抓到臉了
            cv2.rectangle(img_no_face, (x-20, y-50), (x+w+20, y+h+50), (0, 0, 0), -1)
            
        return img_no_face

    def get_person_segmentation(self, frame):
        """
        使用 YOLOv8 鎖定人，回傳 Mask
        """
        results = self.model(frame, verbose=False, stream=True)
        
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        person_found = False
        
        for r in results:
            if r.masks is None:
                continue
            
            boxes = r.boxes
            masks = r.masks
            
            # 尋找最大的 Person (class_id = 0)
            max_area = 0
            best_mask = None

            for i, box in enumerate(boxes):
                if int(box.cls[0]) == 0: # 0 is Person
                    mask_raw = masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
                    
                    area = np.sum(mask_resized)
                    if area > max_area:
                        max_area = area
                        best_mask = (mask_resized * 255).astype(np.uint8)
                        person_found = True
            
            if best_mask is not None:
                combined_mask = best_mask

        return person_found, combined_mask

    def analyze_hand(self, person_img, original_frame):
        # 1. 先把臉塗黑 (關鍵修正: 解決臉被當成手的問題)
        img_no_face = self.remove_face(person_img)
        
        # 2. 轉 HSV 膚色偵測
        hsv = cv2.cvtColor(img_no_face, cv2.COLOR_BGR2HSV)
        mask_skin = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 3. 形態學運算 (關鍵修正: 切斷手腕)
        # 先用 Erode (侵蝕) 把細的手腕變不見，這樣拳頭會更像一個獨立的圓
        kernel = np.ones((5, 5), np.uint8)
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel) # 去雜訊
        
        # 這裡可以根據情況決定要不要做更強的侵蝕
        # mask_skin = cv2.erode(mask_skin, kernel, iterations=1) 
        
        # 4. 尋找輪廓
        contours, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return original_frame, 0, "No Hand Detected"

        # 找最大的輪廓
        cnt = max(contours, key=cv2.contourArea)
        area_cnt = cv2.contourArea(cnt)

        # 過濾太小的區塊 (例如雜訊)
        if area_cnt < 3000: 
            return original_frame, 0, "Area too small"

        # 5. 計算 Convex Hull
        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull)
        
        if area_hull == 0: return original_frame, 0, "Error"

        # === Solidity 計算 ===
        solidity = area_cnt / area_hull

        # === 狀態判定 (根據你的實測調整閥值) ===
        # 實測握拳 0.79 -> 設定閥值 0.76 (保留一點容錯)
        # 張開手通常會在 0.5 ~ 0.65 之間
        
        status = "Moving..."
        color = (0, 255, 255) # 黃色

        if solidity > 0.76: # 下修標準
            status = "FIST (Closed)"
            color = (0, 0, 255) # 紅色
        elif solidity < 0.65: # 下修標準
            status = "OPEN (Relax)"
            color = (0, 255, 0) # 綠色
        
        # === 視覺化 ===
        # 畫在原圖上
        cv2.drawContours(original_frame, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(original_frame, [hull], -1, (255, 0, 0), 2)
        
        # 顯示處理過程的小視窗 (Debug用)
        debug_img = cv2.resize(mask_skin, (200, 200))
        cv2.imshow("Debug: Skin Mask (No Face)", debug_img)

        self.draw_progress_bar(original_frame, solidity, status, color)
        
        return original_frame, solidity, status

    def draw_progress_bar(self, img, solidity, text, color):
        bar_x, bar_y, bar_w, bar_h = 50, 100, 30, 300
        # 重新映射：將 0.60 ~ 0.80 映射到 0% ~ 100%
        progress = np.clip((solidity - 0.60) / (0.80 - 0.60), 0, 1)
        filled_h = int(bar_h * progress)
        
        cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255, 255, 255), 2)
        cv2.rectangle(img, (bar_x, bar_y+bar_h-filled_h), (bar_x+bar_w, bar_y+bar_h), color, -1)
        cv2.putText(img, f"{solidity:.2f}", (bar_x, bar_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, text, (bar_x+40, bar_y+bar_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def main():
    cap = cv2.VideoCapture(0)
    rehab_sys = HandRehabSystem()
    
    # 設定視窗可調整大小
    cv2.namedWindow("Rehabilitation System", cv2.WINDOW_NORMAL)

    print("系統啟動中...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. YOLO 抓人
        found_person, mask = rehab_sys.get_person_segmentation(frame)
        
        output_frame = frame.copy()

        if found_person:
            # 產生「只有人，背景黑」的圖
            person_img = cv2.bitwise_and(frame, frame, mask=mask)
            
            # 顯示 YOLO 抓到的範圍 (Debug用)
            mask_debug = cv2.resize(mask, (200, 200))
            cv2.imshow("Debug: YOLO Mask", mask_debug)
            
            # 2. 手部分析 (含去臉部邏輯)
            output_frame, val, status = rehab_sys.analyze_hand(person_img, output_frame)
        else:
            cv2.putText(output_frame, "YOLO: Searching Person...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Rehabilitation System", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()