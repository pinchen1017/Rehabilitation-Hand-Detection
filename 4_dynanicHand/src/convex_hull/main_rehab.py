import cv2
import numpy as np
from ultralytics import YOLO
import math

class HandRehabSystem:
    def __init__(self):
        # 1. 初始化 YOLOv8 分割模型 (第一次執行會自動下載 yolov8n-seg.pt)
        print("正在載入 YOLOv8-Seg 模型...")
        self.model = YOLO('yolov8n-seg.pt')
        
        # 設定目標類別: 0 是 'person' (COCO dataset)
        self.target_class_id = 0
        
        # 2. 設定膚色偵測範圍 (HSV)
        # 注意: 這裡的數值可能需要根據您的現場光線微調
        self.lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def get_person_segmentation(self, frame):
        """
        使用 YOLOv8 鎖定畫面中最大的「人」，並回傳去背後的影像
        """
        results = self.model(frame, verbose=False, stream=True)
        
        mask_combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        person_found = False

        for r in results:
            if r.masks is None:
                continue
            
            # 取得所有偵測到的物件
            boxes = r.boxes
            masks = r.masks
            
            # 尋找最大的 Person
            max_area = 0
            best_mask = None

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                if cls_id == self.target_class_id: # 確認是人
                    # 取得 mask (轉為 numpy 並調整大小與原圖一致)
                    # ultralytics 的 mask 原始輸出尺寸可能較小，需 resize
                    mask_raw = masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
                    
                    # 計算面積，我們只鎖定畫面中最大的那個人 (主要復健者)
                    area = np.sum(mask_resized)
                    if area > max_area:
                        max_area = area
                        best_mask = (mask_resized * 255).astype(np.uint8)
                        person_found = True

            if person_found and best_mask is not None:
                mask_combined = best_mask

        # 進行去背: 將 Mask 應用到原圖
        # 背景變黑，只保留人
        masked_img = cv2.bitwise_and(frame, frame, mask=mask_combined)
        return person_found, masked_img, mask_combined

    def analyze_hand(self, clean_frame, original_frame):
        """
        在去背後的影像上進行 Convex Hull 分析
        """
        # 1. 轉 HSV 並進行膚色偵測
        hsv = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2HSV)
        mask_skin = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 形態學運算：消除雜訊 (Open) 與 填補破洞 (Close)
        kernel = np.ones((5, 5), np.uint8)
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_CLOSE, kernel)

        # 2. 尋找輪廓
        contours, _ = cv2.findContours(mask_skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return original_frame, 0, "No Hand"

        # 假設最大的膚色區塊是手 (因為已經用 YOLO 過濾掉背景雜訊了)
        # 注意: 如果臉部也在畫面中，可能需要額外邏輯排除臉部 (如位置判斷)
        cnt = max(contours, key=cv2.contourArea)
        area_cnt = cv2.contourArea(cnt)

        if area_cnt < 2000: # 過濾太小的雜訊
            return original_frame, 0, "Too Small"

        # 3. 計算 Convex Hull (凸包)
        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull)
        
        # 避免除以零
        if area_hull == 0:
            return original_frame, 0, "Error"

        # === 核心演算法: Solidity 計算 ===
        solidity = area_cnt / area_hull

        # === 核心演算法: Convexity Defects (指縫偵測 - Check 1) ===
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull_indices)
        
        finger_gaps = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                
                # 利用餘弦定理計算手指間的角度
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57
                
                # 如果角度小於 90 度且深度夠深，視為指縫
                if angle <= 90 and d > 10000: # d 是距離 * 256
                    finger_gaps += 1
                    cv2.circle(original_frame, far, 5, [0, 0, 255], -1) # 畫出指縫紅點

        # === 狀態判定邏輯 ===
        # Check 1: 雜訊過濾 (這裡簡單示意，可根據需求加強)
        # 如果是拳頭，finger_gaps 應該很少；如果是張開手，gaps 應該 >= 3
        
        status = "Analyzing..."
        color = (0, 255, 255) # 黃色

        # Check 2: 復健握力判定 (Solidity)
        # Solidity > 0.90 -> 握緊 (Fist)
        # Solidity < 0.75 -> 張開 (Open)
        
        if solidity > 0.90:
            status = "FIST (Closed)"
            color = (0, 0, 255) # 紅色代表握緊用力
        elif solidity < 0.75:
            status = "OPEN (Relax)"
            color = (0, 255, 0) # 綠色代表放鬆
        else:
            status = "In Progress..."

        # === 視覺化繪圖 ===
        # 畫出輪廓 (綠色)
        cv2.drawContours(original_frame, [cnt], -1, (0, 255, 0), 2)
        # 畫出凸包 (藍色)
        cv2.drawContours(original_frame, [hull], -1, (255, 0, 0), 2)
        
        # 繪製進度條
        self.draw_progress_bar(original_frame, solidity, status, color)

        return original_frame, solidity, status

    def draw_progress_bar(self, img, solidity, text, color):
        """
        繪製握緊程度的進度條
        """
        h, w, _ = img.shape
        # 定義進度條位置
        bar_x, bar_y = 50, 100
        bar_w, bar_h = 30, 300
        
        # 將 Solidity 映射到 0-1 之間的進度 (0.7 ~ 0.95)
        # 0.7以下視為0%，0.95以上視為100%
        progress = np.clip((solidity - 0.70) / (0.95 - 0.70), 0, 1)
        
        filled_h = int(bar_h * progress)
        
        # 畫外框
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        # 畫填充 (由下往上)
        cv2.rectangle(img, (bar_x, bar_y + bar_h - filled_h), (bar_x + bar_w, bar_y + bar_h), color, -1)
        
        # 顯示數值
        cv2.putText(img, f"Solidity: {solidity:.2f}", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, text, (bar_x + 50, bar_y + bar_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main():
    cap = cv2.VideoCapture(0) # 開啟攝影機
    rehab_sys = HandRehabSystem()

    print("程式啟動 - 按 'q' 離開")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. YOLO 去背階段
        found_person, clean_frame, mask = rehab_sys.get_person_segmentation(frame)

        output_frame = frame.copy()

        if found_person:
            # 顯示去背後的效果 (Debug用，可縮小顯示在角落)
            # cv2.imshow("YOLO Segmented", clean_frame)
            
            # 2. 手部 CV 分析階段
            output_frame, solidity, status = rehab_sys.analyze_hand(clean_frame, output_frame)
        else:
            cv2.putText(output_frame, "Searching for Person...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Rehabilitation System", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()