"""
==========================================
STEP 2 — Preprocess Landmarks
圖片 → MediaPipe Landmarks → 正規化 → 42D 座標 → 儲存為影片 JSON

此步驟負責：
1. 讀取 Step1 產生的每張 frame
2. MediaPipe Hands 偵測 21 個手部關節點
3. 以「手腕為中心」做正規化（平移 + 標準化 + 旋轉對齊）
4. 將 21×2 維的 landmark 展平成 42 維
5. 每支影片存成一個 JSON：{ "frames": [42D, 42D, ...] }

輸入資料夾結構：
    1_frames/
        gestureA/
            video1/
                0.jpg
                1.jpg
                ...
            video2/
                ...

輸出資料夾結構：
    2_processed/
        gestureA/
            video1.json
            video2.json

後續：
    Step3 build_csv.py 會從每支影片的 JSON 挑關鍵幀 → 建立 CNN 訓練 CSV
==========================================
"""

import numpy as np
import cv2
from pathlib import Path
import mediapipe as mp
import json

# Step 0: 資料夾設定
FRAMES_DIR = Path("dataset")       # 由 Step1 抽好的 frames 來源
OUT_DIR = Path("2_processed")       # 本步驟輸出：影片 JSON
mp_hands = mp.solutions.hands

# Step 1: Landmark 正規化（中心化 + 尺度化 + 旋轉對齊）
def normalize_landmarks(lm):
    """
    normalize_landmarks()

    功能：
        將 MediaPipe landmark（21 點）轉換成：
        1. 以手腕為原點
        2. 尺度標準化（使用 WRIST→MIDDLE_MCP 長度為基準）
        3. 旋轉對齊，使手掌方向一致（降低角度差）

    輸入：
        lm : mediapipe landmark list, 包含 21 個點

    輸出：
        pts : shape (21,2)，已正規化的 landmark
    """

    # 1. 以 WRIST 當中心
    wrist = np.array([lm[0].x, lm[0].y])
    pts = np.array([[p.x, p.y] for p in lm])
    pts = pts - wrist

    # 2. 以 WRIST→MIDDLE_MCP 做尺度正規化
    middle = np.array([lm[9].x, lm[9].y])
    scale = np.linalg.norm(middle - wrist)
    scale = max(scale, 1e-6)       # 避免除以 0
    pts = pts / scale

    # 3. 旋轉：讓 WRIST→MIDDLE_MCP 對齊 X 軸
    angle = -np.arctan2(pts[9][1], pts[9][0])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    pts = pts.dot(R.T)

    return pts

# Step 2: 資料增強
def augment(points):
    """
    augment()

    功能：
        在 landmark 上加入隨機旋轉與 Gaussian noise，
        用於模型訓練時避免過擬合。

    輸入：
        points : (21,2) 正規化後的 landmark

    輸出：
        pts : (21,2) 加入擾動後的 landmark
    """

    # 隨機旋轉
    angle = np.random.normal(0, np.pi / 10)
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    pts = points.dot(R.T)

    # 高斯雜訊
    noise = np.random.normal(0, 0.01, pts.shape)
    pts = pts + noise

    return pts

# Step 3 — 將 (21,2) 轉成 42 維向量
def to_42d(points):
    """
    to_42d()

    功能：
        將 (21,2) landmark 展平成一維 42 維列表

    回傳：
        list 長度 = 42
    """
    return points.flatten().tolist()

# Step 4 — Main 主流程
def main():
    """
    main()

    功能：
        1. 遍歷 1_frames 底下所有手勢資料夾
        2. 針對每支影片，逐張讀取 frame
        3. 用 MediaPipe Hands 抓 landmark（21 點）
        4. landmark 正規化 → flatten 成 42D
        5. 每支影片存成 1 份 JSON

    最終輸出：
        2_processed/{gesture}/{video}.json
    """
    
    # Step 4-1 — MediaPipe Hands 初始化（只做一次）
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    if not FRAMES_DIR.exists():
        print(f"Dataset dir not found: {FRAMES_DIR}")
        return

    OUT_DIR.mkdir(exist_ok=True)
    
    # 準備用來收集所有處理後的數據 (可以用來最後檢查總量)
    total_frames_count = 0

    # === 修改 1: 讀取所有資料夾 (不管它叫 _left 還是 _right) ===
    # 只要是資料夾，我們都讀進來
    for folder_path in sorted(FRAMES_DIR.iterdir()):
        if not folder_path.is_dir():
            continue

        # === 修改 2: 從資料夾名稱解析 Label ===
        # 資料夾名稱範例: "1_hook_fist_left" -> split('_') -> ["1", "hook", "fist", "left"]
        # 取第 0 個元素 "1" 轉成 int
        try:
            label_idx = int(folder_path.name.split('_')[0])
        except ValueError:
            print(f"[跳過] 無法解析資料夾名稱: {folder_path.name}")
            continue

        print(f"正在處理資料夾: {folder_path.name} (Label: {label_idx}) ...")
        
        # 建立對應的輸出資料夾結構 (只是為了存 json，不影響訓練)
        # 這樣你的 json 也會分開存，方便檢查
        save_dir = OUT_DIR / folder_path.name
        save_dir.mkdir(exist_ok=True, parents=True)

        # 讀取該資料夾內的所有 jpg 圖片 (包含子資料夾)
        image_files = list(folder_path.rglob("*.jpg")) + list(folder_path.rglob("*.jpeg")) + list(folder_path.rglob("*.png"))
        
        if not image_files:
            print(f"  -> 空資料夾，跳過")
            continue

        # 逐張圖片處理
        for img_file in image_files:
            frame = cv2.imread(str(img_file))
            if frame is None: continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)

            if not res.multi_hand_landmarks:
                continue

            # 準備存這張圖生出來的所有數據 (包含擴增的)
            frame_data_list = []

            # 取得骨架 (只取第一隻偵測到的手)
            lm = res.multi_hand_landmarks[0].landmark
            pts = normalize_landmarks(lm)

            # === 左右手判斷與翻轉 (統一轉成右手) ===
            # MediaPipe 判斷結果
            hand_label = res.multi_handedness[0].classification[0].label
            
            # 如果是左手，就翻轉 X 軸
            if hand_label == "Left":
                pts[:, 0] = -pts[:, 0] 

            # 先存一筆原始數據 (已翻轉)
            vec = to_42d(pts)
            frame_data_list.append(vec)

            # 強力擴增 (1張變30張) ===
            for _ in range(29):
                aug_pts = augment(pts)      # 隨機旋轉 + 雜訊
                aug_vec = to_42d(aug_pts)
                frame_data_list.append(aug_vec)

            # 一張圖片存一個 JSON」，這樣比較不會搞混
            # 檔名範例: image01.jpg -> image01.json
            json_name = img_file.stem + ".json"
            out_path = save_dir / json_name
            
            # 寫入 JSON 格式
            # 注意：這裡我們要把 label 也寫進去，或者在 build_csv 時再處理
            # 目前架構是 build_csv 會看資料夾，所以這裡存 features 就好
            with open(out_path, "w") as f:
                json.dump({"frames": frame_data_list}, f)
                
            total_frames_count += len(frame_data_list)

    print(f"前處理完成！總共產出了 {total_frames_count} 筆訓練數據。")

    

if __name__ == "__main__":
    main()