import os
import glob
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import Normalizer
import mediapipe as mp

# ========== 手部關鍵點正規化函 ==========
def normalize_landmarks(lm):
    """
    將 MediaPipe landmark（21 點）轉換成：
    1. 以手腕為原點
    2. 尺度標準化（使用 WRIST→MIDDLE_MCP 長度為基準）
    3. 旋轉對齊，使手掌方向一致（降低角度差）
    
    輸入：lm : mediapipe landmark list, 包含 21 個點
    輸出：pts : shape (21,2)，已正規化的 landmark
    """
    # 1. 以 WRIST 當中心
    wrist = np.array([lm[0].x, lm[0].y])
    pts = np.array([[p.x, p.y] for p in lm])
    pts = pts - wrist

    # 2. 以 WRIST→MIDDLE_MCP 做尺度正規化
    middle = np.array([lm[9].x, lm[9].y])
    scale = np.linalg.norm(middle - wrist)
    scale = max(scale, 1e-6)  # 避免除以 0
    pts = pts / scale

    # 3. 旋轉：讓 WRIST→MIDDLE_MCP 對齊 X 軸
    angle = -np.arctan2(pts[9][1], pts[9][0])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    pts = pts.dot(R.T)

    return pts

def to_42d(points):
    """將 (21,2) landmark 展平成一維 42 維列表"""
    return points.flatten().tolist()

# MediaPipe Hands 初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# 載入訓練好的模型
# 自動尋找最新的模型檔案，或使用指定的檔案名稱
# 嘗試自動尋找最新的模型檔案

# 讀取 models 資料夾底下所有 h5 模型
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # 取得專案根目錄
models_dir = os.path.join(base_dir, "models", "rehab_model_tuned")

# 搜尋模型檔案
model_files = glob.glob(os.path.join(models_dir, "*.h5"))

if not model_files:
    print("錯誤: models 資料夾中找不到任何 h5 模型檔")
    print(f"搜尋位置: {models_dir}")
    exit(1)

# 依照新增順序挑最新
model_filename = max(model_files, key=os.path.getctime)
print(f"自動找到最新模型檔案: {model_filename}")

# 載入模型
try:
    tf_model = load_model(model_filename)
    print(f"成功載入模型: {model_filename}")
except Exception as e:
    print(f"無法載入模型: {e}")
    exit(1)

# 手勢類別名稱
prediction_strings = [
    "hook_fist",  # 類別 1
    "angry_fist",  # 類別 2
    "thumb_flextion",  # 類別 3
    "straight_fist",  # 類別 4
    "the_duck",  # 類別 5
    "fist",  # 類別 6
    "spend_hand",   # 類別 7
    "other"  # 類別 0

]

# 開啟網路攝影機
cap = cv2.VideoCapture(0)
window_name = "Hand Rehabilitation Classification Window"

print("=" * 60)
print("Hand Rehabilitation Gesture Recognition")
print("=" * 60)
print("按 'q' 鍵退出程式")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換 BGR 到 RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 處理手部偵測
    results = hands.process(img_rgb)
    
    x_set = []
    y_set = []
    hand_detected = False
    
    # 確保手部關鍵點存在才進行分類
    if results.multi_hand_landmarks:
        hand_detected = True
        
        # 取得第一隻偵測到的手
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 取得手部標籤（用於左右手判斷）
        hand_label = None
        if results.multi_handedness:
            hand_label = results.multi_handedness[0].classification[0].label
        
        # 繪製手部標記
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # ========== 關鍵修正：使用與訓練時相同的正規化流程 ==========
        try:
            # 1. 取得 MediaPipe landmark 列表
            lm = hand_landmarks.landmark
            
            # 2. 使用 normalize_landmarks 進行正規化（與訓練時一致）
            pts = normalize_landmarks(lm)
            
            # 3. 左右手判斷與翻轉（統一轉成右手，與訓練時一致）
            if hand_label == "Left":
                pts[:, 0] = -pts[:, 0]
            
            # 4. 轉換為 42 維向量
            vec_42d = to_42d(pts)
            xy_set = np.asarray(vec_42d, dtype=np.float32)
            
            # 驗證資料維度
            if len(xy_set) != 42:
                print(f"警告: 資料維度錯誤，預期 42，實際 {len(xy_set)}")
                continue
            
            xy_set = xy_set.reshape(1, -1)
            
            # 5. 使用與訓練時相同的 Normalizer 進行正規化
            transformer = Normalizer().fit(xy_set)
            X_test = transformer.transform(xy_set)
            
            # 6. 進行預測
            predictions = tf_model.predict(X_test, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # 驗證預測類別範圍
            if predicted_class < 0 or predicted_class >= len(prediction_strings):
                print(f"警告: 預測類別超出範圍: {predicted_class}")
                continue
            
            # 7. 顯示預測結果（降低信心度閾值以便測試）
            confidence_threshold = 0.1  # 降低閾值以便測試
            if confidence > confidence_threshold:
                predicted_gesture = prediction_strings[predicted_class]
                
                # 取得像素座標用於顯示（使用中指 MCP 關節點）
                h, w, _ = frame.shape
                text_x = int(hand_landmarks.landmark[9].x * w)
                text_y = int(hand_landmarks.landmark[9].y * h)
                
                # 確保文字不會超出畫面邊界
                text_x = max(10, min(text_x, w - 200))
                text_y = max(30, min(text_y, h - 10))
                
                # 顯示預測文字（帶有陰影效果以提高可讀性）
                cv2.putText(frame, predicted_gesture, (text_x, text_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                cv2.putText(frame, predicted_gesture, (text_x, text_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 顯示信心度
                confidence_text = f"Confidence: {confidence:.2f}"
                cv2.putText(frame, confidence_text, (text_x, text_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 在控制台輸出預測結果（用於調試）
                print(f"預測: {predicted_gesture} (類別: {predicted_class}, 信心度: {confidence:.2f})")
            else:
                # 信心度不足時顯示提示
                h, w, _ = frame.shape
                cv2.putText(frame, f"Low confidence: {confidence:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"預測過程發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            continue
    else:
        # 未偵測到手部時顯示提示
        cv2.putText(frame, "Please put hand in frame", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, "Please put hand in frame", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    
    # 按 'q' 鍵退出
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("程式結束")
        exit()

