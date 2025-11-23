期末專題
===
# 一、 使用規則
推送時從自己的 branch 進行推送，再到 github merge
- branch_J：許鯨魚
- branch_R：Risto
- branch_P：郭品陳
- branch_T：礦採泥
- branch_Z：013

# 二、 完整流程
## 1. 資料前處理
### 1-1 定義復健靜態手勢類別
```
{
  "0": "idle",           // 手放鬆（中立姿勢）
  "1": "hook",           // 勾拳
  "2": "angry_fist",           // 生氣的握拳
  "3": "thumb_flextion",  // 拇指彎曲
  "4": "straight_fist",// 直拳
  "5": "the_duck",   // 鴨子
  "6": "fist",   // 拇指在內的握拳
  "7": "spend_hand"   // 手指伸展
}
```
:::warning
1. 每類錄「3 秒連續」→ 你會自動得到 90 幀（30FPS）
2. 在 CSV 收集程式中，每 3 幀存一筆
:::

### 1-2 骨架來源
以相機影像取得手部關節點，再轉為「21 個手部關節的 3D 座標」做骨架輸入（v0–v20）【手勢偵測模組→關節座標→手勢辨識模組→軌跡處理→字元辨識】的後端流程如論文系統框圖所示。
```
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def get_hand_landmarks(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0].landmark  # list 長度 21
        return None
```
### 1-3 以「手腕」為中心的座標正規化
* **中心對齊**以手腕做為座標原點（v0 與 v9 的中點），可讓資料在空間中更平均、穩定。
* **尺度正規化**：根據手掌中心與指尖的距離進行尺度調整
* **旋轉對齊**：將手指對齊到一個標準方向，以降低旋轉誤差
```
import numpy as np

def normalize_landmarks(lm):
    """
    lm: mediapipe landmark list (21 points)
    return: normalized points, shape (21, 2)
    """

    # 1. 以 WRIST 當原點
    wrist = np.array([lm[0].x, lm[0].y])
    pts = np.array([[p.x, p.y] for p in lm])   # (21, 2)
    pts = pts - wrist

    # 2. 尺度正規化：WRIST → MIDDLE_MCP (v9)
    middle_mcp = np.array([lm[9].x, lm[9].y])
    scale = np.linalg.norm(middle_mcp - wrist)
    if scale < 1e-6:
        scale = 1e-6
    pts = pts / scale

    # 3. 旋轉對齊：讓 v0→v9 對齊 x 軸
    palm_vec = pts[9]  # 已經中心化、尺度化後的 v9
    angle = -np.arctan2(palm_vec[1], palm_vec[0])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    pts = pts.dot(R.T)

    return pts  # shape (21, 2)
```
### 1-4 把座標轉成 42 維 -> 符合 Fasko CNN 輸入
#### a. 取得 21 個關節 (lm.x, lm.y)
#### b. 做中心對齊、尺度正規化
#### c. 展平成 42 維向量： [x0,y0,x1,y1,…,x20,y20]
```
def to_42d_vector(landmarks_xy):
    """landmarks_xy: shape (21, 2) → 回傳 42 維一維向量"""
    return landmarks_xy.flatten().tolist()

...
# 取 landmark → normalize → flatten
lm = extract_landmarks(frame_rgb)
if lm:
    lm_norm = normalize_landmarks(lm)  # shape = (21,2)
    vec_42 = to_42d_vector(lm_norm)    # list 長度 = 42
...

```

### 1-5 資料增強（避免過擬合）
小角度旋轉 + Gaussian noise（對齊你提到的 θ ~ N(0, π/10)）
```
def augment_rotate(points):
    angle = np.random.normal(0, np.pi/10)
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return points.dot(R.T)

def augment_noise(points, sigma=0.01):
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise

def augment(points, use_rotate=True, use_noise=True):
    out = points.copy()
    if use_rotate:
        out = augment_rotate(out)
    if use_noise:
        out = augment_noise(out)
    return out
```
### 1-6 資料收集腳本：錄影 → 產生 CSV
檔名：collect_rehab_dataset.py
```
import cv2
import csv
import mediapipe as mp
import numpy as np

from pathlib import Path

mp_hands = mp.solutions.hands

from normalize_utils import normalize_landmarks, to_42d_vector  # 你可把上面的函式放這個檔

def collect(label_id, output_csv="rehab_dataset.csv"):
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands, open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        print(f"開始收集 label = {label_id}，按 q 結束。")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                pts = normalize_landmarks(lm)       # (21,2)
                # 如果你要資料增強，在這裡可多寫幾筆 augmented 版本
                vec = to_42d_vector(pts)            # 42 維
                vec.append(label_id)                # 最後一欄是 label
                writer.writerow(vec)

            cv2.imshow("Collect Rehab Gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    label_id = int(input("請輸入這次要收集的 label 編號："))
    collect(label_id)
```
:::warning
使用方式：
1. 先跑 python collect_rehab_dataset.py
2. 輸入 0 → 錄一段 idle；按 q 結束。
3. 再跑一次輸入 1 → 錄 fist；依此類推。
->多錄幾輪，讓每一類至少有幾百筆以上。

Fasko 的 train_hand_dataset.py 做的事情是：讀 CSV → Normalizer → CNN (Dense) → 訓練。
:::

## 2. 模型訓練(Fasko CNN)
```
Input: 42 維骨架向量（21點 × (x,y)）
→ Dense(64, relu)
→ Dense(32, relu)
→ Dropout(0.5)
→ Dense(6, softmax)
```
檔名：train_rehab_model.py
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adadelta
import joblib
import json

# 1. 讀資料
df = pd.read_csv("rehab_dataset.csv", header=None)
X = df.iloc[:, :-1].values      # 42 維特徵
y = df.iloc[:, -1].values.astype(int)

num_classes = len(np.unique(y))
print("樣本數:", len(y), "類別數:", num_classes)

# 2. L2 Normalization（與 Fasko 一樣）
normalizer = Normalizer().fit(X)
X_norm = normalizer.transform(X)

# 3. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.1, random_state=0, stratify=y
)

# 4. 建立模型（Fasko 架構改 output 維度）
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=42))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adadelta(),
    metrics=['accuracy']
)

# 5. 訓練
history = model.fit(
    X_train, y_train,
    epochs=85,
    batch_size=32,
    verbose=2,
    validation_data=(X_test, y_test),
    shuffle=True
)

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print("test loss:", test_loss, "test acc:", test_acc)

# 6. 保存模型與 normalizer
model.save("rehab_static_fasko_like.h5")
joblib.dump(normalizer, "rehab_normalizer.joblib")

print("模型與 normalizer 已儲存。")
```
:::warning
1. 測試集 accuracy 至少要 >90%（靜態手勢通常可以到 95%+）。
2. 若某一類特別差，可能要多錄那一類資料或調整手勢定義。
:::

### 3. 微調（Fine-tune Fasko 原始模型）
#### A：復用前兩層 Dense 的權重
1. 從 Fasko 原始 .h5 把前兩層權重搬過來，
2. 最後一層重建成 num_classes，對你的復健資料微調。
檔名：finetune_from_fasko.py
```
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adadelta
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

# 1. 讀你的 rehab 資料
df = pd.read_csv("rehab_dataset.csv", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.astype(int)
num_classes = len(np.unique(y))

normalizer = Normalizer().fit(X)
X_norm = normalizer.transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.1, random_state=0, stratify=y
)

# 2. 讀 Fasko 原本的 model
fasko_model = load_model("normalized_epochs85_42_data_points_extended_9_outputs10_06_2019_09_43_22.h5")

# 3. 取前兩層權重
w1, b1 = fasko_model.layers[0].get_weights()
w2, b2 = fasko_model.layers[1].get_weights()

# 4. 建立新模型（輸出數改成你的類別數）
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=42))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 5. 載入權重
model.layers[0].set_weights([w1, b1])
model.layers[1].set_weights([w2, b2])

# 6. 先凍結前兩層，只訓練最後一層
for layer in model.layers[:2]:
    layer.trainable = False

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adadelta(learning_rate=0.5),
    metrics=['accuracy']
)

print("【Phase 1】只訓練最後一層 softmax")
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2
)

# 7. 再解凍全部一起微調
for layer in model.layers:
    layer.trainable = True

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adadelta(learning_rate=0.3),
    metrics=['accuracy']
)

print("【Phase 2】整個模型 joint fine-tune")
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2
)

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print("Finetune test loss:", test_loss, "acc:", test_acc)

model.save("rehab_static_finetuned_from_fasko.h5")
joblib.dump(normalizer, "rehab_normalizer.joblib")
print("微調後模型已儲存。")
```
#### B：整個模型繼續訓練

## 3. 訓練模型
### (1) 訓練流程
* 使用 SGD 優化器，學習率設為 0.001，每 30 個 epoch 衰減為原本的 0.9。
* 損失函數：CrossEntropyLoss（適合分類任務）。
* 訓練迴圈：使用 5-fold 交叉驗證，來驗證模型的準確性。
```
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 訓練過程
def train_model(model, train_loader, val_loader, epochs=500):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        # 在每個 epoch 結束後進行驗證
        if epoch % 30 == 0:
            validate_model(model, val_loader)
```
## 4. 規則定義
## 4-1 動態手勢規則
{
  "0": "idle",           // 手放鬆（中立姿勢）
  "1": "hook",           // 勾拳
  "2": "angry_fist",           // 生氣的握拳
  "3": "thumb_flextion",  // 拇指彎曲
  "4": "straight_fist",// 直拳
  "5": "the_duck",   // 鴨子
  "6": "fist",   // 拇指在內的握拳
  "7": "spend_hand"   // 手指伸展
}