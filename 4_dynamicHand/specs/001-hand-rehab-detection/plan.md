# 實作計畫：手部復健偵測程式

**分支**: `001-hand-rehab-detection` | **日期**: 2025-12-15 | **規格**: [spec.md](./spec.md)
**輸入**: 功能規格書 `/specs/001-hand-rehab-detection/spec.md`

## 摘要

建立一個即時手部復健偵測程式，使用 OpenCV 與 Mediapipe 擷取手部骨架，透過已訓練的 CNN 模型進行手勢分類，並實作伸展動作狀態機以統計復健次數與類型。程式提供簡易的 OpenCV 視窗介面，在攝影機畫面上疊加顯示手勢名稱與統計資訊。

## 技術背景

**語言/版本**: Python 3.10+
**主要依賴**:
- OpenCV (`opencv-python`) - 攝影機擷取與畫面渲染
- Mediapipe (`mediapipe`) - 手部骨架偵測
- TensorFlow/Keras (`tensorflow`) - 載入已訓練模型
- NumPy (`numpy`) - 數值運算

**儲存**: 無持久化儲存（記憶體內統計）
**測試**: pytest
**目標平台**: Windows/macOS/Linux 桌面環境（需攝影機）
**專案類型**: 單一專案
**效能目標**: 手勢辨識延遲 <100ms，支援即時串流
**約束**: 模型輸入為 63 維向量（21 點 x 3 座標），輸出 8 類手勢
**規模**: 單一使用者桌面應用程式

## 憲章檢查

*門檻：必須在設計前通過。*

| 原則 | 狀態 | 說明 |
|------|------|------|
| I. 品質至上 | ✅ 通過 | 明確的錯誤處理、一致的程式碼風格 |
| II. 可測試性優先 | ✅ 通過 | 核心邏輯（狀態機、手勢分類）可獨立測試 |
| III. MVP | ✅ 通過 | 專注核心功能：手勢辨識、伸展統計、簡易 UI |
| IV. 簡潔原則 | ✅ 通過 | 無過度抽象，直接使用 OpenCV 視窗 |

## 專案結構

### 文件（本功能）

```text
specs/001-hand-rehab-detection/
├── spec.md              # 功能規格書
├── plan.md              # 本文件
├── data-model.md        # 資料模型說明
├── quickstart.md        # 快速入門指南
└── tasks.md             # 任務清單（由 /speckit.tasks 產生）
```

### 原始碼（專案根目錄）

```text
src/
├── main.py              # 程式進入點
├── gesture_classifier.py    # 手勢分類器（載入模型、預測）
├── hand_detector.py     # 手部偵測器（Mediapipe 封裝）
├── stretch_tracker.py   # 伸展追蹤器（狀態機）
├── ui_renderer.py       # UI 渲染器（OpenCV 疊加顯示）
└── config.py            # 設定常數

models/
└── rehab_action_classifier_64_5.h5  # 已訓練模型

tests/
├── test_gesture_classifier.py
├── test_stretch_tracker.py
└── test_hand_detector.py

requirements.txt         # Python 依賴清單
README.md               # 專案說明
```

**結構決策**: 採用單一專案結構，將功能分離為獨立模組（hand_detector、gesture_classifier、stretch_tracker、ui_renderer），每個模組職責單一且可獨立測試。

## 核心模組設計

### 1. 手部偵測器 (hand_detector.py)

**職責**: 封裝 Mediapipe 手部偵測，輸出正規化骨架向量

```python
class HandDetector:
    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.7)
    def detect(self, frame: np.ndarray) -> Optional[HandResult]
    def close(self)

@dataclass
class HandResult:
    landmarks: np.ndarray  # shape: (63,) - 21點 x (x, y, z)
    confidence: float
    raw_landmarks: List[Landmark]  # 原始 Mediapipe 結果，用於繪製
```

### 2. 手勢分類器 (gesture_classifier.py)

**職責**: 載入模型、執行推論、輸出手勢類別

```python
class GestureClassifier:
    def __init__(self, model_path: str)
    def predict(self, skeleton: np.ndarray) -> GesturePrediction

@dataclass
class GesturePrediction:
    class_id: int          # 0-7
    class_name: str        # 手勢名稱
    confidence: float      # 信心度
```

**手勢類別對照**:
```python
GESTURE_NAMES = {
    0: "idle",
    1: "hook",
    2: "angry_fist",
    3: "thumb_flextion",
    4: "straight_fist",
    5: "the_duck",
    6: "fist",
    7: "spend_hand"
}
```

### 3. 伸展追蹤器 (stretch_tracker.py)

**職責**: 實作狀態機，追蹤伸展動作，統計次數與類型

```python
class StretchTracker:
    def __init__(self, hold_duration: float = 2.0)
    def update(self, gesture: GesturePrediction) -> Optional[StretchRecord]
    def get_stats(self) -> StretchStats
    def reset(self)

@dataclass
class StretchRecord:
    stretch_type: str      # 伸展類型名稱
    start_time: float
    end_time: float

@dataclass
class StretchStats:
    total_count: int
    counts_by_type: Dict[str, int]  # 各類型計數
```

**狀態機設計**:

```
狀態: IDLE -> NON_OPEN -> OPEN -> NON_OPEN_FINAL -> (完成或失敗)

轉換規則:
- IDLE: 等待非張開狀態
- NON_OPEN: 偵測到非張開狀態，記錄類型，開始計時
  - 停留 ≥2秒 -> 進入 OPEN_WAITING
  - 中斷 -> 回到 IDLE
- OPEN_WAITING: 等待張開狀態
  - 偵測到 spend_hand -> 進入 OPEN，開始計時
- OPEN: 張開狀態計時中
  - 停留 ≥2秒 -> 進入 NON_OPEN_FINAL_WAITING
  - 中斷 -> 回到 IDLE
- NON_OPEN_FINAL_WAITING: 等待結束非張開狀態
  - 偵測到與起始相同的非張開狀態 -> 進入 NON_OPEN_FINAL，開始計時
  - 偵測到不同非張開狀態 -> 回到 IDLE（混合類型不計）
- NON_OPEN_FINAL: 結束非張開狀態計時中
  - 停留 ≥2秒 -> 完成一次伸展，記錄並回到 IDLE
  - 中斷 -> 回到 IDLE
```

### 4. UI 渲染器 (ui_renderer.py)

**職責**: 在 OpenCV 視窗上疊加顯示資訊

```python
class UIRenderer:
    def render(self, frame: np.ndarray,
               hand_result: Optional[HandResult],
               gesture: Optional[GesturePrediction],
               stats: StretchStats,
               tracker_state: str) -> np.ndarray
```

**顯示內容**:
- 左上角：目前手勢名稱（中英文）
- 右上角：伸展總次數
- 底部：各類型伸展統計
- 畫面中央：手部骨架點連線（如偵測到手部）
- 狀態提示：目前狀態機狀態與計時進度

### 5. 主程式 (main.py)

**職責**: 整合所有模組，執行主迴圈

```python
def main():
    # 初始化
    detector = HandDetector()
    classifier = GestureClassifier("models/rehab_action_classifier_64_5.h5")
    tracker = StretchTracker(hold_duration=2.0)
    renderer = UIRenderer()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 偵測手部
        hand_result = detector.detect(frame)

        # 分類手勢
        gesture = None
        if hand_result:
            gesture = classifier.predict(hand_result.landmarks)

        # 更新狀態機
        if gesture:
            tracker.update(gesture)

        # 渲染 UI
        display_frame = renderer.render(
            frame, hand_result, gesture,
            tracker.get_stats(), tracker.state
        )

        cv2.imshow("Hand Rehab Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
```

## 平滑策略

為降低手勢辨識的抖動，實作移動平均平滑：

```python
class GestureSmoother:
    def __init__(self, window_size: int = 5):
        self.history = deque(maxlen=window_size)

    def smooth(self, prediction: GesturePrediction) -> GesturePrediction:
        self.history.append(prediction.class_id)
        # 使用多數決
        most_common = Counter(self.history).most_common(1)[0][0]
        return GesturePrediction(
            class_id=most_common,
            class_name=GESTURE_NAMES[most_common],
            confidence=prediction.confidence
        )
```

## 複雜度追蹤

無違反憲章原則的情況。

## 依賴清單 (requirements.txt)

```text
opencv-python>=4.8.0
mediapipe==0.10.21
tensorflow>=2.13.0
numpy>=1.24.0
pytest>=7.0.0
```
