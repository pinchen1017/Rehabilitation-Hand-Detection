# 資料模型：手部復健偵測程式

**功能分支**: `001-hand-rehab-detection`
**建立日期**: 2025-12-15

## 核心資料結構

### 手部偵測結果 (HandResult)

代表單一幀的手部偵測結果。

| 欄位 | 型別 | 說明 |
|------|------|------|
| landmarks | np.ndarray (63,) | 正規化骨架向量：21 點 x (x, y, z) |
| confidence | float | 偵測信心度 (0.0-1.0) |
| raw_landmarks | List[Landmark] | 原始 Mediapipe 關鍵點，用於視覺化繪製 |

**骨架向量格式**:
```
[x0, y0, z0, x1, y1, z1, x2, y2, z2, ..., x20, y20, z20]
```

其中點編號對應 Mediapipe Hand Landmarks:
- 0: WRIST（手腕）
- 1-4: THUMB（拇指）
- 5-8: INDEX（食指）
- 9-12: MIDDLE（中指）
- 13-16: RING（無名指）
- 17-20: PINKY（小指）

### 手勢預測結果 (GesturePrediction)

代表單一幀的手勢分類結果。

| 欄位 | 型別 | 說明 |
|------|------|------|
| class_id | int | 手勢類別編號 (0-7) |
| class_name | str | 手勢類別名稱 |
| confidence | float | 分類信心度 (0.0-1.0) |

**手勢類別對照表**:

| class_id | class_name | 中文說明 | 狀態分類 |
|----------|------------|---------|---------|
| 0 | idle | 手放鬆 | 其他 |
| 1 | hook | 勾拳 | 非張開 |
| 2 | angry_fist | 生氣握拳 | 非張開 |
| 3 | thumb_flextion | 拇指彎曲 | 非張開 |
| 4 | straight_fist | 直拳 | 非張開 |
| 5 | the_duck | 鴨子 | 非張開 |
| 6 | fist | 拇指在內握拳 | 非張開 |
| 7 | spend_hand | 手指伸展 | 張開 |

### 伸展記錄 (StretchRecord)

代表一次完成的伸展動作。

| 欄位 | 型別 | 說明 |
|------|------|------|
| stretch_type | str | 伸展類型名稱（如 "hook", "fist"） |
| start_time | float | 伸展開始時間戳（Unix timestamp） |
| end_time | float | 伸展結束時間戳（Unix timestamp） |

**有效伸展類型** (共 6 種):
- hook
- angry_fist
- thumb_flextion
- straight_fist
- the_duck
- fist

### 伸展統計 (StretchStats)

代表目前的伸展統計摘要。

| 欄位 | 型別 | 說明 |
|------|------|------|
| total_count | int | 總伸展次數 |
| counts_by_type | Dict[str, int] | 各類型伸展次數 |

**counts_by_type 結構範例**:
```python
{
    "hook": 3,
    "angry_fist": 2,
    "thumb_flextion": 0,
    "straight_fist": 1,
    "the_duck": 0,
    "fist": 5
}
```

## 狀態機模型

### 追蹤器狀態 (TrackerState)

| 狀態 | 說明 |
|------|------|
| IDLE | 閒置，等待非張開狀態 |
| NON_OPEN_HOLDING | 非張開狀態計時中（需 ≥2 秒） |
| OPEN_WAITING | 等待張開狀態 |
| OPEN_HOLDING | 張開狀態計時中（需 ≥2 秒） |
| NON_OPEN_FINAL_WAITING | 等待結束非張開狀態 |
| NON_OPEN_FINAL_HOLDING | 結束非張開狀態計時中（需 ≥2 秒） |

### 狀態轉換圖

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    v                                         │
    ┌──────┐   非張開    ┌───────────────────┐   ≥2秒    ┌─────────────┐
    │ IDLE │ ────────> │ NON_OPEN_HOLDING  │ ────────> │ OPEN_WAITING │
    └──────┘           └───────────────────┘           └─────────────┘
        ^                      │                              │
        │                      │ 中斷                         │ spend_hand
        │                      v                              v
        │               ┌──────────┐                   ┌─────────────┐
        │<──────────────│   IDLE   │                   │ OPEN_HOLDING │
        │               └──────────┘                   └─────────────┘
        │                                                     │
        │                                                     │ ≥2秒
        │                                                     v
        │                                        ┌────────────────────────┐
        │                                        │ NON_OPEN_FINAL_WAITING │
        │                                        └────────────────────────┘
        │                                                     │
        │            不同類型                                  │ 同類型非張開
        │<────────────────────────────────────────────────────┤
        │                                                     v
        │                                        ┌────────────────────────┐
        │                                        │ NON_OPEN_FINAL_HOLDING │
        │                                        └────────────────────────┘
        │                                                     │
        │                                                     │ ≥2秒
        │                                                     v
        │                                              ┌────────────┐
        └──────────────────────────────────────────────│  完成伸展   │
                                                       └────────────┘
```

### 狀態內部資料

| 欄位 | 型別 | 說明 |
|------|------|------|
| current_state | TrackerState | 目前狀態 |
| start_gesture_type | Optional[str] | 起始非張開手勢類型 |
| state_enter_time | Optional[float] | 進入目前狀態的時間 |
| hold_duration | float | 要求的停留時間（預設 2.0 秒） |

## 設定常數 (config.py)

```python
# 手勢類別
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

GESTURE_NAMES_ZH = {
    0: "手放鬆",
    1: "勾拳",
    2: "生氣握拳",
    3: "拇指彎曲",
    4: "直拳",
    5: "鴨子",
    6: "拇指在內握拳",
    7: "手指伸展"
}

# 狀態分類
NON_OPEN_GESTURES = {1, 2, 3, 4, 5, 6}  # 非張開狀態
OPEN_GESTURE = 7  # 張開狀態
IDLE_GESTURE = 0  # 其他狀態

# 有效伸展類型
VALID_STRETCH_TYPES = ["hook", "angry_fist", "thumb_flextion",
                       "straight_fist", "the_duck", "fist"]

# 時間設定
DEFAULT_HOLD_DURATION = 2.0  # 秒

# 偵測設定
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.7
DEFAULT_MAX_HANDS = 1

# 平滑設定
SMOOTHER_WINDOW_SIZE = 5
```

## 資料流程圖

```
攝影機
   │
   v
┌─────────────────┐
│  HandDetector   │ ──> HandResult (landmarks, confidence)
└─────────────────┘
          │
          v
┌─────────────────┐
│ GestureClassifier│ ──> GesturePrediction (class_id, name, confidence)
└─────────────────┘
          │
          v
┌─────────────────┐
│ GestureSmoother │ ──> GesturePrediction (平滑後)
└─────────────────┘
          │
          v
┌─────────────────┐
│ StretchTracker  │ ──> StretchStats (total_count, counts_by_type)
└─────────────────┘         │
          │                 │
          v                 v
┌─────────────────┐   ┌─────────────────┐
│   UIRenderer    │   │  StretchRecord  │ (若完成伸展)
└─────────────────┘   └─────────────────┘
          │
          v
      OpenCV 視窗
```
