# 快速入門：手部復健偵測程式

**功能分支**: `001-hand-rehab-detection`
**建立日期**: 2025-12-15

## 環境需求

- Python 3.10 或以上
- 攝影機（內建或外接）
- 作業系統：Windows / macOS / Linux

## 安裝步驟

### 1. 建立虛擬環境

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 3. 確認模型檔案

確保模型檔案已放置於正確位置：

```
models/
└── rehab_action_classifier_64_5.h5
```

## 執行程式

```bash
python src/main.py
```

## 使用說明

### 基本操作

1. 啟動程式後，攝影機畫面會出現在視窗中
2. 將手放入畫面中，系統會自動偵測並顯示手部骨架
3. 畫面左上角顯示目前辨識的手勢名稱
4. 畫面右上角顯示伸展總次數
5. 按下 `q` 鍵退出程式

### 手勢說明

| 手勢 | 說明 |
|------|------|
| idle | 手放鬆狀態 |
| hook | 勾拳 |
| angry_fist | 生氣握拳 |
| thumb_flextion | 拇指彎曲 |
| straight_fist | 直拳 |
| the_duck | 鴨子手勢 |
| fist | 拇指在內握拳 |
| spend_hand | 手指完全伸展 |

### 伸展動作判定

有效伸展動作需滿足以下條件：

1. **起始**：維持任一非張開手勢（hook、angry_fist、thumb_flextion、straight_fist、the_duck、fist）≥2 秒
2. **張開**：維持手指伸展（spend_hand）≥2 秒
3. **結束**：維持與起始**相同**的非張開手勢 ≥2 秒

**範例**：
- ✅ fist（2秒）→ spend_hand（2秒）→ fist（2秒）= 1 次 fist 類型伸展
- ❌ fist（2秒）→ spend_hand（2秒）→ hook（2秒）= 不計入（起始與結束手勢不同）

### 畫面資訊說明

```
┌─────────────────────────────────────────┐
│ 目前手勢: fist (拇指在內握拳)      總次數: 5 │
│                                         │
│                                         │
│            [手部骨架顯示]                │
│                                         │
│                                         │
│ 狀態: 張開中 (1.5秒/2.0秒)              │
│─────────────────────────────────────────│
│ hook: 1 | angry_fist: 2 | fist: 2       │
└─────────────────────────────────────────┘
```

## 快捷鍵

| 按鍵 | 功能 |
|------|------|
| q | 退出程式 |
| r | 重置統計計數 |

## 常見問題

### 攝影機無法開啟

1. 確認攝影機已連接且未被其他程式佔用
2. 嘗試指定攝影機編號：修改 `src/main.py` 中的 `cv2.VideoCapture(0)` 為其他編號（如 1, 2）

### 手部偵測不穩定

1. 確保光線充足
2. 手部完整出現在畫面中
3. 避免背景過於複雜

### 模型載入失敗

1. 確認模型檔案路徑正確
2. 確認 TensorFlow 版本相容

## 驗證測試

執行以下命令確認安裝正確：

```bash
# 執行單元測試
pytest tests/ -v

# 測試攝影機連接
python -c "import cv2; cap = cv2.VideoCapture(0); print('攝影機正常' if cap.isOpened() else '攝影機異常'); cap.release()"

# 測試模型載入
python -c "from tensorflow.keras.models import load_model; m = load_model('models/rehab_action_classifier_64_5.h5'); print('模型載入成功')"
```
