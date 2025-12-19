# 手部復健偵測程式

即時偵測手部伸展動作，自動統計復健次數與類型。

## 功能特點

- 即時手勢辨識（8 種手勢）
- 伸展動作自動計數
- 6 種伸展類型統計
- 視覺化 UI 顯示（支援中文）
- 資料蒐集與模型訓練

## 環境需求

- Python 3.10
- 攝影機
- Windows 系統（中文字體支援）

## 安裝

```bash
# 建立虛擬環境
py -3.10 -m venv venv

# 啟動虛擬環境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

## 使用方式

### 即時辨識模式

```bash
cd src
python main.py
```

#### 快捷鍵

| 按鍵 | 功能 |
|------|------|
| q | 退出程式 |
| r | 重置統計 |

### 資料蒐集模式

蒐集手勢特徵點資料以訓練模型。

```bash
cd src
python data_collector.py
```

#### 快捷鍵

| 按鍵 | 功能 |
|------|------|
| 0-7 | 選擇目標手勢類別 |
| 空白鍵 | 切換蒐集狀態(開/關)，以蒐集當前手勢資料 |
| q | 退出程式 |

### 資料前處理

處理蒐集的原始資料，進行正規化與資料分割。

```bash
cd src
python data_preprocessor.py
```

### 模型訓練

訓練手勢分類模型。

```bash
cd src
python model_trainer.py              # 訓練基礎與增強模型
python model_trainer.py -m basic     # 僅訓練基礎模型
python model_trainer.py -m enhanced  # 僅訓練增強模型
```

## 手勢類別

| 編號 | 名稱 | 說明 |
|------|------|------|
| 0 | idle | 手放鬆 |
| 1 | hook | 勾拳 |
| 2 | angry_fist | 生氣握拳 |
| 3 | thumb_flextion | 拇指彎曲 |
| 4 | straight_fist | 直拳 |
| 5 | the_duck | 鴨子 |
| 6 | fist | 拇指在內握拳 |
| 7 | spend_hand | 手指伸展 |

## 伸展動作判定

有效伸展動作需滿足：

1. 維持非張開手勢 ≥2 秒
2. 維持手指伸展 ≥2 秒
3. 維持**相同**非張開手勢 ≥2 秒

## 專案結構

```
src/
├── main.py              # 主程式（即時辨識）
├── config.py            # 設定常數
├── hand_detector.py     # 手部偵測器
├── gesture_classifier.py # 手勢分類器
├── stretch_tracker.py   # 伸展追蹤器
├── ui_renderer.py       # UI 渲染器
├── data_collector.py    # 資料蒐集腳本
├── data_preprocessor.py # 前處理工具
├── model_trainer.py     # 訓練流程
└── training_utils.py    # 訓練工具

data/
├── raw/                 # 原始蒐集的 CSV 檔案
└── processed/           # 前處理後的資料集

models/
├── basic_model.h5       # 基礎訓練模型
└── enhanced_model.h5    # 增強訓練模型

training_logs/           # 訓練歷史和圖表
└── plots/
```

## 授權

MIT License
