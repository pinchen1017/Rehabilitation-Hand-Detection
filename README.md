期末專題：印度人太有料了
===
## 一、 使用規則
推送時從自己的 branch 進行推送，再到 github merge
- branch_J：許鯨魚
- branch_R：Risto
- branch_P：郭品陳
- branch_T：礦採泥
- branch_Z：013

## 二、 復健手勢系統

這是一個基於 **MediaPipe**、**深度神經網路 (DNN)** 與 **凸包幾何分析 (Convex Hull)** 的手部復健輔助系統。

本專案 (**Branch P**) 專注於 **「教練模式 (Coach Mode)」** 的實現：
1.  **DNN 模型**：負責權威性的手勢辨識與復健次數計數，確保動態手勢的流暢性。
2.  **幾何分析 (CV)**：作為輔助教練，即時分析手部物理特徵 (如指縫、實心度)，並在畫面上提示使用者修正動作。

## 三、 主要功能

- **即時手部偵測**：使用 MediaPipe 高效偵測手部關鍵點。
- **動態手勢計數**：內建狀態機 (State Machine)，可計算復健動作的完成次數 (如：張開 -> 握拳 -> 張開)。
- **雙重驗證教練模式**：結合 DNN 的辨識結果與 CV 的幾何特徵，提供即時動作指導（如：「請握緊」、「請張開手指」）。
- **完整訓練流程**：包含資料蒐集、前處理、模型訓練到即時推論的完整 Pipeline。

## 四、 環境安裝

請確保您的環境已安裝 Python (建議 3.8+)。

1.  **複製專案**
    ```bash
    git clone -b branch_P [https://github.com/pinchen1017/Rehabilitation-Hand-Detection.git](https://github.com/pinchen1017/Rehabilitation-Hand-Detection.git)
    cd Rehabilitation-Hand-Detection
    ```

2.  **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```
    > 若無 `requirements.txt`，主要依賴套件為：
    > `opencv-python`, `mediapipe`, `numpy`, `tensorflow`, `scikit-learn`, `pandas`, `ultralytics`

## 五、 使用方法 (Workflow)

本分支的所有核心程式碼皆位於 `4_dynanicHand` 資料夾內。請在專案根目錄下執行以下指令。

### 1. 資料蒐集 (Data Collection)
開啟攝影機，蒐集手部關鍵點資料以建立客製化資料集。

* **操作說明**：
    * 數字鍵 `0-7`：切換目標手勢類別。
    * `空白鍵`：開始 / 暫停錄製。
    * `S`：儲存資料。
    * `Q`：退出程式。

```bash
python 4_dynanicHand/src/data_collector.py
```

### 2. 資料前處理 (Data Preprocessing)
將蒐集到的原始座標數據進行正規化、增強與分割 (Train/Val/Test)。
```bash
python 4_dynanicHand/src/train_model.py
```

### 3. 模型訓練 (Model Training)
使用處理後的資料訓練 DNN 模型，並轉換為 TFLite 格式。
```bash
cd src
python model_trainer.py              # 訓練基礎與增強模型
python model_trainer.py -m basic     # 僅訓練基礎模型
python model_trainer.py -m enhanced  # 僅訓練增強模型
```

### 4. 啟動主程式 (Real-time Detection)
啟動即時偵測系統 (教練模式)。這是最終整合的版本 (v29)，包含流暢計數與 CV 提示功能。
```bash
python 4_dynanicHand/src/main_final_v29.py
```

## 六、 專案結構
```
4_dynanicHand/
├── data/                  # 存放蒐集與處理後的資料 (.csv)
├── model/                 # 存放訓練好的模型 (.tflite, .h5)
├── src/                   # 核心程式碼
│   ├── config.py          # 設定檔 (路徑、參數)
│   ├── data_collector.py  # 資料蒐集
│   ├── data_preprocessor.py # 資料前處理
│   ├── train_model.py     # 模型訓練
│   ├── main_final_v29.py  # [主程式] 最終整合版 (教練模式)
│   ├── hand_detector.py   # MediaPipe 手部偵測封裝
│   ├── gesture_classifier.py # DNN 模型推論
│   ├── stretch_tracker.py # 復健動作計數邏輯
│   ├── ui_renderer.py     # 畫面繪製
│   ├── data_collector.py    # 資料蒐集腳本
│   ├── data_preprocessor.py # 前處理工具
│   ├── model_trainer.py     # 訓練流程
│   └── training_utils.py    # 訓練工具
└── data/
│    ├── raw/                 # 原始蒐集的 CSV 檔案
│    └── processed/           # 前處理後的資料集
└── models/
│   ├── basic_model.h5       # 基礎訓練模型
│   └── enhanced_model.h5    # 增強訓練模型
└── training_logs/           # 訓練歷史和圖表
    └── plots/
```

## 七、 手部類別定義
ID,類別名稱,描述,提示範例 (教練模式)
0,Idle,手部放鬆 / 無效動作,-
1,Hook,勾手 (爪狀),"""Bend Fingers!"""
2,Angry Fist,握拳 (類似生氣的手勢),"""Grip Tighter!"""
3,Thumb Flexion,拇指彎曲 / 四指併攏,"""Keep Long!"""
4,Straight Hand,手掌伸直併攏,"""Flatten Hand!"""
5,The Duck,鴨嘴手勢,-
6,Fist,標準握拳 (拇指在內),"""Close Gaps!"""
7,Spread Hand,五指張開 (伸展),"""Open Fingers!"""

## 八、 備註與除錯
* 關於 YOLO 去背：系統整合了 YOLOv8-Seg 進行手部去背，以提高 Convex Hull 計算的抗干擾能力。程式初始化時需要載入 YOLO 模型，請稍候片刻。
* 教練模式視窗：程式運行時會開啟兩個視窗。
* Rehab System：主畫面，顯示計數與骨架。
* Debug: ROI Mask：教練視窗，顯示幾何分析結果與提示文字。
* 重置計數：在執行過程中，隨時按下鍵盤上的 R 鍵可重置計數器。
* 結束程式：按下 Q 鍵可安全退出程式。