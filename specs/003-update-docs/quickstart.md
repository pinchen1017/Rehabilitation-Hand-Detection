# Quick Start: 說明文件更新

**Branch**: `003-update-docs` | **Date**: 2025-12-19

## 概述

本功能為純文件更新，不涉及程式碼修改。以下是需要更新的文件清單與變更內容。

## 需要更新的文件

### 1. specs/002-data-collection-and-training/spec.md

**變更位置**: FR-007

**原始內容**:
```
- **FR-007**: 系統必須（MUST）使用最小-最大正規化將特徵點座標正規化至 [0, 1] 範圍
```

**更新為**:
```
- **FR-007**: 系統必須（MUST）使用以下正規化方法：
  - x, y 座標：相對於手腕位置的偏移，再進行 min-max 正規化至 [0, 1] 範圍
  - z 座標：使用 wrist-relative 正規化（相對於手腕 z 座標的差值）

  > 注意：原本的 min-max 正規化在即時推論時無法得知全域極值，
  > wrist-relative 方法確保訓練與推論使用相同的前處理邏輯。
```

### 2. README.md

**需要新增的章節**:

#### 使用方式 - 資料蒐集模式
```bash
cd src
python data_collector.py
```

**快捷鍵**:
| 按鍵 | 功能 |
|------|------|
| 0-7 | 選擇目標手勢類別 |
| 空白鍵 | 按住以蒐集當前手勢資料 |
| q | 退出程式 |

#### 使用方式 - 資料前處理
```bash
cd src
python data_preprocessor.py
```

#### 使用方式 - 模型訓練
```bash
cd src
python model_trainer.py              # 訓練基礎與增強模型
python model_trainer.py -m basic     # 僅訓練基礎模型
python model_trainer.py -m enhanced  # 僅訓練增強模型
```

#### 專案結構更新
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

#### UI 相關注意事項（可選）
- 程式使用 msyh.ttc（微軟雅黑）字體支援中文顯示
- 此字體為 Windows 系統預設字體
- 非 Windows 使用者可能需要調整 `config.py` 中的字體設定

## 驗證清單

- [ ] specs/002-data-collection-and-training/spec.md FR-007 已更新
- [ ] README.md 包含資料蒐集模式說明
- [ ] README.md 包含資料前處理說明
- [ ] README.md 包含模型訓練說明
- [ ] README.md 專案結構已更新
- [ ] 文件中未提及 rehab_action_classifier_64_5.h5
