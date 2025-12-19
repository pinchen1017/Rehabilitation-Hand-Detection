# 任務清單：資料蒐集與模型訓練流程

**輸入**: 設計文件 `/specs/002-data-collection-and-training/`
**前置需求**: plan.md（必要）、spec.md（必要）

**測試**: 本規格未明確要求自動化測試，故不包含測試任務。各階段提供手動驗證檢查點。

**組織方式**: 任務依使用者故事分組，以支援獨立實作與測試。

## 格式說明：`[ID] [P?] [Story] 描述`

- **[P]**: 可平行執行（不同檔案、無依賴）
- **[Story]**: 任務所屬的使用者故事（例如 US1、US2、US3）
- 描述中包含確切的檔案路徑

## 路徑慣例

- **單一專案**: `src/`、`data/`、`models/` 位於專案根目錄
- 本專案延續既有結構，最小化修改

---

## Phase 1: 環境設置（共用基礎設施）

**目的**: 專案初始化與目錄結構建立

- [x] T001 建立資料目錄結構 data/raw/ 和 data/processed/
- [x] T002 建立訓練紀錄目錄 training_logs/ 和 training_logs/plots/
- [x] T003 更新 requirements.txt 新增依賴：pandas、scikit-learn、matplotlib

---

## Phase 2: 基礎設定（阻塞性前置作業）

**目的**: 所有使用者故事開始前必須完成的核心設定

**⚠️ 重要**: 此階段完成前，無法開始任何使用者故事的實作

- [x] T004 在 src/config.py 新增資料蒐集設定常數（DATA_RAW_DIR、DATA_PROCESSED_DIR、RAW_DATA_FILE、TARGET_SAMPLES_PER_CLASS、NUM_CLASSES）
- [x] T005 在 src/config.py 新增資料前處理設定常數（TRAIN_RATIO、VAL_RATIO、TEST_RATIO、RANDOM_SEED、AUGMENT_* 設定）
- [x] T006 在 src/config.py 新增模型訓練設定常數（TRAINING_LOGS_DIR、BASIC_MODEL_CONFIG、ENHANCED_MODEL_CONFIG、模型路徑）

**檢查點**: 基礎設定完成 - 可開始使用者故事實作

---

## Phase 3: 使用者故事 1 - 資料蒐集 (優先級: P1) 🎯 MVP

**目標**: 提供簡易 OpenCV 前端，蒐集 8 種手勢類別的手部特徵點資料

**獨立測試**: 啟動 `python src/data_collector.py`，按數字鍵切換類別，做出手勢並按下空白鍵切換至蒐集狀態，確認資料已儲存至 `data/raw/gesture_data.csv`

### 實作任務

- [x] T007 [US1] 建立 src/data_collector.py 基本結構與 DataCollector 類別框架
- [x] T008 [US1] 實作 DataCollector.__init__() 初始化：重用 HandDetector、設定輸出路徑、初始化計數器
- [x] T009 [US1] 實作 DataCollector._collect_sample() 方法：將 landmarks 轉換為 dict 並加入 samples 列表
- [x] T010 [US1] 實作 DataCollector.save_data() 方法：使用 pandas 儲存 CSV 檔案
- [x] T011 [US1] 實作 DataCollector._render_ui() 方法：繪製類別選擇、統計數量、手部骨架、操作說明
- [x] T012 [US1] 實作 DataCollector.run() 主迴圈：攝影機擷取、手部偵測、按鍵處理（0-7 切換、空白鍵切換蒐集狀態、s 儲存、q 退出）
- [x] T013 [US1] 實作無手部偵測時的警告顯示與資料拒絕邏輯
- [x] T014 [US1] 實作蒐集結束時的摘要顯示（各類別數量統計）
- [x] T015 [US1] 新增 src/data_collector.py 的 __main__ 入口點

**檢查點**: 使用者故事 1 應可獨立運作並測試

---

## Phase 4: 使用者故事 2 - 資料前處理 (優先級: P1)

**目標**: 清理、正規化、分割與增強已蒐集的資料

**獨立測試**: 執行 `python src/data_preprocessor.py`，確認 `data/processed/` 目錄下產生 train.csv、val.csv、test.csv

### 實作任務

- [x] T016 [US2] 建立 src/data_preprocessor.py 基本結構與 DataPreprocessor 類別框架
- [x] T017 [US2] 實作 DataPreprocessor.load_raw_data() 方法：載入原始 CSV 資料
- [x] T018 [US2] 實作 DataPreprocessor.validate_samples() 方法：移除 NaN 和異常值，記錄移除數量
- [x] T019 [US2] 實作 DataPreprocessor.normalize() 方法：正規化 z 座標至 [0, 1] 範圍
- [x] T020 [US2] 實作 DataPreprocessor.augment() 方法：隨機旋轉、縮放、平移增強
- [x] T021 [US2] 實作 DataPreprocessor.split_data() 方法：使用 sklearn 進行分層分割（70/15/15）
- [x] T022 [US2] 實作 DataPreprocessor.process() 完整流程：載入→驗證→正規化→增強→分割→儲存
- [x] T023 [US2] 實作類別不平衡警告與樣本不足警告
- [x] T024 [US2] 新增 src/data_preprocessor.py 的 __main__ 入口點

**檢查點**: 使用者故事 2 應可獨立運作並測試

---

## Phase 5: 使用者故事 3 - 基礎模型訓練 (優先級: P2)

**目標**: 訓練基礎全連接神經網路模型，建立基準分類器

**獨立測試**: 執行 `python src/model_trainer.py --model basic`，確認 `models/basic_model.h5` 已儲存且測試準確率 ≥85%

### 實作任務

- [x] T025 [US3] 建立 src/model_trainer.py 基本結構與 ModelTrainer 類別框架
- [x] T026 [US3] 實作 ModelTrainer.load_data() 方法：載入訓練/驗證/測試 CSV 並分離特徵與標籤
- [x] T027 [US3] 實作 ModelTrainer.build_basic_model() 方法：建立 2 層隱藏層 + Dropout 的全連接網路
- [x] T028 [US3] 實作 ModelTrainer.train() 基礎版：使用 EarlyStopping callback、訓練並評估
- [x] T029 [US3] 實作 ModelTrainer.save_model() 方法：儲存模型為 .h5 格式
- [x] T030 [US3] 實作基礎模型訓練後的混淆矩陣產生
- [x] T031 [US3] 實作基礎模型訓練後的準確率顯示

**檢查點**: 使用者故事 3 應可獨立運作並測試

---

## Phase 6: 使用者故事 4 - 增強模型訓練 (優先級: P2)

**目標**: 訓練增強版模型，達到比基礎模型更高的準確率

**獨立測試**: 執行 `python src/model_trainer.py --model enhanced`，確認 `models/enhanced_model.h5` 已儲存且測試準確率 > 基礎模型

### 實作任務

- [x] T032 [US4] 實作 ModelTrainer.build_enhanced_model() 方法：建立 3 層隱藏層 + BatchNorm + L2 正規化的網路
- [x] T033 [US4] 實作 ModelTrainer.train() 增強版：新增 ReduceLROnPlateau callback
- [x] T034 [US4] 實作增強模型訓練後的混淆矩陣產生
- [x] T035 [US4] 實作模型比較功能：顯示基礎 vs 增強模型準確率差異
- [x] T036 [US4] 新增 src/model_trainer.py 的 __main__ 入口點（支援 --model 參數：basic/enhanced/all）

**檢查點**: 使用者故事 4 應可獨立運作並測試

---

## Phase 7: 使用者故事 5 - 訓練進度視覺化 (優先級: P3)

**目標**: 產生訓練過程的視覺化圖表

**獨立測試**: 訓練完成後，確認 `training_logs/plots/` 目錄下產生損失曲線和混淆矩陣圖表

### 實作任務

- [x] T037 [P] [US5] 建立 src/training_utils.py 基本結構
- [x] T038 [P] [US5] 實作 plot_training_history() 函數：繪製訓練/驗證損失與準確率曲線
- [x] T039 [P] [US5] 實作 plot_confusion_matrix() 函數：繪製混淆矩陣熱力圖
- [x] T040 [P] [US5] 實作 print_classification_report() 函數：印出各類別精確率/召回率/F1
- [x] T041 [US5] 實作 save_training_history() 函數：儲存訓練歷史為 JSON 檔案
- [x] T042 [US5] 整合視覺化至 ModelTrainer：訓練完成後自動產生圖表

**檢查點**: 所有使用者故事應可獨立運作

---

## Phase 8: 收尾與整合

**目的**: 跨故事的改進與整合驗證

- [x] T043 驗證訓練好的模型可在 src/main.py 中正常載入（修改 config.py 中的 DEFAULT_MODEL_PATH）
- [x] T044 確認所有模組的錯誤處理與例外訊息完整
- [x] T045 執行完整流程驗證：蒐集 → 前處理 → 基礎訓練 → 增強訓練 → 辨識

---

## 依賴關係與執行順序

### 階段依賴

- **環境設置 (Phase 1)**: 無依賴 - 可立即開始
- **基礎設定 (Phase 2)**: 依賴 Phase 1 完成 - **阻塞所有使用者故事**
- **使用者故事 (Phase 3-7)**: 皆依賴 Phase 2 完成
  - US1（資料蒐集）和 US2（前處理）可平行開發
  - US3（基礎訓練）和 US4（增強訓練）依賴 US2 完成（需要處理後的資料）
  - US5（視覺化）可與 US3/US4 平行開發
- **收尾 (Phase 8)**: 依賴所有使用者故事完成

### 使用者故事依賴

```
Phase 2（基礎設定）
    │
    ├──> US1（資料蒐集）──> US2（前處理）──┬──> US3（基礎訓練）
    │                                      │
    │                                      └──> US4（增強訓練）
    │
    └──> US5（視覺化）- 可獨立開發，整合時與 US3/US4 合併
```

### 各故事內部依賴

- 框架建立 → 核心方法 → 輔助功能 → 入口點
- 依序完成，無法平行

### 平行機會

- Phase 1 所有任務可平行執行
- Phase 2 任務 T004、T005、T006 皆在同一檔案，須依序執行
- US5 的 T037-T040 標記 [P]，可平行執行

---

## 平行範例：基礎設定

```bash
# Phase 1 可平行執行：
任務: "建立資料目錄結構 data/raw/ 和 data/processed/"
任務: "建立訓練紀錄目錄 training_logs/ 和 training_logs/plots/"
任務: "更新 requirements.txt"
```

## 平行範例：使用者故事 5

```bash
# US5 視覺化函數可平行開發：
任務: "實作 plot_training_history() 函數"
任務: "實作 plot_confusion_matrix() 函數"
任務: "實作 print_classification_report() 函數"
```

---

## 實作策略

### MVP 優先（僅使用者故事 1 + 2）

1. 完成 Phase 1: 環境設置
2. 完成 Phase 2: 基礎設定（**重要 - 阻塞所有故事**）
3. 完成 Phase 3: 使用者故事 1（資料蒐集）
4. 完成 Phase 4: 使用者故事 2（資料前處理）
5. **停止並驗證**: 獨立測試資料蒐集與前處理功能
6. 若準備好可開始蒐集實際資料

### 增量交付

1. 完成環境設置 + 基礎設定 → 基礎就緒
2. 新增使用者故事 1 → 可開始蒐集資料
3. 新增使用者故事 2 → 可處理資料
4. 新增使用者故事 3 → 可訓練基礎模型
5. 新增使用者故事 4 → 可訓練增強模型
6. 新增使用者故事 5 → 可視覺化分析
7. 每個故事獨立增加價值，不破壞既有功能

---

## 備註

- [P] 任務 = 不同檔案、無依賴，可平行執行
- [Story] 標籤將任務對應至特定使用者故事，便於追蹤
- 每個使用者故事應可獨立完成與測試
- 每完成一個任務或邏輯群組後提交
- 可在任何檢查點停止以獨立驗證故事
- 避免：模糊任務、同檔案衝突、破壞獨立性的跨故事依賴
