# 任務清單：手部復健偵測程式

**輸入**: 設計文件 `/specs/001-hand-rehab-detection/`
**前置條件**: plan.md (必要), spec.md (必要), data-model.md (已讀取)

**測試**: 本功能規格未明確要求測試，故不包含測試任務。如需測試，請執行 `/speckit.tasks` 並指定測試需求。

**組織方式**: 任務按使用者故事分組，支援獨立實作與測試。

## 格式說明: `[ID] [P?] [Story?] 描述`

- **[P]**: 可平行執行（不同檔案、無依賴）
- **[Story]**: 所屬使用者故事（如 [US1], [US2], [US3], [US4]）
- 所有任務包含確切檔案路徑

## 路徑慣例

本專案為單一專案結構：
- 原始碼：`src/`
- 模型檔案：`models/`
- 測試：`tests/`（本次不產生）

---

## Phase 1: 專案初始化 (Setup) ✅ 完成

**目的**: 建立專案結構與依賴環境

- [x] T001 建立專案目錄結構 `src/`, `models/`, `tests/`
- [x] T002 建立 requirements.txt 並列出依賴套件（opencv-python, mediapipe, tensorflow, numpy）
- [x] T003 [P] 建立設定常數模組 src/config.py（手勢類別、狀態分類、時間設定）
- [x] T004 [P] 確認模型檔案 models/rehab_action_classifier_64_5.h5 存在

**檢查點**: 專案結構就緒，依賴已定義 ✅

---

## Phase 2: 基礎建設 (Foundational) ✅ 完成

**目的**: 建立所有使用者故事共用的核心模組

**⚠️ 重要**: 必須完成此階段才能開始任何使用者故事

- [x] T005 實作 HandResult 資料類別於 src/hand_detector.py
- [x] T006 實作 GesturePrediction 資料類別於 src/gesture_classifier.py
- [x] T007 [P] 實作 StretchRecord 資料類別於 src/stretch_tracker.py
- [x] T008 [P] 實作 StretchStats 資料類別於 src/stretch_tracker.py

**檢查點**: 核心資料結構就緒，使用者故事可開始實作 ✅

---

## Phase 3: 使用者故事 1 - 即時手勢辨識 (Priority: P1) 🎯 MVP ✅ 完成

**目標**: 即時顯示手勢名稱，讓使用者確認動作是否正確

**獨立測試**: 啟動程式後，對著攝影機做出任意手勢，畫面上應顯示對應的手勢名稱

### 實作任務

- [x] T009 [US1] 實作 HandDetector 類別於 src/hand_detector.py（封裝 Mediapipe 手部偵測）
- [x] T010 [US1] 實作 HandDetector.detect() 方法，將 21 個關鍵點轉換為 42 維向量
- [x] T011 [US1] 實作 GestureClassifier 類別於 src/gesture_classifier.py（載入 Keras 模型）
- [x] T012 [US1] 實作 GestureClassifier.predict() 方法，輸出手勢類別與信心度
- [x] T013 [US1] 實作 GestureSmoother 類別於 src/gesture_classifier.py（移動平均平滑）
- [x] T014 [US1] 實作基礎 UIRenderer 類別於 src/ui_renderer.py（顯示手勢名稱）
- [x] T015 [US1] 實作 UIRenderer.render() 方法，在畫面左上角疊加手勢名稱（中英文）
- [x] T016 [US1] 實作 UIRenderer 繪製手部骨架點連線
- [x] T017 [US1] 實作主程式框架於 src/main.py（攝影機迴圈、手部偵測、手勢辨識、畫面顯示）
- [x] T018 [US1] 處理未偵測到手部的情況，顯示「未偵測到手部」提示
- [x] T019 [US1] 處理模型載入失敗的錯誤，顯示明確錯誤訊息並安全退出

**檢查點**: 使用者故事 1 完成，程式可即時辨識並顯示手勢名稱 ✅

---

## Phase 4: 使用者故事 2 - 伸展次數統計 (Priority: P2) ✅ 完成

**目標**: 自動計算完成的伸展次數，追蹤復健進度

**獨立測試**: 執行完整的「非張開 → 張開 → 非張開」動作序列（每階段 ≥2 秒），確認計數器增加 1

### 實作任務

- [x] T020 [US2] 實作 TrackerState 列舉於 src/stretch_tracker.py（6 種狀態）
- [x] T021 [US2] 實作 StretchTracker 類別於 src/stretch_tracker.py（狀態機核心）
- [x] T022 [US2] 實作 StretchTracker.update() 方法（狀態轉換邏輯）
- [x] T023 [US2] 實作 IDLE → NON_OPEN_HOLDING 狀態轉換（偵測非張開狀態）
- [x] T024 [US2] 實作 NON_OPEN_HOLDING → OPEN_WAITING 狀態轉換（停留 ≥2 秒）
- [x] T025 [US2] 實作 OPEN_WAITING → OPEN_HOLDING 狀態轉換（偵測 spend_hand）
- [x] T026 [US2] 實作 OPEN_HOLDING → NON_OPEN_FINAL_WAITING 狀態轉換（停留 ≥2 秒）
- [x] T027 [US2] 實作 NON_OPEN_FINAL_WAITING → NON_OPEN_FINAL_HOLDING 狀態轉換（同類型非張開）
- [x] T028 [US2] 實作 NON_OPEN_FINAL_HOLDING → 完成伸展（停留 ≥2 秒，回到 IDLE）
- [x] T029 [US2] 實作混合類型檢測，若起始與結束手勢不同則回到 IDLE
- [x] T030 [US2] 實作 StretchTracker.get_stats() 方法，返回 StretchStats
- [x] T031 [US2] 實作 StretchTracker.reset() 方法，重置統計
- [x] T032 [US2] 整合 StretchTracker 至 src/main.py 主迴圈

**檢查點**: 使用者故事 2 完成，程式可正確統計伸展次數 ✅

---

## Phase 5: 使用者故事 3 - 伸展類型記錄 (Priority: P3) ✅ 完成

**目標**: 記錄每次伸展的類型，分析動作模式

**獨立測試**: 完成數次不同類型的伸展動作，確認系統記錄了正確的伸展類型

### 實作任務

- [x] T033 [US3] 擴充 StretchTracker 以記錄每次伸展的類型（6 種：hook, angry_fist, thumb_flextion, straight_fist, the_duck, fist）
- [x] T034 [US3] 實作 StretchStats.counts_by_type 字典，統計各類型次數
- [x] T035 [US3] 實作伸展完成時建立 StretchRecord 並更新統計

**檢查點**: 使用者故事 3 完成，程式可記錄並統計各類型伸展 ✅

---

## Phase 6: 使用者故事 4 - 視覺化顯示 (Priority: P2) ✅ 完成

**目標**: 在畫面上同時顯示手勢辨識結果與統計資訊

**獨立測試**: 啟動程式後，確認畫面上同時顯示手勢名稱、伸展總次數、各類型統計

### 實作任務

- [x] T036 [US4] 擴充 UIRenderer 於右上角顯示伸展總次數
- [x] T037 [US4] 擴充 UIRenderer 於底部顯示各類型伸展統計
- [x] T038 [US4] 擴充 UIRenderer 顯示狀態機目前狀態與計時進度
- [x] T039 [US4] 整合完整 UI 渲染至 src/main.py

**檢查點**: 使用者故事 4 完成，程式提供完整視覺化介面 ✅

---

## Phase 7: 收尾與優化 (Polish) ✅ 完成

**目的**: 跨功能優化與文件整理

- [x] T040 [P] 新增快捷鍵 'r' 重置統計計數於 src/main.py
- [x] T041 [P] 建立 README.md 說明專案概述、安裝步驟、使用範例
- [x] T042 驗證 quickstart.md 所有步驟可正確執行
- [x] T043 程式碼風格檢查與整理

---

## 完成摘要

**總任務數**: 43
**已完成**: 43
**跳過**: 0

所有任務已完成！程式可以正常執行。

---

## 執行方式

```bash
cd src
python main.py
```
