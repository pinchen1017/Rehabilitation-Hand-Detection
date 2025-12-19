# Implementation Plan: 說明文件更新

**Branch**: 003-update-docs | **Date**: 2025-12-19 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from /specs/003-update-docs/spec.md

## Summary

更新專案說明文件以反映最新的程式碼變更，包括：
1. 更正 z 座標正規化方法說明（min-max → wrist-relative）
2. 整合 README.md，納入資料蒐集與訓練功能的使用說明
3. 記錄 UI 介面修改（位置調整、中文字體支援）

**技術方法**: 純文件編輯，無程式碼修改。

## Technical Context

**Language/Version**: Markdown 文件
**Primary Dependencies**: N/A（純文件更新）
**Storage**: N/A
**Testing**: N/A（文件審閱）
**Target Platform**: 文件（GitHub README、規格說明）
**Project Type**: single（既有專案結構）
**Performance Goals**: N/A
**Constraints**: 文件需清晰易讀，使用者能在 5 分鐘內理解使用方式
**Scale/Scope**: 3 個文件更新（README.md、spec.md (002)、變更記錄）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**狀態**: 通過 - 此功能為純文件更新，不涉及程式碼架構或依賴變更。

憲法檔案尚未設定專案特定規則，採用預設指南：
- [x] 簡單性原則：僅更新必要文件
- [x] 一致性原則：確保文件與程式碼一致
- [x] 可維護性原則：保持文件結構清晰

## Project Structure

### Documentation (this feature)

text
specs/003-update-docs/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)


### Source Code (repository root)

text
hand-stretches/
├── README.md                    # [更新] 主要說明文件
├── src/
│   ├── main.py                  # 主程式（即時辨識）
│   ├── config.py                # 設定常數
│   ├── hand_detector.py         # 手部偵測器
│   ├── gesture_classifier.py    # 手勢分類器
│   ├── stretch_tracker.py       # 伸展追蹤器
│   ├── ui_renderer.py           # UI 渲染器
│   ├── data_collector.py        # [新增] 資料蒐集腳本
│   ├── data_preprocessor.py     # [新增] 前處理工具
│   ├── model_trainer.py         # [新增] 訓練流程
│   └── training_utils.py        # [新增] 訓練工具
├── data/
│   ├── raw/                     # 原始蒐集的 CSV 檔案
│   └── processed/               # 前處理後的資料集
├── models/
│   ├── basic_model.h5           # 基礎訓練模型
│   └── enhanced_model.h5        # 增強訓練模型
├── training_logs/               # 訓練歷史和圖表
│   └── plots/
└── specs/
    ├── 001-hand-rehab-detection/
    ├── 002-data-collection-and-training/  # [更新] FR-007 正規化說明
    └── 003-update-docs/


**Structure Decision**: 既有單一專案結構，本次僅更新文件內容。

## Implementation Tasks

### Task 1: 更新 specs/002-data-collection-and-training/spec.md

**目標**: 更正 z 座標正規化方法說明

**變更內容**:
- FR-007: 將「最小-最大正規化」改為「wrist-relative z 正規化」
- 新增說明：為何使用 wrist-relative（推論時無法得知全域極值）

### Task 2: 更新 README.md

**目標**: 整合資料蒐集與訓練功能的使用說明

**變更內容**:
1. 使用方式章節：
   - 即時辨識模式（既有）
   - 資料蒐集模式（新增）
   - 模型訓練（新增）
2. 專案結構：
   - 新增檔案：data_collector.py、data_preprocessor.py、model_trainer.py、training_utils.py
   - 新增目錄：data/、training_logs/
   - models/ 目錄僅列出 basic_model.h5 與 enhanced_model.h5
3. 快捷鍵說明：
   - 資料蒐集模式快捷鍵

### Task 3: 記錄 UI 變更

**目標**: 在 README.md 中記錄 UI 修改

**變更內容**:
- UI 位置調整（避免文字重疊）
- 中文字體支援（msyh.ttc）

## Complexity Tracking

無複雜度違規 - 此功能為簡單的文件更新任務。

## Notes

- 程式碼已完成，本計畫僅涉及文件編輯
- 不需要 data-model.md 或 contracts/（無資料模型或 API 變更）
- 不需要執行代理上下文更新腳本（無技術堆疊變更）
- 忽略 rehab_action_classifier_64_5.h5，不列入說明文件
