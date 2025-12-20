# Tasks: 說明文件更新

**Branch**: `003-update-docs` | **Date**: 2025-12-19

## Phase 1: Setup

- [x] T1.1: 確認所有需要更新的文件存在

## Phase 2: Core Documentation Updates

- [x] T2.1: 更新 specs/002-data-collection-and-training/spec.md FR-007
- [x] T2.2: 更新 README.md - 新增資料蒐集模式說明
- [x] T2.3: 更新 README.md - 新增模型訓練說明
- [x] T2.4: 更新 README.md - 更新專案結構

## Phase 3: Validation

- [x] T3.1: 驗證所有文件更新完成
- [x] T3.2: 確認 rehab_action_classifier_64_5.h5 未出現在文件中

## Dependencies

- T2.2, T2.3, T2.4 依賴 T1.1
- T3.1, T3.2 依賴 T2.1, T2.2, T2.3, T2.4
