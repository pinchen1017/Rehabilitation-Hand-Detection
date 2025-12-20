"""
手部復健偵測程式 - 訓練輔助工具模組
提供視覺化與評估輔助函數
"""

import os
import json
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from config import (
    TRAINING_LOGS_DIR,
    GESTURE_NAMES,
    NUM_CLASSES
)


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = False
) -> str:
    """
    繪製訓練歷史曲線

    Args:
        history: 訓練歷史字典，包含 loss, val_loss, accuracy, val_accuracy
        save_path: 儲存路徑，若為 None 則使用預設路徑
        show: 是否顯示圖表

    Returns:
        儲存的檔案路徑
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    # 損失曲線
    axes[0].plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    axes[0].set_title("Loss Curve", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 準確率曲線
    axes[1].plot(epochs, history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2)
    axes[1].set_title("Accuracy Curve", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 儲存圖表
    if save_path is None:
        os.makedirs(os.path.join(TRAINING_LOGS_DIR, "plots"), exist_ok=True)
        save_path = os.path.join(TRAINING_LOGS_DIR, "plots", "training_history.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"訓練歷史圖表已儲存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
    normalize: bool = True
) -> str:
    """
    繪製混淆矩陣熱力圖

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        save_path: 儲存路徑
        show: 是否顯示圖表
        normalize: 是否正規化

    Returns:
        儲存的檔案路徑
    """
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    # 取得類別名稱
    labels = [GESTURE_NAMES.get(i, f"Class {i}") for i in range(NUM_CLASSES)]

    # 繪製熱力圖
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_display,
        display_labels=labels
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=fmt,
        colorbar=True
    )

    plt.title("Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # 旋轉 x 軸標籤
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # 儲存圖表
    if save_path is None:
        os.makedirs(os.path.join(TRAINING_LOGS_DIR, "plots"), exist_ok=True)
        save_path = os.path.join(TRAINING_LOGS_DIR, "plots", "confusion_matrix.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"混淆矩陣圖表已儲存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> str:
    """
    印出並儲存分類報告

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        save_path: 儲存路徑

    Returns:
        分類報告字串
    """
    # 取得類別名稱
    target_names = [GESTURE_NAMES.get(i, f"Class {i}") for i in range(NUM_CLASSES)]

    # 產生報告
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )

    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    # 儲存報告
    if save_path is None:
        os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)
        save_path = os.path.join(TRAINING_LOGS_DIR, "classification_report.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    print(f"分類報告已儲存至: {save_path}")

    return report


def save_training_history(
    history: Dict,
    model_type: str = "model",
    save_path: Optional[str] = None
) -> str:
    """
    儲存訓練歷史為 JSON 檔案

    Args:
        history: 訓練歷史字典
        model_type: 模型類型名稱
        save_path: 儲存路徑

    Returns:
        儲存的檔案路徑
    """
    if save_path is None:
        os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)
        save_path = os.path.join(TRAINING_LOGS_DIR, f"{model_type}_history.json")

    # 確保所有值都是可序列化的
    history_serializable = {}
    for key, values in history.items():
        if isinstance(values, (list, tuple)):
            history_serializable[key] = [float(v) for v in values]
        elif isinstance(values, np.ndarray):
            history_serializable[key] = values.tolist()
        else:
            history_serializable[key] = values

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history_serializable, f, indent=2)

    print(f"訓練歷史已儲存至: {save_path}")
    return save_path


def load_training_history(path: str) -> Dict:
    """
    載入訓練歷史

    Args:
        path: JSON 檔案路徑

    Returns:
        訓練歷史字典
    """
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    return history


def plot_model_comparison(
    basic_history: Dict,
    enhanced_history: Dict,
    save_path: Optional[str] = None,
    show: bool = False
) -> str:
    """
    繪製模型比較圖

    Args:
        basic_history: 基礎模型訓練歷史
        enhanced_history: 增強模型訓練歷史
        save_path: 儲存路徑
        show: 是否顯示圖表

    Returns:
        儲存的檔案路徑
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 損失比較
    basic_epochs = range(1, len(basic_history["val_loss"]) + 1)
    enhanced_epochs = range(1, len(enhanced_history["val_loss"]) + 1)

    axes[0].plot(basic_epochs, basic_history["val_loss"], "b-", label="Basic Model", linewidth=2)
    axes[0].plot(enhanced_epochs, enhanced_history["val_loss"], "r-", label="Enhanced Model", linewidth=2)
    axes[0].set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Validation Loss", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 準確率比較
    axes[1].plot(basic_epochs, basic_history["val_accuracy"], "b-", label="Basic Model", linewidth=2)
    axes[1].plot(enhanced_epochs, enhanced_history["val_accuracy"], "r-", label="Enhanced Model", linewidth=2)
    axes[1].set_title("Validation Accuracy Comparison", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation Accuracy", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 儲存圖表
    if save_path is None:
        os.makedirs(os.path.join(TRAINING_LOGS_DIR, "plots"), exist_ok=True)
        save_path = os.path.join(TRAINING_LOGS_DIR, "plots", "model_comparison.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"模型比較圖表已儲存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def generate_all_visualizations(
    model,
    history: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "model"
) -> Dict[str, str]:
    """
    產生所有視覺化圖表

    Args:
        model: 訓練好的模型
        history: 訓練歷史
        X_test: 測試特徵
        y_test: 測試標籤
        model_type: 模型類型

    Returns:
        包含所有輸出檔案路徑的字典
    """
    plots_dir = os.path.join(TRAINING_LOGS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 預測
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    results = {}

    # 訓練歷史
    history_path = os.path.join(plots_dir, f"{model_type}_loss.png")
    results["history"] = plot_training_history(history, save_path=history_path)

    # 混淆矩陣
    cm_path = os.path.join(plots_dir, f"{model_type}_confusion.png")
    results["confusion_matrix"] = plot_confusion_matrix(
        y_test, y_pred_classes, save_path=cm_path
    )

    # 分類報告
    report_path = os.path.join(TRAINING_LOGS_DIR, f"{model_type}_classification_report.txt")
    results["classification_report"] = print_classification_report(
        y_test, y_pred_classes, save_path=report_path
    )

    # 儲存訓練歷史
    results["history_json"] = save_training_history(history, model_type)

    return results
