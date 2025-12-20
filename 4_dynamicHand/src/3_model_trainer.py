"""
手部復健偵測程式 - 模型訓練模組
訓練基礎與增強版手勢分類模型
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

# TensorFlow 設定
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 減少 TensorFlow 警告
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report

from config import (
    DATA_PROCESSED_DIR,
    TRAINING_LOGS_DIR,
    BASIC_MODEL_CONFIG,
    ENHANCED_MODEL_CONFIG,
    BASIC_MODEL_PATH,
    ENHANCED_MODEL_PATH,
    NUM_CLASSES,
    GESTURE_NAMES
)


class ModelTrainer:
    """模型訓練器"""

    def __init__(self, data_dir: str = DATA_PROCESSED_DIR):
        """
        初始化模型訓練器

        Args:
            data_dir: 處理後資料目錄路徑
        """
        self.data_dir = data_dir
        self.history = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        # 確保目錄存在
        os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)
        os.makedirs(os.path.join(TRAINING_LOGS_DIR, "plots"), exist_ok=True)
        os.makedirs(os.path.dirname(BASIC_MODEL_PATH), exist_ok=True)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        載入訓練/驗證/測試資料

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test) 元組
        """
        train_path = os.path.join(self.data_dir, "train.csv")
        val_path = os.path.join(self.data_dir, "val.csv")
        test_path = os.path.join(self.data_dir, "test.csv")

        # 檢查檔案是否存在
        for path in [train_path, val_path, test_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"資料檔案不存在: {path}")

        # 載入資料
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # 分離特徵與標籤
        self.X_train = train_df.drop("label", axis=1).values.astype(np.float32)
        self.y_train = train_df["label"].values.astype(np.int32)

        self.X_val = val_df.drop("label", axis=1).values.astype(np.float32)
        self.y_val = val_df["label"].values.astype(np.int32)

        self.X_test = test_df.drop("label", axis=1).values.astype(np.float32)
        self.y_test = test_df["label"].values.astype(np.int32)

        print(f"載入資料完成:")
        print(f"  訓練集: {len(self.X_train)} 筆")
        print(f"  驗證集: {len(self.X_val)} 筆")
        print(f"  測試集: {len(self.X_test)} 筆")
        print(f"  特徵維度: {self.X_train.shape[1]}")

        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def build_basic_model(self) -> keras.Model:
        """
        建立基礎模型

        Returns:
            編譯好的 Keras 模型
        """
        config = BASIC_MODEL_CONFIG

        model = keras.Sequential([
            layers.Input(shape=(63,)),
            layers.Dense(config["hidden_layers"][0], activation="relu"),
            layers.Dropout(config["dropout_rate"]),
            layers.Dense(config["hidden_layers"][1], activation="relu"),
            layers.Dropout(config["dropout_rate"]),
            layers.Dense(NUM_CLASSES, activation="softmax")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("\n基礎模型架構:")
        model.summary()

        return model

    def build_enhanced_model(self) -> keras.Model:
        """
        建立增強模型

        Returns:
            編譯好的 Keras 模型
        """
        config = ENHANCED_MODEL_CONFIG

        model = keras.Sequential()
        model.add(layers.Input(shape=(63,)))

        for units in config["hidden_layers"]:
            model.add(layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(config["l2_regularization"])
            ))
            if config["use_batch_norm"]:
                model.add(layers.BatchNormalization())
            model.add(layers.Dropout(config["dropout_rate"]))

        model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("\n增強模型架構:")
        model.summary()

        return model

    def train(
        self,
        model: keras.Model,
        model_type: str = "basic"
    ) -> Dict:
        """
        訓練模型

        Args:
            model: Keras 模型
            model_type: 模型類型 ("basic" 或 "enhanced")

        Returns:
            訓練結果字典
        """
        if self.X_train is None:
            self.load_data()

        config = BASIC_MODEL_CONFIG if model_type == "basic" else ENHANCED_MODEL_CONFIG

        print(f"\n開始訓練 {model_type} 模型...")
        print(f"  批次大小: {config['batch_size']}")
        print(f"  最大訓練輪數: {config['epochs']}")
        print(f"  早停耐心值: {config['early_stopping_patience']}")

        # 設定回調函數
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=config["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            )
        ]

        # 增強模型額外加入學習率調度
        if model_type == "enhanced":
            callbacks.append(ReduceLROnPlateau(
                monitor="val_loss",
                factor=config["lr_reduce_factor"],
                patience=config["lr_reduce_patience"],
                min_lr=1e-6,
                verbose=1
            ))

        # 訓練模型
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=callbacks,
            verbose=1
        )

        self.history = history.history

        # 評估模型
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test, verbose=0)

        print(f"\n{model_type} 模型訓練完成!")
        print(f"  測試損失: {test_loss:.4f}")
        print(f"  測試準確率: {test_acc:.4f} ({test_acc*100:.2f}%)")

        return {
            "model_type": model_type,
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "epochs_trained": len(history.history["loss"]),
            "history": self.history
        }

    def save_model(self, model: keras.Model, path: str) -> None:
        """
        儲存模型

        Args:
            model: Keras 模型
            path: 儲存路徑
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        print(f"模型已儲存至: {path}")

    def generate_confusion_matrix(
        self,
        model: keras.Model,
        model_type: str = "basic"
    ) -> np.ndarray:
        """
        產生混淆矩陣

        Args:
            model: 已訓練的模型
            model_type: 模型類型

        Returns:
            混淆矩陣陣列
        """
        if self.X_test is None:
            self.load_data()

        # 預測
        y_pred = model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # 計算混淆矩陣
        cm = confusion_matrix(self.y_test, y_pred_classes)

        print(f"\n{model_type} 模型混淆矩陣:")
        print(cm)

        return cm

    def print_classification_report(
        self,
        model: keras.Model,
        model_type: str = "basic"
    ) -> str:
        """
        印出分類報告

        Args:
            model: 已訓練的模型
            model_type: 模型類型

        Returns:
            分類報告字串
        """
        if self.X_test is None:
            self.load_data()

        # 預測
        y_pred = model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # 產生報告
        target_names = [GESTURE_NAMES.get(i, f"class_{i}") for i in range(NUM_CLASSES)]
        report = classification_report(
            self.y_test,
            y_pred_classes,
            target_names=target_names
        )

        print(f"\n{model_type} 模型分類報告:")
        print(report)

        return report

    def save_training_history(self, model_type: str = "basic") -> str:
        """
        儲存訓練歷史

        Args:
            model_type: 模型類型

        Returns:
            儲存的檔案路徑
        """
        if self.history is None:
            print("警告：沒有訓練歷史可儲存")
            return ""

        history_path = os.path.join(TRAINING_LOGS_DIR, f"{model_type}_history.json")

        # 轉換為可序列化的格式
        history_serializable = {
            k: [float(v) for v in vals]
            for k, vals in self.history.items()
        }

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_serializable, f, indent=2)

        print(f"訓練歷史已儲存至: {history_path}")
        return history_path

    def compare_models(
        self,
        basic_acc: float,
        enhanced_acc: float
    ) -> None:
        """
        比較基礎與增強模型

        Args:
            basic_acc: 基礎模型準確率
            enhanced_acc: 增強模型準確率
        """
        print("\n" + "=" * 50)
        print("模型比較")
        print("=" * 50)
        print(f"基礎模型準確率:   {basic_acc:.4f} ({basic_acc*100:.2f}%)")
        print(f"增強模型準確率:   {enhanced_acc:.4f} ({enhanced_acc*100:.2f}%)")

        diff = enhanced_acc - basic_acc
        if diff > 0:
            print(f"增強模型提升:     +{diff:.4f} (+{diff*100:.2f}%)")
        else:
            print(f"增強模型差異:     {diff:.4f} ({diff*100:.2f}%)")

        if enhanced_acc > basic_acc:
            print("\n✓ 增強模型優於基礎模型")
        else:
            print("\n⚠ 警告：增強模型未優於基礎模型，可能需要調整超參數")


def train_basic_model(trainer: ModelTrainer, generate_plots: bool = True) -> Tuple[keras.Model, Dict]:
    """訓練基礎模型"""
    print("\n" + "=" * 50)
    print("訓練基礎模型")
    print("=" * 50)

    model = trainer.build_basic_model()
    result = trainer.train(model, "basic")

    trainer.save_model(model, BASIC_MODEL_PATH)
    trainer.generate_confusion_matrix(model, "basic")
    trainer.print_classification_report(model, "basic")
    trainer.save_training_history("basic")

    # 產生視覺化圖表
    if generate_plots:
        try:
            from training_utils import generate_all_visualizations
            generate_all_visualizations(
                model, trainer.history,
                trainer.X_test, trainer.y_test,
                model_type="basic"
            )
        except ImportError:
            print("警告：無法載入 training_utils，跳過視覺化")

    return model, result


def train_enhanced_model(trainer: ModelTrainer, generate_plots: bool = True) -> Tuple[keras.Model, Dict]:
    """訓練增強模型"""
    print("\n" + "=" * 50)
    print("訓練增強模型")
    print("=" * 50)

    model = trainer.build_enhanced_model()
    result = trainer.train(model, "enhanced")

    trainer.save_model(model, ENHANCED_MODEL_PATH)
    trainer.generate_confusion_matrix(model, "enhanced")
    trainer.print_classification_report(model, "enhanced")
    trainer.save_training_history("enhanced")

    # 產生視覺化圖表
    if generate_plots:
        try:
            from training_utils import generate_all_visualizations
            generate_all_visualizations(
                model, trainer.history,
                trainer.X_test, trainer.y_test,
                model_type="enhanced"
            )
        except ImportError:
            print("警告：無法載入 training_utils，跳過視覺化")

    return model, result


def main():
    """主程式進入點"""
    parser = argparse.ArgumentParser(description="模型訓練工具")
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["basic", "enhanced", "all"],
        default="all",
        help="訓練模型類型 (預設: all)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=DATA_PROCESSED_DIR,
        help=f"處理後資料目錄 (預設: {DATA_PROCESSED_DIR})"
    )

    args = parser.parse_args()

    try:
        trainer = ModelTrainer(data_dir=args.data_dir)
        trainer.load_data()

        basic_acc = None
        enhanced_acc = None

        if args.model in ["basic", "all"]:
            _, basic_result = train_basic_model(trainer)
            basic_acc = basic_result["test_accuracy"]

        if args.model in ["enhanced", "all"]:
            _, enhanced_result = train_enhanced_model(trainer)
            enhanced_acc = enhanced_result["test_accuracy"]

        if args.model == "all" and basic_acc is not None and enhanced_acc is not None:
            trainer.compare_models(basic_acc, enhanced_acc)

        print("\n" + "=" * 50)
        print("訓練完成！")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"\n錯誤: {e}")
        print("請先使用 data_preprocessor.py 處理資料。")
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
