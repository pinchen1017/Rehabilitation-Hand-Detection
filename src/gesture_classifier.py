"""
手部復健偵測程式 - 手勢分類器模組
載入 Keras 模型並執行手勢分類
"""

from dataclasses import dataclass
from collections import deque, Counter
from typing import Optional
import numpy as np

from config import GESTURE_NAMES, SMOOTHER_WINDOW_SIZE


@dataclass
class GesturePrediction:
    """手勢預測結果"""
    class_id: int          # 0-7
    class_name: str        # 手勢名稱
    confidence: float      # 信心度


class GestureClassifier:
    """手勢分類器 - 載入 Keras 模型並執行推論"""

    def __init__(self, model_path: str):
        """
        初始化分類器

        Args:
            model_path: Keras 模型檔案路徑 (.h5)

        Raises:
            FileNotFoundError: 模型檔案不存在
            Exception: 模型載入失敗
        """
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
        except Exception as e:
            raise Exception(f"模型載入失敗: {model_path}\n錯誤: {str(e)}")

    def predict(self, skeleton: np.ndarray) -> GesturePrediction:
        """
        執行手勢分類

        Args:
            skeleton: 63 維骨架向量 (21 點 x 3 座標)

        Returns:
            GesturePrediction 包含類別 ID、名稱與信心度
        """
        skeleton = self._normalize_relative_z(skeleton)  # relative-z 正規化
        
        # 調整輸入形狀為 (1, 63)
        input_data = skeleton.reshape(1, -1)

        # 執行推論
        predictions = self.model.predict(input_data, verbose=0)

        # 取得最高信心度的類別
        class_id = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_id])

        return GesturePrediction(
            class_id=class_id,
            class_name=GESTURE_NAMES.get(class_id, "unknown"),
            confidence=confidence
        )
    
    def _normalize_relative_z(self, skeleton: np.ndarray) -> np.ndarray:
        sk = skeleton.copy()
        z = sk[2::3]
        z0 = z[0]

        z_rel = z - z0
        std = np.std(z_rel)
        if std > 1e-6:
            z_rel = z_rel / std

        sk[2::3] = z_rel
        return sk


class GestureSmoother:
    """手勢平滑器 - 使用移動平均降低抖動"""

    def __init__(self, window_size: int = SMOOTHER_WINDOW_SIZE):
        self.history = deque(maxlen=window_size)

    def smooth(self, prediction: GesturePrediction) -> GesturePrediction:
        """
        平滑手勢預測結果（使用多數決）

        Args:
            prediction: 原始預測結果

        Returns:
            平滑後的預測結果
        """
        self.history.append(prediction.class_id)

        # 使用多數決
        if len(self.history) == 0:
            return prediction

        most_common = Counter(self.history).most_common(1)[0][0]

        return GesturePrediction(
            class_id=most_common,
            class_name=GESTURE_NAMES.get(most_common, "unknown"),
            confidence=prediction.confidence
        )

    def reset(self):
        """重置歷史記錄"""
        self.history.clear()
