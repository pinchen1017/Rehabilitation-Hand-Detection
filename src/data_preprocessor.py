"""
手部復健偵測程式 - 資料前處理模組
清理、正規化、分割與增強已蒐集的資料
"""

import os
import sys
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RAW_DATA_FILE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    AUGMENT_ROTATION_RANGE,
    AUGMENT_SCALE_RANGE,
    AUGMENT_TRANSLATION_RANGE,
    TARGET_SAMPLES_PER_CLASS,
    NUM_CLASSES,
    GESTURE_NAMES
)


class DataPreprocessor:
    """資料前處理器"""

    def __init__(
        self,
        raw_path: Optional[str] = None,
        output_dir: str = DATA_PROCESSED_DIR
    ):
        """
        初始化資料前處理器

        Args:
            raw_path: 原始 CSV 檔案路徑
            output_dir: 輸出目錄路徑
        """
        if raw_path is None:
            self.raw_path = os.path.join(DATA_RAW_DIR, RAW_DATA_FILE)
        else:
            self.raw_path = raw_path

        self.output_dir = output_dir
        self.removed_count = 0
        self.original_count = 0

    def load_raw_data(self) -> pd.DataFrame:
        """
        載入原始資料

        Returns:
            載入的 DataFrame

        Raises:
            FileNotFoundError: 檔案不存在
        """
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"原始資料檔案不存在: {self.raw_path}")

        df = pd.read_csv(self.raw_path)
        self.original_count = len(df)
        print(f"載入原始資料: {self.original_count} 筆")
        return df

    def validate_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        驗證並清理樣本

        Args:
            df: 原始 DataFrame

        Returns:
            清理後的 DataFrame
        """
        initial_count = len(df)

        # 移除包含 NaN 的行
        df = df.dropna()
        nan_removed = initial_count - len(df)
        if nan_removed > 0:
            print(f"  移除 NaN 樣本: {nan_removed} 筆")

        # 檢查座標異常值 (x, y 應在 [0, 1] 範圍，z 可以略微超出)
        x_cols = [f"x{i}" for i in range(21)]
        y_cols = [f"y{i}" for i in range(21)]

        # 找出 x, y 超出 [0, 1] 範圍的行
        x_valid = (df[x_cols] >= 0).all(axis=1) & (df[x_cols] <= 1).all(axis=1)
        y_valid = (df[y_cols] >= 0).all(axis=1) & (df[y_cols] <= 1).all(axis=1)
        valid_mask = x_valid & y_valid

        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"  移除座標異常樣本: {invalid_count} 筆")
            df = df[valid_mask]

        self.removed_count = initial_count - len(df)
        print(f"  驗證完成: 保留 {len(df)} 筆，移除 {self.removed_count} 筆")
        return df.reset_index(drop=True)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        正規化座標

        MediaPipe 的 x, y 已經是 [0, 1]，只需處理 z 座標

        Args:
            df: 輸入 DataFrame

        Returns:
            正規化後的 DataFrame
        """
        df = df.copy()
        z_cols = [f"z{i}" for i in range(21)]

        # 對 z 座標進行 min-max 正規化
        z_values = df[z_cols].values
        z_min = z_values.min()
        z_max = z_values.max()

        if z_max - z_min > 0:
            df[z_cols] = (z_values - z_min) / (z_max - z_min)
        else:
            df[z_cols] = 0.5  # 如果所有值相同，設為中間值

        print(f"  正規化完成: z 座標範圍 [{z_min:.4f}, {z_max:.4f}] -> [0, 1]")
        return df

    def augment(self, df: pd.DataFrame, multiplier: int = 2) -> pd.DataFrame:
        """
        資料增強

        Args:
            df: 輸入 DataFrame
            multiplier: 增強倍數

        Returns:
            增強後的 DataFrame
        """
        if multiplier <= 1:
            return df

        augmented_samples = []
        original_count = len(df)

        for _ in range(multiplier - 1):
            for _, row in df.iterrows():
                augmented_row = self._augment_sample(row)
                augmented_samples.append(augmented_row)

        augmented_df = pd.DataFrame(augmented_samples)
        result_df = pd.concat([df, augmented_df], ignore_index=True)

        print(f"  資料增強完成: {original_count} -> {len(result_df)} 筆 (x{multiplier})")
        return result_df

    def _augment_sample(self, row: pd.Series) -> Dict:
        """
        對單一樣本進行增強

        Args:
            row: 原始樣本

        Returns:
            增強後的樣本字典
        """
        augmented = {"label": row["label"]}

        # 取得所有座標
        x_coords = np.array([row[f"x{i}"] for i in range(21)])
        y_coords = np.array([row[f"y{i}"] for i in range(21)])
        z_coords = np.array([row[f"z{i}"] for i in range(21)])

        # 計算手部中心
        center_x = x_coords.mean()
        center_y = y_coords.mean()

        # 隨機旋轉
        angle = np.random.uniform(
            -AUGMENT_ROTATION_RANGE,
            AUGMENT_ROTATION_RANGE
        ) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # 平移至原點
        x_centered = x_coords - center_x
        y_centered = y_coords - center_y

        # 旋轉
        x_rotated = x_centered * cos_a - y_centered * sin_a
        y_rotated = x_centered * sin_a + y_centered * cos_a

        # 平移回原位
        x_coords = x_rotated + center_x
        y_coords = y_rotated + center_y

        # 隨機縮放
        scale = np.random.uniform(*AUGMENT_SCALE_RANGE)
        x_coords = center_x + (x_coords - center_x) * scale
        y_coords = center_y + (y_coords - center_y) * scale

        # 隨機平移
        tx = np.random.uniform(-AUGMENT_TRANSLATION_RANGE, AUGMENT_TRANSLATION_RANGE)
        ty = np.random.uniform(-AUGMENT_TRANSLATION_RANGE, AUGMENT_TRANSLATION_RANGE)
        x_coords = np.clip(x_coords + tx, 0, 1)
        y_coords = np.clip(y_coords + ty, 0, 1)

        # 儲存增強後的座標
        for i in range(21):
            augmented[f"x{i}"] = float(x_coords[i])
            augmented[f"y{i}"] = float(y_coords[i])
            augmented[f"z{i}"] = float(z_coords[i])

        return augmented

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        分層分割資料

        Args:
            df: 輸入 DataFrame

        Returns:
            (train_df, val_df, test_df) 元組
        """
        # 計算分割比例
        test_val_ratio = TEST_RATIO + VAL_RATIO
        val_ratio_adjusted = VAL_RATIO / test_val_ratio

        # 第一次分割：分出訓練集
        train_df, temp_df = train_test_split(
            df,
            test_size=test_val_ratio,
            stratify=df["label"],
            random_state=RANDOM_SEED
        )

        # 第二次分割：分出驗證集和測試集
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_df["label"],
            random_state=RANDOM_SEED
        )

        print(f"  資料分割完成:")
        print(f"    訓練集: {len(train_df)} 筆 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"    驗證集: {len(val_df)} 筆 ({len(val_df)/len(df)*100:.1f}%)")
        print(f"    測試集: {len(test_df)} 筆 ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    def _check_class_balance(self, df: pd.DataFrame) -> None:
        """
        檢查類別平衡

        Args:
            df: 輸入 DataFrame
        """
        class_counts = df["label"].value_counts().sort_index()
        min_count = class_counts.min()
        max_count = class_counts.max()

        print("\n類別分布:")
        for class_id, count in class_counts.items():
            name = GESTURE_NAMES.get(int(class_id), "unknown")
            bar = "█" * int(count / max_count * 20)
            print(f"  {class_id}: {name:15s} {count:5d} {bar}")

        # 警告不平衡
        if max_count > min_count * 2:
            print("\n⚠️ 警告: 類別不平衡！建議為少數類別蒐集更多資料。")

        # 警告樣本不足
        if min_count < 100:
            print("\n⚠️ 警告: 部分類別樣本不足 100 筆，結果可能不可靠。")

    def process(self, augment: bool = True, augment_multiplier: int = 2) -> Dict[str, str]:
        """
        執行完整前處理流程

        Args:
            augment: 是否進行資料增強
            augment_multiplier: 增強倍數

        Returns:
            包含輸出檔案路徑的字典
        """
        print("=" * 50)
        print("資料前處理流程")
        print("=" * 50)

        # 載入原始資料
        print("\n[1/5] 載入原始資料...")
        df = self.load_raw_data()

        # 檢查類別平衡
        self._check_class_balance(df)

        # 驗證樣本
        print("\n[2/5] 驗證樣本...")
        df = self.validate_samples(df)

        # 正規化
        print("\n[3/5] 正規化座標...")
        df = self.normalize(df)

        # 資料增強
        if augment:
            print("\n[4/5] 資料增強...")
            df = self.augment(df, augment_multiplier)
        else:
            print("\n[4/5] 跳過資料增強")

        # 分割資料
        print("\n[5/5] 分割資料...")
        train_df, val_df, test_df = self.split_data(df)

        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 儲存檔案
        train_path = os.path.join(self.output_dir, "train.csv")
        val_path = os.path.join(self.output_dir, "val.csv")
        test_path = os.path.join(self.output_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print("\n" + "=" * 50)
        print("前處理完成！")
        print("=" * 50)
        print(f"輸出檔案:")
        print(f"  訓練集: {train_path}")
        print(f"  驗證集: {val_path}")
        print(f"  測試集: {test_path}")

        return {
            "train": train_path,
            "val": val_path,
            "test": test_path
        }


def main():
    """主程式進入點"""
    import argparse

    parser = argparse.ArgumentParser(description="資料前處理工具")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help=f"輸入的原始 CSV 檔案路徑 (預設: {DATA_RAW_DIR}/{RAW_DATA_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DATA_PROCESSED_DIR,
        help=f"輸出目錄路徑 (預設: {DATA_PROCESSED_DIR})"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="不進行資料增強"
    )
    parser.add_argument(
        "--augment-multiplier", "-m",
        type=int,
        default=2,
        help="資料增強倍數 (預設: 2)"
    )

    args = parser.parse_args()

    try:
        preprocessor = DataPreprocessor(
            raw_path=args.input,
            output_dir=args.output
        )
        preprocessor.process(
            augment=not args.no_augment,
            # augment=False,  # 暫時硬編碼為不進行資料增強
            augment_multiplier=args.augment_multiplier
        )
    except FileNotFoundError as e:
        print(f"\n錯誤: {e}")
        print("請先使用 data_collector.py 蒐集資料。")
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
