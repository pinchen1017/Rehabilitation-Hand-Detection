# 實作計畫：資料蒐集與模型訓練流程

**分支**: `002-data-collection-and-training` | **日期**: 2025-12-19 | **規格**: [spec.md](./spec.md)
**輸入**: 功能規格書 `/specs/002-data-collection-and-training/spec.md`

## 摘要

在既有手部復健偵測專案上擴展資料蒐集與模型訓練功能。重用現有的 `HandDetector` 模組進行手部特徵點擷取，新增資料蒐集腳本（含簡易 OpenCV 前端）、資料前處理工具、以及基礎/增強模型訓練流程。所有新增模組皆為 Python 實作，並最小化對原有程式碼的修改。

## 技術背景

**語言/版本**: Python 3.10+（與原專案一致）
**主要依賴**:
- OpenCV (`opencv-python`) - 攝影機擷取與 UI 顯示（既有）
- Mediapipe (`mediapipe`) - 手部骨架偵測（既有）
- TensorFlow/Keras (`tensorflow`) - 模型訓練與儲存（既有）
- NumPy (`numpy`) - 數值運算（既有）
- Pandas (`pandas`) - CSV 資料處理（新增）
- scikit-learn (`scikit-learn`) - 資料分割與評估（新增）
- Matplotlib (`matplotlib`) - 訓練視覺化（新增）

**儲存**: CSV 檔案（資料）、H5 檔案（模型）
**測試**: pytest
**目標平台**: Windows/macOS/Linux 桌面環境（需攝影機）
**專案類型**: 單一專案（擴展既有結構）
**效能目標**: 資料蒐集 ≥100 筆/分鐘，訓練 <5 分鐘
**約束**: 模型格式需與既有 `GestureClassifier` 相容（.h5, 63 輸入, 8 輸出）

## 憲章檢查

*門檻：必須在設計前通過。*

| 原則 | 狀態 | 說明 |
|------|------|------|
| I. 品質至上 | ✅ 通過 | 重用既有模組，保持一致性 |
| II. 可測試性優先 | ✅ 通過 | 各模組獨立可測試 |
| III. MVP | ✅ 通過 | 專注核心功能：蒐集、前處理、訓練 |
| IV. 簡潔原則 | ✅ 通過 | 最小化修改，使用 OpenCV 簡易前端 |

## 專案結構

### 文件（本功能）

```text
specs/002-data-collection-and-training/
├── spec.md              # 功能規格書
├── plan.md              # 本文件
└── tasks.md             # 任務清單（由 /speckit.tasks 產生）
```

### 原始碼（專案根目錄）

```text
src/
├── config.py               # 既有 - 新增訓練相關常數
├── hand_detector.py        # 既有 - 重用，不修改
├── gesture_classifier.py   # 既有 - 不修改
├── stretch_tracker.py      # 既有 - 不修改
├── ui_renderer.py          # 既有 - 不修改
├── main.py                 # 既有 - 不修改
│
├── data_collector.py       # 新增 - 資料蒐集主程式（含 UI）
├── data_preprocessor.py    # 新增 - 資料前處理工具
├── model_trainer.py        # 新增 - 模型訓練流程
└── training_utils.py       # 新增 - 訓練輔助函數

data/
├── raw/                    # 新增 - 原始蒐集資料
│   └── gesture_data.csv    # 蒐集的原始資料檔
└── processed/              # 新增 - 前處理後資料
    ├── train.csv           # 訓練集
    ├── val.csv             # 驗證集
    └── test.csv            # 測試集

models/
├── rehab_action_classifier_64_5.h5  # 既有
├── basic_model.h5                   # 新增 - 基礎模型
└── enhanced_model.h5                # 新增 - 增強模型

training_logs/              # 新增 - 訓練紀錄
├── basic_history.json      # 基礎模型訓練歷史
├── enhanced_history.json   # 增強模型訓練歷史
└── plots/                  # 視覺化圖表
    ├── basic_loss.png
    ├── basic_confusion.png
    ├── enhanced_loss.png
    └── enhanced_confusion.png
```

**結構決策**: 延續既有單一專案結構，新增模組放置於 `src/` 目錄，資料與訓練紀錄分別放置於獨立目錄。

## 核心模組設計

### 1. 設定擴展 (config.py 修改)

**修改內容**: 新增訓練相關常數，不影響既有功能

```python
# ===== 新增：資料蒐集設定 =====
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
RAW_DATA_FILE = "gesture_data.csv"

# 資料蒐集目標
TARGET_SAMPLES_PER_CLASS = 1000
NUM_CLASSES = 8

# ===== 新增：資料前處理設定 =====
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# 資料增強設定
AUGMENT_ROTATION_RANGE = 15  # 度
AUGMENT_SCALE_RANGE = (0.9, 1.1)
AUGMENT_TRANSLATION_RANGE = 0.1

# ===== 新增：模型訓練設定 =====
TRAINING_LOGS_DIR = "training_logs"

# 基礎模型設定
BASIC_MODEL_CONFIG = {
    "hidden_layers": [128, 64],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "early_stopping_patience": 10
}

# 增強模型設定
ENHANCED_MODEL_CONFIG = {
    "hidden_layers": [256, 128, 64],
    "dropout_rate": 0.4,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 15,
    "use_batch_norm": True,
    "l2_regularization": 0.001,
    "lr_reduce_factor": 0.5,
    "lr_reduce_patience": 5
}

# 模型儲存路徑
BASIC_MODEL_PATH = "models/basic_model.h5"
ENHANCED_MODEL_PATH = "models/enhanced_model.h5"
```

### 2. 資料蒐集器 (data_collector.py)

**職責**: 提供簡易 OpenCV 前端，蒐集手勢特徵點資料

```python
class DataCollector:
    """資料蒐集器 - 含簡易 OpenCV 前端"""

    def __init__(self, output_path: str = None):
        self.detector = HandDetector()  # 重用既有模組
        self.current_class: int = 0
        self.samples: List[Dict] = []
        self.counts_per_class: Dict[int, int] = {i: 0 for i in range(8)}
        self.output_path = output_path or os.path.join(DATA_RAW_DIR, RAW_DATA_FILE)
        self.is_collecting: bool = False

    def run(self):
        """執行蒐集主迴圈"""
        # 開啟攝影機
        # 主迴圈：
        #   - 偵測手部
        #   - 繪製 UI（類別選擇、統計、手部骨架）
        #   - 按數字鍵 0-7 切換類別
        #   - 按住空白鍵蒐集資料
        #   - 按 's' 儲存資料
        #   - 按 'q' 退出

    def _collect_sample(self, landmarks: np.ndarray, label: int):
        """蒐集一筆樣本"""
        sample = {"label": label}
        for i in range(21):
            sample[f"x{i}"] = landmarks[i * 3]
            sample[f"y{i}"] = landmarks[i * 3 + 1]
            sample[f"z{i}"] = landmarks[i * 3 + 2]
        self.samples.append(sample)
        self.counts_per_class[label] += 1

    def save_data(self):
        """儲存資料至 CSV"""
        df = pd.DataFrame(self.samples)
        df.to_csv(self.output_path, index=False)
```

**UI 設計**:
- 左上角：目前選擇的類別（編號與名稱）
- 右上角：各類別已蒐集數量
- 中央：手部骨架與狀態提示
- 底部：操作說明（數字鍵切換、空白鍵蒐集、s 儲存、q 退出）
- 蒐集時顯示綠色邊框提示

### 3. 資料前處理器 (data_preprocessor.py)

**職責**: 清理、正規化、分割與增強資料

```python
class DataPreprocessor:
    """資料前處理器"""

    def __init__(self, raw_path: str, output_dir: str = DATA_PROCESSED_DIR):
        self.raw_path = raw_path
        self.output_dir = output_dir

    def load_raw_data(self) -> pd.DataFrame:
        """載入原始資料"""
        return pd.read_csv(self.raw_path)

    def validate_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """驗證並清理樣本"""
        # 移除 NaN
        # 移除座標異常值
        # 記錄移除數量

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """正規化座標至 [0, 1]"""
        # 對每筆樣本的 x, y, z 分別正規化
        # x, y 已經是 [0, 1]（MediaPipe 輸出）
        # z 需要正規化

    def augment(self, df: pd.DataFrame, multiplier: int = 2) -> pd.DataFrame:
        """資料增強"""
        # 隨機旋轉
        # 隨機縮放
        # 隨機平移

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """分層分割資料"""
        from sklearn.model_selection import train_test_split
        # 保持類別比例的分割

    def process(self, augment: bool = True) -> Dict[str, str]:
        """執行完整前處理流程"""
        df = self.load_raw_data()
        df = self.validate_samples(df)
        df = self.normalize(df)

        if augment:
            df = self.augment(df)

        train_df, val_df, test_df = self.split_data(df)

        # 儲存
        train_path = os.path.join(self.output_dir, "train.csv")
        val_path = os.path.join(self.output_dir, "val.csv")
        test_path = os.path.join(self.output_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        return {"train": train_path, "val": val_path, "test": test_path}
```

### 4. 模型訓練器 (model_trainer.py)

**職責**: 訓練基礎與增強模型

```python
class ModelTrainer:
    """模型訓練器"""

    def __init__(self, data_dir: str = DATA_PROCESSED_DIR):
        self.data_dir = data_dir
        self.history = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """載入訓練/驗證/測試資料"""
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(self.data_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))

        # 分離特徵與標籤
        X_train = train_df.drop("label", axis=1).values
        y_train = train_df["label"].values
        # ... 同理處理 val 和 test

        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_basic_model(self) -> tf.keras.Model:
        """建立基礎模型"""
        config = BASIC_MODEL_CONFIG
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(63,)),
            tf.keras.layers.Dense(config["hidden_layers"][0], activation="relu"),
            tf.keras.layers.Dropout(config["dropout_rate"]),
            tf.keras.layers.Dense(config["hidden_layers"][1], activation="relu"),
            tf.keras.layers.Dropout(config["dropout_rate"]),
            tf.keras.layers.Dense(8, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def build_enhanced_model(self) -> tf.keras.Model:
        """建立增強模型"""
        config = ENHANCED_MODEL_CONFIG
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(63,)))

        for units in config["hidden_layers"]:
            model.add(tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(config["l2_regularization"])
            ))
            if config["use_batch_norm"]:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(config["dropout_rate"]))

        model.add(tf.keras.layers.Dense(8, activation="softmax"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(self, model: tf.keras.Model, model_type: str = "basic") -> Dict:
        """訓練模型"""
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()

        config = BASIC_MODEL_CONFIG if model_type == "basic" else ENHANCED_MODEL_CONFIG

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=config["early_stopping_patience"],
                restore_best_weights=True
            )
        ]

        if model_type == "enhanced":
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                factor=config["lr_reduce_factor"],
                patience=config["lr_reduce_patience"]
            ))

        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=callbacks,
            verbose=1
        )

        # 評估
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        return {
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "history": self.history.history
        }

    def save_model(self, model: tf.keras.Model, path: str):
        """儲存模型"""
        model.save(path)

    def evaluate_and_visualize(self, model: tf.keras.Model, model_type: str):
        """評估並產生視覺化"""
        # 產生混淆矩陣
        # 產生損失/準確率曲線
        # 計算各類別指標
```

### 5. 訓練輔助函數 (training_utils.py)

**職責**: 視覺化與評估輔助函數

```python
def plot_training_history(history: Dict, save_path: str):
    """繪製訓練歷史曲線"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 損失曲線
    axes[0].plot(history["loss"], label="訓練損失")
    axes[0].plot(history["val_loss"], label="驗證損失")
    axes[0].set_title("損失曲線")
    axes[0].legend()

    # 準確率曲線
    axes[1].plot(history["accuracy"], label="訓練準確率")
    axes[1].plot(history["val_accuracy"], label="驗證準確率")
    axes[1].set_title("準確率曲線")
    axes[1].legend()

    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """繪製混淆矩陣"""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(GESTURE_NAMES.values()))
    disp.plot(cmap="Blues")
    plt.title("混淆矩陣")
    plt.savefig(save_path)
    plt.close()

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """印出分類報告"""
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true, y_pred,
        target_names=list(GESTURE_NAMES.values())
    )
    print(report)
```

## 資料流程圖

```
┌──────────────────────────────────────────────────────────────────────┐
│                           資料蒐集階段                                │
└──────────────────────────────────────────────────────────────────────┘
     攝影機
        │
        v
┌───────────────────┐
│   HandDetector    │  (重用既有模組)
│   (hand_detector) │
└───────────────────┘
        │
        v
┌───────────────────┐      按住空白鍵       ┌───────────────────┐
│  DataCollector    │ ──────────────────> │   gesture_data.csv │
│  (data_collector) │                      │   (data/raw/)      │
└───────────────────┘                      └───────────────────┘
                                                    │
┌──────────────────────────────────────────────────────────────────────┐
│                           資料前處理階段                              │
└──────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    v
                                         ┌───────────────────┐
                                         │  DataPreprocessor │
                                         └───────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    v                               v                               v
            ┌─────────────┐                 ┌─────────────┐                 ┌─────────────┐
            │  train.csv  │                 │   val.csv   │                 │  test.csv   │
            │   (70%)     │                 │   (15%)     │                 │   (15%)     │
            └─────────────┘                 └─────────────┘                 └─────────────┘
                    │                               │                               │
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                      模型訓練階段                                         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                    │                               │                               │
                    └───────────────────────────────┼───────────────────────────────┘
                                                    v
                                         ┌───────────────────┐
                                         │   ModelTrainer    │
                                         └───────────────────┘
                                                    │
                            ┌───────────────────────┴───────────────────────┐
                            v                                               v
                    ┌───────────────┐                               ┌───────────────┐
                    │  基礎模型訓練  │                               │  增強模型訓練  │
                    └───────────────┘                               └───────────────┘
                            │                                               │
                            v                                               v
                    ┌───────────────┐                               ┌───────────────┐
                    │basic_model.h5 │                               │enhanced_model │
                    └───────────────┘                               │     .h5       │
                                                                    └───────────────┘
                                                    │
                                                    v
                                         ┌───────────────────┐
                                         │ GestureClassifier │  (既有模組，直接載入使用)
                                         │  (main.py 辨識)   │
                                         └───────────────────┘
```

## 模型架構比較

### 基礎模型

```
輸入層 (63)
    │
    v
Dense (128, ReLU)
    │
    v
Dropout (0.3)
    │
    v
Dense (64, ReLU)
    │
    v
Dropout (0.3)
    │
    v
Dense (8, Softmax) ─────> 輸出
```

### 增強模型

```
輸入層 (63)
    │
    v
Dense (256, ReLU, L2)
    │
    v
BatchNormalization
    │
    v
Dropout (0.4)
    │
    v
Dense (128, ReLU, L2)
    │
    v
BatchNormalization
    │
    v
Dropout (0.4)
    │
    v
Dense (64, ReLU, L2)
    │
    v
BatchNormalization
    │
    v
Dropout (0.4)
    │
    v
Dense (8, Softmax) ─────> 輸出
```

## 依賴清單更新 (requirements.txt)

```text
# 既有依賴
opencv-python>=4.8.0
mediapipe==0.10.21
tensorflow>=2.13.0
numpy>=1.24.0
pytest>=7.0.0

# 新增依賴
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

## 使用流程

### 1. 資料蒐集

```bash
python src/data_collector.py
```

操作：
- 數字鍵 0-7：切換目標類別
- 空白鍵（按住）：蒐集資料
- s：儲存資料
- q：退出

### 2. 資料前處理

```bash
python src/data_preprocessor.py
```

或在 Python 中：

```python
from data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor("data/raw/gesture_data.csv")
paths = preprocessor.process(augment=True)
print(f"處理完成: {paths}")
```

### 3. 模型訓練

```bash
# 訓練基礎模型
python src/model_trainer.py --model basic

# 訓練增強模型
python src/model_trainer.py --model enhanced

# 訓練兩者
python src/model_trainer.py --model all
```

### 4. 使用訓練好的模型

修改 `config.py` 中的模型路徑：

```python
# 使用基礎模型
DEFAULT_MODEL_PATH = "models/basic_model.h5"

# 或使用增強模型
DEFAULT_MODEL_PATH = "models/enhanced_model.h5"
```

然後正常執行辨識程式：

```bash
python src/main.py
```

## 複雜度追蹤

無違反憲章原則的情況。所有設計皆遵循最小修改原則，重用既有模組。

## 風險與緩解

| 風險 | 影響 | 緩解策略 |
|------|------|----------|
| 資料蒐集不足 | 模型準確率低 | UI 顯示各類別數量，提示不足類別 |
| 類別不平衡 | 模型偏向多數類別 | 前處理時檢查並警告，可使用增強補足 |
| 模型不相容 | 既有辨識程式無法載入 | 使用相同的輸入(63)/輸出(8)維度，相同 .h5 格式 |
| 增強模型未優於基礎 | 不符合規格要求 | 調整增強模型超參數，增加正規化 |
