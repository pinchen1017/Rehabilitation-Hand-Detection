"""
手部復健偵測程式 - 設定常數模組
"""

# 手勢類別（英文）
GESTURE_NAMES = {
    0: "idle",
    1: "hook",
    2: "angry_fist",
    3: "thumb_flextion",
    4: "straight_fist",
    5: "the_duck",
    6: "fist",
    7: "spend_hand"
}

# 手勢類別（中文）
GESTURE_NAMES_ZH = {
    0: "手放鬆",
    1: "勾拳",
    2: "生氣握拳",
    3: "拇指彎曲",
    4: "直拳",
    5: "鴨子",
    6: "拇指在內握拳",
    7: "手指伸展"
}

# 狀態分類
NON_OPEN_GESTURES = {1, 2, 3, 4, 5, 6}  # 非張開狀態
OPEN_GESTURE = 7  # 張開狀態
IDLE_GESTURE = 0  # 其他狀態

# 有效伸展類型
VALID_STRETCH_TYPES = ["hook", "angry_fist", "thumb_flextion",
                       "straight_fist", "the_duck", "fist"]

# 時間設定
DEFAULT_HOLD_DURATION = 2.0  # 秒

# 偵測設定
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.7
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5
DEFAULT_MAX_HANDS = 1

# 平滑設定
SMOOTHER_WINDOW_SIZE = 5

# 模型路徑
DEFAULT_MODEL_PATH = "models/basic_model.h5"

# UI 設定
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # 白色
BG_COLOR = (0, 0, 0)  # 黑色
LANDMARK_COLOR = (0, 255, 0)  # 綠色
CONNECTION_COLOR = (255, 0, 0)  # 藍色

# ===== 資料蒐集設定 =====
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
RAW_DATA_FILE = "gesture_data.csv"

# 資料蒐集目標
TARGET_SAMPLES_PER_CLASS = 1000
NUM_CLASSES = 8

# ===== 資料前處理設定 =====
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# 資料增強設定
AUGMENT_ROTATION_RANGE = 15  # 度
AUGMENT_SCALE_RANGE = (0.9, 1.1)
AUGMENT_TRANSLATION_RANGE = 0.1

# ===== 模型訓練設定 =====
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
