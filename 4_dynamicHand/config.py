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
DEFAULT_MODEL_PATH = "models/rehab_action_classifier_64_6.h5"

# UI 設定
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # 白色
BG_COLOR = (0, 0, 0)  # 黑色
LANDMARK_COLOR = (0, 255, 0)  # 綠色
CONNECTION_COLOR = (255, 0, 0)  # 藍色
