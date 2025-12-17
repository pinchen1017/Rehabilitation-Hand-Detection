"""
手部復健偵測程式 - 伸展追蹤器模組
實作狀態機，追蹤伸展動作並統計次數與類型
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
import time

from config import (
    NON_OPEN_GESTURES,
    OPEN_GESTURE,
    GESTURE_NAMES,
    DEFAULT_HOLD_DURATION,
    VALID_STRETCH_TYPES
)
from gesture_classifier import GesturePrediction


@dataclass
class StretchRecord:
    """伸展記錄"""
    stretch_type: str      # 伸展類型名稱
    start_time: float      # 開始時間戳
    end_time: float        # 結束時間戳


@dataclass
class StretchStats:
    """伸展統計"""
    total_count: int = 0
    counts_by_type: Dict[str, int] = field(default_factory=lambda: {
        t: 0 for t in VALID_STRETCH_TYPES
    })
    history: List[StretchRecord] = field(default_factory=list)


class TrackerState(Enum):
    """追蹤器狀態"""
    IDLE = "閒置"
    NON_OPEN_HOLDING = "非張開計時中"
    OPEN_WAITING = "等待張開"
    OPEN_HOLDING = "張開計時中"
    NON_OPEN_FINAL_WAITING = "等待結束非張開"
    NON_OPEN_FINAL_HOLDING = "結束非張開計時中"


class StretchTracker:
    """伸展追蹤器 - 狀態機核心"""

    def __init__(self, hold_duration: float = DEFAULT_HOLD_DURATION):
        self.hold_duration = hold_duration
        self.state = TrackerState.IDLE
        self.start_gesture_type: Optional[str] = None
        self.state_enter_time: Optional[float] = None
        self.stretch_start_time: Optional[float] = None
        self.stats = StretchStats()

    def _get_elapsed_time(self) -> float:
        """取得目前狀態已經過的時間"""
        if self.state_enter_time is None:
            return 0.0
        return time.time() - self.state_enter_time

    def _enter_state(self, new_state: TrackerState):
        """進入新狀態"""
        self.state = new_state
        self.state_enter_time = time.time()

    def _reset_to_idle(self):
        """重置至閒置狀態"""
        self.state = TrackerState.IDLE
        self.start_gesture_type = None
        self.state_enter_time = None
        self.stretch_start_time = None

    def update(self, gesture: GesturePrediction) -> Optional[StretchRecord]:
        """
        更新狀態機

        Args:
            gesture: 目前的手勢預測結果

        Returns:
            StretchRecord 如果完成一次伸展，否則 None
        """
        class_id = gesture.class_id
        gesture_name = gesture.class_name
        current_time = time.time()
        elapsed = self._get_elapsed_time()

        # 狀態轉換邏輯
        if self.state == TrackerState.IDLE:
            # IDLE: 等待非張開狀態
            if class_id in NON_OPEN_GESTURES:
                self.start_gesture_type = gesture_name
                self.stretch_start_time = current_time
                self._enter_state(TrackerState.NON_OPEN_HOLDING)

        elif self.state == TrackerState.NON_OPEN_HOLDING:
            # NON_OPEN_HOLDING: 非張開狀態計時中
            if class_id in NON_OPEN_GESTURES and gesture_name == self.start_gesture_type:
                # 維持同類型非張開狀態
                if elapsed >= self.hold_duration:
                    self._enter_state(TrackerState.OPEN_WAITING)
            else:
                # 中斷（換成其他狀態）
                self._reset_to_idle()

        elif self.state == TrackerState.OPEN_WAITING:
            # OPEN_WAITING: 等待張開狀態
            if class_id == OPEN_GESTURE:
                self._enter_state(TrackerState.OPEN_HOLDING)
            elif class_id in NON_OPEN_GESTURES and gesture_name == self.start_gesture_type:
                # 維持原本的非張開狀態，繼續等待
                pass
            else:
                # 其他情況重置
                self._reset_to_idle()

        elif self.state == TrackerState.OPEN_HOLDING:
            # OPEN_HOLDING: 張開狀態計時中
            if class_id == OPEN_GESTURE:
                # 維持張開狀態
                if elapsed >= self.hold_duration:
                    self._enter_state(TrackerState.NON_OPEN_FINAL_WAITING)
            else:
                # 中斷
                self._reset_to_idle()

        elif self.state == TrackerState.NON_OPEN_FINAL_WAITING:
            # NON_OPEN_FINAL_WAITING: 等待結束非張開狀態
            if class_id == OPEN_GESTURE:
                # 維持張開狀態，繼續等待
                pass
            elif class_id in NON_OPEN_GESTURES:
                if gesture_name == self.start_gesture_type:
                    # 同類型非張開狀態
                    self._enter_state(TrackerState.NON_OPEN_FINAL_HOLDING)
                else:
                    # 不同類型（混合類型不計）
                    self._reset_to_idle()
            else:
                # 其他情況（如 idle）重置
                self._reset_to_idle()

        elif self.state == TrackerState.NON_OPEN_FINAL_HOLDING:
            # NON_OPEN_FINAL_HOLDING: 結束非張開狀態計時中
            if class_id in NON_OPEN_GESTURES and gesture_name == self.start_gesture_type:
                # 維持同類型非張開狀態
                if elapsed >= self.hold_duration:
                    # 完成一次伸展！
                    record = StretchRecord(
                        stretch_type=self.start_gesture_type,
                        start_time=self.stretch_start_time,
                        end_time=current_time
                    )
                    # 更新統計
                    self.stats.total_count += 1
                    if self.start_gesture_type in self.stats.counts_by_type:
                        self.stats.counts_by_type[self.start_gesture_type] += 1
                    self.stats.history.append(record)

                    # 重置狀態
                    self._reset_to_idle()
                    return record
            else:
                # 中斷
                self._reset_to_idle()

        return None

    def get_stats(self) -> StretchStats:
        """取得伸展統計"""
        return self.stats

    def get_state_info(self) -> Dict:
        """取得目前狀態資訊（用於 UI 顯示）"""
        return {
            "state": self.state.value,
            "state_name": self.state.name,
            "elapsed": self._get_elapsed_time(),
            "hold_duration": self.hold_duration,
            "progress": min(1.0, self._get_elapsed_time() / self.hold_duration) if self.state != TrackerState.IDLE else 0.0,
            "start_gesture": self.start_gesture_type
        }

    def reset(self):
        """重置統計"""
        self._reset_to_idle()
        self.stats = StretchStats()
