"""
手部復健偵測程式 - 主程式
即時偵測手部伸展動作並統計次數與類型
"""

import sys
import cv2

from config import DEFAULT_MODEL_PATH
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier, GestureSmoother
from stretch_tracker import StretchTracker
from ui_renderer import UIRenderer


def main():
    """主程式進入點"""
    print("=" * 50)
    print("手部復健偵測程式")
    print("=" * 50)
    print("按 'q' 退出程式")
    print("按 'r' 重置統計")
    print("=" * 50)

    # 初始化模組
    print("\n正在初始化...")

    # 1. 載入手勢分類模型
    try:
        print(f"  載入模型: {DEFAULT_MODEL_PATH}")
        classifier = GestureClassifier(DEFAULT_MODEL_PATH)
        print("  模型載入成功!")
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        print("請確認模型檔案存在且格式正確。")
        sys.exit(1)

    # 2. 初始化手部偵測器
    print("  初始化手部偵測器...")
    detector = HandDetector()

    # 3. 初始化其他模組
    smoother = GestureSmoother()
    tracker = StretchTracker()
    renderer = UIRenderer()

    print("  初始化完成!")

    # 開啟攝影機
    print("\n正在開啟攝影機...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("錯誤: 無法開啟攝影機")
        print("請確認攝影機已連接且未被其他程式佔用。")
        detector.close()
        sys.exit(1)

    print("攝影機已開啟!")
    print("\n開始偵測...\n")

    try:
        while True:
            # 讀取影像
            ret, frame = cap.read()
            if not ret:
                print("警告: 無法讀取攝影機畫面")
                break

            # 水平翻轉（鏡像效果，更直覺）
            frame = cv2.flip(frame, 1)

            # 偵測手部
            hand_result = detector.detect(frame)

            # 分類手勢
            gesture = None
            if hand_result is not None:
                raw_prediction = classifier.predict(hand_result.landmarks)
                gesture = smoother.smooth(raw_prediction)

                # 更新狀態機
                stretch_record = tracker.update(gesture)
                if stretch_record:
                    print(f"完成伸展! 類型: {stretch_record.stretch_type}, "
                          f"總次數: {tracker.get_stats().total_count}")

            # 渲染 UI
            display_frame = renderer.render(
                frame,
                hand_result,
                gesture,
                tracker.get_stats(),
                tracker.get_state_info()
            )

            # 顯示畫面
            cv2.imshow("Hand Rehab Detection", display_frame)

            # 處理按鍵
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n使用者退出程式")
                break
            elif key == ord('r'):
                tracker.reset()
                smoother.reset()
                print("\n統計已重置!")

    except KeyboardInterrupt:
        print("\n程式被中斷")

    finally:
        # 釋放資源
        print("\n正在釋放資源...")
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print("程式結束")

        # 顯示最終統計
        stats = tracker.get_stats()
        print("\n" + "=" * 50)
        print("最終統計")
        print("=" * 50)
        print(f"總伸展次數: {stats.total_count}")
        print("各類型次數:")
        for stretch_type, count in stats.counts_by_type.items():
            if count > 0:
                print(f"  {stretch_type}: {count}")
        print("=" * 50)


if __name__ == "__main__":
    main()
