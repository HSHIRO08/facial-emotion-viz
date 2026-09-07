"""
Script chạy chính của toàn bộ hệ thống: webcam -> landmark + geometric features + emotion
-> ghi log ra data/emotion_log.csv, đồng thời hiển thị video real-time có overlay cảm xúc.

Nhấn 'q' hoặc Ctrl+C để thoát sạch sẽ (đóng camera, đóng file).
"""

import csv
import os
import time
from datetime import datetime

import cv2

from face_landmarks import create_face_landmarker, detect_landmarks, get_landmark_coords, draw_face_regions
from geometric_features import extract_all_features
from emotion_detector import detect_emotion, EMOTIONS

DETECT_INTERVAL = 1.0  # giây giữa mỗi lần ghi log

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "emotion_log.csv")
CSV_HEADER = (
    ["timestamp", "dominant_emotion"]
    + EMOTIONS
    + [
        "ear_left",
        "ear_right",
        "mar",
        "eyebrow_raise_left",
        "eyebrow_raise_right",
        "mouth_corner_angle",
    ]
)


def _ensure_csv_header(path):
    """Tạo file CSV kèm header nếu chưa tồn tại, giữ nguyên nếu đã có (để append)."""
    file_exists = os.path.isfile(path)
    if not file_exists:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def _append_log_row(path, timestamp, emotion_result, features):
    row = (
        [timestamp, emotion_result["dominant_emotion"]]
        + [emotion_result["scores"][emotion] for emotion in EMOTIONS]
        + [
            features["ear_left"],
            features["ear_right"],
            features["mar"],
            features["eyebrow_raise_left"],
            features["eyebrow_raise_right"],
            features["mouth_corner_angle"],
        ]
    )
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    _ensure_csv_header(CSV_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam. Kiểm tra camera đã kết nối chưa.")
        return
    start_time = time.time()
    last_run_time = 0
    last_dominant_emotion = None

    try:
        with create_face_landmarker() as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                timestamp_ms = int((time.time() - start_time) * 1000)
                face_landmarks = detect_landmarks(landmarker, frame, timestamp_ms)

                landmark_coords = None
                if face_landmarks is not None:
                    landmark_coords = get_landmark_coords(face_landmarks, frame.shape)
                    draw_face_regions(frame, landmark_coords)

                now = time.time()
                if landmark_coords is not None and now - last_run_time >= DETECT_INTERVAL:
                    last_run_time = now

                    features = extract_all_features(landmark_coords)
                    emotion_result = detect_emotion(frame)

                    if emotion_result is not None:
                        last_dominant_emotion = emotion_result["dominant_emotion"]
                        timestamp = datetime.now().isoformat(timespec="seconds")
                        _append_log_row(CSV_PATH, timestamp, emotion_result, features)

                # Overlay text hiển thị cảm xúc gần nhất lên video real-time
                if last_dominant_emotion:
                    cv2.putText(
                        frame,
                        f"Emotion: {last_dominant_emotion}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Facial Emotion Logger", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
