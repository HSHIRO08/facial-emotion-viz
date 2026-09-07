"""
Module nhận diện cảm xúc khuôn mặt bằng DeepFace, kết hợp với pipeline landmark hình học.
"""

import time
from datetime import datetime

import cv2
from deepface import DeepFace

from face_landmarks import create_face_landmarker, detect_landmarks, get_landmark_coords, draw_face_regions
from geometric_features import extract_all_features

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def detect_emotion(frame):
    """
    Chạy DeepFace để phát hiện cảm xúc trên frame hiện tại.

    Returns:
        dict {"dominant_emotion": str, "scores": {emotion: float, ...}} hoặc None nếu
        không phát hiện được khuôn mặt / có lỗi xảy ra.
    """
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
        )
        # DeepFace có thể trả về list (nhiều mặt) hoặc dict tuỳ phiên bản
        result = results[0] if isinstance(results, list) else results

        return {
            "dominant_emotion": result["dominant_emotion"],
            "scores": {emotion: result["emotion"].get(emotion, 0.0) for emotion in EMOTIONS},
        }
    except Exception:
        # Không tìm thấy khuôn mặt hoặc lỗi phân tích -> bỏ qua frame này, không crash chương trình
        return None


def _run_main():
    """Chạy webcam, mỗi ~1 giây lấy landmark + geometric features + emotion, in log ra console."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam. Kiểm tra camera đã kết nối chưa.")
        return
    start_time = time.time()
    last_run_time = 0
    interval_seconds = 1.0

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
            if landmark_coords is not None and now - last_run_time >= interval_seconds:
                last_run_time = now

                features = extract_all_features(landmark_coords)
                emotion_result = detect_emotion(frame)

                if emotion_result is not None:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    dominant = emotion_result["dominant_emotion"]
                    confidence = emotion_result["scores"][dominant]
                    ear_avg = (features["ear_left"] + features["ear_right"]) / 2
                    print(
                        f"[{timestamp}] Emotion: {dominant} ({confidence:.0f}%) | "
                        f"EAR: {ear_avg:.2f} | MAR: {features['mar']:.2f} | "
                        f"Eyebrow: {(features['eyebrow_raise_left'] + features['eyebrow_raise_right']) / 2:.2f} | "
                        f"MouthAngle: {features['mouth_corner_angle']:.1f}°"
                    )

            cv2.imshow("Emotion Detector Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_main()
