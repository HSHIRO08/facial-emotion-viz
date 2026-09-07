"""
Module tính các chỉ số hình học (geometric features) từ landmark khuôn mặt MediaPipe.

Toạ độ đầu vào là dict {index: (x, y)} lấy từ face_landmarks.get_landmark_coords().

LƯU Ý: các chỉ số landmark MediaPipe dưới đây theo hệ quy chiếu ảnh (đã mirror qua webcam),
nếu nghi ngờ sai vùng, hãy chạy script debug vẽ số landmark lên ảnh để kiểm tra lại bằng mắt.
"""

import math
from datetime import datetime
import time

import cv2

from face_landmarks import create_face_landmarker, detect_landmarks, get_landmark_coords, draw_face_regions

# --- Bộ chỉ số landmark chuẩn dùng để tính EAR (Eye Aspect Ratio) ---
# Thứ tự: [p1 = khoé mắt ngoài, p2, p3 = mí trên, p4 = khoé mắt trong, p5, p6 = mí dưới]
EYE_EAR_INDICES = {
    "left": [362, 385, 387, 263, 373, 380],
    "right": [33, 160, 158, 133, 153, 144],
}

# Điểm đại diện cho tâm lông mày và tâm mắt (mí trên), dùng để đo độ nhướn lông mày
EYEBROW_INDICES = {
    "left": {"eyebrow_center": 285, "eye_top": 386},
    "right": {"eyebrow_center": 55, "eye_top": 159},
}

# Điểm khoé mắt ngoài trái/phải, dùng làm khoảng cách chuẩn hoá (interocular distance)
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33

# Điểm môi: khoé miệng trái/phải, tâm môi trên/dưới (đường viền trong)
MOUTH_LEFT_CORNER = 291
MOUTH_RIGHT_CORNER = 61
MOUTH_TOP_CENTER = 13
MOUTH_BOTTOM_CENTER = 14


def _euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _interocular_distance(landmarks):
    """Khoảng cách giữa 2 khoé mắt ngoài, dùng để chuẩn hoá các chỉ số theo kích thước khuôn mặt."""
    return _euclidean(landmarks[LEFT_EYE_OUTER], landmarks[RIGHT_EYE_OUTER])


def calculate_ear(landmarks, eye_side):
    """
    Eye Aspect Ratio (EAR) = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    EAR càng nhỏ = mắt càng nhắm, EAR càng lớn = mắt càng mở to.
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[idx] for idx in EYE_EAR_INDICES[eye_side]]
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = 2.0 * _euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def calculate_mar(landmarks):
    """
    Mouth Aspect Ratio (MAR) = khoảng cách môi trên-dưới / khoảng cách khoé miệng trái-phải.
    MAR càng lớn = miệng mở càng rộng.
    """
    vertical = _euclidean(landmarks[MOUTH_TOP_CENTER], landmarks[MOUTH_BOTTOM_CENTER])
    horizontal = _euclidean(landmarks[MOUTH_LEFT_CORNER], landmarks[MOUTH_RIGHT_CORNER])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def calculate_eyebrow_raise(landmarks, eye_side):
    """
    Khoảng cách giữa tâm lông mày và mí mắt trên tương ứng, đã chuẩn hoá theo interocular distance.
    Số càng lớn = lông mày nhướn càng cao.
    """
    idx = EYEBROW_INDICES[eye_side]
    distance = _euclidean(landmarks[idx["eyebrow_center"]], landmarks[idx["eye_top"]])
    interocular = _interocular_distance(landmarks)
    if interocular == 0:
        return 0.0
    return distance / interocular


def calculate_mouth_corner_angle(landmarks):
    """
    Góc lệch (độ) của khóe miệng so với đường ngang qua tâm miệng.
    Dương = khóe miệng nhếch lên (cười), âm = khóe miệng hạ xuống (buồn).
    Lấy trung bình góc của 2 bên khoé miệng so với điểm tâm môi.
    """
    mouth_center = (
        (landmarks[MOUTH_TOP_CENTER][0] + landmarks[MOUTH_BOTTOM_CENTER][0]) / 2,
        (landmarks[MOUTH_TOP_CENTER][1] + landmarks[MOUTH_BOTTOM_CENTER][1]) / 2,
    )

    angles = []
    for corner_idx in (MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER):
        corner = landmarks[corner_idx]
        dx = abs(corner[0] - mouth_center[0])
        # Trục y ảnh hướng xuống dưới, nên khoé miệng cao hơn tâm (y nhỏ hơn) => dy dương => cười
        dy = mouth_center[1] - corner[1]
        if dx == 0:
            continue
        angles.append(math.degrees(math.atan2(dy, dx)))

    if not angles:
        return 0.0
    return sum(angles) / len(angles)


def extract_all_features(landmarks):
    """Trả về dict đầy đủ 4 nhóm chỉ số hình học, kèm chuẩn hoá theo interocular distance."""
    return {
        "ear_left": calculate_ear(landmarks, "left"),
        "ear_right": calculate_ear(landmarks, "right"),
        "mar": calculate_mar(landmarks),
        "eyebrow_raise_left": calculate_eyebrow_raise(landmarks, "left"),
        "eyebrow_raise_right": calculate_eyebrow_raise(landmarks, "right"),
        "mouth_corner_angle": calculate_mouth_corner_angle(landmarks),
    }


def _run_test():
    """Chạy webcam, in ra console 4 chỉ số hình học mỗi giây để kiểm tra logic tính toán."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam. Kiểm tra camera đã kết nối chưa.")
        return
    start_time = time.time()
    last_print_time = 0
    interval_seconds = 1.0

    with create_face_landmarker() as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            timestamp_ms = int((time.time() - start_time) * 1000)
            face_landmarks = detect_landmarks(landmarker, frame, timestamp_ms)

            if face_landmarks is not None:
                landmark_coords = get_landmark_coords(face_landmarks, frame.shape)
                draw_face_regions(frame, landmark_coords)

                now = time.time()
                if now - last_print_time >= interval_seconds:
                    last_print_time = now
                    features = extract_all_features(landmark_coords)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{timestamp}] EAR(L/R): {features['ear_left']:.2f}/{features['ear_right']:.2f} | "
                        f"MAR: {features['mar']:.2f} | "
                        f"Eyebrow(L/R): {features['eyebrow_raise_left']:.2f}/{features['eyebrow_raise_right']:.2f} | "
                        f"MouthAngle: {features['mouth_corner_angle']:.1f}°"
                    )

            cv2.imshow("Geometric Features Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_test()
