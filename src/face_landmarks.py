"""
Module phát hiện landmark khuôn mặt bằng MediaPipe (Tasks API - FaceLandmarker).

LƯU Ý: mediapipe >= 1.0 đã bỏ API cũ `mp.solutions.face_mesh`, thay bằng
`mediapipe.tasks.python.vision.FaceLandmarker` dùng file model .task (đã tải sẵn
trong src/models/face_landmarker.task).

Mục tiêu bước này: chỉ phát hiện + vẽ landmark theo 3 vùng (lông mày, mắt, môi).
Việc tính chỉ số cảm xúc (EAR, MAR...) sẽ làm ở bước sau (geometric_features.py).
"""

import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarksConnections

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

# Bộ chỉ số landmark chuẩn của MediaPipe cho từng vùng khuôn mặt (Tasks API).
LEFT_EYEBROW = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
RIGHT_EYEBROW = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
LEFT_EYE = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
RIGHT_EYE = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
LIPS = FaceLandmarksConnections.FACE_LANDMARKS_LIPS

# Màu vẽ theo chuẩn BGR của OpenCV
COLOR_EYEBROW = (0, 255, 255)   # vàng
COLOR_EYE = (255, 0, 0)         # xanh dương
COLOR_LIPS = (0, 0, 255)        # đỏ


def _connections_to_indices(connections):
    """Chuyển 1 tập các cạnh (Connection.start, Connection.end) thành tập chỉ số điểm duy nhất."""
    indices = set()
    for connection in connections:
        indices.add(connection.start)
        indices.add(connection.end)
    return indices


# Gom sẵn các tập chỉ số điểm theo từng vùng, kèm màu tương ứng, để dùng khi vẽ.
REGION_INDICES = [
    (_connections_to_indices(LEFT_EYEBROW) | _connections_to_indices(RIGHT_EYEBROW), COLOR_EYEBROW),
    (_connections_to_indices(LEFT_EYE) | _connections_to_indices(RIGHT_EYE), COLOR_EYE),
    (_connections_to_indices(LIPS), COLOR_LIPS),
]


def create_face_landmarker():
    """Tạo đối tượng FaceLandmarker (dùng như context manager, hỗ trợ chế độ VIDEO)."""
    # Đọc model qua buffer thay vì model_asset_path vì mediapipe (native C++) không đọc được
    # đường dẫn chứa ký tự Unicode (ví dụ thư mục "Máy tính") trên Windows.
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(landmarker, frame, timestamp_ms):
    """
    Chạy FaceLandmarker trên 1 frame OpenCV (BGR).

    Returns:
        list các NormalizedLandmark (mỗi phần tử có .x, .y chuẩn hoá 0..1) của khuôn mặt đầu
        tiên, hoặc None nếu không phát hiện được khuôn mặt.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def get_landmark_coords(face_landmarks, frame_shape):
    """
    Chuyển toạ độ landmark chuẩn hoá (0..1) sang toạ độ pixel thực tế.

    Args:
        face_landmarks: list các NormalizedLandmark trả về từ detect_landmarks()
        frame_shape: shape của frame OpenCV, dạng (height, width, channels)

    Returns:
        dict: {index_landmark: (x_pixel, y_pixel)}
    """
    height, width = frame_shape[0], frame_shape[1]
    coords = {}
    for idx, landmark in enumerate(face_landmarks):
        x_pixel = int(landmark.x * width)
        y_pixel = int(landmark.y * height)
        coords[idx] = (x_pixel, y_pixel)
    return coords


def draw_face_regions(frame, landmark_coords):
    """Vẽ các điểm landmark lên frame, tô màu riêng theo từng vùng (lông mày/mắt/môi)."""
    for indices, color in REGION_INDICES:
        for idx in indices:
            point = landmark_coords.get(idx)
            if point is not None:
                cv2.circle(frame, point, radius=1, color=color, thickness=-1)
    return frame


def run_face_mesh_webcam(camera_index=0):
    """Mở webcam, chạy FaceLandmarker real-time, vẽ landmark theo vùng. Nhấn 'q' để thoát."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Không thể mở webcam (index={camera_index}). Kiểm tra camera đã kết nối chưa.")
        return
    start_time = time.time()

    with create_face_landmarker() as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            timestamp_ms = int((time.time() - start_time) * 1000)
            landmarks = detect_landmarks(landmarker, frame, timestamp_ms)

            if landmarks is not None:
                landmark_coords = get_landmark_coords(landmarks, frame.shape)
                draw_face_regions(frame, landmark_coords)

            cv2.imshow("Face Mesh - Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_mesh_webcam()
