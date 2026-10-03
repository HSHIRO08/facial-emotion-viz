"""
Suy luận real-time: nhận dạng biểu cảm khuôn mặt từ webcam, mỗi khung hình được crop
khuôn mặt (qua MediaPipe FaceLandmarker) rồi đưa vào model để phân loại.

Để chống hiện tượng nhảy nhãn (flickering) giữa các frame liên tiếp, xác suất từng lớp
và toạ độ bounding box đều được làm mượt bằng EMA (exponential moving average) thay vì
dùng trực tiếp kết quả dự đoán thô của từng frame.

Nhấn 'q' để thoát, 's' để bật/tắt ghi lịch sử cảm xúc ra CSV (EMOTION_HISTORY_CSV).
Cửa sổ hiển thị tự co vừa màn hình (MAX_DISPLAY_*), có nút Start/Pause (bấm chuột) để tạm dừng/
tiếp tục xử lý video mà không cần đóng ứng dụng.
"""

import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np
import torch

from config import (
    BBOX_SMOOTHING_ALPHA,
    CHECKPOINT_DIR,
    CLASS_LABELS,
    DEVICE,
    EMOTION_HISTORY_CSV,
    IDX_TO_LABEL,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROB_SMOOTHING_ALPHA,
    REALTIME_CONFIDENCE_THRESHOLD,
    USE_TRANSFER_LEARNING,
)
from face_landmarks import create_face_landmarker, detect_landmarks, get_face_bbox, get_landmark_coords
from model import build_model

BAR_CHART_COLOR = (255, 180, 0)
BAR_CHART_WIDTH = 220
BAR_CHART_ROW_HEIGHT = 26

WINDOW_NAME = "Facial Expression Recognition (Real-time)"
MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT = 1100, 650  # giới hạn kích thước cửa sổ để vừa màn hình
BUTTON_WIDTH, BUTTON_HEIGHT = 110, 44
BUTTON_MARGIN = 16


class _UIState:
    """Trạng thái UI chia sẻ giữa vòng lặp chính và callback chuột (nút Start/Pause)."""

    def __init__(self):
        self.paused = False


def _button_rects(display_width, display_height):
    """Toạ độ 2 nút Start/Pause (x1, y1, x2, y2) trên frame đã phóng to, góc dưới trái."""
    y2 = display_height - BUTTON_MARGIN
    y1 = y2 - BUTTON_HEIGHT
    start_x1 = BUTTON_MARGIN
    start_x2 = start_x1 + BUTTON_WIDTH
    pause_x1 = start_x2 + BUTTON_MARGIN
    pause_x2 = pause_x1 + BUTTON_WIDTH
    return {
        "start": (start_x1, y1, start_x2, y2),
        "pause": (pause_x1, y1, pause_x2, y2),
    }


def _on_mouse(event, x, y, flags, state):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    rects = state["rects"]
    ui_state = state["ui_state"]

    def _inside(rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    if _inside(rects["start"]):
        ui_state.paused = False
    elif _inside(rects["pause"]):
        ui_state.paused = True


def _draw_buttons(display_frame, rects, paused):
    start_color = (0, 180, 0) if paused else (0, 100, 0)
    pause_color = (0, 140, 230) if not paused else (0, 70, 120)

    for key, color, text in (("start", start_color, "Start"), ("pause", pause_color, "Pause")):
        x1, y1, x2, y2 = rects[key]
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (BUTTON_WIDTH - text_size[0]) // 2
        text_y = y1 + (BUTTON_HEIGHT + text_size[1]) // 2
        cv2.putText(display_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if paused:
        cv2.putText(
            display_frame, "PAUSED", (BUTTON_MARGIN, rects["start"][1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 230), 2,
        )


def _preprocess_face_crop(frame, bbox):
    x1, y1, x2, y2 = bbox
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
    face = face.astype(np.float32) / 255.0
    # Phải khớp với chuẩn hoá dùng khi train (xem dataset.py).
    if USE_TRANSFER_LEARNING:
        face = (face - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    else:
        face = (face - 0.5) / 0.5
    return face


def _frame_to_tensor(face):
    tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, C, H, W)
    return tensor


def _ema_update(previous, current, alpha):
    """Exponential moving average: previous=None nghĩa là khởi tạo trực tiếp bằng current."""
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous


def _draw_probability_bar_chart(frame, class_probs):
    """Vẽ side panel nhỏ hiển thị % của tất cả các lớp cảm xúc, sắp xếp giảm dần."""
    height, width = frame.shape[0], frame.shape[1]
    panel_x = width - BAR_CHART_WIDTH - 10
    panel_y = 10
    order = np.argsort(class_probs)[::-1]

    overlay = frame.copy()
    panel_h = BAR_CHART_ROW_HEIGHT * len(CLASS_LABELS) + 10
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + BAR_CHART_WIDTH, panel_y + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for row, class_idx in enumerate(order):
        label = CLASS_LABELS[class_idx]
        prob = float(class_probs[class_idx])
        y = panel_y + 10 + row * BAR_CHART_ROW_HEIGHT
        bar_max_width = BAR_CHART_WIDTH - 100
        bar_width = int(bar_max_width * prob)
        cv2.rectangle(frame, (panel_x + 85, y), (panel_x + 85 + bar_width, y + 16), BAR_CHART_COLOR, -1)
        cv2.putText(frame, label[:8], (panel_x + 5, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(
            frame, f"{prob:.0%}", (panel_x + 85 + bar_max_width + 5, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (255, 255, 255), 1,
        )


def _append_history_row(label, confidence):
    file_exists = os.path.isfile(EMOTION_HISTORY_CSV)
    with open(EMOTION_HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "label", "confidence"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), label, f"{confidence:.4f}"])


def run(checkpoint_path, camera_index=0, log_history=False):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại {checkpoint_path}. Hãy chạy train.py trước, "
            "hoặc truyền --checkpoint tới file .pt khác."
        )

    model = build_model().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Không thể mở webcam (index={camera_index}). Kiểm tra camera đã kết nối chưa.")
        return

    capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    display_scale = min(MAX_DISPLAY_WIDTH / capture_width, MAX_DISPLAY_HEIGHT / capture_height)
    display_width = int(capture_width * display_scale)
    display_height = int(capture_height * display_scale)
    button_rects = _button_rects(display_width, display_height)

    ui_state = _UIState()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse, {"rects": button_rects, "ui_state": ui_state})

    smoothed_probs = None
    smoothed_bbox = None
    last_logged_label = None
    last_rendered_frame = None

    try:
        with create_face_landmarker() as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                if ui_state.paused:
                    # Vẫn đọc frame để không làm nghẽn buffer camera, nhưng không xử lý/cập nhật kết quả.
                    display_frame = (
                        cv2.resize(last_rendered_frame, (display_width, display_height))
                        if last_rendered_frame is not None
                        else cv2.resize(frame, (display_width, display_height))
                    )
                    _draw_buttons(display_frame, button_rects, ui_state.paused)
                    cv2.imshow(WINDOW_NAME, display_frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord("q"):
                        break
                    continue

                timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                face_landmarks = detect_landmarks(landmarker, frame, timestamp_ms)

                if face_landmarks is not None:
                    landmark_coords = get_landmark_coords(face_landmarks, frame.shape)
                    raw_bbox = np.array(get_face_bbox(landmark_coords, frame.shape), dtype=np.float32)
                    smoothed_bbox = _ema_update(smoothed_bbox, raw_bbox, BBOX_SMOOTHING_ALPHA)
                    bbox = tuple(int(v) for v in smoothed_bbox)

                    face_crop = _preprocess_face_crop(frame, bbox)

                    if face_crop is not None:
                        image_tensor = _frame_to_tensor(face_crop).to(DEVICE)
                        with torch.no_grad():
                            logits = model(image_tensor)
                            raw_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        smoothed_probs = _ema_update(smoothed_probs, raw_probs, PROB_SMOOTHING_ALPHA)
                else:
                    # Mất mặt: reset trạng thái smoothing để không "kéo dài" dự đoán cũ sai lệch.
                    smoothed_probs = None
                    smoothed_bbox = None

                last_rendered_frame = frame
                display_frame = cv2.resize(frame, (display_width, display_height))
                
                # Vẽ bbox nếu có khuôn mặt được phát hiện
                if smoothed_bbox is not None:
                    scaled_bbox = tuple(int(v * display_scale) for v in smoothed_bbox)
                    cv2.rectangle(display_frame, (scaled_bbox[0], scaled_bbox[1]), (scaled_bbox[2], scaled_bbox[3]), (0, 255, 0), 2)
                
                # Vẽ emotion text nếu có dự đoán
                if smoothed_probs is not None:
                    pred_idx = int(np.argmax(smoothed_probs))
                    label = IDX_TO_LABEL[pred_idx]
                    confidence = float(smoothed_probs[pred_idx])
                    _draw_probability_bar_chart(display_frame, smoothed_probs)
                    if confidence >= REALTIME_CONFIDENCE_THRESHOLD:
                        cv2.putText(
                            display_frame, f"Emotion: {label} ({confidence:.0%})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        )
                        if log_history and label != last_logged_label:
                            _append_history_row(label, confidence)
                            last_logged_label = label

                _draw_buttons(display_frame, button_rects, ui_state.paused)
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    log_history = not log_history
                    print(f"Ghi lịch sử cảm xúc: {'BẬT' if log_history else 'TẮT'} -> {EMOTION_HISTORY_CSV}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best_model.pt"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--log-history", action="store_true", help="Ghi lịch sử cảm xúc ra CSV ngay từ đầu")
    args = parser.parse_args()

    run(args.checkpoint, args.camera, args.log_history)


