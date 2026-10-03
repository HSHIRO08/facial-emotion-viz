"""
Cấu hình dùng chung cho pipeline nhận dạng Macro-Expression (ảnh tĩnh, 2D-CNN).

Tất cả path, hằng số kiến trúc model và hyperparameter tập trung ở đây để
dataset.py / model.py / train.py / evaluate.py / realtime_inference.py cùng dùng.
"""

import os

import torch

# --- Đường dẫn ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
FACE_LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

# File manifest CSV mô tả dataset, xem cấu trúc trong build_manifest_template.py / README.
TRAIN_MANIFEST = os.path.join(PROCESSED_DIR, "train_manifest.csv")
VAL_MANIFEST = os.path.join(PROCESSED_DIR, "val_manifest.csv")
TEST_MANIFEST = os.path.join(PROCESSED_DIR, "test_manifest.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Nhãn cảm xúc (khớp tên các thư mục con trong data/raw/) ---
CLASS_LABELS = ["disgust", "happiness", "repression", "surprise", "fear", "sadness"]
NUM_CLASSES = len(CLASS_LABELS)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

# --- Tham số ảnh đầu vào ---
IMAGE_SIZE = 112  # kích thước ảnh khuôn mặt vuông sau khi resize (H = W = IMAGE_SIZE)
NUM_CHANNELS = 3

# --- Transfer Learning (TL) ---
# Dataset hiện tại chỉ ~7.5k ảnh train / 6 lớp, không nhiều cho CNN học từ đầu (from scratch).
# Dùng backbone đã pretrain trên ImageNet (torchvision) thường hội tụ nhanh hơn và cho
# accuracy cao hơn với lượng dữ liệu nhỏ như vậy.
USE_TRANSFER_LEARNING = True
BACKBONE_NAME = "resnet18"  # resnet18 | mobilenet_v2
FREEZE_BACKBONE = True  # True: chỉ train lớp classifier cuối (fine-tune nhanh, ít overfit)
# Chuẩn hoá ảnh theo thống kê ImageNet khi dùng backbone pretrain, bắt buộc để khớp với
# phân phối dữ liệu mà backbone đã học.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Hyperparameter huấn luyện ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tham số suy luận real-time ---
REALTIME_CONFIDENCE_THRESHOLD = 0.5
# Crop khuôn mặt sát hơn (ít margin) để tránh lẫn cổ/nền vào ảnh đưa vào model.
# Margin không đối xứng: trán cần nhiều hơn 1 chút, cằm/cổ cần ít để không "ôm" quá sâu.
FACE_BBOX_TOP_MARGIN_RATIO = 0.12
FACE_BBOX_BOTTOM_MARGIN_RATIO = 0.03
FACE_BBOX_SIDE_MARGIN_RATIO = 0.08
# Smoothing theo thời gian (temporal filtering) để chống nhảy nhãn (flickering) giữa các
# frame liên tiếp: EMA (exponential moving average) trên xác suất từng lớp và trên bbox.
PROB_SMOOTHING_ALPHA = 0.3  # càng nhỏ càng mượt (ít nhạy với thay đổi tức thời)
BBOX_SMOOTHING_ALPHA = 0.4
EMOTION_HISTORY_CSV = os.path.join(CHECKPOINT_DIR, "emotion_history.csv")

