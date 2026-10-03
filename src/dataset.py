"""
Dataset cho bài toán Macro-Expression Recognition dựa trên ảnh khuôn mặt tĩnh.

Dataset được mô tả qua file "manifest" CSV với các cột:
    image_path  : đường dẫn tới 1 ảnh khuôn mặt đã crop sẵn
    label       : nhãn cảm xúc (phải nằm trong config.CLASS_LABELS)

Xem build_manifest_template.py để biết cách tạo manifest từ data/raw/<label>/*.jpg.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CLASS_LABELS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LABEL_TO_IDX, USE_TRANSFER_LEARNING


def _imread_unicode(path):
    """cv2.imread trả None trên Windows nếu đường dẫn chứa ký tự Unicode (vd "Máy tính").

    Đọc file qua np.fromfile + cv2.imdecode để tránh giới hạn này.
    """
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _augment_image(image):
    """Random flip + xoay nhẹ + đổi độ sáng, tăng đa dạng dữ liệu cho các lớp ít ảnh."""
    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

    if np.random.rand() < 0.5:
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    return image



class FacialExpressionDataset(Dataset):
    """Đọc manifest CSV, trả về (tensor ảnh, nhãn) cho mỗi ảnh khuôn mặt."""

    def __init__(self, manifest_path, image_size=IMAGE_SIZE, augment=False):
        self.manifest = pd.read_csv(manifest_path)
        required_columns = {"image_path", "label"}
        missing = required_columns - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest thiếu cột bắt buộc: {missing}")

        unknown_labels = set(self.manifest["label"].unique()) - set(CLASS_LABELS)
        if unknown_labels:
            raise ValueError(f"Manifest chứa nhãn không nằm trong config.CLASS_LABELS: {unknown_labels}")

        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        image_path = row["image_path"]

        image = _imread_unicode(image_path)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.augment:
            image = _augment_image(image)

        image = image.astype(np.float32) / 255.0
        if USE_TRANSFER_LEARNING:
            # Chuẩn hoá theo thống kê ImageNet để khớp với backbone pretrain (model.py).
            image = (image - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        else:
            image = (image - 0.5) / 0.5  # chuẩn hoá về [-1, 1]
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()  # (C, H, W)

        label_idx = LABEL_TO_IDX[row["label"]]
        return image, label_idx


def build_dataloader(manifest_path, batch_size, shuffle, num_workers, augment=False):
    from torch.utils.data import DataLoader

    dataset = FacialExpressionDataset(manifest_path, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

