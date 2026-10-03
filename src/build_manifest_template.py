"""
Sinh train/val/test manifest CSV từ dataset ảnh tĩnh đặt tại data/raw/.

Cấu trúc thư mục RAW yêu cầu: mỗi lớp cảm xúc là 1 thư mục con, chứa các ảnh khuôn mặt
đã crop sẵn (không quan trọng định dạng tên file):

    data/raw/
    ├── happiness/*.jpg
    ├── disgust/*.jpg
    ├── repression/*.jpg
    ├── surprise/*.jpg
    ├── fear/*.jpg
    ├── sadness/*.jpg
    └── others/*.jpg

Tên thư mục con phải khớp với config.CLASS_LABELS.

Cách dùng: python build_manifest_template.py
    -> sinh ra data/processed/{train,val,test}_manifest.csv theo tỉ lệ chia mặc định
       (70/15/15), chia ngẫu nhiên có stratify theo nhãn (giữ tỉ lệ lớp đồng đều giữa
       các tập).
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLASS_LABELS, PROCESSED_DIR, RAW_DIR

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _iter_raw_images():
    """Quét data/raw/<label>/*.jpg, trả về DataFrame (image_path, label)."""
    rows = []
    for label in CLASS_LABELS:
        label_dir = os.path.join(RAW_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if filename.lower().endswith(_IMAGE_EXTENSIONS):
                rows.append({"image_path": os.path.join(label_dir, filename), "label": label})

    if not rows:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh nào trong {RAW_DIR}/<label>/. Hãy đặt ảnh theo cấu trúc "
            "mô tả ở đầu file này trước khi chạy build_manifest_template.py."
        )
    return pd.DataFrame(rows)


def build_manifests(train_ratio=0.7, val_ratio=0.15, seed=42):
    data = _iter_raw_images()

    train_df, remaining_df = train_test_split(
        data, train_size=train_ratio, random_state=seed, stratify=data["label"]
    )
    val_share_of_remaining = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        remaining_df, train_size=val_share_of_remaining, random_state=seed, stratify=remaining_df["label"]
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        output_path = os.path.join(PROCESSED_DIR, f"{name}_manifest.csv")
        df.to_csv(output_path, index=False)
        print(f"Đã ghi {len(df)} ảnh vào {output_path}")


if __name__ == "__main__":
    build_manifests()

