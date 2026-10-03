# facial-emotion-viz — Facial Expression Recognition (ảnh tĩnh, 2D-CNN / Transfer Learning)

Dự án nhận dạng **biểu cảm khuôn mặt** từ ảnh tĩnh (macro-expression: happiness, disgust,
repression, surprise, fear, sadness), sử dụng model **2D-CNN** hoặc **backbone pretrain
(ResNet18/MobileNetV2) qua Transfer Learning** huấn luyện trên tập ảnh khuôn mặt đã crop sẵn.

> Lưu ý: hướng ban đầu của dự án là Micro-Expression Recognition theo chuỗi khung hình
> (3D-CNN + LSTM), nhưng dữ liệu thực tế hiện có (`data/raw/<label>/*.jpg`) là các ảnh tĩnh
> độc lập, không có thông tin clip/thứ tự thời gian — không đủ để huấn luyện model
> spatio-temporal. Dự án đã chuyển sang hướng macro-expression (ảnh tĩnh) để phù hợp với
> dữ liệu đang có. Nếu sau này có dataset video micro-expression chuẩn (CASME II/SAMM/SMIC,
> giữ nguyên thứ tự frame theo từng clip), có thể khôi phục hướng spatio-temporal.

## Cấu trúc thư mục

```
facial-emotion-viz/
├── src/
│   ├── config.py                  # Đường dẫn, nhãn, hyperparameter dùng chung
│   ├── face_landmarks.py          # Phát hiện landmark + crop khuôn mặt (MediaPipe Tasks API)
│   ├── dataset.py                 # Dataset đọc manifest CSV -> ảnh khuôn mặt tĩnh
│   ├── model.py                   # Kiến trúc 2D-CNN (4 khối Conv+BN+ReLU+MaxPool) + classifier
│   ├── train.py                   # Huấn luyện model (early stopping, checkpoint tốt nhất)
│   ├── evaluate.py                # Đánh giá: accuracy, F1 macro, confusion matrix
│   ├── realtime_inference.py      # Suy luận real-time qua webcam (phân loại từng khung hình)
│   ├── build_manifest_template.py # Quét data/raw/<label>/*.jpg -> sinh train/val/test manifest
│   └── models/face_landmarker.task
├── data/
│   ├── raw/<label>/*.jpg          # Ảnh khuôn mặt gốc, 1 thư mục / nhãn cảm xúc
│   └── processed/                 # train_manifest.csv, val_manifest.csv, test_manifest.csv
├── checkpoints/                   # Model đã huấn luyện (best_model.pt) + confusion_matrix.png
├── notebooks/                     # Notebook phân tích/EDA
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Lưu ý: dùng Python 3.10–3.12 (một số phiên bản mediapipe/torch chưa hỗ trợ Python 3.13+
> trên Windows tại thời điểm viết). Cài PyTorch bản GPU (CUDA) riêng theo hướng dẫn tại
> https://pytorch.org/get-started/locally/ nếu có GPU, để tăng tốc huấn luyện.

## Quy trình

### 1. Chuẩn bị dataset

Đặt ảnh khuôn mặt đã crop sẵn vào `data/raw/<label>/`, với `<label>` khớp đúng tên trong
`config.CLASS_LABELS` (disgust, happiness, repression, surprise, fear, sadness).
Tên file ảnh không quan trọng. Sau đó chạy:

```bash
python src/build_manifest_template.py
```

để tự động chia train/val/test (70/15/15, stratify theo nhãn) và sinh 3 file manifest
trong `data/processed/`.

### 2. Huấn luyện

```bash
cd src
python train.py
```

Model tốt nhất (val_loss thấp nhất) được lưu tại `checkpoints/best_model.pt`, có early
stopping sau `EARLY_STOPPING_PATIENCE` epoch không cải thiện (cấu hình trong `config.py`).

### 3. Đánh giá

```bash
python evaluate.py --checkpoint ../checkpoints/best_model.pt
```

In ra precision/recall/F1 (macro) theo từng lớp và lưu confusion matrix (`.png`) cạnh
checkpoint.

### 4. Suy luận real-time qua webcam

```bash
python realtime_inference.py --checkpoint ../checkpoints/best_model.pt
```

Mỗi khung hình webcam được crop khuôn mặt (qua MediaPipe FaceLandmarker) rồi đưa vào
model để phân loại. Kết quả (nhãn, bounding box, biểu đồ xác suất các lớp) được làm mượt
bằng EMA để chống nhảy nhãn giữa các frame. Cửa sổ hiển thị phóng to x2 (`DISPLAY_SCALE`
trong `realtime_inference.py`), có nút **Start/Pause** (bấm chuột ở góc dưới trái) để tạm
dừng/tiếp tục xử lý video. Nhấn `q` để thoát, `s` để bật/tắt ghi lịch sử cảm xúc ra
`checkpoints/emotion_history.csv`.

## Kiến trúc model

Cấu hình qua `config.USE_TRANSFER_LEARNING`:

- **`True` (mặc định, khuyến nghị cho dataset nhỏ):** dùng backbone `BACKBONE_NAME`
  (`resnet18` hoặc `mobilenet_v2`) đã pretrain trên ImageNet, thay lớp FC cuối bằng
  classifier `NUM_CLASSES` lớp. `FREEZE_BACKBONE = True` nghĩa là chỉ huấn luyện lớp
  classifier cuối (fine-tune nhanh, ít overfit); đặt `False` để fine-tune toàn bộ backbone.
  Ảnh được chuẩn hoá theo thống kê ImageNet (`IMAGENET_MEAN`/`IMAGENET_STD`) để khớp với
  phân phối dữ liệu backbone đã học.
- **`False`:** dùng `FacialExpressionCNN` tự thiết kế, học from scratch:

```
Ảnh khuôn mặt (B, C, H, W)
        │
        ▼
  4 khối Conv2D + BatchNorm + ReLU + MaxPool (32 -> 64 -> 128 -> 256 kênh)
        │
        ▼
  Global Average Pooling
        │
        ▼
  Fully-Connected Classifier -> logits (B, NUM_CLASSES)
```

