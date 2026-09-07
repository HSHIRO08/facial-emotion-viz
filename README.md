# facial-emotion-viz

Dự án thu thập dữ liệu cảm xúc khuôn mặt qua webcam, kết hợp landmark hình học
(lông mày, mắt, môi) với model nhận diện cảm xúc (DeepFace), sau đó trực quan hóa
và phân tích dữ liệu thu được.

## Cấu trúc thư mục

```
facial-emotion-viz/
├── src/            # Code chính (face landmarks, geometric features, emotion detector, logger)
├── data/           # File CSV log dữ liệu thu thập từ webcam
├── notebooks/      # Notebook phân tích EDA & trực quan hóa
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Quy trình

1. Phát hiện landmark khuôn mặt bằng MediaPipe Face Mesh
2. Tính các chỉ số hình học: EAR, MAR, eyebrow raise, mouth corner angle
3. Nhận diện cảm xúc bằng DeepFace
4. Ghi log toàn bộ dữ liệu ra CSV
5. Phân tích & trực quan hóa dữ liệu thu thập được (EDA)
