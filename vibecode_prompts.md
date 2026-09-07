# Bộ prompt vibecode — Dự án phân tích & trực quan hóa cảm xúc khuôn mặt

Cách dùng: copy từng prompt theo đúng thứ tự, dán vào AI coding tool (Claude Code, Cursor...).
Chạy thử và xác nhận bước đó hoạt động đúng trước khi sang prompt tiếp theo.
Nếu lỗi, dán nguyên thông báo lỗi vào cùng cửa sổ chat để AI tự sửa tiếp.

---

## BƯỚC 0 — Khởi tạo project

```
Tạo cho tôi cấu trúc project Python tên "facial-emotion-viz" với:
- Thư mục src/ chứa code chính
- Thư mục data/ để lưu file CSV log
- Thư mục notebooks/ cho phân tích EDA sau này
- File requirements.txt gồm: opencv-python, mediapipe, deepface, pandas, numpy, matplotlib, seaborn, plotly, streamlit
- File README.md mô tả ngắn gọn mục tiêu dự án: thu thập dữ liệu cảm xúc khuôn mặt qua webcam
  kết hợp landmark hình học (lông mày, mắt, môi) với model nhận diện cảm xúc, sau đó trực quan hóa.
- File .gitignore chuẩn cho Python (bỏ qua venv, __pycache__, file CSV lớn trong data/)

Đừng viết logic xử lý ảnh/model vội, bước này chỉ cần khung project.
```

---

## BƯỚC 1 — Phát hiện landmark khuôn mặt bằng MediaPipe

```
Trong file src/face_landmarks.py, viết module dùng MediaPipe Face Mesh để:

1. Mở webcam bằng OpenCV (VideoCapture(0))
2. Với mỗi frame, chạy MediaPipe Face Mesh (refine_landmarks=True) để lấy 468 điểm landmark
3. Vẽ các điểm landmark lên frame, TÔ MÀU RIÊNG theo 3 vùng:
   - Lông mày (trái + phải): màu vàng
   - Mắt (viền mắt trái + phải): màu xanh dương
   - Môi (viền môi trên + dưới): màu đỏ
   Dùng đúng bộ chỉ số landmark chuẩn của MediaPipe cho từng vùng (FACEMESH_LEFT_EYEBROW,
   FACEMESH_RIGHT_EYEBROW, FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE, FACEMESH_LIPS)
4. Hiển thị video real-time trong cửa sổ OpenCV, nhấn 'q' để thoát
5. Viết hàm get_landmark_coords(face_landmarks, frame_shape) trả về dict toạ độ (x, y) pixel
   cho từng điểm landmark, để module khác tái sử dụng

Chỉ tập trung vào việc phát hiện + vẽ landmark ở bước này, CHƯA cần tính toán chỉ số cảm xúc.
Thêm comment giải thích rõ từng đoạn vì tôi cần hiểu code để báo cáo đồ án.
```

---

## BƯỚC 2 — Tính các chỉ số hình học (geometric features)

```
Tạo file src/geometric_features.py, import từ face_landmarks.py, viết các hàm tính chỉ số sau
từ dict toạ độ landmark (dùng công thức khoảng cách Euclid giữa các điểm mốc MediaPipe tương ứng):

1. calculate_ear(landmarks, eye_side) -> float
   Eye Aspect Ratio (EAR): tỉ lệ giữa chiều cao và chiều rộng của mắt.
   Dùng công thức chuẩn EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
   Map sang đúng chỉ số landmark của MediaPipe cho mắt trái/phải.

2. calculate_mar(landmarks) -> float
   Mouth Aspect Ratio (MAR): tỉ lệ độ mở miệng, dùng khoảng cách môi trên-dưới
   chia cho khoảng cách khóe miệng trái-phải.

3. calculate_eyebrow_raise(landmarks, eye_side) -> float
   Khoảng cách (đã chuẩn hoá theo kích thước khuôn mặt) giữa điểm giữa lông mày
   và điểm giữa mắt tương ứng. Số càng lớn = lông mày nhướn càng cao.

4. calculate_mouth_corner_angle(landmarks) -> float
   Góc lệch của khóe miệng so với đường ngang qua tâm miệng, dương = khóe miệng nhếch lên (cười),
   âm = khóe miệng hạ xuống (buồn).

5. Viết hàm extract_all_features(landmarks) -> dict trả về đủ 4 chỉ số trên trong 1 dict,
   kèm normalize theo khoảng cách 2 mắt (interocular distance) để chỉ số không phụ thuộc
   khoảng cách camera tới mặt.

Viết thêm 1 script test đơn giản: chạy webcam, in ra console 4 chỉ số này real-time mỗi giây
để tôi kiểm tra logic tính toán có hợp lý không trước khi đi tiếp.
```

---

## BƯỚC 3 — Kết hợp với model nhận diện cảm xúc (DeepFace)

```
Tạo file src/emotion_detector.py:

1. Viết hàm detect_emotion(frame) dùng DeepFace.analyze(actions=["emotion"], enforce_detection=False)
   trả về dict gồm dominant_emotion và điểm số của cả 7 cảm xúc (angry, disgust, fear, happy, sad,
   surprise, neutral)
2. Xử lý gracefully trường hợp DeepFace không phát hiện được khuôn mặt (trả về None thay vì crash)
3. Viết hàm main kết hợp cả 2 module: mỗi ~1 giây, lấy landmark (từ face_landmarks.py),
   tính geometric features (từ geometric_features.py), VÀ chạy detect_emotion() trên cùng 1 frame
4. In ra console dòng log dạng:
   [14:32:05] Emotion: happy (82%) | EAR: 0.28 | MAR: 0.45 | Eyebrow: 0.12 | MouthAngle: 8.3°

Mục tiêu: xác nhận cả 2 pipeline (landmark hình học + deep learning) chạy đồng bộ được trên
cùng 1 frame mà không bị lag quá nặng.
```

---

## BƯỚC 4 — Ghi log đầy đủ ra CSV

```
Tạo file src/main_logger.py, hoàn thiện thành script chạy chính của toàn bộ hệ thống:

1. Mở webcam, chạy vòng lặp real-time
2. Mỗi DETECT_INTERVAL giây (biến cấu hình, mặc định 1.0s):
   - Lấy landmark, tính 4 chỉ số hình học
   - Chạy detect_emotion()
   - Ghi 1 dòng vào file data/emotion_log.csv với các cột:
     timestamp, dominant_emotion, angry, disgust, fear, happy, sad, surprise, neutral,
     ear_left, ear_right, mar, eyebrow_raise_left, eyebrow_raise_right, mouth_corner_angle
3. Vẽ landmark + text overlay hiển thị dominant_emotion lên video real-time (tái sử dụng bước 1)
4. Tạo header CSV tự động nếu file chưa tồn tại, append nếu đã có
5. Xử lý thoát sạch sẽ (release camera, đóng file) khi nhấn 'q' hoặc Ctrl+C

Sau khi xong, tôi sẽ chạy script này 3-5 phút để thu thập dữ liệu thật, dùng cho bước
phân tích & trực quan hóa tiếp theo.
```

---

## BƯỚC 5 — EDA & trực quan hóa (sau khi đã có file CSV thật)

```
Tôi đã có file data/emotion_log.csv với các cột: timestamp, dominant_emotion,
[7 cột xác suất cảm xúc], ear_left, ear_right, mar, eyebrow_raise_left, eyebrow_raise_right,
mouth_corner_angle.

Viết notebook notebooks/eda_visualization.ipynb (hoặc file src/dashboard.py dùng Streamlit,
tôi sẽ nói rõ sau) thực hiện:

1. Đọc CSV bằng pandas, parse timestamp thành datetime, kiểm tra dữ liệu thiếu/lỗi
2. Biểu đồ cột: phân bố tần suất từng loại cảm xúc (dominant_emotion) trong toàn bộ phiên
3. Line chart: xác suất của 7 cảm xúc biến đổi theo thời gian (trục x = timestamp)
4. Line chart thứ 2 chồng lên: giá trị EAR, MAR theo thời gian, để so sánh xem chỉ số hình học
   có tương quan với thời điểm đổi cảm xúc không
5. Scatter plot: mouth_corner_angle vs xác suất "happy" — kiểm tra tương quan
6. Heatmap: ma trận tương quan (correlation) giữa các chỉ số hình học và xác suất từng cảm xúc
7. Radar chart: so sánh trung bình các chỉ số hình học (EAR, MAR, eyebrow_raise,
   mouth_corner_angle) ứng với từng nhóm dominant_emotion khác nhau

Với mỗi biểu đồ, thêm 1-2 dòng markdown giải thích insight rút ra được, để tôi dùng
trực tiếp cho phần báo cáo.
```

---

## Ghi chú khi vibecode

- Luôn yêu cầu AI **giải thích code**, đừng chỉ copy-paste — bạn cần hiểu để bảo vệ đồ án
- Sau mỗi bước, tự chạy thử ngay, đừng dồn nhiều bước rồi mới test — dễ khó xác định lỗi ở đâu
- Nếu AI sinh code dùng landmark index sai (MediaPipe có nhiều bộ chỉ số dễ nhầm), yêu cầu nó
  in ra hình ảnh debug có đánh số landmark để bạn tự kiểm tra bằng mắt
- Bước 5 nên tách riêng, chỉ làm sau khi đã thu thập được dữ liệu thật — đừng để AI tạo dữ liệu giả
  để test, vì insight rút ra sẽ không có giá trị thực cho báo cáo
