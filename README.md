# 🎙 VoiceCtrl — Điều Khiển Đối Tượng Bằng Giọng Nói

> Đề tài: **Ứng dụng Machine Learning trong điều khiển đối tượng trên Web bằng lệnh thoại thời gian thực**

---

## Mục tiêu đề tài

- Xây dựng hệ thống nhận dạng 4 lệnh thoại cơ bản: `left`, `right`, `up`, `down` trong môi trường thời gian thực
- Tích hợp mô hình SVM vào giao diện web để điều khiển trực tiếp đối tượng đồ họa trên màn hình
- Đánh giá hiệu năng mô hình: độ chính xác, độ trễ xử lý, khả năng hoạt động thực tế
- Tạo nền tảng mở rộng sang điều khiển game, robot ảo hoặc hệ thống thông minh

---

## Cấu trúc project

```
voicectrl-project/
├── backend/
│   ├── main.py                      # FastAPI app — endpoint /predict và /health
│   ├── requirements.txt             # Python dependencies
│   └── svm_speech_model.joblib      # Model SVM đã train (đặt ở đây)
│
├── frontend/
│   └── index.html                   # Giao diện web điều khiển (all-in-one)
│
├── training/                        # (tuỳ chọn) dataset, notebook, script train
│   ├── train.py
│   └── dataset/
│
└── README.md
```

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| Python | 3.9 trở lên |
| pip | mới nhất |
| Trình duyệt | Chrome / Edge (hỗ trợ Web Audio API) |
| Microphone | Bắt buộc |

---

## Cài đặt & Chạy

### Bước 1 — Tạo môi trường ảo

```bash
cd voicectrl-project/

# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate
```

### Bước 2 — Cài dependencies

```bash
cd backend/
pip install -r requirements.txt
winget install ffmpeg
```

### Bước 3 — Đặt model đúng vị trí

Sao chép file `svm_speech_model.joblib` vào thư mục `backend/`:

```bash
# Kiểm tra file đã ở đúng chỗ chưa
ls backend/svm_speech_model.joblib
```

### Bước 4 — Chạy API server

```bash
# Vẫn trong thư mục backend/
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra server đang chạy tốt bằng cách mở trình duyệt và truy cập:

```
http://localhost:8000/health
```

Kết quả mong đợi:

```json
{
  "status": "ok",
  "model_loaded": true,
  "commands": ["left", "right", "up", "down"]
}
```

> ⚠️ Nếu thấy `"model_loaded": false` → kiểm tra lại đường dẫn file `.joblib`

### Bước 5 — Mở giao diện web

Mở **tab terminal mới** (giữ nguyên backend đang chạy), rồi chọn một trong hai cách:

**Cách 1 — Đơn giản (double-click):**
```
Mở thẳng file frontend/index.html trong trình duyệt
```

**Cách 2 — Dùng local server (khuyến nghị, tránh lỗi CORS):**
```bash
cd frontend/
npx serve . -p 3000
# Truy cập http://localhost:3000
```

### Bước 6 — Sử dụng

1. Nhấn nút **"Kiểm Tra Kết Nối"** → chờ thông báo "✓ Đã kết nối"
2. Nhấn phím `SPACE` hoặc click nút microphone
3. Nói một trong 4 lệnh: **left**, **right**, **up**, **down**
4. Quan sát đối tượng di chuyển trên màn hình

---

## Tóm tắt — 2 terminal chạy song song

| Terminal | Thư mục | Lệnh |
|---|---|---|
| Terminal 1 | `backend/` | `uvicorn main:app --port 8000 --reload` |
| Terminal 2 | `frontend/` | `npx serve . -p 3000` |

---

## API Reference

### `POST /predict`

Nhận file audio, trả về lệnh điều khiển được nhận dạng.

**Request:** `multipart/form-data`

| Field | Type | Mô tả |
|---|---|---|
| `audio` | File | File âm thanh `.wav` hoặc `.webm` |

**Response:**

```json
{
  "command": "left",
  "confidence": 0.923,
  "latency_ms": 45.2,
  "probabilities": {
    "left": 0.923,
    "right": 0.041,
    "up": 0.024,
    "down": 0.012
  },
  "status": "success"
}
```

### `GET /health`

Kiểm tra trạng thái server và model.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "svm_speech_model.joblib",
  "commands": ["left", "right", "up", "down"]
}
```

---

## Phím tắt (Frontend)

| Phím | Chức năng |
|---|---|
| `SPACE` | Bắt đầu ghi âm (1.5 giây) |
| `↑ ↓ ← →` | Di chuyển thủ công (test không cần mic) |

---

## Lưu ý kỹ thuật

### Feature extraction

Backend trích xuất **78 features** từ mỗi đoạn audio:

| Feature | Chiều | Thống kê | Tổng |
|---|---|---|---|
| MFCC | 13 | mean + std | 26 |
| MFCC delta | 13 | mean + std | 26 |
| MFCC delta² | 13 | mean + std | 26 |

> ⚠️ **Quan trọng:** Nếu model được train với pipeline feature extraction khác (số MFCC, thêm chroma, spectral centroid...), cần chỉnh lại hàm `extract_features()` trong `main.py` cho khớp.

### Tuỳ chỉnh

| Tham số | File | Mặc định | Ý nghĩa |
|---|---|---|---|
| `STEP` | `index.html` | `40` | Số pixel mỗi lần di chuyển |
| `RECORD_MS` | `index.html` | `1500` | Thời gian ghi âm (ms) |
| `MODEL_PATH` | biến môi trường | `svm_speech_model.joblib` | Đường dẫn đến model |

Thay đổi `MODEL_PATH` bằng biến môi trường:

```bash
MODEL_PATH=/path/to/model.joblib uvicorn main:app --port 8000
```

---

## Xử lý lỗi thường gặp

**Lỗi: `ModuleNotFoundError: No module named 'librosa'`**
```bash
pip install -r requirements.txt
```

**Lỗi: Microphone không hoạt động**
- Trình duyệt phải được cấp quyền truy cập microphone
- Dùng Chrome hoặc Edge (Firefox đôi khi có vấn đề với `MediaRecorder`)

**Lỗi: CORS khi fetch API**
- Không mở `index.html` bằng `file://` — dùng `npx serve` hoặc live server

**Lỗi: `model_loaded: false`**
- Kiểm tra file `svm_speech_model.joblib` có nằm trong `backend/` không
- Kiểm tra tên file có đúng chính xác không (phân biệt hoa thường)

---

## Công nghệ sử dụng

| Tầng | Công nghệ |
|---|---|
| Machine Learning | scikit-learn (SVM), joblib |
| Feature Extraction | librosa (MFCC) |
| Backend | FastAPI, uvicorn |
| Frontend | HTML5, Web Audio API, Vanilla JS |
| Giao tiếp | REST API, multipart/form-data |
