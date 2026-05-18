from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

import numpy as np
import joblib
import librosa
import os
import time
import logging
import subprocess

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Load ffmpeg =====
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info(f"✅ ffmpeg loaded: {FFMPEG_PATH}")
except Exception as e:
    FFMPEG_PATH = None
    logger.error(f"❌ Không tìm thấy ffmpeg: {e}")

# ===== PATH =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(BASE_DIR, "models/svm_speech_model_v3_svm.joblib")
SCALER_PATH  = os.path.join(BASE_DIR, "models/scaler_v3.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models/label_encoder_v3.joblib")
PCA_PATH     = os.path.join(BASE_DIR, "models/pca_v3.joblib")

STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

# ===== GLOBAL =====
model = None
scaler = None
label_encoder = None
pca = None


# ===== LIFESPAN =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, label_encoder, pca

    try:
        model         = joblib.load(MODEL_PATH)
        scaler        = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        pca           = joblib.load(PCA_PATH)

        logger.info("✅ Model + Scaler + LabelEncoder + PCA loaded")

    except Exception as e:
        logger.error(f"❌ Load model failed: {e}")

    yield


# ===== FASTAPI APP =====
app = FastAPI(
    title="Speech Command API",
    version="1.0.0",
    lifespan=lifespan
)

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== SERVE FRONTEND =====
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    logger.info(f"Static files served from: {STATIC_DIR}")
else:
    logger.warning(f"Static dir không tồn tại: {STATIC_DIR}")


# ===== AUDIO LOADING =====
def load_audio_bytes(audio_bytes: bytes):
    if FFMPEG_PATH is None:
        raise RuntimeError("FFmpeg chưa sẵn sàng")

    command = [
        FFMPEG_PATH,
        "-loglevel", "error",
        "-i", "pipe:0",
        "-ac", "1",
        "-ar", "16000",
        "-f", "f32le",
        "pipe:1"
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    out, err = process.communicate(input=audio_bytes)

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg decode failed:\n{err.decode()}")

    audio_data = np.frombuffer(out, dtype=np.float32)

    if len(audio_data) == 0:
        raise RuntimeError("Audio rỗng sau decode")

    return audio_data, 16000


# ===== FEATURE =====
def extract_features(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(y=audio_data)
    rms = librosa.feature.rms(y=audio_data)
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)

    features = np.concatenate([
        np.mean(mfcc, axis=1),        np.std(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),  np.std(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1), np.std(delta2_mfcc, axis=1),
        np.mean(chroma, axis=1),      np.std(chroma, axis=1),
        np.mean(zcr, axis=1),         np.std(zcr, axis=1),
        np.mean(rms, axis=1),         np.std(rms, axis=1),
        np.mean(contrast, axis=1),    np.std(contrast, axis=1),
    ])

    return features


# ===== API =====
@app.post("/api/predict")
async def predict_command(audio: UploadFile = File(...)):
    start_time = time.time()

    try:
        audio_bytes = await audio.read()

        audio_data, sr = load_audio_bytes(audio_bytes)
        features = extract_features(audio_data, sr).reshape(1, -1)

        if model is not None:
            features = scaler.transform(features)
            features = pca.transform(features)

            pred = model.predict(features)[0]
            prediction = label_encoder.inverse_transform([pred])[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                confidence = float(max(proba))
            else:
                confidence = 0.85
        else:
            prediction = "unknown"
            confidence = 0.0

        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "command": str(prediction),
            "confidence": confidence,
            "latency_ms": latency,
            "status": "success"
        }

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }
