"""Backend Neural TTS offline dùng mô hình VITS (Piper ONNX: vi_VN-vivos-x_low).
Chạy hoàn toàn offline từ tệp model đóng gói trong lab4_data/neural_model/.
API theo chuẩn Lab 4 (Mục 5.6):
    from neural_tts_backend import synthesize
    y_nn, sr, extra = synthesize("Mời khách hàng số 23 đến quầy 5")
"""
import os
import re
import time
import numpy as np
from piper import PiperVoice, SynthesisConfig

from dsp_utils import mel_spectrogram_db

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "lab4_data", "neural_model", "vi_VN-vivos-x_low.onnx")

DIGIT_MAP = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

_VOICE_CACHE = {}


def normalize_for_neural(text: str) -> str:
    """Chuẩn hóa câu miền hẹp để Neural TTS đọc cùng nội dung từ với Concat/HMM."""
    clean = re.sub(r"[.,!?;:\"'()\[\]]+", " ", text.lower())
    out = []
    for tok in clean.split():
        if tok.isdigit():
            out.extend(DIGIT_MAP[c] for c in tok)
        else:
            out.append(tok)
    return " ".join(out)


def get_voice(model_path: str = DEFAULT_MODEL_PATH) -> PiperVoice:
    if model_path not in _VOICE_CACHE:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Không tìm thấy model neural TTS offline tại: {model_path}"
            )
        _VOICE_CACHE[model_path] = PiperVoice.load(model_path, use_cuda=False)
    return _VOICE_CACHE[model_path]


def synthesize(
    text: str,
    speaker_id: int = 22,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_w_scale: float = 0.8,
    model_path: str = DEFAULT_MODEL_PATH,
    normalize_digits: bool = True,
):
    """Tổng hợp tiếng nói từ văn bản bằng mô hình neural VITS offline.

    Returns:
        y_nn (np.ndarray float32): waveform mono
        sr (int): tần số lấy mẫu (16000 Hz)
        extra (dict): mel_spectrogram, latency_s, rtf, normalized_text, model_size_mb
    """
    voice = get_voice(model_path)
    text_in = normalize_for_neural(text) if normalize_digits else text.strip()
    syn_cfg = SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        normalize_audio=True,
    )

    t0 = time.perf_counter()
    chunks = list(voice.synthesize(text_in, syn_cfg))
    latency_s = time.perf_counter() - t0

    if not chunks:
        raise RuntimeError(f"Piper không sinh được audio cho văn bản: {text_in!r}")

    sr = int(chunks[0].sample_rate)
    y_nn = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)

    # Trích xuất log-mel spectrogram (80 mel bins, dB) để lưu và trực quan hóa
    mel_db = mel_spectrogram_db(y_nn, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)

    dur_s = len(y_nn) / sr
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    extra = {
        "normalized_text": text_in,
        "latency_s": float(latency_s),
        "audio_duration_s": float(dur_s),
        "rtf": float(latency_s / max(dur_s, 1e-6)),
        "mel_spectrogram": mel_db,
        "model_path": model_path,
        "model_size_mb": float(model_size_mb),
        "speaker_id": int(speaker_id),
    }
    return y_nn, sr, extra
