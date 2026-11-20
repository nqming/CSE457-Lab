"""Sinh bộ dữ liệu word-unit miền hẹp 'Thông báo xếp hàng' (16 từ x 3 candidate = 48 file WAV).
Chạy: python tools/prepare_data.py
Yêu cầu: pip install edge-tts soundfile librosa numpy
"""
import asyncio
import csv
import os
import tempfile
import edge_tts
import librosa
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS_DIR = os.path.join(ROOT, "lab4_data", "units")
META_PATH = os.path.join(ROOT, "lab4_data", "metadata.csv")
os.makedirs(UNITS_DIR, exist_ok=True)

SR = 16000
VOICE = "vi-VN-HoaiMyNeural"
SPEAKER = "HoaiMy"

# 16 đơn vị từ vựng miền hẹp "thông báo xếp hàng" (>= 15 từ theo yêu cầu Lab 4)
LEXICON = [
    ("mời", "moi"),
    ("khách", "khach"),
    ("hàng", "hang"),
    ("số", "so"),
    ("không", "khong"),
    ("một", "mot"),
    ("hai", "hai"),
    ("ba", "ba"),
    ("bốn", "bon"),
    ("năm", "nam"),
    ("sáu", "sau"),
    ("bảy", "bay"),
    ("tám", "tam"),
    ("chín", "chin"),
    ("đến", "den"),
    ("quầy", "quay"),
]

# 3 candidate cho mỗi từ: mô phỏng 3 lần thu của cùng người nói với biến thiên tự nhiên về tốc độ, cao độ, biên độ
TAKES = [
    ("01", 1.00, 0.0, 0.085),   # chuẩn
    ("02", 0.92, -0.6, 0.068),  # chậm hơn, trầm nhẹ, nhỏ hơn
    ("03", 1.08, 0.6, 0.102),   # nhanh hơn, cao nhẹ, lớn hơn
]


def trim_silence_energy(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Cắt khoảng lặng đầu/cuối dựa trên năng lượng ngắn hạn, giữ biên an toàn cho phụ âm."""
    win = int(0.015 * sr)  # 15 ms
    hop = int(0.005 * sr)  # 5 ms
    if len(x) <= win:
        return x
    frames = librosa.util.frame(x, frame_length=win, hop_length=hop)
    frm_rms = np.sqrt(np.mean(frames ** 2, axis=0) + 1e-12)
    thresh = max(float(np.max(frm_rms)) * 0.025, 1.5e-3)
    active = np.where(frm_rms >= thresh)[0]
    if len(active) == 0:
        return x
    start_sample = max(0, active[0] * hop - int(0.012 * sr))
    end_sample = min(len(x), active[-1] * hop + win + int(0.018 * sr))
    return x[start_sample:end_sample]


async def synth_base(text: str, tmp_mp3: str) -> np.ndarray:
    comm = edge_tts.Communicate(text=text, voice=VOICE)
    await comm.save(tmp_mp3)
    x, sr_in = sf.read(tmp_mp3, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr_in != SR:
        x = librosa.resample(x, orig_sr=sr_in, target_sr=SR)
    x = x - np.mean(x)
    return trim_silence_energy(x, SR)


def make_take(x_base: np.ndarray, speed_rate: float, pitch_st: float, target_rms: float, rng: np.random.Generator) -> np.ndarray:
    y = x_base.astype(np.float64).copy()
    if abs(speed_rate - 1.0) > 1e-3:
        y = librosa.effects.time_stretch(y, rate=speed_rate)
    if abs(pitch_st) > 1e-3:
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=pitch_st)
    # Độ lệch RMS tự nhiên nhỏ (+/- 5%) giữa các lần thu
    jitter_rms = target_rms * rng.uniform(0.95, 1.05)
    cur_rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    if cur_rms > 1e-6:
        y = y * (jitter_rms / cur_rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.95:
        y = y * (0.95 / peak)
    return y


async def main():
    rng = np.random.default_rng(457)
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for word, ascii_name in LEXICON:
            tmp_mp3 = os.path.join(tmpdir, f"{ascii_name}.mp3")
            x_base = None
            for attempt in range(3):
                try:
                    x_base = await synth_base(word + ".", tmp_mp3)
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError(f"Lỗi sinh từ '{word}': {exc}") from exc
                    await asyncio.sleep(1.0)

            for take_id, speed_rate, pitch_st, target_rms in TAKES:
                fname = f"{ascii_name}_{take_id}.wav"
                out_path = os.path.join(UNITS_DIR, fname)
                y = make_take(x_base, speed_rate, pitch_st, target_rms, rng)
                sf.write(out_path, y.astype(np.float32), SR, subtype="PCM_16")

                y_chk, fs_chk = sf.read(out_path, dtype="float32")
                assert fs_chk == SR
                dur = len(y_chk) / SR
                rms_val = float(np.sqrt(np.mean(y_chk ** 2) + 1e-12))
                f0 = librosa.yin(y_chk, fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=160)
                valid_f0 = f0[(f0 > 75) & (f0 < 380)]
                med_f0 = float(np.median(valid_f0)) if len(valid_f0) > 0 else 0.0

                rows.append({
                    "label": word,
                    "ascii_label": ascii_name,
                    "candidate_id": take_id,
                    "speaker": SPEAKER,
                    "duration": round(dur, 4),
                    "rms": round(rms_val, 6),
                    "f0_median": round(med_f0, 2),
                    "file_path": f"lab4_data/units/{fname}",
                })
                print(f"  {fname:14s} | dur={dur:.3f}s | rms={rms_val:.4f} | f0_med={med_f0:.1f} Hz")

    with open(META_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "ascii_label", "candidate_id", "speaker", "duration", "rms", "f0_median", "file_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Hoàn tất: {len(rows)} unit WAV -> {META_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
