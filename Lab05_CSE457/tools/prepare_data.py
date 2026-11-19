"""Chuẩn bị dữ liệu Lab 5 từ TTS thô (lab5_data/raw) -> cấu trúc lab5_data/ theo đề.
- spk01: giọng David, spk02: giọng Zira
- spk03: David dịch cao độ+formant +4 semitone, spk04: Zira dịch -3 semitone (giả lập người nói khác)
- Câu 0-3: enroll, 4-5: dev, 6-7: test, 8/9: nguồn cho separation
- Nhiễu tổng hợp: fan.wav (hum + nhiễu màu điều chế cánh quạt), cafeteria.wav (babble + pink)
"""
import os, numpy as np, librosa, soundfile as sf

SR = 16000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "lab5_data", "raw")
D = os.path.join(ROOT, "lab5_data")
rng = np.random.default_rng(0)

def load(p):
    x, _ = librosa.load(p, sr=SR, mono=True)
    x, _ = librosa.effects.trim(x, top_db=35)
    return x

def shift(x, n):  # resample-based: dịch cả pitch lẫn formant (đổi "kích thước tuyến âm")
    return librosa.effects.pitch_shift(x, sr=SR, n_steps=n)

def norm(x, peak=0.5):
    return (peak * x / (np.max(np.abs(x)) + 1e-12)).astype(np.float32)

for sub in ["clean", "noise", "separation"]:
    os.makedirs(os.path.join(D, sub), exist_ok=True)

spk = {"spk01": ("david", 0), "spk02": ("zira", 0), "spk03": ("david", 4), "spk04": ("zira", -3)}
split = {"enroll": [0, 1, 2, 3], "dev": [4, 5], "test": [6, 7]}
for name, (voice, st) in spk.items():
    for part, idx in split.items():
        for k, i in enumerate(idx, 1):
            x = load(os.path.join(RAW, f"{voice}_{i:02d}.wav"))
            if st: x = shift(x, st)
            sf.write(os.path.join(D, "clean", f"{name}_{part}_{k:02d}.wav"), norm(x), SR, subtype="PCM_16")

# separation sources: 2 người khác nhau, câu khác nhau
sf.write(os.path.join(D, "separation", "source_A.wav"), norm(load(os.path.join(RAW, "david_08.wav"))), SR, subtype="PCM_16")
sf.write(os.path.join(D, "separation", "source_B.wav"), norm(load(os.path.join(RAW, "zira_09.wav"))), SR, subtype="PCM_16")

# ---- noise: fan ----
T = 10 * SR
t = np.arange(T) / SR
w = rng.standard_normal(T)
brown = np.cumsum(w); brown -= np.convolve(brown, np.ones(4001) / 4001, mode="same")  # bỏ trôi DC
brown /= np.std(brown)
import scipy.signal as ss
b, a = ss.butter(2, [80, 2500], btype="band", fs=SR)
broad = ss.lfilter(b, a, w); broad /= np.std(broad)
hum = sum(0.3 / h * np.sin(2 * np.pi * 50 * h * t + rng.uniform(0, 6.28)) for h in (1, 2, 3, 4))
blade = 1 + 0.3 * np.sin(2 * np.pi * 12 * t)  # điều chế biên độ do cánh quạt
fan = blade * (0.6 * broad + 0.4 * brown) + hum
sf.write(os.path.join(D, "noise", "fan.wav"), norm(fan), SR, subtype="PCM_16")

# ---- noise: cafeteria (babble) ----
bab = np.zeros(T)
for v in ["david", "zira"]:
    for i in range(10):
        x = load(os.path.join(RAW, f"{v}_{i:02d}.wav"))
        x = shift(x, rng.uniform(-5, 5))
        pos = rng.integers(0, T - len(x))
        bab[pos:pos + len(x)] += x / (np.std(x) + 1e-9) * rng.uniform(0.5, 1.0)
pink = ss.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], w)
cafe = bab / np.std(bab) + 0.3 * pink / np.std(pink)
sf.write(os.path.join(D, "noise", "cafeteria.wav"), norm(cafe), SR, subtype="PCM_16")
print("Prepared data in", D)
