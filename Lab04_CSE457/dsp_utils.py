"""Các hàm xử lý tín hiệu tiếng nói thuần NumPy/SciPy (không phụ thuộc Numba/Librosa).
Bao gồm: STFT, Mel-spectrogram, YIN F0 (FFT-CMNDF), Phase Vocoder và Pitch Shift bảo toàn formant.
"""
import numpy as np
from scipy import signal


def stft_mag(x: np.ndarray, n_fft: int = 512, win_length: int = 400, hop_length: int = 160) -> np.ndarray:
    """Tính biên độ STFT (shape: n_fft//2 + 1, n_frames)."""
    x = np.asarray(x, dtype=np.float64)
    pad = win_length // 2
    xp = np.pad(x, (pad, pad))
    if len(xp) < win_length:
        xp = np.pad(xp, (0, win_length - len(xp)))
    n_frames = 1 + (len(xp) - win_length) // hop_length
    win = np.hanning(win_length)
    frames = np.lib.stride_tricks.sliding_window_view(xp, win_length)[::hop_length][:n_frames]
    spec = np.fft.rfft(frames * win[None, :], n=n_fft, axis=1)
    return np.abs(spec).T


def spec_db(x: np.ndarray, n_fft: int = 512, win_length: int = 400, hop_length: int = 160) -> np.ndarray:
    mag = stft_mag(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    ref = max(float(np.max(mag)), 1e-10)
    return 20.0 * np.log10(np.maximum(mag, ref * 1e-4) / ref)


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int = 16000, n_fft: int = 512, n_mels: int = 80, fmin: float = 50.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    m_pts = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_pts = mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        b_left, b_center, b_right = bins[m - 1], bins[m], bins[m + 1]
        if b_center > b_left:
            for k in range(b_left, b_center):
                fb[m - 1, k] = (k - b_left) / (b_center - b_left)
        if b_right > b_center:
            for k in range(b_center, min(b_right, n_fft // 2 + 1)):
                fb[m - 1, k] = (b_right - k) / (b_right - b_center)
    return fb


def mel_spectrogram_db(x: np.ndarray, sr: int = 16000, n_fft: int = 512, win_length: int = 400, hop_length: int = 160, n_mels: int = 80) -> np.ndarray:
    power = stft_mag(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length) ** 2
    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel = fb @ power
    ref = max(float(np.max(mel)), 1e-10)
    return 10.0 * np.log10(np.maximum(mel, ref * 1e-8) / ref)


def yin_f0(
    x: np.ndarray,
    sr: int = 16000,
    fmin: float = 75.0,
    fmax: float = 400.0,
    frame_length: int = 1024,
    hop_length: int = 160,
    threshold: float = 0.15,
) -> np.ndarray:
    """Ước lượng F0 theo thuật toán YIN (CMNDF + nội suy parabolic) thuần NumPy."""
    x = np.asarray(x, dtype=np.float64)
    pad = frame_length // 2
    xp = np.pad(x, (pad, pad))
    if len(xp) < frame_length:
        xp = np.pad(xp, (0, frame_length - len(xp)))
    n_frames = 1 + (len(xp) - frame_length) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(xp, frame_length)[::hop_length][:n_frames]

    tau_min = max(2, int(np.floor(sr / fmax)))
    tau_max = min(frame_length // 2 - 1, int(np.ceil(sr / fmin)))
    half = frame_length // 2

    # Tính difference function d(tau) vector hóa theo từng khung bằng FFT
    n_fft = 1 << int(np.ceil(np.log2(frame_length + half)))
    x_first = frames[:, :half]
    sq_first = np.sum(x_first ** 2, axis=1, keepdims=True)
    cs_sq = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    sq_shifted = cs_sq[:, half : half + tau_max + 1] - cs_sq[:, : tau_max + 1]

    F1 = np.fft.rfft(x_first, n=n_fft, axis=1)
    F2 = np.fft.rfft(frames, n=n_fft, axis=1)
    acf = np.fft.irfft(F2 * np.conj(F1), n=n_fft, axis=1)[:, : tau_max + 1]
    d = np.maximum(sq_first + sq_shifted - 2.0 * acf, 0.0)

    # Cumulative Mean Normalized Difference Function (CMNDF)
    cmndf = np.ones_like(d)
    cumsum_d = np.cumsum(d[:, 1:], axis=1)
    taus = np.arange(1, tau_max + 1, dtype=np.float64)[None, :]
    cmndf[:, 1:] = d[:, 1:] / np.maximum(cumsum_d / taus, 1e-12)

    f0 = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        sub = cmndf[t, tau_min : tau_max + 1]
        below = np.where(sub < threshold)[0]
        if len(below) > 0:
            idx = below[0]
            while idx + 1 < len(sub) and sub[idx + 1] < sub[idx]:
                idx += 1
            tau = tau_min + idx
        else:
            tau = tau_min + int(np.argmin(sub))

        # Nội suy parabolic quanh cực tiểu
        if 1 <= tau < tau_max:
            y0, y1, y2 = cmndf[t, tau - 1], cmndf[t, tau], cmndf[t, tau + 1]
            denom = 2.0 * (y0 - 2.0 * y1 + y2)
            if abs(denom) > 1e-12:
                tau = tau + np.clip((y0 - y2) / denom, -0.5, 0.5)
        f0[t] = sr / max(tau, 1e-6)
    return f0


def phase_vocoder_stretch(x: np.ndarray, rate: float, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Kéo giãn thời gian bằng Phase Vocoder (rate > 1: nhanh hơn/ngắn hơn)."""
    if abs(rate - 1.0) < 1e-4:
        return x.copy()
    x = np.asarray(x, dtype=np.float64)
    win = np.hanning(n_fft)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad))
    n_frames = 1 + max(0, (len(xp) - n_fft) // hop)
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:n_frames]
    stft_mat = np.fft.rfft(frames * win[None, :], axis=1).T  # (n_bins, n_frames)

    n_bins, n_f = stft_mat.shape
    time_steps = np.arange(0, n_f - 1, rate)
    omega = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft
    phase_acc = np.angle(stft_mat[:, 0])
    out_stft = np.zeros((n_bins, len(time_steps)), dtype=np.complex128)

    for idx, step in enumerate(time_steps):
        t0 = int(np.floor(step))
        t1 = min(t0 + 1, n_f - 1)
        alpha = step - t0
        mag = (1.0 - alpha) * np.abs(stft_mat[:, t0]) + alpha * np.abs(stft_mat[:, t1])
        dphi = np.angle(stft_mat[:, t1]) - np.angle(stft_mat[:, t0]) - omega
        dphi = dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        phase_acc = phase_acc + omega + dphi
        out_stft[:, idx] = mag * np.exp(1j * phase_acc)

    # ISTFT overlap-add
    out_len = len(time_steps) * hop + n_fft
    y = np.zeros(out_len, dtype=np.float64)
    w_sum = np.zeros(out_len, dtype=np.float64)
    for idx in range(len(time_steps)):
        frm = np.fft.irfft(out_stft[:, idx], n=n_fft) * win
        s = idx * hop
        y[s : s + n_fft] += frm
        w_sum[s : s + n_fft] += win ** 2
    valid = w_sum > 1e-6
    y[valid] /= w_sum[valid]
    return y[pad : max(pad + 1, out_len - pad)]


def spectral_envelope_cepstral(mag: np.ndarray, order: int = 24) -> np.ndarray:
    """Tách đường bao phổ mượt từ biên độ STFT bằng cepstral liftering (real cepstrum)."""
    log_mag = np.log(np.maximum(mag, 1e-6))
    ceps = np.fft.irfft(log_mag, axis=0)
    lifter = np.zeros(ceps.shape[0], dtype=np.float64)
    lifter[0] = 1.0
    lifter[1:order] = 2.0
    lifter[order] = 1.0
    smooth_log = np.real(np.fft.rfft(ceps * lifter[:, None], axis=0))
    return np.exp(smooth_log[: mag.shape[0]])


def pitch_shift_preserve_formant(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Dịch cao độ ±semitone bằng phase-vocoder kết hợp bảo toàn đường bao phổ (formant)."""
    if abs(semitones) < 1e-4:
        return x.copy()
    r = 2.0 ** (semitones / 12.0)
    # Bước 1: giãn thời gian bằng 1/r, sau đó resample để dịch F0 theo tỷ lệ r
    y_stretch = phase_vocoder_stretch(x, rate=1.0 / r, n_fft=512, hop=128)
    n_target = len(x)
    y_shifted = signal.resample(y_stretch, n_target)

    # Bước 2: ghép lại đường bao phổ gốc (source-filter envelope preservation)
    n_fft, hop = 512, 128
    win = np.hanning(n_fft)
    pad = n_fft // 2
    x_pad = np.pad(x, (pad, pad))
    y_pad = np.pad(y_shifted, (pad, pad))
    n_f = 1 + (len(x_pad) - n_fft) // hop
    fx = np.lib.stride_tricks.sliding_window_view(x_pad, n_fft)[::hop][:n_f]
    fy = np.lib.stride_tricks.sliding_window_view(y_pad, n_fft)[::hop][:n_f]

    Sx = np.fft.rfft(fx * win[None, :], axis=1).T
    Sy = np.fft.rfft(fy * win[None, :], axis=1).T
    env_orig = spectral_envelope_cepstral(np.abs(Sx), order=24)
    env_shft = spectral_envelope_cepstral(np.abs(Sy), order=24)
    Sy_corr = Sy * (env_orig / np.maximum(env_shft, 1e-5))

    out = np.zeros(len(y_pad), dtype=np.float64)
    w_sum = np.zeros(len(y_pad), dtype=np.float64)
    for t in range(n_f):
        frm = np.fft.irfft(Sy_corr[:, t], n=n_fft) * win
        s = t * hop
        out[s : s + n_fft] += frm
        w_sum[s : s + n_fft] += win ** 2
    valid = w_sum > 1e-6
    out[valid] /= w_sum[valid]
    return out[pad : pad + len(x)]
