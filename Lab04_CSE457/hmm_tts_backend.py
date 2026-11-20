"""Backend tổng hợp tiếng nói tham số dựa trên HMM (HMM-based Parametric TTS).
Bao gồm:
  1. Trích xuất tham số âm học theo khung 10 ms: log-F0, log-energy, LSF bậc 16 + Delta.
  2. Huấn luyện HMM 5 trạng thái left-to-right cho từng từ trong inventory.
  3. Sinh tham số bằng MLPG (Maximum Likelihood Parameter Generation) với ràng buộc Delta.
  4. Bộ tổng hợp source-filter (glottal pulse train / noise + vocal-tract IIR filter 1/A(z)).

API theo chuẩn Lab 4 (Mục 5.5):
    from hmm_tts_backend import synthesize_hmm
    y_hmm, sr, tracks = synthesize_hmm("Mời khách hàng số 23 đến quầy 5")
"""
import csv
import os
import re
import time
import numpy as np
import soundfile as sf
from scipy import linalg, signal

from dsp_utils import yin_f0

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "lab4_data", "hmm_demo", "hmm_model.npz")
DEFAULT_META_PATH = os.path.join(ROOT, "lab4_data", "metadata.csv")

SR = 16000
WIN_LEN = 400   # 25 ms
HOP_LEN = 160   # 10 ms
LPC_ORDER = 16
N_STATES = 5
PRE_EMPH = 0.97
FEAT_DIM = 2 + LPC_ORDER  # [log_f0, log_energy, lsf_0..lsf_15]

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

_HMM_CACHE = {}


def normalize_queue_text(text: str) -> list[str]:
    clean = re.sub(r"[.,!?;:\"'()\[\]]+", " ", text.lower())
    out = []
    for tok in clean.split():
        if tok.isdigit():
            out.extend(DIGIT_MAP[c] for c in tok)
        else:
            out.append(tok)
    return out


def levinson_lpc(frame: np.ndarray, order: int = LPC_ORDER) -> np.ndarray:
    """Tính hệ số LPC [1, a_1, ..., a_p] bằng hàm tự tương quan và thuật toán Levinson-Durbin."""
    r = np.correlate(frame, frame, mode="full")[len(frame) - 1 : len(frame) + order]
    r[0] = max(float(r[0]), 1e-10)
    # Lag window nhẹ để mở rộng bandwidth ~15 Hz, giúp ổn định nghiệm phổ
    lags = np.arange(order + 1)
    r = r * np.exp(-0.5 * (2 * np.pi * 15.0 / SR * lags) ** 2)

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1])
        k = -acc / (e + 1e-12)
        k = float(np.clip(k, -0.995, 0.995))
        a_new = a.copy()
        a_new[1:i] = a[1:i] + k * a[i - 1 : 0 : -1]
        a_new[i] = k
        a = a_new
        e = e * (1.0 - k * k)
        if e <= 1e-12:
            break
    return a


def lpc_to_lsf(a: np.ndarray) -> np.ndarray:
    """Chuyển đổi hệ số LPC bậc chẵn sang Line Spectral Frequencies (radian trong (0, pi))."""
    p = len(a) - 1
    a_pad = np.concatenate([a, [0.0]])
    a_rev = a_pad[::-1]
    P = a_pad + a_rev  # đối xứng, bậc p+1, có nghiệm tại z = -1
    Q = a_pad - a_rev  # phản đối xứng, bậc p+1, có nghiệm tại z = +1

    # Chia bỏ nghiệm tầm thường z = -1 khỏi P và z = +1 khỏi Q
    P_red, _ = np.polydiv(P, np.array([1.0, 1.0]))
    Q_red, _ = np.polydiv(Q, np.array([1.0, -1.0]))

    rp = np.roots(P_red)
    rq = np.roots(Q_red)
    ang_p = np.sort(np.angle(rp[np.imag(rp) > 0]))
    ang_q = np.sort(np.angle(rq[np.imag(rq) > 0]))

    if len(ang_p) != p // 2 or len(ang_q) != p // 2:
        # Dự phòng nếu có suy biến số học
        return np.linspace(0.15, np.pi - 0.15, p)

    lsf = np.empty(p, dtype=np.float64)
    lsf[0::2] = ang_p
    lsf[1::2] = ang_q
    lsf = np.sort(np.clip(lsf, 0.02, np.pi - 0.02))
    return lsf


def lsf_to_lpc(lsf: np.ndarray) -> np.ndarray:
    """Khôi phục hệ số LPC ổn định [1, a_1, ..., a_p] từ vector LSF."""
    p = len(lsf)
    lsf = np.sort(np.clip(lsf, 0.02, np.pi - 0.02))
    # Đảm bảo khoảng cách tối thiểu giữa các LSF liên tiếp để bộ lọc luôn ổn định
    for i in range(1, p):
        if lsf[i] <= lsf[i - 1] + 0.01:
            lsf[i] = min(np.pi - 0.015, lsf[i - 1] + 0.01)

    ang_p = lsf[0::2]
    ang_q = lsf[1::2]
    P_red = np.array([1.0])
    for w in ang_p:
        P_red = np.convolve(P_red, [1.0, -2.0 * np.cos(w), 1.0])
    Q_red = np.array([1.0])
    for w in ang_q:
        Q_red = np.convolve(Q_red, [1.0, -2.0 * np.cos(w), 1.0])

    P = np.convolve(P_red, [1.0, 1.0])
    Q = np.convolve(Q_red, [1.0, -1.0])
    a = 0.5 * (P + Q)
    return a[: p + 1]


def lpc_formants(a: np.ndarray, sr: int = SR) -> tuple[float, float]:
    """Ước lượng formant F1, F2 (Hz) từ nghiệm đa thức LPC A(z)."""
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0.01]
    freqs = np.angle(roots) * (sr / (2.0 * np.pi))
    bws = -0.5 * (sr / (2.0 * np.pi)) * np.log(np.maximum(np.abs(roots), 1e-6))
    valid = freqs[(freqs > 180) & (freqs < 4000) & (bws < 500)]
    valid = np.sort(valid)
    f1 = float(valid[0]) if len(valid) >= 1 else 500.0
    f2 = float(valid[1]) if len(valid) >= 2 else 1500.0
    return f1, f2


def compute_delta(x: np.ndarray) -> np.ndarray:
    """Tính đạo hàm bậc 1 (Delta feature) theo công thức sai phân trung tâm."""
    T = len(x)
    d = np.zeros_like(x)
    if T == 1:
        return d
    d[0] = x[1] - x[0]
    d[-1] = x[-1] - x[-2]
    if T > 2:
        d[1:-1] = 0.5 * (x[2:] - x[:-2])
    return d


def extract_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trích xuất ma trận đặc trưng tĩnh [log_f0, log_rms, LSF(16)] và mặt nạ VUV."""
    x = x.astype(np.float64)
    # Pad để khớp số khung theo center=True của YIN
    pad = WIN_LEN // 2
    xp = np.pad(x, (pad, pad))
    n_frames = 1 + (len(xp) - WIN_LEN) // HOP_LEN

    f0_raw = yin_f0(x, sr=SR, fmin=75.0, fmax=400.0, frame_length=1024, hop_length=HOP_LEN)
    if len(f0_raw) < n_frames:
        f0_raw = np.pad(f0_raw, (0, n_frames - len(f0_raw)), mode="edge")
    f0_raw = f0_raw[:n_frames]

    win = np.hamming(WIN_LEN)
    feats = np.zeros((n_frames, FEAT_DIM), dtype=np.float64)
    vuv = np.zeros(n_frames, dtype=np.float64)

    for t in range(n_frames):
        s = t * HOP_LEN
        frm = xp[s : s + WIN_LEN]
        rms_val = float(np.sqrt(np.mean(frm ** 2) + 1e-12))
        feats[t, 1] = np.log(max(rms_val, 1e-5))

        # Kiểm tra hữu thanh dựa trên năng lượng & độ tuần hoàn tự tương quan chuẩn hóa
        frm_w = frm * win
        ac = np.correlate(frm_w, frm_w, mode="full")[WIN_LEN - 1 :]
        ac0 = ac[0] + 1e-12
        lag_min, lag_max = int(SR / 400), int(SR / 75)
        peak_ac = float(np.max(ac[lag_min : min(lag_max, len(ac))]) / ac0) if len(ac) > lag_min else 0.0
        is_voiced = (rms_val > 0.012) and (75.0 < f0_raw[t] < 395.0) and (peak_ac > 0.28)
        vuv[t] = 1.0 if is_voiced else 0.0

        # Pre-emphasis trước khi phân tích LPC để tách formant sắc nét
        frm_pe = np.empty_like(frm)
        frm_pe[0] = frm[0]
        frm_pe[1:] = frm[1:] - PRE_EMPH * frm[:-1]
        a = levinson_lpc(frm_pe * win, LPC_ORDER)
        feats[t, 2:] = lpc_to_lsf(a)

    # Nội suy log-F0 liên tục qua vùng vô thanh để MLPG mượt
    voiced_idx = np.where(vuv > 0.5)[0]
    if len(voiced_idx) >= 2:
        log_f0 = np.interp(np.arange(n_frames), voiced_idx, np.log(f0_raw[voiced_idx]))
    elif len(voiced_idx) == 1:
        log_f0 = np.full(n_frames, np.log(f0_raw[voiced_idx[0]]))
    else:
        log_f0 = np.full(n_frames, np.log(200.0))
    feats[:, 0] = log_f0
    return feats, vuv


def train_hmm_model(meta_path: str = DEFAULT_META_PATH, save_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Huấn luyện mô hình HMM 5 trạng thái left-to-right cho từng từ từ metadata.csv."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    word_files: dict[str, list[str]] = {}
    with open(meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row["file_path"]
            if not os.path.isabs(p):
                p = os.path.join(ROOT, p)
            word_files.setdefault(row["label"], []).append(p)

    model_data = {}
    for word, paths in word_files.items():
        seq_feats = []
        seq_deltas = []
        seq_vuv = []
        for p in paths:
            x, fs = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            assert fs == SR
            ft, vuv = extract_features(x)
            seq_feats.append(ft)
            seq_deltas.append(compute_delta(ft))
            seq_vuv.append(vuv)

        # Khởi tạo phân đoạn đều theo N_STATES, sau đó tinh chỉnh bằng Viterbi-EM (3 vòng)
        seg_bounds = [np.linspace(0, len(ft), N_STATES + 1, dtype=int) for ft in seq_feats]
        for _ in range(3):
            state_obs = [[] for _ in range(N_STATES)]
            state_dlt = [[] for _ in range(N_STATES)]
            state_vuv = [[] for _ in range(N_STATES)]
            state_dur = [[] for _ in range(N_STATES)]
            for ft, dl, vv, bnd in zip(seq_feats, seq_deltas, seq_vuv, seg_bounds):
                for s in range(N_STATES):
                    i0, i1 = bnd[s], max(bnd[s] + 1, bnd[s + 1])
                    state_obs[s].append(ft[i0:i1])
                    state_dlt[s].append(dl[i0:i1])
                    state_vuv[s].append(vv[i0:i1])
                    state_dur[s].append(i1 - i0)

            mu_s = np.zeros((N_STATES, FEAT_DIM))
            var_s = np.ones((N_STATES, FEAT_DIM))
            mu_d = np.zeros((N_STATES, FEAT_DIM))
            var_d = np.ones((N_STATES, FEAT_DIM))
            vuv_p = np.zeros(N_STATES)
            dur_m = np.zeros(N_STATES)

            min_var = np.full(FEAT_DIM, 1e-4)
            min_var[0] = 0.002   # log-F0
            min_var[1] = 0.02    # log-energy

            for s in range(N_STATES):
                obs_cat = np.vstack(state_obs[s])
                dlt_cat = np.vstack(state_dlt[s])
                vuv_cat = np.concatenate(state_vuv[s])
                mu_s[s] = np.mean(obs_cat, axis=0)
                var_s[s] = np.maximum(np.var(obs_cat, axis=0), min_var)
                mu_d[s] = np.mean(dlt_cat, axis=0)
                var_d[s] = np.maximum(np.var(dlt_cat, axis=0), min_var * 0.25)
                vuv_p[s] = float(np.mean(vuv_cat))
                dur_m[s] = float(np.mean(state_dur[s]))

            # Cập nhật biên bằng Viterbi left-to-right trên khung của từng bản ghi
            new_bounds = []
            for ft in seq_feats:
                T = len(ft)
                dp = np.full((T, N_STATES), 1e18)
                bp = np.zeros((T, N_STATES), dtype=int)
                dp[0, 0] = 0.5 * np.sum((ft[0] - mu_s[0]) ** 2 / var_s[0])
                for t in range(1, T):
                    for s in range(N_STATES):
                        emit = 0.5 * np.sum((ft[t] - mu_s[s]) ** 2 / var_s[s])
                        best_prev = dp[t - 1, s]
                        best_s = s
                        if s > 0 and dp[t - 1, s - 1] < best_prev:
                            best_prev = dp[t - 1, s - 1]
                            best_s = s - 1
                        dp[t, s] = best_prev + emit
                        bp[t, s] = best_s
                path = np.zeros(T, dtype=int)
                path[-1] = N_STATES - 1
                for t in range(T - 2, -1, -1):
                    path[t] = bp[t + 1, path[t + 1]]
                bnd = [0]
                for s in range(N_STATES):
                    idx = np.where(path == s)[0]
                    bnd.append(int(idx[-1] + 1) if len(idx) > 0 else bnd[-1] + 1)
                bnd[-1] = T
                new_bounds.append(np.array(bnd))
            seg_bounds = new_bounds

        # Ma trận chuyển trạng thái A (left-to-right Bakis)
        A = np.zeros((N_STATES, N_STATES))
        for s in range(N_STATES):
            if s < N_STATES - 1:
                p_exit = min(0.9, 1.0 / max(dur_m[s], 1.5))
                A[s, s] = 1.0 - p_exit
                A[s, s + 1] = p_exit
            else:
                A[s, s] = 1.0

        model_data[word] = {
            "mu_s": mu_s,
            "var_s": var_s,
            "mu_d": mu_d,
            "var_d": var_d,
            "vuv_p": vuv_p,
            "dur_m": dur_m,
            "A": A,
        }

    # Lưu nén vào file npz
    flat = {}
    words = sorted(model_data.keys())
    flat["words"] = np.array(words)
    for w in words:
        for k, v in model_data[w].items():
            flat[f"{w}::{k}"] = v
    np.savez(save_path, **flat)
    return model_data


def load_hmm_model(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    if model_path in _HMM_CACHE:
        return _HMM_CACHE[model_path]
    if not os.path.isfile(model_path):
        data = train_hmm_model(DEFAULT_META_PATH, model_path)
        _HMM_CACHE[model_path] = data
        return data
    npz = np.load(model_path, allow_pickle=False)
    words = list(npz["words"])
    data = {}
    for w in words:
        data[w] = {
            k: npz[f"{w}::{k}"]
            for k in ("mu_s", "var_s", "mu_d", "var_d", "vuv_p", "dur_m", "A")
        }
    _HMM_CACHE[model_path] = data
    return data


def mlpg_solve_1d(mu_s: np.ndarray, var_s: np.ndarray, mu_d: np.ndarray, var_d: np.ndarray) -> np.ndarray:
    """Giải hệ tuyến tính MLPG 1 chiều: (W^T Sigma^-1 W) c = W^T Sigma^-1 mu."""
    T = len(mu_s)
    if T <= 2:
        return mu_s.copy()

    # Xây dựng ma trận W (2T x T): T hàng đầu là tĩnh I, T hàng sau là ma trận sai phân D
    W_s = np.eye(T)
    W_d = np.zeros((T, T))
    W_d[0, 0], W_d[0, 1] = -1.0, 1.0
    W_d[-1, -2], W_d[-1, -1] = -1.0, 1.0
    for t in range(1, T - 1):
        W_d[t, t - 1] = -0.5
        W_d[t, t + 1] = 0.5

    inv_vs = 1.0 / np.maximum(var_s, 1e-6)
    inv_vd = 1.0 / np.maximum(var_d, 1e-6)

    R_mat = np.diag(inv_vs) + W_d.T @ np.diag(inv_vd) @ W_d
    r_vec = inv_vs * mu_s + W_d.T @ (inv_vd * mu_d)
    return linalg.solve(R_mat, r_vec, assume_a="pos")


def source_filter_vocoder(
    f0_traj: np.ndarray,
    energy_traj: np.ndarray,
    lsf_traj: np.ndarray,
    vuv_traj: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bộ tổng hợp source-filter liên tục pha và trạng thái bộ lọc IIR qua các khung."""
    T = len(f0_traj)
    y_pre = np.zeros(T * HOP_LEN, dtype=np.float64)
    zi = np.zeros(LPC_ORDER, dtype=np.float64)
    phase = 0.0

    for t in range(T):
        f0 = float(f0_traj[t])
        voiced = bool(vuv_traj[t] >= 0.5)
        exc = np.zeros(HOP_LEN, dtype=np.float64)

        if voiced and f0 > 50.0:
            dphi = 2.0 * np.pi * f0 / SR
            for n in range(HOP_LEN):
                phase += dphi
                if phase >= 2.0 * np.pi:
                    phase -= 2.0 * np.pi
                    exc[n] = 1.0
            # Làm mềm xung glottal + pha thêm chút nhiễu thở tự nhiên
            exc = exc + 0.05 * rng.standard_normal(HOP_LEN)
        else:
            exc = 0.35 * rng.standard_normal(HOP_LEN)

        a = lsf_to_lpc(lsf_traj[t])
        # Lọc qua vocal-tract IIR H(z) = 1 / A(z) giữ nguyên trạng thái zi
        frm_syn, zi = signal.lfilter([1.0], a, exc, zi=zi)
        cur_rms = float(np.sqrt(np.mean(frm_syn ** 2) + 1e-12))
        target_rms = float(energy_traj[t])
        if cur_rms > 1e-6:
            frm_syn = frm_syn * (target_rms / cur_rms)
        y_pre[t * HOP_LEN : (t + 1) * HOP_LEN] = frm_syn

    # De-emphasis nhẹ 1/(1 - 0.90 z^-1) để khôi phục độ ấm giọng tự nhiên
    y = signal.lfilter([1.0], [1.0, -0.90], y_pre)
    # Loại DC và tránh clipping
    y = y - np.mean(y)
    peak = float(np.max(np.abs(y)))
    if peak > 0.92:
        y = y * (0.92 / peak)
    return y.astype(np.float32)


def synthesize_hmm(
    text: str,
    f0_scale: float = 1.0,
    duration_scale: float = 1.0,
    use_delta: bool = True,
    model_path: str = DEFAULT_MODEL_PATH,
    pause_ms: int = 35,
):
    """Tổng hợp câu tiếng nói bằng HMM generation + MLPG + Source-Filter Vocoder.

    Returns:
        y_hmm (np.ndarray float32): waveform đầu ra
        sr (int): 16000 Hz
        tracks (dict): state_sequence, f0, f0_nodelta, energy, f1, f2, lsf, word_spans, latency_s
    """
    t0 = time.perf_counter()
    model = load_hmm_model(model_path)
    tokens = normalize_queue_text(text)
    if not tokens:
        raise ValueError("Văn bản đầu vào rỗng sau chuẩn hóa.")

    pause_frames = max(0, int(round(pause_ms / 10.0)))
    rng = np.random.default_rng(457)

    frame_mu_s, frame_var_s = [], []
    frame_mu_d, frame_var_d = [], []
    frame_vuv, state_seq = [], []
    word_spans = []

    cursor = 0
    for idx, tok in enumerate(tokens):
        if tok not in model:
            raise KeyError(f"Từ '{tok}' không có trong mô hình HMM.")
        w_m = model[tok]
        w_start = cursor
        for s in range(N_STATES):
            dur_f = max(2, int(round(w_m["dur_m"][s] * duration_scale)))
            for _ in range(dur_f):
                frame_mu_s.append(w_m["mu_s"][s])
                frame_var_s.append(w_m["var_s"][s])
                frame_mu_d.append(w_m["mu_d"][s])
                frame_var_d.append(w_m["var_d"][s])
                frame_vuv.append(w_m["vuv_p"][s])
                state_seq.append(idx * N_STATES + s)
            cursor += dur_f
        word_spans.append((tok, w_start, cursor))

        if idx < len(tokens) - 1 and pause_frames > 0:
            # Khoảng lặng chuyển từ ngắn
            sil_s = w_m["mu_s"][-1].copy()
            sil_s[1] = np.log(0.004)
            for _ in range(pause_frames):
                frame_mu_s.append(sil_s)
                frame_var_s.append(w_m["var_s"][-1])
                frame_mu_d.append(np.zeros(FEAT_DIM))
                frame_var_d.append(np.full(FEAT_DIM, 0.01))
                frame_vuv.append(0.0)
                state_seq.append(-1)
            cursor += pause_frames

    mu_s = np.array(frame_mu_s)
    var_s = np.array(frame_var_s)
    mu_d = np.array(frame_mu_d)
    var_d = np.array(frame_var_d)
    vuv_traj = np.array(frame_vuv)

    T = len(mu_s)
    gen_static = np.zeros((T, FEAT_DIM), dtype=np.float64)
    if use_delta:
        for d in range(FEAT_DIM):
            gen_static[:, d] = mlpg_solve_1d(mu_s[:, d], var_s[:, d], mu_d[:, d], var_d[:, d])
    else:
        gen_static = mu_s.copy()

    f0_nodelta = np.exp(mu_s[:, 0]) * f0_scale
    f0_traj = np.exp(gen_static[:, 0]) * f0_scale
    energy_traj = np.exp(gen_static[:, 1])
    lsf_traj = gen_static[:, 2:]

    # Ước lượng formant F1, F2 từ LSF để vẽ và phân tích
    f1_traj = np.zeros(T, dtype=np.float64)
    f2_traj = np.zeros(T, dtype=np.float64)
    for t in range(T):
        a = lsf_to_lpc(lsf_traj[t])
        f1_traj[t], f2_traj[t] = lpc_formants(a, SR)

    y_hmm = source_filter_vocoder(f0_traj, energy_traj, lsf_traj, vuv_traj, rng)
    latency_s = time.perf_counter() - t0

    tracks = {
        "tokens": tokens,
        "state_sequence": np.array(state_seq),
        "f0": f0_traj,
        "f0_nodelta": f0_nodelta,
        "vuv": vuv_traj,
        "energy": energy_traj,
        "lsf": lsf_traj,
        "f1": f1_traj,
        "f2": f2_traj,
        "word_spans": word_spans,
        "frame_shift_ms": 10.0,
        "latency_s": float(latency_s),
    }
    return y_hmm, SR, tracks
