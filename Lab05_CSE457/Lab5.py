"""CSE457 - Lab 5: Speech Enhancement, Speaker Recognition and Speaker Separation.
Chạy: python Lab5.py   (dữ liệu trong lab5_data/, kết quả trong results/)
"""
import os, json, itertools
import numpy as np
import librosa, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.decomposition import FastICA

# ============================ CONFIG ============================
CFG = dict(
    sr=16000, n_fft=512, win_length=400, hop_length=160,     # 25 ms / 10 ms
    snr_list=[0, 5, 10], noises=["fan", "cafeteria"],
    case_clean="spk01_test_01", case_noise="fan", case_snr=5,
    beta_list=[1.0, 1.5, 2.0], floor_list=[0.0, 0.01, 0.05],
    beta_main=1.5, floor_main=0.02,
    n_mfcc=20, speakers=["spk01", "spk02", "spk03", "spk04"],
    n_enroll=4, n_dev=2, n_test=2,
    A_list={"A_easy": [[1.0, 0.6], [0.4, 1.0]],
            "A_medium": [[1.0, 0.8], [0.7, 1.0]],
            "A_near_singular": [[1.0, 0.95], [0.9, 1.0]]},
    ica_noise_snr_db=20, seed=0,
)
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "lab5_data")
AUD = os.path.join(ROOT, "results", "audio"); FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(AUD, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SR = CFG["sr"]; rng = np.random.default_rng(CFG["seed"])
R = {"config": CFG}

# ============================ UTILS ============================
def load16k(path):
    x, _ = librosa.load(path, sr=SR, mono=True)
    return x.astype(np.float64)

def clean_path(name): return os.path.join(DATA, "clean", name + ".wav")
def power(x): return np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12

def fit_len(v, L):
    if len(v) < L: v = np.tile(v, int(np.ceil(L / len(v))))
    return v[:L]

def add_noise_at_snr(s, v, snr):
    v = fit_len(v, len(s))
    alpha = np.sqrt(power(s) / (power(v) * 10 ** (snr / 10)))
    return s + alpha * v, alpha

def snr_db(ref, test):
    L = min(len(ref), len(test)); ref, test = ref[:L], test[:L]
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum((test - ref) ** 2) + 1e-12))

def save(name, x, peak_norm=False):
    x = np.asarray(x, dtype=np.float64)
    if peak_norm: x = 0.9 * x / (np.max(np.abs(x)) + 1e-12)
    elif np.max(np.abs(x)) > 0.999: x = x / np.max(np.abs(x)) * 0.999  # tránh clip khi ghi file
    sf.write(os.path.join(AUD, name), x.astype(np.float32), SR, subtype="PCM_16")

def stft(x):
    return librosa.stft(x, n_fft=CFG["n_fft"], hop_length=CFG["hop_length"], win_length=CFG["win_length"], window="hann")

def istft(X, L):
    return librosa.istft(X, hop_length=CFG["hop_length"], win_length=CFG["win_length"], window="hann", length=L)

def noise_psd(noise):
    return np.mean(np.abs(stft(noise)) ** 2, axis=1, keepdims=True)   # P̂v[k] = mean_m |V[m,k]|^2

def spectral_subtraction(y, Pn, beta, floor):
    Y = stft(y); Py = np.abs(Y) ** 2
    Pss = np.maximum(Py - beta * Pn, floor * Pn)
    return istft(np.sqrt(Pss) * np.exp(1j * np.angle(Y)), len(y))

def wiener(y, Pn):
    Y = stft(y); Py = np.abs(Y) ** 2
    Ps = np.maximum(Py - Pn, 0.0)
    H = Ps / (Ps + Pn + 1e-12)
    return istft(H * Y, len(y))

def spec_db(x):
    return librosa.amplitude_to_db(np.abs(stft(x)) + 1e-10, ref=1.0)

def plot_specs(sigs, titles, fname, wave=False):
    n = len(sigs)
    fig, ax = plt.subplots(n, 2 if wave else 1, figsize=(12 if wave else 8, 2.3 * n), squeeze=False)
    vmax = max(spec_db(s).max() for s in sigs)
    for i, (s, t) in enumerate(zip(sigs, titles)):
        a = ax[i, -1]
        im = a.imshow(spec_db(s), origin="lower", aspect="auto", cmap="magma", vmin=vmax - 80, vmax=vmax,
                      extent=[0, len(s) / SR, 0, SR / 2000])
        a.set_title(t + " - spectrogram"); a.set_ylabel("kHz")
        if wave:
            ax[i, 0].plot(np.arange(len(s)) / SR, s, lw=0.5); ax[i, 0].set_title(t + " - waveform")
            ax[i, 0].set_xlim(0, len(s) / SR)
    ax[-1, -1].set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=120); plt.close(fig)

# ======================= PART 1: NOISY SPEECH =======================
print("== Part 1: noisy speech ==")
s = load16k(clean_path(CFG["case_clean"]))
noises = {n: load16k(os.path.join(DATA, "noise", n + ".wav")) for n in CFG["noises"]}
# Dùng nửa đầu file noise để trộn, nửa sau để ước lượng PSD (không "nhìn trộm" chính mẫu nhiễu đã cộng)
noise_mix = {n: v[: len(v) // 2] for n, v in noises.items()}
noise_est = {n: v[len(v) // 2:] for n, v in noises.items()}
save("clean.wav", s)
R["part1"] = []
noisy = {}
for n in CFG["noises"]:
    for snr in CFG["snr_list"]:
        y, alpha = add_noise_at_snr(s, noise_mix[n], snr)
        meas = snr_db(s, y)
        noisy[(n, snr)] = y
        save(f"noisy_{n}_{snr}dB.wav", y)
        R["part1"].append(dict(noise=n, target=snr, measured=round(meas, 3), err=round(meas - snr, 4),
                               alpha=round(alpha, 4), P_s=power(s), P_v=power(fit_len(noise_mix[n], len(s)))))
        print(f"  {n:9s} target {snr:2d} dB -> measured {meas:.3f} dB (alpha={alpha:.3f})")
save("noisy_5dB.wav", noisy[(CFG["case_noise"], 5)])
plot_specs([s] + [noisy[("fan", k)] for k in CFG["snr_list"]],
           ["clean"] + [f"noisy fan {k} dB" for k in CFG["snr_list"]], "fig1_clean_noisy_fan.png", wave=True)
plot_specs([s] + [noisy[("cafeteria", k)] for k in CFG["snr_list"]],
           ["clean"] + [f"noisy cafeteria {k} dB" for k in CFG["snr_list"]], "fig1_clean_noisy_cafeteria.png", wave=True)

# ======================= PART 2-3: ENHANCEMENT =======================
print("== Part 2-3: spectral subtraction & Wiener ==")
R["enh_main"] = []; R["beta_sweep"] = []; R["floor_sweep"] = []
for n in CFG["noises"]:
    Pn = noise_psd(noise_est[n])
    for snr in CFG["snr_list"]:
        y = noisy[(n, snr)]; si = snr_db(s, y)
        xss = spectral_subtraction(y, Pn, CFG["beta_main"], CFG["floor_main"]); xw = wiener(y, Pn)
        R["enh_main"].append(dict(noise=n, snr_in=round(si, 2), ss_out=round(snr_db(s, xss), 2),
                                  ss_d=round(snr_db(s, xss) - si, 2), w_out=round(snr_db(s, xw), 2),
                                  w_d=round(snr_db(s, xw) - si, 2)))
        save(f"enhanced_ss_{n}_{snr}dB.wav", xss); save(f"enhanced_wiener_{n}_{snr}dB.wav", xw)
        for b in CFG["beta_list"]:
            for g in CFG["floor_list"]:
                xo = spectral_subtraction(y, Pn, b, g)
                R["beta_sweep"].append(dict(noise=n, snr_in=snr, beta=b, floor=g, snr_out=round(snr_db(s, xo), 2),
                                            dsnr=round(snr_db(s, xo) - si, 2)))
                if n == "fan" and snr == 5: save(f"ss_fan5dB_beta{b}_floor{g}.wav", xo)
        if n == CFG["case_noise"] and snr == CFG["case_snr"]:
            save("enhanced_ss.wav", xss); save("enhanced_wiener.wav", xw)
            plot_specs([s, y, xss, xw], ["clean", f"noisy {n} {snr} dB", f"spectral subtraction (β={CFG['beta_main']}, γ={CFG['floor_main']})", "Wiener"],
                       "fig2_spec_clean_noisy_enhanced.png")
            # gain map minh họa musical noise: so sánh gain SS vs Wiener
            Y = stft(y); Py = np.abs(Y) ** 2
            Gss = np.sqrt(np.maximum(Py - 2.0 * Pn, 0.0 * Pn) / (Py + 1e-12))
            Gw = np.maximum(Py - Pn, 0) / (np.maximum(Py - Pn, 0) + Pn + 1e-12)
            fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
            for a, G, t in zip(ax, [Gss, Gw], ["SS gain β=2, γ=0 (lỗ phổ → musical noise)", "Wiener gain"]):
                a.imshow(G, origin="lower", aspect="auto", cmap="gray", vmin=0, vmax=1, extent=[0, len(y) / SR, 0, 8]); a.set_title(t)
                a.set_xlabel("Time (s)"); a.set_ylabel("kHz")
            fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_gain_maps.png"), dpi=120); plt.close(fig)
for r in R["enh_main"]: print("  ", r)

# Hình ΔSNR theo β và γ (fan & cafeteria)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for a, n in zip(ax, CFG["noises"]):
    for g in CFG["floor_list"]:
        for snr, ls in zip(CFG["snr_list"], ["-", "--", ":"]):
            v = [r["dsnr"] for r in R["beta_sweep"] if r["noise"] == n and r["floor"] == g and r["snr_in"] == snr]
            a.plot(CFG["beta_list"], v, ls, marker="o", label=f"γ={g}, SNRin={snr}")
    a.set_title(f"ΔSNR spectral subtraction - {n}"); a.set_xlabel("β"); a.set_ylabel("ΔSNR (dB)"); a.grid(alpha=.3)
ax[1].legend(fontsize=7, ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_dsnr_beta_floor.png"), dpi=120); plt.close(fig)

# ======================= PART 4: SPEAKER ID =======================
print("== Part 4: speaker identification ==")
SPK = CFG["speakers"]
def files(part, n): return {sp: [f"{sp}_{part}_{k:02d}" for k in range(1, n + 1)] for sp in SPK}
ENR, DEV, TEST = files("enroll", CFG["n_enroll"]), files("dev", CFG["n_dev"]), files("test", CFG["n_test"])
AUDIO = {f: load16k(clean_path(f)) for d in (ENR, DEV, TEST) for fl in d.values() for f in fl}

def raw_embedding(x, use_delta=True, cmn=False):
    M = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=CFG["n_mfcc"] + 1, n_fft=CFG["n_fft"], win_length=CFG["win_length"],
                             hop_length=CFG["hop_length"], window="hann")[1:]   # bỏ c0 (năng lượng) -> dùng c1..cD
    if cmn: M = M - M.mean(axis=1, keepdims=True)   # c_t - c̄
    parts = [M.mean(axis=1), M.std(axis=1)]
    if use_delta:
        Dl = librosa.feature.delta(M); parts += [Dl.mean(axis=1), Dl.std(axis=1)]
    e = np.concatenate(parts)
    return e

_MU = {}
def speaker_embedding(x, use_delta=True, cmn=False):
    """Embedding = raw stats - mean toàn cục của tập ENROLLMENT (chỉ dùng dữ liệu train), rồi L2-normalize.
    Centering giúp cosine không bị 'nén' về ~1 do các thành phần dương lớn chung (σ, μ(c1))."""
    key = (use_delta, cmn)
    if key not in _MU:
        _MU[key] = np.mean([raw_embedding(AUDIO[f], use_delta, cmn) for fl in ENR.values() for f in fl], axis=0)
    e = raw_embedding(x, use_delta, cmn) - _MU[key]
    return e / (norm(e) + 1e-12)

def cosine(a, b): return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))

def build_models(opts, n_enroll=None):
    m = {}
    for sp in SPK:
        E = np.array([speaker_embedding(AUDIO[f], **opts) for f in ENR[sp][: n_enroll or CFG["n_enroll"]]])
        mu = E.mean(axis=0); m[sp] = mu / norm(mu)
    return m

def identify(models, x, opts):
    e = speaker_embedding(x, **opts)
    sc = {sp: cosine(e, models[sp]) for sp in SPK}
    srt = sorted(sc.values(), reverse=True)
    return max(sc, key=sc.get), sc, srt[0] - srt[1]

def eval_id(models, opts, setd=TEST, transform=None):
    cm = np.zeros((len(SPK), len(SPK)), int); rows = []; margins_true = []
    for sp in SPK:
        for f in setd[sp]:
            x = AUDIO[f] if transform is None else transform(f, AUDIO[f])
            p, sc, _ = identify(models, x, opts)
            cm[SPK.index(sp), SPK.index(p)] += 1
            other = max(v for k, v in sc.items() if k != sp)
            margins_true.append(sc[sp] - other)
            rows.append(dict(file=f, true=sp, pred=p, **{k: round(v, 4) for k, v in sc.items()}))
    return dict(acc=float(np.trace(cm) / cm.sum()), cm=cm.tolist(), rows=rows,
                mean_margin=float(np.mean(margins_true)), min_margin=float(np.min(margins_true)))

VARIANTS = {"MFCC": dict(use_delta=False, cmn=False), "MFCC+delta": dict(use_delta=True, cmn=False),
            "MFCC+delta+CMN": dict(use_delta=True, cmn=True)}
R["spkid"] = {}
for vn, opts in VARIANTS.items():
    res = eval_id(build_models(opts), opts)
    R["spkid"][vn] = res
    print(f"  {vn:15s} acc={res['acc']:.3f} mean margin={res['mean_margin']:.4f} min margin={res['min_margin']:.4f}")
MAIN = VARIANTS["MFCC+delta"]
MODELS = build_models(MAIN)
R["enroll_tokens"] = []
for ne in [1, 2, 4]:
    res = eval_id(build_models(MAIN, ne), MAIN)
    R["enroll_tokens"].append(dict(n=ne, acc=res["acc"], mean_margin=round(res["mean_margin"], 4), min_margin=round(res["min_margin"], 4)))
    print(f"  enroll={ne}: acc={res['acc']:.3f}, mean margin={res['mean_margin']:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, vn in zip(ax, VARIANTS):
    cm = np.array(R["spkid"][vn]["cm"])
    a.imshow(cm, cmap="Blues"); a.set_xticks(range(len(SPK)), SPK); a.set_yticks(range(len(SPK)), SPK)
    for i, j in itertools.product(range(len(SPK)), repeat=2): a.text(j, i, cm[i, j], ha="center", va="center")
    a.set_title(f"{vn} (acc={R['spkid'][vn]['acc']:.2f})"); a.set_xlabel("Predicted"); a.set_ylabel("True")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_confusion_matrix.png"), dpi=120); plt.close(fig)

# ======================= PART 5: VERIFICATION =======================
print("== Part 5: speaker verification ==")
def trials(setd):
    out = []
    for sp in SPK:
        for f in setd[sp]:
            e = speaker_embedding(AUDIO[f], **MAIN)
            for claim in SPK: out.append((cosine(e, MODELS[claim]), int(claim == sp)))
    return out
dev_scores, test_scores = trials(DEV), trials(TEST)
def far_frr(scores, th):
    g = np.array([s for s, l in scores if l == 1]); im = np.array([s for s, l in scores if l == 0])
    return float(np.mean(im >= th)), float(np.mean(g < th))
ths = np.linspace(-1, 1, 4001)
dev_curve = np.array([far_frr(dev_scores, t) for t in ths])
# θ gần EER trên dev; nếu có cả dải EER=0 thì lấy trung điểm dải đó (bền hơn)
gap = np.abs(dev_curve[:, 0] - dev_curve[:, 1]); best_gap = gap.min()
cand = ths[gap <= best_gap + 1e-12]
cand_err = [sum(far_frr(dev_scores, t)) for t in cand]
cand = cand[np.array(cand_err) <= min(cand_err) + 1e-12]
theta = float(np.median(cand))
far_d, frr_d = far_frr(dev_scores, theta); far_t, frr_t = far_frr(test_scores, theta)
R["verif"] = dict(theta=theta, dev_FAR=far_d, dev_FRR=frr_d, test_FAR=far_t, test_FRR=frr_t,
                  n_dev_gen=sum(l for _, l in dev_scores), n_dev_imp=sum(1 - l for _, l in dev_scores),
                  n_test_gen=sum(l for _, l in test_scores), n_test_imp=sum(1 - l for _, l in test_scores),
                  dev_gen_min=min(s for s, l in dev_scores if l), dev_imp_max=max(s for s, l in dev_scores if not l),
                  test_gen_min=min(s for s, l in test_scores if l), test_imp_max=max(s for s, l in test_scores if not l))
print("  ", R["verif"])
test_curve = np.array([far_frr(test_scores, t) for t in ths])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for a, c, t in zip(ax, [dev_curve, test_curve], ["Dev (chọn θ)", "Test (θ đã khóa)"]):
    a.plot(ths, c[:, 0], label="FAR"); a.plot(ths, c[:, 1], label="FRR"); a.axvline(theta, color="k", ls="--", label=f"θ={theta:.3f}")
    a.set_xlim(min(s for s, _ in dev_scores + test_scores) - 0.02, 1.0); a.set_xlabel("θ (cosine)"); a.set_title(t); a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_far_frr.png"), dpi=120); plt.close(fig)
fig, a = plt.subplots(figsize=(6, 3.5))
a.hist([s for s, l in dev_scores + test_scores if l], bins=20, alpha=.7, label="genuine")
a.hist([s for s, l in dev_scores + test_scores if not l], bins=40, alpha=.7, label="impostor")
a.axvline(theta, color="k", ls="--"); a.legend(); a.set_xlabel("cosine score"); a.set_title("Phân bố score")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_score_hist.png"), dpi=120); plt.close(fig)

# ======================= PART 6-7: ICA =======================
print("== Part 6-7: ICA source separation ==")
def rms_normalize(x): return x / (np.sqrt(np.mean(x * x)) + 1e-12)
s1 = load16k(os.path.join(DATA, "separation", "source_A.wav")); s2 = load16k(os.path.join(DATA, "separation", "source_B.wav"))
L = min(len(s1), len(s2)); S = np.vstack([rms_normalize(s1[:L]), rms_normalize(s2[:L])])
R["ica"] = []

def corr_mat(S, Sh): return np.array([[np.corrcoef(S[i], Sh[j])[0, 1] for j in range(2)] for i in range(2)])

def sdr_like(ref, est):  # SNR sau khi khớp scale tối ưu (least squares)
    g = np.dot(est, ref) / np.dot(est, est); return snr_db(ref, g * est)

def run_ica(A, noise_snr=None, tag=""):
    A = np.array(A); X = A @ S
    if noise_snr is not None:
        X = X + np.vstack([rng.standard_normal(L) * np.sqrt(power(X[i]) / 10 ** (noise_snr / 10)) for i in range(2)])
    ica = FastICA(n_components=2, whiten="unit-variance", random_state=CFG["seed"], max_iter=2000)
    Sh = ica.fit_transform(X.T).T
    C = corr_mat(S, Sh); Cabs = np.abs(C)
    perm = (0, 1) if Cabs[0, 0] + Cabs[1, 1] >= Cabs[0, 1] + Cabs[1, 0] else (1, 0)
    Sm = np.vstack([np.sign(C[i, perm[i]]) * Sh[perm[i]] for i in range(2)])
    Sm = np.vstack([rms_normalize(Sm[i]) for i in range(2)])
    Cafter = corr_mat(S, Sm)
    mix_sdr = [sdr_like(S[i], X[i]) for i in range(2)]
    out_sdr = [sdr_like(S[i], Sm[i]) for i in range(2)]
    r = dict(tag=tag, A=A.tolist(), det=float(np.linalg.det(A)), cond=float(np.linalg.cond(A)), noise_snr=noise_snr,
             C_before=np.round(Cabs, 4).tolist(), C_after=np.round(Cafter, 4).tolist(), perm=list(perm),
             mix_sdr=[round(v, 2) for v in mix_sdr], sep_sdr=[round(v, 2) for v in out_sdr],
             improvement=[round(o - m, 2) for o, m in zip(out_sdr, mix_sdr)])
    R["ica"].append(r)
    print(f"  {tag:30s} det={r['det']:.3f} |C|before={r['C_before']} SDR mix={r['mix_sdr']} sep={r['sep_sdr']}")
    return X, Sm, r

for name, A in CFG["A_list"].items():
    X, Sm, r = run_ica(A, None, name)
    if name == "A_easy":
        save("source_A.wav", S[0], True); save("source_B.wav", S[1], True)
        save("mix_mic1.wav", X[0], True); save("mix_mic2.wav", X[1], True)
        save("separated_1.wav", Sm[0], True); save("separated_2.wav", Sm[1], True)
        sigs = [S[0], S[1], X[0], X[1], Sm[0], Sm[1]]
        tit = ["source A", "source B", "mix mic1", "mix mic2", "separated 1", "separated 2"]
        fig, ax = plt.subplots(6, 2, figsize=(12, 12))
        for i, (sg, t) in enumerate(zip(sigs, tit)):
            ax[i, 0].plot(np.arange(L) / SR, sg, lw=.4); ax[i, 0].set_title(t); ax[i, 0].set_xlim(0, L / SR)
            ax[i, 1].imshow(spec_db(sg), origin="lower", aspect="auto", cmap="magma", extent=[0, L / SR, 0, 8]); ax[i, 1].set_title(t)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_ica_waveforms.png"), dpi=110); plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        for a, M, t in zip(ax, [np.array(r["C_before"]), np.abs(np.array(r["C_after"]))], ["|ρ| trước ghép", "|ρ| sau ghép permutation/sign"]):
            a.imshow(M, cmap="viridis", vmin=0, vmax=1)
            for i, j in itertools.product(range(2), repeat=2): a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="w")
            a.set_xticks([0, 1], ["est 1", "est 2"]); a.set_yticks([0, 1], ["source A", "source B"]); a.set_title(t)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_ica_correlation.png"), dpi=120); plt.close(fig)
for snr in [30, CFG["ica_noise_snr_db"], 10]:
    X, Sm, r = run_ica(CFG["A_list"]["A_easy"], snr, f"A_easy + noise {snr} dB")
    if snr == CFG["ica_noise_snr_db"]:
        save("mix_mic1_noisy.wav", X[0], True); save("separated_1_noisy.wav", Sm[0], True)
for name in ["A_medium", "A_near_singular"]:
    run_ica(CFG["A_list"][name], CFG["ica_noise_snr_db"], f"{name} + noise {CFG['ica_noise_snr_db']} dB")

# ======================= PART 8: INTEGRATED =======================
print("== Part 8: speaker-ID on noisy vs enhanced ==")
R["integrated"] = []
Pn_all = {n: noise_psd(noise_est[n]) for n in CFG["noises"]}
for n in CFG["noises"]:
    for snr in CFG["snr_list"]:
        def mk(kind):
            def tf(f, x):
                y, _ = add_noise_at_snr(x, noise_mix[n], snr)
                if kind == "noisy": return y
                if kind == "ss": return spectral_subtraction(y, Pn_all[n], CFG["beta_main"], CFG["floor_main"])
                return wiener(y, Pn_all[n])
            return tf
        row = dict(noise=n, snr=snr)
        for kind in ["noisy", "ss", "wiener"]:
            res = eval_id(MODELS, MAIN, TEST, mk(kind))
            row[kind + "_acc"] = res["acc"]; row[kind + "_margin"] = round(res["mean_margin"], 4)
        R["integrated"].append(row); print("  ", row)
R["integrated_clean"] = dict(acc=R["spkid"]["MFCC+delta"]["acc"], margin=round(R["spkid"]["MFCC+delta"]["mean_margin"], 4))

# Case study: một utterance qua clean/noisy/SS/Wiener
case = []
y5 = noisy[(CFG["case_noise"], CFG["case_snr"])]; Pn = Pn_all[CFG["case_noise"]]
for kind, x in [("clean", s), ("noisy_5dB", y5), ("enhanced_ss", spectral_subtraction(y5, Pn, CFG["beta_main"], CFG["floor_main"])),
                ("enhanced_wiener", wiener(y5, Pn))]:
    p, sc, mg = identify(MODELS, x, MAIN)
    case.append(dict(input=kind, pred=p, score_true=round(sc["spk01"], 4), margin=round(mg, 4), **{k: round(v, 4) for k, v in sc.items()}))
R["case_study_id"] = case
for c in case: print("  ", c)

with open(os.path.join(ROOT, "results", "results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=float)
print("Done. Results in results/")
