"""Sinh report.docx từ results/results.json + results/figures. Chạy sau Lab5.py: python make_report.py"""
import os, json
from docx import Document
from docx.shared import Pt, Cm

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "results", "results.json"), encoding="utf-8"))
FIG = os.path.join(ROOT, "results", "figures")
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)

def H(t, l=1): doc.add_heading(t, level=l)
def P(t): doc.add_paragraph(t)
def B(t): doc.add_paragraph(t, style="List Bullet")
def IMG(f, w=16): doc.add_picture(os.path.join(FIG, f), width=Cm(w))
def T(header, rows):
    tb = doc.add_table(rows=1, cols=len(header)); tb.style = "Light Grid Accent 1"
    for c, h in zip(tb.rows[0].cells, header): c.text = str(h)
    for r in rows:
        cs = tb.add_row().cells
        for c, v in zip(cs, r): c.text = f"{v:.3f}" if isinstance(v, float) else str(v)

doc.add_heading("LAB 5 – Speech Enhancement, Speaker Recognition và Speaker Separation", 0)
P("Học phần CSE457 – Xử lý âm thanh và tiếng nói. Mã nguồn: Lab5.py (chạy end-to-end), dữ liệu: lab5_data/, kết quả: results/.")

H("1. Dữ liệu và cấu hình")
P("Dữ liệu được tạo tự động (tools/gen_tts.ps1, tools/prepare_data.py) vì không có bản thu thật: "
  "spk01 = giọng TTS David, spk02 = Zira, spk03 = David dịch pitch+formant +4 semitone, spk04 = Zira dịch −3 semitone. "
  "Mỗi người: 4 enrollment, 2 dev, 2 test (câu khác nhau, không trùng giữa các tập). Mono, 16 kHz, PCM WAV, 2–4 s. "
  "Nhiễu tổng hợp: fan.wav (hum 50 Hz + nhiễu băng rộng/brown điều chế 12 Hz) và cafeteria.wav (babble 20 câu + pink noise). "
  "Nửa đầu file nhiễu dùng để trộn, nửa sau dùng ước lượng PSD (tránh ước lượng trên đúng mẫu nhiễu đã cộng).")
c = R["config"]
T(["Tham số", "Giá trị"], [["sr", c["sr"]], ["win / hop / NFFT", f"{c['win_length']} / {c['hop_length']} / {c['n_fft']} (25 ms/10 ms)"],
   ["SNR", c["snr_list"]], ["β (sweep) / γ (sweep)", f"{c['beta_list']} / {c['floor_list']}"], ["β, γ chính", f"{c['beta_main']}, {c['floor_main']}"],
   ["MFCC D", f"{c['n_mfcc']} (c1..c20, bỏ c0)"], ["Embedding", "[μ,σ của MFCC, μ,σ của Δ], trừ mean toàn cục tập enroll, L2-norm"],
   ["Mixing A", "; ".join(f"{k}={v}" for k, v in c["A_list"].items())]])

H("2. Tạo noisy speech và kiểm tra SNR")
P("y[n] = s[n] + α·v[n], α = sqrt(Ps / (Pv·10^(SNR/10))). SNR đo lại = 10log10(Σs² / Σ(y−s)²).")
T(["Noise", "Target (dB)", "Measured (dB)", "Sai lệch", "α"], [[r["noise"], r["target"], r["measured"], r["err"], r["alpha"]] for r in R["part1"]])
P("Sai lệch ≈ 0 dB (≪ ±0.2 dB) vì không peak-normalize sau khi cộng và noise được cắt đúng độ dài.")
IMG("fig1_clean_noisy_fan.png")

H("3. Spectral subtraction và Wiener")
P("|Ŝ|² = max(|Y|² − βP̂v, γP̂v), giữ phase noisy, ISTFT overlap-add. Wiener: P̂s = max(Py − P̂v, 0), H = P̂s/(P̂s+P̂v+ε).")
T(["Noise", "SNR_in", "SS out", "ΔSNR SS", "Wiener out", "ΔSNR Wiener"],
  [[r["noise"], r["snr_in"], r["ss_out"], r["ss_d"], r["w_out"], r["w_d"]] for r in R["enh_main"]])
IMG("fig2_spec_clean_noisy_enhanced.png", 14)
P("Sweep β, γ (fan noise, ΔSNR dB):")
rows = []
for snr in c["snr_list"]:
    for g in c["floor_list"]:
        rows.append([snr, g] + [next(r["dsnr"] for r in R["beta_sweep"] if r["noise"] == "fan" and r["snr_in"] == snr and r["floor"] == g and r["beta"] == b) for b in c["beta_list"]])
T(["SNR_in", "γ"] + [f"β={b}" for b in c["beta_list"]], rows)
IMG("fig4_dsnr_beta_floor.png"); IMG("fig3_gain_maps.png")
P("Nhận xét nghe/spectrogram: SS (β=1.5, γ=0.02) còn một 'thảm' nhiễu nền dư (do floor) và các đốm năng lượng cô lập "
  "– musical noise – rõ nhất khi γ=0 và β=2 (bản đồ gain lốm đốm các ô 0/1). Wiener cho gain liên tục hơn nên nền sạch hơn, "
  "nhưng ở 0 dB làm mất phụ âm xát năng lượng thấp và có âm sắc 'khô/kim loại'. "
  "Ở SNR 0–5 dB cả hai cải thiện SNR (+2 đến +5 dB); ở 10 dB ΔSNR âm: lượng méo speech do trừ phổ (gain < 1 trên cả bin speech-dominant "
  "và sai số pha/PSD) lớn hơn lượng nhiễu còn phải loại bỏ. Như vậy, metric SNR 'thưởng' thuật toán nhiều nhất khi nhiễu chiếm ưu thế, "
  "còn ở SNR cao nên giảm β hoặc tăng floor.")

H("4. Speaker identification")
T(["Biến thể", "Accuracy", "Margin TB", "Margin min"], [[k, v["acc"], v["mean_margin"], v["min_margin"]] for k, v in R["spkid"].items()])
IMG("fig5_confusion_matrix.png")
P("Bảng score (MFCC+delta, test clean):")
rows = R["spkid"]["MFCC+delta"]["rows"]
T(["File", "True", "Pred"] + c["speakers"], [[r["file"], r["true"], r["pred"]] + [r[s] for s in c["speakers"]] for r in rows])
P("Ablation số enrollment tokens:")
T(["# enroll", "Accuracy", "Margin TB", "Margin min"], [[r["n"], r["acc"], r["mean_margin"], r["min_margin"]] for r in R["enroll_tokens"]])
P("Δ không làm tăng margin trên dữ liệu này (0.808 → 0.805) vì TTS có prosody/động học rất đều giữa các người nói; "
  "CMN làm giảm mạnh margin và accuracy vì nó xoá μ(c) – chính là thông tin đường bao phổ trung bình (độ dài tuyến âm, formant) "
  "phân biệt người nói trong baseline thống kê. Tăng số enrollment từ 1 → 4 làm margin TB tăng 0.50 → 0.80 (model ổn định hơn).")
P("Lưu ý: c0 bị loại và embedding được trừ mean toàn cục tính chỉ trên tập enrollment; nếu không, mọi cosine đều ≈ 0.99 "
  "(margin ≈ 0.006) do các thành phần dương lớn chung, khiến verification gần như không hoạt động.")

H("5. Speaker verification")
v = R["verif"]
P(f"Trials: dev {v['n_dev_gen']} genuine / {v['n_dev_imp']} impostor, test {v['n_test_gen']} / {v['n_test_imp']}. "
  f"θ chọn gần EER trên DEV: θ = {v['theta']:.4f} (dev FAR = {v['dev_FAR']:.3f}, FRR = {v['dev_FRR']:.3f}). "
  f"Khoá θ và đánh giá TEST: FAR = {v['test_FAR']:.3f}, FRR = {v['test_FRR']:.3f}.")
P("Test tốt hơn dev do test set nhỏ (32 trial) và tình cờ dễ hơn; với ít trial, mỗi lỗi = 4–12.5 %, nên kết quả có phương sai lớn. "
  "Đây chính là lý do không chọn θ trên test: θ tối ưu trên test sẽ lạc quan và không tổng quát.")
IMG("fig6_far_frr.png"); IMG("fig7_score_hist.png", 10)

H("6. ICA source separation")
P("x = A·s (2 nguồn RMS-normalize, 2 mic). FastICA(n_components=2, whiten='unit-variance'); A KHÔNG dùng trong ICA. "
  "Permutation/sign giải bằng |ρ| với ground truth; SDR tính sau khi khớp scale tối ưu.")
T(["Thí nghiệm", "det(A)", "cond(A)", "|ρ| trước ghép", "SDR mix (dB)", "SDR tách (dB)"],
  [[r["tag"], round(r["det"], 3), round(r["cond"], 1), str(r["C_before"]), str(r["mix_sdr"]), str(r["sep_sdr"])] for r in R["ica"]])
IMG("fig9_ica_correlation.png", 12); IMG("fig8_ica_waveforms.png", 15)
P("Không nhiễu: ICA tách gần hoàn hảo (|ρ| ≈ 1.000, SDR ≈ 14–15 dB) với MỌI A khả nghịch – ICA có tính equivariant, kết quả chỉ phụ thuộc "
  "nguồn chứ không phụ thuộc A. Output 1 khớp source B → permutation ambiguity hiện ra rõ. "
  "Khi thêm nhiễu cảm biến, det(A) nhỏ trở thành vấn đề: W ≈ A⁻¹ có chuẩn lớn và khuếch đại nhiễu; với A gần suy biến + nhiễu 20 dB "
  "ICA thất bại (|ρ| chỉ 0.64–0.77, SDR ≈ 2–4 dB) trong khi A_easy vẫn đạt ≈ 12 dB.")

H("7. Thí nghiệm tích hợp: speaker-ID trên noisy vs enhanced (8 test utterance)")
P(f"Tham chiếu clean: accuracy = {R['integrated_clean']['acc']:.2f}.")
T(["Noise", "SNR", "Noisy acc", "SS acc", "Wiener acc", "Noisy margin", "SS margin", "Wiener margin"],
  [[r["noise"], r["snr"], r["noisy_acc"], r["ss_acc"], r["wiener_acc"], r["noisy_margin"], r["ss_margin"], r["wiener_margin"]] for r in R["integrated"]])
P("Case study (spk01_test_01, fan 5 dB):")
T(["Input", "Dự đoán", "score spk01", "margin"], [[r["input"], r["pred"], r["score_true"], r["margin"]] for r in R["case_study_id"]])

H("8. Ba kết luận kỹ thuật")
B("Enhancement có lợi khi SNR thấp: ΔSNR SS/Wiener ≈ +4.6/+4.8 dB tại 0 dB (fan) nhưng âm (−2.2/−3.0 dB) tại 10 dB; β lớn hơn và γ = 0 tăng musical noise.")
B("Speaker-ID baseline rất nhạy với mismatch noise: accuracy 1.00 (clean) → 0.25 (noisy fan, mọi SNR). Wiener khôi phục tới 0.75–1.00 với fan "
  "(nhiễu dừng, PSD ước lượng tốt), SS chỉ 0.375; với cafeteria (babble không dừng) cải thiện ít hơn. Enhancement không phải lúc nào cũng giúp.")
B("ICA tách hoàn hảo khi mô hình tức thời đúng và không nhiễu, độc lập với det(A); nhưng khi có nhiễu, det(A) nhỏ làm SDR rơi từ 12.3 dB (det 0.76) "
  "xuống 8.5 dB (det 0.44) và 2–4 dB (det 0.145).")

H("9. Trả lời câu hỏi báo cáo")
QA = [
 "Tai người kém nhạy với pha ngắn hạn, còn magnitude mang phần lớn thông tin cảm nhận; ước lượng pha sạch rất khó. Ở SNR rất thấp, pha noisy bị nhiễu chi phối "
 "nên dù magnitude đúng, tín hiệu tái tạo vẫn có méo và nhiễu dư; giả thiết Py ≈ Ps + Pv (bỏ qua số hạng chéo) cũng sai lệch nhiều khi Ps ≈ Pv.",
 "β (over-subtraction) bù sai số ước lượng PSD và dao động của periodogram: β lớn → trừ nhiều nhiễu hơn nhưng cũng trừ vào bin có speech yếu (phụ âm, formant cao) → méo. "
 "γ là sàn phổ: ngăn giá trị âm và 'lỗ phổ' gây musical noise, đổi lại giữ lại nền nhiễu dư (bảng sweep: γ=0.05 cho ΔSNR khác và nền êm hơn).",
 "Speech-dominant (Ps ≫ Pv): H → 1, giữ nguyên tín hiệu. Noise-dominant: H → 0, triệt bin. Ở giữa, H là trọng số MMSE theo SNR cục bộ, thay đổi trơn nên ít musical noise.",
 "SNR so sánh mẫu-theo-mẫu nên lệch thời gian biến chính speech thành 'lỗi'. Delay 10 ms (160 mẫu) làm tương quan giảm mạnh, SNR_out bị đánh giá THẤP hơn thực tế (có thể âm). "
 "Ở đây ISTFT dùng cùng window/hop và length=len(y) nên không có delay.",
 "Identification: output là nhãn trong N người đã đăng ký (bài toán N-lớp), metric accuracy/confusion matrix. Verification: output accept/reject cho một claim (nhị phân), metric FAR, FRR, EER/DET.",
 "Chọn θ trên test là 'tune trên test', cho số đo lạc quan, không phản ánh dữ liệu mới. θ tăng → FAR giảm, FRR tăng; θ phụ thuộc chi phí ứng dụng (ngân hàng ưu tiên FAR thấp).",
 "MFCC mô tả đường bao phổ, tức hàm truyền tuyến âm; độ dài/hình dạng tuyến âm, vị trí formant trung bình và độ nghiêng phổ khác nhau giữa người nói, nên thống kê dài hạn của MFCC chứa thông tin người nói.",
 "CMN xoá thành phần nhân tính cố định (kênh/micro) → robust với kênh. Nhưng μ(c) cũng chứa đặc trưng tuyến âm của người nói; baseline mean/std mất phần này. Kết quả: acc 1.00 → 0.75, margin 0.80 → 0.24.",
 "x = As = (AP⁻¹D⁻¹)(DPs) với P hoán vị, D đường chéo bất kỳ: mọi cặp như vậy cho cùng quan sát và các thành phần vẫn độc lập, nên ICA chỉ xác định nguồn tới mức hoán vị và scale/dấu.",
 "Hai hàng gần tỉ lệ → hai mic thu gần như cùng một hỗn hợp, det(A) → 0, A gần suy biến; thông tin để phân biệt hai nguồn rất ít, A⁻¹ có chuẩn lớn và khuếch đại nhiễu (thấy rõ ở A_near_singular + noise 20 dB).",
 "Phòng thật: mỗi nguồn tới mic qua đáp ứng xung h_ij[n] (trễ + phản xạ), x_i = Σ_j h_ij * s_j (convolutive), không phải hệ số vô hướng; cần ICA miền tần số (mỗi bin một ICA, rồi giải permutation giữa các bin) hoặc beamforming.",
 "Không. Với fan noise, Wiener tăng accuracy 0.25 → 0.75–1.00 nhưng SS chỉ 0.375; với cafeteria 0 dB cả hai không giúp (0.25), và ở case study SS vẫn đoán sai trong khi Wiener đúng. "
 "Enhancement tạo méo/artifact làm lệch thống kê MFCC so với model enrollment sạch; hiệu quả tuỳ loại nhiễu và thuật toán.",
]
for i, a in enumerate(QA, 1): P(f"Câu {i}. {a}")

H("10. Giới hạn")
P("Dữ liệu là TTS + giọng biến đổi pitch, ít biến thiên hơn người thật; số trial nhỏ nên FAR/FRR có phương sai lớn. "
  "Khuyến nghị thay lab5_data/clean bằng bản thu thật (cùng tên file) và chạy lại Lab5.py.")
doc.save(os.path.join(ROOT, "report.docx")); print("report.docx saved")
