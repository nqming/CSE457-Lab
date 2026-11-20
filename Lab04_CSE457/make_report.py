"""Sinh report.docx từ results/results.json, ratings.csv và results/figures/.
Chạy sau Lab04.py: python make_report.py
"""
import csv
import json
import os
from docx import Document
from docx.shared import Cm, Pt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "results", "results.json"), encoding="utf-8"))
FIG = os.path.join(ROOT, "results", "figures")
with open(os.path.join(ROOT, "ratings.csv"), encoding="utf-8") as f:
    RATINGS = list(csv.DictReader(f))

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)


def H(t, l=1):
    doc.add_heading(t, level=l)


def P(t):
    doc.add_paragraph(t)


def B(t):
    doc.add_paragraph(t, style="List Bullet")


def IMG(f, w=15.5):
    doc.add_picture(os.path.join(FIG, f), width=Cm(w))


def T(header, rows):
    tb = doc.add_table(rows=1, cols=len(header))
    tb.style = "Light Grid Accent 1"
    for c, h in zip(tb.rows[0].cells, header):
        c.text = str(h)
    for r in rows:
        cs = tb.add_row().cells
        for c, v in zip(cs, r):
            c.text = f"{v:.3f}" if isinstance(v, float) else str(v)


doc.add_heading("LAB 4 – Tổng hợp tiếng nói (Text-to-Speech Synthesis)", 0)
P("Học phần CSE457 – Xử lý âm thanh và tiếng nói. "
  "Mã nguồn chính: Lab04.py (chạy end-to-end), backend HMM: hmm_tts_backend.py, "
  "backend Neural TTS offline: neural_tts_backend.py, tiện ích DSP: dsp_utils.py, "
  "dữ liệu: lab4_data/, đầu ra âm thanh: audio_output/, kết quả: results/ và ratings.csv.")

# ---- 1 ----
H("1. Kiến trúc TTS, Text Normalization và Inventory (Mục A)")
P("Chuỗi xử lý tổng quát của một hệ thống Text-to-Speech gồm: "
  "Text Analysis (chuẩn hóa văn bản) → Phonetic Analysis (G2P/lexicon → chuỗi đơn vị phát âm) → "
  "Prosodic Analysis (F0, duration, năng lượng) → Synthesis Backend → Waveform.")
P("Trong miền hẹp 'thông báo xếp hàng', hàm normalize_queue_text() chuyển văn bản về chữ thường, "
  "loại bỏ dấu câu (.,!?;:), tách token theo khoảng trắng và quy đổi mọi chuỗi chữ số sang từng chữ số "
  "('0'→'không' ... '9'→'chín'). Ví dụ:")
for s, toks in R["part_a"]["norm_examples"].items():
    B(f"\"{s}\" → {toks}")

pa = R["part_a"]
P(f"Inventory gồm {pa['n_words']} từ miền hẹp (mời, khách, hàng, số, không–chín, đến, quầy), "
  f"mỗi từ có 3 candidate (_01 chuẩn, _02 chậm/trầm hơn, _03 nhanh/cao hơn) từ cùng người nói HoaiMy, "
  f"tổng cộng {pa['n_files']} tệp WAV mono 16 kHz, 16-bit PCM đã cắt silence đầu/cuối và lưu thông tin trong lab4_data/metadata.csv.")
T(["Thống kê Inventory", "Min", "Mean", "Max"], [
    ["Duration (s)", round(pa["duration_min"], 4), round(pa["duration_mean"], 4), round(pa["duration_max"], 4)],
    ["RMS amplitude", round(pa["rms_min"], 4), round(pa["rms_mean"], 4), round(pa["rms_max"], 4)],
    ["F0 median (Hz)", "-", round(pa["f0_mean"], 2), "-"],
])
P("Nhận xét histogram (Hình 1): duration tập trung quanh 230–360 ms, ngoại trừ từ 'khách' ngắn nhất (~143–169 ms) "
  "do chứa âm tắc-xát vô thanh ngắn và âm cuối khép nhanh; RMS phân bố quanh 3 cụm tương ứng 3 lần thu (0.065–0.107) "
  "và không có tệp bị clipping hay silence kéo dài bất thường.")
IMG("fig1_inventory_stats.png")

# ---- 2 ----
H("2. Baseline Concatenation (Mục B)")
pb = R["part_b"]
P(f"Ghép trực tiếp chuỗi candidate cố định _01 cho câu \"{pb['sentence']}\" với xfade = 0 ms và chưa match_rms "
  f"thu được audio_output/raw_concat.wav dài {pb['duration_s']:.3f} s. "
  f"Độ lệch phổ biên trung bình mean D_spec = {pb['mean_d_spec']:.2f} dB² và bước nhảy mẫu trung bình = {pb['mean_sample_jump']:.5f}.")
T(["Biên nối", "Thời điểm (s)", "D_spec (dB²)", "ΔRMS (dB)", "Sample jump"], [
    [b["pair"], round(b["time_s"], 3), round(b["d_spec"], 2), round(b["d_rms_db"], 2), round(b["sample_jump"], 5)]
    for b in pb["boundaries"]
])
P("Khi nghe và quan sát biên nối (Hình 2, cột trái), tại các cặp như 'ba → đến' (D_spec = 1041.0 dB²) "
  "hay 'quầy → năm' (D_spec = 974.9 dB²), có hiện tượng gãy pha nhẹ, chênh năng lượng cục bộ (tới 12.7 dB ở biên 'hai → ba') "
  "và đứt quãng đường bao phổ trên spectrogram.")

# ---- 3 ----
H("3. Unit Selection (DP) và Boundary Smoothing (Mục C – TN 1 & TN 2)")
P("Hàm chi phí chọn chuỗi đơn vị tối ưu theo công thức:\n"
  "θ* = arg min_{θ₁…θJ} [ Σⱼ C_target(θⱼ, tⱼ) + λ Σⱼ C_join(θⱼ, θⱼ₊₁) ], với λ = 1.5.\n"
  "Trong đó C_target đo độ lệch duration (trọng số 0.45 so với nhịp 0.96×mean), F0 (0.45) và RMS (0.10) sau chuẩn hóa theo độ lệch chuẩn; "
  "C_join kết hợp D_spec = (1/K) Σ (L_left[k] − L_right[k])² trên 25 ms biên (0.65), chênh F0 (0.20) và chênh RMS biên (0.15). "
  "Tìm đường đi tối ưu trên lưới 3 candidate/vị trí bằng Quy hoạch động Viterbi.")

us_rows = []
for u in R["part_c"]["unit_selection"]:
    for mname, key in [("Fixed (_01)", "fixed_01"), ("Target-only", "target_only"), ("DP (λ=1.5)", "dp_selected")]:
        d = u[key]
        us_rows.append([u["sentence"][:24] + "...", mname, round(d["target_cost_sum"], 3),
                        round(d["join_cost_sum"], 3), round(d["total_cost"], 3), round(d["mean_d_spec"], 2)])
T(["Câu test", "Phương pháp", "Target cost", "Join cost", "Total cost", "Mean D_spec"], us_rows)

P("Thí nghiệm 1 – Cross-fade (0, 5, 15, 30 ms) với công thức y[n] = (1 − α[n])x₁[n] + α[n]x₂[n]:")
T(["Cross-fade (ms)", "Duration (s)", "Max |Δx| quanh biên", "Spectral flux biên (dB²)"], [
    [r["xfade_ms"], round(r["duration_s"], 3), round(r["mean_boundary_diff"], 5), round(r["mean_boundary_spec_flux"], 2)]
    for r in R["part_c"]["xfade_sweep"]
])
P("Phân tích (Hình 2 & Hình 3): (1) Quy hoạch động DP giảm mạnh Total Cost và Mean D_spec trên cả 3 câu "
  "(câu 1 giảm D_spec từ 594.0 xuống 315.0 dB², giảm 47%; Total Cost từ 30.57 xuống 22.03). "
  "(2) Với cross-fade, 0 ms còn bước nhảy biên độ (0.00477) và flux cao (336.56 dB²); 5 ms cải thiện vừa phải; "
  "15 ms đạt điểm cân bằng tối ưu (Max |Δx| thấp nhất = 0.00297, spectral flux giảm còn 270.12 dB²) "
  "mà không làm mờ phụ âm đầu/cuối; 30 ms tiếp tục giảm flux nhưng làm phần chồng lấn lấn sâu vào âm vị đầu, "
  "tăng Max |Δx| cục bộ do giao thoa trực tiếp các mẫu hữu thanh có biên độ lớn.")
IMG("fig2_concat_waveform_spec.png")
IMG("fig3_unit_selection_crossfade.png")

# ---- 4 ----
H("4. Prosody và Mô hình Source–Filter (Mục D – TN 3 & TN 4)")
pd = R["part_d"]
P(f"Kiểm chứng F0 ≈ Fs / Np trên đoạn hữu thanh ổn định của từ 'hàng' (hang_01.wav): "
  f"khoảng cách trung bình giữa các đỉnh chu kỳ pitch đo được Np = {pd['hang_np_samples']:.2f} mẫu ở Fs = 16000 Hz, "
  f"cho F0 = Fs / Np = {pd['hang_f0_from_np_hz']:.2f} Hz, khớp sát với kết quả thuật toán YIN ({pd['hang_f0_yin_hz']:.2f} Hz).")

P("Thí nghiệm 3 – Thay đổi tốc độ bằng WSOLA (0.8×, 1.0×, 1.2×):")
T(["Tốc độ (rate)", "Duration đầu ra (s)", "F0 median (Hz)"], [
    [r["speed"], round(r["duration_s"], 3), round(r["f0_median_hz"], 2)] for r in pd["speed_exp"]
])
P("WSOLA dịch các khung bằng overlap-add tại điểm tương quan cực đại nên thời lượng tỷ lệ nghịch tốc độ (2.99 s → 2.41 s → 1.99 s) "
  "trong khi F0 giữ ổn định (~245 Hz). Tốc độ 0.8× nghe chậm rõ nhưng hơi kéo dài phụ âm/nguyên âm; 1.2× gọn và linh hoạt hơn "
  "nhưng nếu tăng quá cao sẽ giảm thời gian xử lý ngữ âm của người nghe.")

P("Thí nghiệm 4 – Dịch cao độ ±2 semitone (r = 2^(Δs/12), F0_new = r · F0_old):")
T(["Δs (semitone)", "Tỷ số r = 2^(Δs/12)", "F0 median đo được (Hz)"], [
    [r["semitone"], round(r["ratio"], 4), round(r["f0_median_hz"], 2)] for r in pd["pitch_exp"]
])
P("Theo mô hình source–filter X(z) = E(z)H(z), nguồn kích thích e[n] quyết định F0 còn bộ lọc tuyến âm H(z) quyết định "
  "đường bao phổ / formant (tính chất nguyên âm và đặc trưng người nói). Hình 4 (trên phải) chứng minh: "
  "khi dùng thuật toán bảo toàn đường bao phổ cepstral (đường nét đứt xanh), các đỉnh formant (~1.05, 2.25, 3.30 kHz) "
  "giữ nguyên vị trí như giọng gốc (xanh đậm) trong khi F0 tăng đúng tỷ lệ 2^(2/12) = 1.1225 (245.0 → 275.1 Hz). "
  "Ngược lại, resample thô (chấm đỏ) kéo toàn bộ formant dịch lên theo F0 gây méo âm sắc ('hiệu ứng chipmunk'). "
  "Với ±2 semitone, mức độ artifact/buzziness rất nhẹ; nếu dịch lớn hơn (±6 st) hiện tượng phasiness sẽ rõ hơn.")
IMG("fig4_prosody_source_filter.png")

# ---- 5 ----
H("5. Tổng hợp tham số dựa trên HMM (Mục E)")
pe = R["part_e"]
P("Backend hmm_tts_backend.py huấn luyện HMM 5 trạng thái left-to-right cho mỗi từ từ 3 candidate: "
  "đặc trưng mỗi khung 10 ms gồm [log F0, log RMS, 16 hệ số LSF] (18 chiều tĩnh + 18 chiều Δ). "
  "Ở chế độ generation, hệ thống nối chuỗi trạng thái theo mô hình duration rồi giải hệ phương trình MLPG "
  "(W^T Σ⁻¹ W) ĉ = W^T Σ⁻¹ μ để sinh chuỗi tham số liên tục, sau đó đưa qua bộ vocoder source–filter (xung glottal/nhiễu + IIR 1/A(z)).")
T(["Câu test HMM", "Duration (s)", "Latency (s)", "Số frame (10 ms)", "F0 mean (Hz)"], [
    [r["sentence"], round(r["duration_s"], 3), round(r["latency_s"], 4), r["n_frames"], round(r["f0_mean_hz"], 2)]
    for r in pe["runs"]
])
P("Thí nghiệm điều khiển tham số HMM trên câu 1 (dur × 0.8/1.2 và F0 × 0.9/1.1):")
T(["Cấu hình biến thiên", "Duration (s)", "F0 mean (Hz)"], [
    [k, round(v["duration_s"], 3), round(v["f0_mean_hz"], 2)] for k, v in pe["variations"].items()
])
P("Quan sát Hình 5: khi KHÔNG có Δ-feature (đường nét đứt đỏ), quỹ đạo F0 nhảy bậc thang đột ngột tại mọi biên trạng thái; "
  "khi bổ sung Δ + MLPG (đường xanh liền), quỹ đạo F0 và formant F1/F2 chuyển tiếp mượt mà giữa các state và giữa các từ. "
  "Điều này minh họa rõ khác biệt cốt lõi: ghép nối chọn waveform tự nhiên sẵn có nên giàu chi tiết pha/âm sắc, "
  "còn HMM sinh quỹ đạo tham số trung bình từ phân bố thống kê (mượt và điều khiển prosody dễ nhưng bị làm mượt quá mức / buzzy qua vocoder).")
IMG("fig5_hmm_trajectories.png")

# ---- 6 ----
H("6. TTS học sâu bằng mô hình Pretrained Offline (Mục F)")
pf = R["part_f"]
P(f"Backend neural_tts_backend.py sử dụng mô hình VITS huấn luyện sẵn ({pf['model_name']}, "
  f"dung lượng {pf['model_size_mb']:.2f} MB) đóng gói hoàn toàn offline trong lab4_data/neural_model/. "
  "Pipeline nội tại: văn bản chuẩn hóa → chuyển âm vị eSpeak-ng → encoder văn bản + duration predictor + flow/VITS decoder → waveform trực tiếp 16 kHz.")
T(["Câu test Neural TTS", "Duration (s)", "Latency (s)", "RTF", "Mel shape"], [
    [r["sentence"], round(r["duration_s"], 3), round(r["latency_s"], 4), round(r["rtf"], 3), str(r["mel_shape"])]
    for r in pf["runs"]
])
P("Tốc độ suy luận CPU rất nhanh (latency ~53–66 ms cho câu ~2.4–2.8 s, tương ứng RTF ≈ 0.023, nhanh hơn thời gian thực ~43 lần).")

# ---- 7 ----
H("7. Đánh giá và so sánh 3 hướng tổng hợp (Mục G – TN 5 & Listening Test)")
P("Bảng tổng hợp thông số đo được trên cùng máy cho câu 1 (đã chuẩn hóa loudness trước khi đánh giá):")
T(["Backend", "Duration (s)", "Latency (s)", "RTF", "Bộ nhớ/dữ liệu (MB)", "Spec Flux (dB/frame)", "F0 step (Hz)"], [
    [r["backend"], round(r["duration_s"], 3), round(r["latency_s"], 4), round(r["rtf"], 4),
     round(r["storage_mb"], 3), round(r["mean_spec_flux_db"], 2), round(r["f0_median_step_hz"], 2)]
    for r in R["part_g"]["comparison"]
])

# Tổng hợp ratings.csv theo backend
by_bk = {}
for row in RATINGS:
    by_bk.setdefault(row["backend"], []).append(row)
P("Bảng tổng hợp đánh giá nghe mù A/B & MOS mini (pilot expert evaluation trên 3 câu test, file ratings.csv):")
sum_rows = []
for bk, items in by_bk.items():
    intel = sum(float(x["intelligibility_pct"]) for x in items) / len(items)
    mos_n = sum(float(x["mos_naturalness"]) for x in items) / len(items)
    mos_c = sum(float(x["mos_continuity"]) for x in items) / len(items)
    sum_rows.append([bk, f"{intel:.1f}%", round(mos_n, 2), round(mos_c, 2)])
T(["Phương pháp", "Intelligibility", "MOS Naturalness (1–5)", "MOS Continuity (1–5)"], sum_rows)

P("Bảng so sánh tổng thể theo 5 tiêu chí (trả lời TN 5):")
T(["Phương pháp", "Intelligibility", "Naturalness", "Continuity", "Latency", "Bộ nhớ / dữ liệu"], [
    ["Concatenation (DP + xfade)", "Rất cao (100%, giọng thu thật)", "Khá (3.4/5, ngữ điệu từng từ)", "Khá (3.6/5, còn ngắt nhịp nhẹ)", "Thấp nhất (~8 ms)", "Nhỏ (0.43 MB cho 48 unit)"],
    ["HMM Parametric (MLPG)", "Khá (89.3%, hơi nghẹt phụ âm)", "Thấp (2.4/5, giọng vocoder)", "Tốt (3.9/5, quỹ đạo MLPG mượt)", "Trung bình (~157 ms)", "Rất nhỏ (0.076 MB)"],
    ["Neural TTS (Piper VITS)", "Rất cao (100%)", "Cao nhất (4.4/5, ngữ điệu tự nhiên)", "Cao nhất (4.6/5, liền mạch)", "Thấp (~56 ms, RTF 0.024)", "Lớn nhất (26.5 MB ONNX)"],
])
IMG("fig6_backend_comparison.png", 16.0)

# ---- 8 ----
H("8. Trả lời 10 câu hỏi báo cáo (Mục 7)")
QA = [
    ("1. Vì sao một câu ghép từ các đoạn 'tự nhiên' vẫn có thể nghe kém tự nhiên?",
     "Do thiếu đồng nhất về ngữ cảnh đồng cấu âm (coarticulation) và prosody cấp câu: F0 nhảy bậc tại biên từ, "
     "nhịp độ (duration) và cường độ từng từ thu rời không tạo được đường ngữ điệu chung, kèm theo lệch pha/biên độ tại điểm nối."),
    ("2. Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?",
     "Target cost đo mức phù hợp của candidate với ngữ điệu/duration/âm sắc mục tiêu tại vị trí j; join cost đo độ mượt "
     "khi nối liên tiếp hai candidate θ_j và θ_{j+1}. Nếu chỉ tối ưu join cost, hệ thống có thể chọn chuỗi toàn các unit lệch hẳn "
     "tốc độ, cao độ hoặc cảm xúc mong muốn miễn là chúng nối êm với nhau."),
    ("3. Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?",
     "Phụ âm bật (p, t, k, b, d) và phụ âm xát (s, kh, ch) thường rất ngắn (15–50 ms) và nằm ở biên đầu/cuối từ. "
     "Cross-fade quá dài (≥ 30–50 ms) sẽ làm suy giảm biên độ âm cuối của từ trước đồng thời trộn chồng lên pha nổ/xát đầu của từ sau, làm mờ ranh giới âm vị."),
    ("4. Giải thích bằng source–filter model vì sao thay F0 không đồng nghĩa với thay formant.",
     "Trong mô hình X(z) = E(z)H(z), F0 là tần số cơ bản của nguồn kích thích glottal e[n] (xác định khoảng cách hài), "
     "còn formant là các tần số cộng hưởng (cực) của bộ lọc tuyến âm H(z). Giữ nguyên H(z) và chỉ đổi chu kỳ xung e[n] sẽ dịch F0 mà bảo toàn vị trí formant và bản sắc nguyên âm."),
    ("5. PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?",
     "TD-PSOLA cắt các khung cửa sổ đồng bộ quanh các pitch epoch gốc (khoảng cách T_0) rồi dịch và overlap-add lại với khoảng cách T mới "
     "(T < T_0 để tăng F0, T > T_0 để giảm F0). Khi kéo dài voiced fricative (như /z/, /v/) quá mức, phần nhiễu xát ngẫu nhiên bị lặp lại tuần hoàn theo mỗi chu kỳ ghép nên tai cảm nhận thành tiếng rè kim loại (buzziness)."),
    ("6. Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?",
     "Nếu chỉ cực đại hóa xác suất đặc trưng tĩnh, nghiệm ML là giá trị trung bình từng state nên tạo đường bậc thang giật cục tại biên state. "
     "Khi đưa thêm ràng buộc Δc_t ≈ μ_Δ (với phương sai σ_Δ²), bài toán MLPG tạo liên kết tuyến tính giữa các khung lân cận qua ma trận cửa sổ W, ép quỹ đạo biến thiên liên tục và khớp độ dốc tự nhiên."),
    ("7. Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?",
     "Vì tiếng nói tự nhiên có vô số biến thể về ngữ cảnh âm vị (trước/sau), trọng âm, vị trí trong câu (đầu/giữa/cuối) và cao độ. "
     "Nếu inventory nhỏ, bộ chọn không tìm được unit khớp ngữ cảnh nên join cost và target cost đều cao."),
    ("8. Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.",
     "(1) Biểu diễn và tổng hợp: Concatenative lưu và ghép trực tiếp các đoạn waveform thu sẵn; Neural TTS học ánh xạ phi tuyến end-to-end "
     "từ văn bản/âm vị sang biểu diễn âm học/waveform nhờ mạng nơ-ron sâu. (2) Khả năng khái quát & dung lượng: Concatenative cần kho audio lớn cho mọi đơn vị/ngữ cảnh "
     "và khó thay đổi ngữ điệu; Neural TTS tổng hợp được mọi câu mới với ngữ điệu tự nhiên từ mô hình trọng số cố định."),
    ("9. Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?",
     "Vì khác biệt về phiên bản thư viện, tập dữ liệu huấn luyện, người nói, tần số lấy mẫu, phần cứng/latency mạng và nguy cơ không tái lập được kết quả; "
     "cần đóng gói cố định model offline trước giờ Lab để so sánh công bằng và lặp lại được."),
    ("10. Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.",
     "Có thể dùng trong các ứng dụng ưu tiên cảm giác âm sắc/không khí hơn nội dung chữ chính xác, như giọng nền phim/game, nhạc thể nghiệm, "
     "tạo hiệu ứng giọng nhân vật giả lập hoặc mô phỏng tiếng nói nền (babble/crowd ambience) — nhưng KHÔNG dùng cho thông báo công cộng, điều hướng hay đọc tin."),
]
for q, a in QA:
    P(q)
    B(a)

# ---- 9 ----
H("9. Kết luận")
B("Trong miền hẹp 'thông báo xếp hàng' (16 từ × 3 candidate), Concatenation kết hợp Quy hoạch động DP + match_rms + cross-fade 15 ms "
  "giảm 47% độ lệch phổ biên (D_spec: 594.0 → 315.0 dB²) và cho độ rõ 100% với độ trễ cực thấp (8.1 ms).")
B("Thí nghiệm Source–Filter, WSOLA và HMM-MLPG chứng minh vai trò độc lập giữa nguồn kích thích F0 và bộ lọc formant H(z), "
  "cũng như tác dụng làm trơn quỹ đạo của đặc trưng động Δ.")
B("Neural TTS offline (Piper VITS vi_VN-vivos-x_low) đạt chất lượng toàn diện tốt nhất (MOS tự nhiên 4.43, liền mạch 4.57, RTF 0.024) "
  "khi có sẵn mô hình pretrained đóng gói cục bộ.")

out_docx = os.path.join(ROOT, "report.docx")
doc.save(out_docx)
print(f"Đã tạo {out_docx}")
