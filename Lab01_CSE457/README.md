# CSE457 — Lab 1: Phân tích và xử lý tín hiệu âm thanh số

## Pipeline thực hiện

```
audio/input.mp3
   ↓ A. đọc metadata → mono hóa → chuẩn hóa [−1,1]
   ↓ B. waveform, peak/RMS/energy, kiểm tra clipping
   ↓ C. FFT: Δf vs true resolution, đỉnh phổ, băng thông hiệu dụng
   ↓ D. STFT/spectrogram, so sánh frame 10/25/50 ms
   ↓ E. cửa sổ rectangular vs Hamming (main-lobe / side-lobe / leakage)
   ↓ F. FIR low-pass / high-pass / band-pass, bù group delay
   ↓ G. lượng tử hóa + SNR, resampling + aliasing, PCM vs MP3
   → figures/ + audio/ + results_summary.json
```

## Một số kết quả chính

| Đại lượng | Giá trị đo được |
|---|---|
| F_s / Nyquist | 24,000 Hz / 12,000 Hz |
| Băng thông hiệu dụng | ≈ 10,951 Hz — **thấp hơn Nyquist**, dấu vết của bộ mã hóa MP3 |
| Peak / RMS / Crest factor | −1.23 dBFS / −21.58 dBFS / 20.34 dB |
| f_0 và hoạ âm | 189 → 376 → 561 → 758 Hz |
| Cửa sổ: main-lobe / side-lobe | rect 79.8 Hz / −13.3 dB · Hamming 160.4 Hz / −42.7 dB |
| Leakage đo được (5–10 kHz) | rectangular cao hơn Hamming **14.9 dB** |
| SNR lượng tử 4 / 8 / 16 bit | 8.9 / 31.7 / 78.4 dB — độ dốc **5.81 dB/bit** |
| Aliasing khi bỏ mẫu thủ công | +5.1 dB mức nền trên 3.4 kHz (chirp kiểm chứng: hình 12c) |
| Compression ratio MP3 gốc | **4.80 : 1** (768 → 160 kbps, tiết kiệm 79.2%) |

## Chạy lại

```bash
pip install numpy scipy matplotlib soundfile pydub pandas jupyter
# cần ffmpeg trong PATH (đọc MP3 và nén MP3 ở phần G)
jupyter notebook Lab01_CSE457.ipynb    # rồi Run All
```

Notebook dùng **đường dẫn tương đối** (`audio/`, `figures/`) nên phải chạy từ thư mục gốc của repo.
Mọi hình và mọi tệp âm thanh đều được sinh lại từ code — không có kết quả nào chèn thủ công.

## Ghi chú phương pháp

- Mọi phép so sánh đều là **controlled experiment**: khi so frame length và cửa sổ, dữ liệu / NFFT /
  dynamic range được giữ nguyên.
- Group delay của FIR (100 mẫu) **đã được bù** trước khi so sánh hay tính sai số.
- Tệp gốc là MP3 (lossy) nên bản WAV giải mã **không phải ground truth không mất mát**; mọi phép nén trong
  phần G là transcoding và chỉ dùng để so sánh tương đối.
