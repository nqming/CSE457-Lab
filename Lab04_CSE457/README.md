# Lab 4 – Tổng hợp tiếng nói (Text-to-Speech Synthesis)

Học phần **CSE457 – Xử lý âm thanh và tiếng nói** (Khoa CNTT, Trường ĐH Thủy Lợi).

## Cài đặt
```bash
pip install numpy scipy matplotlib soundfile librosa piper-tts edge-tts python-docx
```

## Cấu trúc thư mục
```text
Lab04_CSE457/
  Lab 4.pdf                     # Đề bài Lab 4
  Lab04.py                      # Script chạy toàn bộ pipeline A -> G
  dsp_utils.py                  # STFT, Mel-spectrogram, YIN F0, Phase Vocoder, Pitch Shift bảo toàn formant
  hmm_tts_backend.py            # Backend HMM 5 trạng thái + MLPG (Δ-feature) + Source-Filter Vocoder
  neural_tts_backend.py         # Backend Neural TTS offline (Piper VITS vi_VN-vivos-x_low)
  make_report.py                # Sinh report.docx từ results/results.json, ratings.csv, results/figures/
  ratings.csv                   # Bảng đánh giá mù A/B & MOS mini trên 3 câu test
  report.docx                   # Báo cáo tổng hợp kết quả Lab 4
  tools/
    prepare_data.py             # Sinh 48 word-unit WAV (16 từ x 3 candidate) và metadata.csv
  lab4_data/
    units/                      # 48 file WAV mono 16 kHz, 16-bit PCM
    metadata.csv                # Thống kê label, candidate_id, speaker, duration, rms, f0_median
    hmm_demo/hmm_model.npz      # Tham số HMM 5-state đã huấn luyện
    neural_model/               # Mô hình Piper VITS tiếng Việt chạy hoàn toàn offline
  audio_output/                 # raw_concat.wav, best_concat.wav, hmm_tts.wav, neural_tts.wav, ...
  results/
    results.json                # Toàn bộ số liệu thí nghiệm
    figures/                    # 6 biểu đồ phân tích (fig1 .. fig6)
```

## Chạy lại từ đầu
```bash
python tools/prepare_data.py    # (tuỳ chọn) sinh lại bộ word-unit inventory
python Lab04.py                 # chạy toàn bộ thí nghiệm và xuất audio + hình
python make_report.py           # tạo report.docx
```

Muốn thay bằng giọng thu âm thật: thay các file `.wav` trong `lab4_data/units/` (16 kHz, mono, 16-bit PCM), xóa `lab4_data/hmm_demo/hmm_model.npz` rồi chạy lại `python Lab04.py`.
