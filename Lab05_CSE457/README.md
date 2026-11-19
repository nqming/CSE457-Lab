# Lab 5 – Speech Enhancement, Speaker Recognition, Speaker Separation

## Cài đặt
```
pip install numpy scipy matplotlib librosa soundfile scikit-learn python-docx
```

## Cấu trúc
```
lab5_data/
  raw/          # TTS thô (David, Zira) – sinh bởi tools/gen_tts.ps1
  clean/        # spk01..spk04 _enroll_01..04, _dev_01..02, _test_01..02 (16 kHz mono)
  noise/        # fan.wav, cafeteria.wav (tổng hợp)
  separation/   # source_A.wav, source_B.wav
results/audio/  # noisy, enhanced_ss, enhanced_wiener, mix_mic1/2, separated_1/2, ...
results/figures/# spectrogram, confusion matrix, FAR-FRR, ICA correlation
results/results.json
```

## Chạy
```
powershell -ExecutionPolicy Bypass -File tools\gen_tts.ps1   # (tuỳ chọn) sinh lại giọng TTS
python tools\prepare_data.py                                  # (tuỳ chọn) tạo lab5_data
python Lab5.py                                                # toàn bộ thí nghiệm
python make_report.py                                         # tạo report.docx
```
Muốn dùng giọng thật: ghi đè file trong `lab5_data/clean/` (giữ nguyên tên) rồi chạy lại `Lab5.py`.
Toàn bộ tham số nằm trong dict `CFG` đầu file `Lab5.py`.
