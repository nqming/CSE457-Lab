# Lab 2 - MFCC + DTW nhận dạng từ đơn

Bộ code này được xây dựng theo Lab 2 CSE457: 5 từ `không, một, hai, ba, bốn`, mỗi từ 5 lần ghi.
- Chuẩn hóa WAV về mono 16 kHz
- Frame 25 ms, hop 10 ms, Hamming
- Short-time energy/RMS + ZCR
- Endpoint detection bằng energy tương đối (`top_db=35`) + margin 50 ms
- Pre-emphasis alpha=0.97
- MFCC: 24 Mel filters, 13 coefficients, NFFT=512, CMN
- Tự cài Euclidean local distance + DTW dynamic programming + backtracking + normalization
- 3 templates/từ, 2 test/từ
- Accuracy + confusion matrix + top-3 scores
- E1: có/không endpoint detection
- E2: MFCC 13 vs MFCC 13 + Delta

## Chạy
```bash
pip install -r requirements.txt
python lab2.py
```

Các hình nằm trong `figures/`, kết quả CSV nằm trong `results/`.

## Ghi chú
Dataset gốc của người học đã được đặt trong `dataset/`. Code không dùng ASR trực tuyến hoặc mô hình end-to-end.
