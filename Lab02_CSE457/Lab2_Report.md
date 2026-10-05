# Lab 2 – Kết quả chạy trên bộ dữ liệu Voice

## Cấu hình
- 5 lớp: không, một, hai, ba, bốn
- 5 file/lớp; 3 template + 2 test
- Fs = 16 kHz; frame = 25 ms; hop = 10 ms
- Hamming; pre-emphasis α = 0.97
- NFFT = 512; 24 Mel filters; 13 MFCC; CMN
- DTW Euclidean, dynamic programming, backtracking, chuẩn hóa theo path length
- Endpoint: `top_db = 35`, margin = 50 ms

## Kết quả thực tế
- E1 no endpoint trim: **90.00%**
- Baseline trim + MFCC13: **90.00%**
- E2 trim + MFCC13+Delta: **100.00%**
- DTW same word (ba): **20.2040%**
- DTW different word (ba vs bon): **27.7368%**

## Lưu ý
Kết quả này là kết quả chạy trực tiếp trên 25 WAV được cung cấp. Khi chạy lại trên môi trường khác, phiên bản thư viện hoặc cách xử lý audio có thể làm số liệu thay đổi rất nhỏ.
