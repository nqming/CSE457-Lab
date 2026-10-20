# BÁO CÁO LAB 3 — HMM

## 1. Mục tiêu và dữ liệu
- Vocabulary, số train/test/lớp, thiết bị, nguồn TTS eSpeak, giọng/tốc độ/cao độ và phương pháp tạo noise mô phỏng (xem DATASET_CARD.md).
- Chỉ ra cách chống data leakage.

## 2. Feature và HMM
- 16 kHz; frame 25 ms, hop 10 ms; 26 Mel; 13 MFCC+Δ+Δ², CMVN utterance.
- Left-to-right 5 states, diag, self-loop 0.7, next 0.3; A cố định, EM cập nhật Gaussian.

## 3. Kết quả trên dữ liệu kiểm thử WAV tổng hợp
- Chèn `figures/features_*.png`, `figures/viterbi_*.png`, confusion matrix.
- Bảng acoustic score, training convergence; giải thích nhãn bị nhầm.

## 4. Các thí nghiệm
- E1: 3/5/7 trạng thái, accuracy, convergence.
- E2: 13-D vs 39-D accuracy.
- E3: diag vs full nếu đủ dữ liệu; nêu rõ nếu không chạy.
- E4: DTW vs HMM, accuracy và thời gian.
- E5: simulated noise/reverb mismatch accuracy drop với dataset/robustness.
- E6: λLM với corpus minh họa giả lập; nêu giới hạn không phải corpus thật.

## 5. Phân tích và kết luận
- Đặt số liệu và quan sát thực từ kết quả, không điền giả định như kết quả thực.
- Giới hạn whole-word, người nói, dữ liệu ít, reject threshold.
- Trả lời 10 câu hỏi báo cáo trong notebook.
