# Báo Cáo Experiment ww_hand_crafted Cho So Sánh Context Mode

## 1. Thiết Lập Thí Nghiệm
- Tập dữ liệu: `ww_hand_crafted`
- Mô hình sử dụng: `gpt-4o-mini`
- Kích thước cửa sổ cố định: `5`
- Các chế độ ngữ cảnh được so sánh:
  - `surrounding_w5`
  - `prev_w5`
  - `next_w5`
- Baseline dùng để đối chiếu:
  - `all_at_once`
  - `step_by_step`

## 2. Kết Luận Chính
- Về `step_accuracy`, phương án tốt nhất trong ba context mode là `prev_w5` với giá trị `0.172`.
- Về `agent_accuracy`, `surrounding_w5` là phương án tốt nhất trong ba context mode với giá trị `0.552`.
- So với baseline, `step_by_step` có `step_accuracy` bằng với `prev_w5` ở mức `0.172`, trong khi `all_at_once` thấp hơn ở mức `0.086`.
- `next_w5` là phương án yếu nhất trong ba context mode theo `step_accuracy`, chỉ đạt `0.086`.

## 3. Bảng Tóm Tắt Accuracy

| method | agent_accuracy | step_accuracy |
| --- | --- | --- |
| all_at_once | 0.5 | 0.086 |
| step_by_step | 0.483 | 0.172 |
| step_based_multi_step_w5 | 0.552 | 0.138 |
| step_based_multi_step_prev_w5 | 0.414 | 0.172 |
| step_based_multi_step_next_w5 | 0.448 | 0.086 |

## 4. Phân Tích Giao Nhau Giữa Các Case Đúng
- Chỉ `surrounding` đúng: `2`
- Chỉ `prev` đúng: `4`
- Chỉ `next` đúng: `2`
- `surrounding + prev` cùng đúng: `3`
- `surrounding + next` cùng đúng: `0`
- `prev + next` cùng đúng: `0`
- Cả ba cùng đúng: `3`
- Không phương án nào đúng: `44`
- Diễn giải:
  - `prev_w5` tạo ra nhiều chiến thắng riêng hơn `next_w5` (`4` so với `2`).
  - Không xuất hiện case chỉ đúng theo cặp `surrounding+next` hoặc `prev+next` nếu các giá trị này vẫn bằng 0.

## 5. Phân Tích Độ Lệch Step

| method | mean_signed_error | median_signed_error | mean_abs_error | median_abs_error | early_rate | exact_rate | late_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prev_w5 | 0.414 | 0 | 11.103 | 7.5 | 0.414 | 0.172 | 0.414 |
| next_w5 | -1.897 | -1.5 | 10.862 | 8.5 | 0.5 | 0.086 | 0.414 |

- `prev_w5` có `mean signed error` bằng `0.414` và `median signed error` bằng `0.0`. Điều này cho thấy phương án này không có thiên lệch mạnh theo hướng đoán sớm hay đoán muộn trên toàn bộ tập.
- `next_w5` có `mean signed error` bằng `-1.897` và `median signed error` bằng `-1.5`. Dấu âm cho thấy phương án này có xu hướng đoán sớm hơn ground-truth step, chứ không phải muộn hơn.
- `mean absolute error` của hai phương án khá gần nhau: `prev_w5 = 11.103`, `next_w5 = 10.862`.

## 6. Phân Tích Theo Vị Trí Lỗi Trong Trajectory

| position_bucket | count | prev_step_accuracy | next_step_accuracy | mean_relative_position | accuracy_gap_prev_minus_next |
| --- | --- | --- | --- | --- | --- |
| head | 29 | 0.103 | 0.034 | 0.157 | 0.069 |
| middle | 10 | 0.3 | 0.1 | 0.497 | 0.2 |
| tail | 19 | 0.211 | 0.158 | 0.939 | 0.053 |

- Ở bucket `head`: `prev_w5 = 0.103`, `next_w5 = 0.034`.
- Ở bucket `middle`: `prev_w5 = 0.300`, `next_w5 = 0.100`.
- Ở bucket `tail`: `prev_w5 = 0.211`, `next_w5 = 0.158`.
- Diễn giải:
  - Giả thuyết `prev_w5` hoạt động tốt hơn khi lỗi nằm cuối trajectory được ủng hộ, nhưng mức chênh không lớn: khoảng cách ở bucket `tail` là `0.053`.
  - Giả thuyết `next_w5` hoạt động tốt hơn khi lỗi nằm đầu trajectory không được ủng hộ trên tập này, vì ngay cả ở bucket `head`, `prev_w5` vẫn tốt hơn với khoảng cách `0.069`.
  - Khoảng cách lớn nhất giữa `prev_w5` và `next_w5` xuất hiện ở bucket `middle`, với chênh lệch `0.200` nghiêng về `prev_w5`.

## 7. Bảng Qualitative Error Cases
- Đã sinh riêng một bảng qualitative với tối đa 5 ví dụ cho mỗi nhóm:
  - `prev_beats_next`
  - `next_beats_prev`
  - `both_wrong`
- Tổng số dòng qualitative được chọn: `12`.

## 8. Các Artifact Được Sinh Ra
- `ww_hand_crafted_agent_accuracy.png`
- `ww_hand_crafted_step_accuracy.png`
- `ww_hand_crafted_context_mode_step_correct_venn.png`
- `ww_hand_crafted_step_error_distribution.png`
- `ww_hand_crafted_position_bucket_comparison.png`
- `ww_hand_crafted_context_mode_cases.csv`
- `ww_hand_crafted_step_error_cases.csv`
- `ww_hand_crafted_position_bucket_cases.csv`
- `ww_hand_crafted_qualitative_error_cases.csv`
- `qualitative_error_cases.md`
