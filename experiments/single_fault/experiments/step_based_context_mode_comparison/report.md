# Báo Cáo Kết Quả: Step-Based Context Mode Comparison

## Mục tiêu

Experiment này đánh giá ba cách lấy ngữ cảnh cho phương pháp step-based attribution trên tập `ww_hand_crafted`, với cấu hình cố định:

- Mô hình: `gpt-4o-mini`
- Window size: `w = 5`
- Context modes:
  - `surrounding_w5`
  - `prev_w5`
  - `next_w5`

Hai baseline được dùng để đối chiếu là `all_at_once` và `step_by_step`.

Mục tiêu chính là kiểm tra xem việc thay đổi hướng lấy ngữ cảnh có làm thay đổi khả năng xác định đúng step lỗi và agent gây lỗi hay không.

## Kết quả tổng quan

Bảng accuracy tổng hợp như sau:

| method | agent_accuracy | step_accuracy |
| --- | --- | --- |
| all_at_once | 0.500 | 0.086 |
| step_by_step | 0.483 | 0.172 |
| step_based_multi_step_w5 | 0.552 | 0.138 |
| step_based_multi_step_prev_w5 | 0.414 | 0.172 |
| step_based_multi_step_next_w5 | 0.448 | 0.086 |

Từ bảng trên có thể rút ra bốn điểm chính:

1. `prev_w5` là context mode tốt nhất về `step_accuracy`, đạt `0.172`.
2. `prev_w5` đạt `step_accuracy` ngang với baseline `step_by_step`.
3. `surrounding_w5` là context mode tốt nhất về `agent_accuracy`, đạt `0.552`.
4. `next_w5` là phương án yếu nhất về `step_accuracy`, chỉ đạt `0.086`, tương đương `all_at_once`.

Điều này cho thấy nếu mục tiêu là dự đoán đúng step lỗi, ngữ cảnh phía trước (`prev_w5`) hữu ích hơn ngữ cảnh phía sau (`next_w5`). Ngược lại, nếu mục tiêu là nhận diện đúng agent gây lỗi, thì việc có ngữ cảnh hai phía (`surrounding_w5`) lại có lợi thế hơn.

## Phân tích giao nhau giữa các case đúng

Thống kê giao nhau giữa ba context mode theo `step_correct`:

- Chỉ `surrounding` đúng: `2`
- Chỉ `prev` đúng: `4`
- Chỉ `next` đúng: `2`
- `surrounding + prev` cùng đúng: `3`
- `surrounding + next` cùng đúng: `0`
- `prev + next` cùng đúng: `0`
- Cả ba cùng đúng: `3`
- Không phương án nào đúng: `44`

Các con số này cho thấy:

- `prev_w5` tạo ra nhiều chiến thắng riêng hơn `next_w5`.
- `next_w5` không tạo ra vùng giao riêng nào với `surrounding` hoặc với `prev` ngoài phần giao ba.
- Số lượng case mà không phương án nào đúng vẫn rất lớn, cho thấy bài toán trên tập `ww_hand_crafted` còn khó với cả ba thiết kế ngữ cảnh.

Về mặt thực nghiệm, overlap này củng cố kết luận rằng `prev_w5` mang lại tín hiệu hữu ích hơn `next_w5` cho việc xác định đúng step lỗi.

## Phân tích độ lệch step

Để vượt ra ngoài chỉ số `step_accuracy` nhị phân, experiment tính thêm:

- `pred_step - gt_step`: độ lệch có dấu
- `abs(pred_step - gt_step)`: độ lệch tuyệt đối

Bảng tóm tắt:

| method | mean_signed_error | median_signed_error | mean_abs_error | median_abs_error | early_rate | exact_rate | late_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prev_w5 | 0.414 | 0.0 | 11.103 | 7.5 | 0.414 | 0.172 | 0.414 |
| next_w5 | -1.897 | -1.5 | 10.862 | 8.5 | 0.500 | 0.086 | 0.414 |

Diễn giải kết quả:

### `prev_w5`

- `mean signed error = 0.414`
- `median signed error = 0.0`
- `early_rate = 0.414`
- `late_rate = 0.414`

Kết quả này cho thấy `prev_w5` không có thiên lệch rõ rệt về hướng đoán sớm hay đoán muộn. Lỗi của nó phân bố tương đối cân bằng quanh ground-truth step.

### `next_w5`

- `mean signed error = -1.897`
- `median signed error = -1.5`
- `early_rate = 0.500`
- `late_rate = 0.414`

Dấu âm ở cả mean và median cho thấy `next_w5` có xu hướng đoán sớm hơn ground-truth step. Đây là kết quả ngược với trực giác ban đầu rằng dùng ngữ cảnh phía sau có thể khiến mô hình đoán muộn hơn.

### Kết luận từ step error

Mặc dù `mean_abs_error` của hai phương án khá gần nhau (`11.103` so với `10.862`), `prev_w5` có `exact_rate` cao hơn đáng kể (`0.172` so với `0.086`). Nói cách khác, `prev_w5` chạm đúng step lỗi thường xuyên hơn, còn `next_w5` không tạo ra lợi thế thực sự ở mức exact step prediction.

## Phân tích theo vị trí lỗi trong trajectory

Experiment chia vị trí lỗi theo tỉ lệ `gt_step / trajectory_length` thành ba vùng:

- `head`: nhỏ hơn `1/3`
- `middle`: từ `1/3` đến nhỏ hơn `2/3`
- `tail`: lớn hơn hoặc bằng `2/3`

Bảng kết quả:

| position_bucket | count | prev_step_accuracy | next_step_accuracy | mean_relative_position | accuracy_gap_prev_minus_next |
| --- | --- | --- | --- | --- | --- |
| head | 29 | 0.103 | 0.034 | 0.157 | 0.069 |
| middle | 10 | 0.300 | 0.100 | 0.497 | 0.200 |
| tail | 19 | 0.211 | 0.158 | 0.939 | 0.053 |

Nhận xét theo từng bucket:

### `head`

- `prev_w5 = 0.103`
- `next_w5 = 0.034`

Giả thuyết rằng `next_w5` sẽ tốt hơn khi lỗi nằm ở đầu trajectory không được dữ liệu ủng hộ. Ngay cả trong vùng đầu, `prev_w5` vẫn cho kết quả tốt hơn rõ rệt.

### `middle`

- `prev_w5 = 0.300`
- `next_w5 = 0.100`

Đây là vùng có khoảng cách lớn nhất giữa hai phương án. Chênh lệch `0.200` nghiêng mạnh về `prev_w5`.

### `tail`

- `prev_w5 = 0.211`
- `next_w5 = 0.158`

Giả thuyết rằng `prev_w5` hoạt động tốt hơn khi lỗi nằm cuối trajectory được ủng hộ, nhưng mức chênh không lớn.

### Kết luận từ position bucket

`prev_w5` tốt hơn `next_w5` ở cả ba vùng `head`, `middle`, và `tail`. Do đó, trên tập `ww_hand_crafted`, không có bằng chứng cho thấy `next_w5` có lợi thế cục bộ ở đầu trajectory. Ngược lại, `prev_w5` duy trì ưu thế tương đối nhất quán, đặc biệt rõ ở vùng giữa.

## Phân tích qualitative

Experiment cũng sinh một bảng qualitative để kiểm tra thủ công các sample tiêu biểu, gồm tối đa:

- 5 case `prev_beats_next`
- 5 case `next_beats_prev`
- 5 case `both_wrong`

Trong dữ liệu hiện tại:

- Có đủ `5` case cho nhóm `prev_beats_next`
- Chỉ có `2` case cho nhóm `next_beats_prev`
- Có đủ `5` case cho nhóm `both_wrong`

Điều này tiếp tục củng cố kết quả định lượng: `prev_w5` vượt `next_w5` thường xuyên hơn không chỉ ở mức trung bình, mà cả ở mức từng sample cụ thể.

## Kết luận chung

Từ toàn bộ kết quả của experiment này, có thể rút ra các finding chính như sau:

1. `prev_w5` là context mode mạnh nhất cho mục tiêu dự đoán đúng step lỗi trên tập `ww_hand_crafted`.
2. `surrounding_w5` mạnh nhất về `agent_accuracy`, nhưng không tốt nhất về `step_accuracy`.
3. `next_w5` không cho thấy lợi thế kỳ vọng từ việc dùng ngữ cảnh phía sau; ngược lại, nó còn có xu hướng đoán sớm hơn ground truth step.
4. `prev_w5` tốt hơn `next_w5` ở cả ba vùng vị trí lỗi trong trajectory.
5. Khi xem ở mức case-by-case, `prev_w5` cũng thắng `next_w5` nhiều hơn về số lượng sample đúng riêng lẻ.

Tổng hợp lại, nếu mục tiêu chính là tối ưu `step_accuracy` cho bài toán failure attribution trên tập `ww_hand_crafted`, thì `prev_w5` là lựa chọn hợp lý hơn `next_w5`. Nếu mục tiêu là tối ưu nhận diện agent gây lỗi, `surrounding_w5` là ứng viên mạnh hơn.

## Hạn chế

- Số mẫu chỉ là `58`, nên một số chênh lệch chưa đủ để đưa ra kết luận quá mạnh.
- Số lượng case `none_correct` rất lớn (`44/58`), cho thấy toàn bộ bài toán vẫn khó với cả ba context mode.
- Phân tích hiện tại chỉ áp dụng cho cấu hình cố định `w = 5`, nên chưa cho biết xu hướng này có giữ nguyên khi thay đổi window size hay không.

## Hướng phân tích tiếp theo

Các bước hợp lý tiếp theo là:

1. Phân tích sâu hơn các case `prev_beats_next` và `both_wrong` để tìm nguyên nhân lỗi cụ thể.
2. Kiểm tra xem xu hướng của `prev_w5` có lặp lại trên tập `ww_algorithm_generated` hay không.
3. So sánh thêm theo độ dài trajectory để xem lợi thế của `prev_w5` có phụ thuộc vào độ dài trace.
4. Mở rộng sang nhiều giá trị `w` để kiểm tra tính ổn định của kết luận.

## Các artifact liên quan

Các artifact chi tiết của experiment nằm trong:

- `single_fault/experiments/step_based_context_mode_comparison/output/analysis/`

Bao gồm:

- `report.md`
- `qualitative_error_cases.md`
- `ww_hand_crafted_agent_accuracy.png`
- `ww_hand_crafted_step_accuracy.png`
- `ww_hand_crafted_context_mode_step_correct_venn.png`
- `ww_hand_crafted_step_error_distribution.png`
- `ww_hand_crafted_position_bucket_comparison.png`
- các file `.csv` summary và case-level tương ứng
