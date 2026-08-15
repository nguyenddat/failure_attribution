# Experiment 4: Fixed Window Segmentation — Finding Notes

Chưa chạy full — chỉ mới smoke-test `--limit 1` (2 file, 24 method-config
mỗi file, không lỗi). File này ghi lại quyết định thiết kế phát sinh khi
cài đặt.

## Thiết kế

- Window size `k ∈ {5, 7, 9, 11}` (lẻ, kiểu kernel CNN — center = step
  hiện tại).
- 3 context mode, mỗi mode dùng **prompt riêng** phản ánh logic đánh giá
  khác nhau (không dùng chung 1 prompt generic như
  `single_fault/methods/baselines/step_based_multi_step.py`):
  - `both`: đánh giá bước hiện tại dựa cả ngữ cảnh trước lẫn sau.
  - `prev_only`: giả định các bước trước đã được xác nhận đúng (đúng theo
    ngữ nghĩa quét tuần tự early-stop) — chỉ hỏi bước hiện tại có phải là
    NEW mistake hay không.
  - `next_only`: root-cause reasoning — nhìn hệ quả/dấu hiệu lỗi ở các
    bước sau để suy ngược xem bước hiện tại có phải nguồn gốc hay không.
- Ngân sách ngữ cảnh (không tính bước hiện tại) = `k - 1` cho cả 3 mode,
  để so sánh công bằng giữa các mode ở cùng window size. Biên trajectory
  bị cắt (clip), không đệm giả (valid-convolution style).
- Quét tuần tự, dừng sớm khi `error_found=True` (giống `step_by_step`
  baseline).

## Lệch so với hạ tầng có sẵn

Codebase đã có `step_based_multi_step.py` + `step_based_segmentation`/
`step_based_context_mode_comparison` (window 5-8, 1 prompt chung 3 mode,
chỉ chạy trên `who_and_when`). Theo yêu cầu, experiment 4 viết **code mới
hoàn toàn** (datasets/windows/system_prompt/methods/llm/export_excel/run),
không import từ các module đó — chỉ tham khảo logic. Có tái dùng 2 hạ
tầng chung không đặc thù method: `experiments.chat_models.get_model` và
`experiments.single_fault.utils.file.load_json`.

## Dataset

`who_and_when__hand-crafted` (58 file) + `trace_elephant` (220 file) —
theo lựa chọn người dùng, bỏ `telbench` (agent_accuracy vô nghĩa, quá to)
và bỏ qua path chết `ww_algorithm_generated` (đã đổi thành
`agent_error_bench`, schema khác, ngoài phạm vi).

## Chưa làm

- Chưa chạy full 278 file × 12 method-config (~3336 lượt file×method, mỗi
  lượt trung bình chạy nhiều LLM call/step tới khi early-stop). Lệnh chạy:
  `python experiments/4.fixed_window_segmentation/src/run.py`
- Chưa có script vẽ biểu đồ `step_accuracy`/`agent_accuracy` theo
  `window_size` × `context_mode` × `dataset`.
