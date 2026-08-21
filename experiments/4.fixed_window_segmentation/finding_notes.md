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

## Kết quả who_and_when__hand-crafted (58 file, đủ 12 config)

Vẽ bằng `src/plot_who_and_when.py` (cùng dạng 2-panel short/long với
experiment 5) → `results/figures/who_and_when_overall_accuracy_by_length.png`.
Chỉ so `step_accuracy`, ngưỡng chia `num_steps = 22`.

- **Trace ngắn (<= 22 steps, n=19)**: không config nào vượt all_at_once
  (0.316). Tốt nhất `w5_prev_only` = 0.263.
- **Trace dài (> 22 steps, n=39)**: baseline sụp (all_at_once 0.077,
  step_by_step 0.128) và **`w11_prev_only` = 0.184** vượt cả hai — config
  duy nhất thắng. Đây là hướng đáng đào tiếp: fixed window chỉ có giá trị
  ở vùng trace dài, đúng với giả thuyết của experiment 3.
- `next_only` hỏng hoàn toàn ở step-level: **0.000 trên cả 232 lượt**
  (4 window size × 58 file). Đã kiểm tra không phải lỗi off-by-one
  (`step` trong dataset là 0-based, khớp index của `enumerate`) —
  `pred_step - gt_step` phân tán rộng và không bao giờ bằng 0. Cần soi
  lại `next_only_prompt` trước khi kết luận.

## Chưa làm

- `trace_elephant` mới xong 6.3/12 config (1860/3324 row toàn thí nghiệm).
  Job pm2 `exp4-fixed-window` chết lúc 03:01 16-08-2026 do DNS fail tạm
  thời (`httpx2.ConnectError: Temporary failure in name resolution` →
  `openai.APIConnectionError`) — `llm.py` không retry, `run.py` không bọc
  try/except quanh `process_file`, `autorestart: false`. Chạy tiếp:
  `pm2 restart exp4-fixed-window` (đã có `is_row_done` skip row cũ).
