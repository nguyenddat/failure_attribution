# Experiment 3: Trace Length Performance Cliff — Finding Notes

Chạy chưa xong / chưa có kết quả — file này ghi lại các quyết định thiết
kế phát sinh khi cài đặt (nằm ngoài spec gốc) để người đọc kết quả sau
này hiểu rõ giới hạn dữ liệu.

## Sai lệch so với spec gốc (phát hiện khi cài đặt)

- **telbench**: raw JSON không có `trajectory`/`mistake_agent`/`mistake_step`
  (có `spans`/`gold.error_span_ids` thay vào) và là multi-fault thật (nhiều
  `error_span_ids`). Đã map: 1 span = 1 step, `agent_name="agent"` (giả),
  `mistake_step` = vị trí của phần tử **đầu tiên** trong `error_span_ids`.
  `agent_accuracy` trên dataset này vô nghĩa (luôn khớp) — chỉ
  `step_accuracy` dùng được.
- **trace_elephant**: dùng `task_instruction` thay vì `question`; mỗi step
  trong `trajectory` không có field `content` sẵn (có `input`/`output` dạng
  OpenAI chat-completion object, khác nhau giữa 3 hệ thống nguồn). Đã map
  `content = output.choices[0].message.content`, fallback sang tóm tắt
  `tool_calls` khi rỗng. 1 file có `mistake_step: None` — bị skip, không
  ghi row.

## Data quality — mistake_step vượt quá độ dài trajectory

Kiểm tra toàn bộ 3 dataset sau khi normalize (không phải lỗi do code
normalize — `who_and_when__hand-crafted` dùng `trajectory`/`mistake_step`
gốc, không biến đổi gì):

- `who_and_when__hand-crafted`: 3/58 file có `mistake_step >= num_steps`
  (`25.json`: 51 vs 28 step; `39.json`: 8 vs 5 step; `50.json`: 24 vs 19
  step).
- `trace_elephant`: 5/220 file có `mistake_step == num_steps` (lệch đúng 1
  — có thể do nguồn đếm step 1-indexed cho nhãn nhưng 0-indexed cho
  trajectory, hoặc step gây lỗi không nằm trong trajectory được ghi lại):
  `97.json`, `107.json`, `133.json`, `158.json`, `160.json`.
- `telbench`: 0/1000 file lệch.

Không crash `all_at_once_single_file` (không có indexing theo `gt_step`,
chỉ so sánh số) — các file này vẫn chạy được, `agent_accuracy`/
`step_accuracy` sẽ tự động = 0 (không match với `pred_step`/`pred_agent`
hợp lệ nào). Không sửa trong phạm vi spec này (chỉ thu thập dữ liệu) —
nếu phân tích sau này thấy các file này gây nhiễu, lọc theo cột `status`/
`gt_step`/`num_steps` trong `accuracy.xlsx`.

## Chưa chạy được trong phiên cài đặt này

Không có conda env nào sẵn có đủ `langchain_classic` + `pandas` +
`openpyxl` + `huggingface_hub` + `tiktoken` cùng lúc — `run.py` chưa được
smoke-test end-to-end (3 module con `datasets.py`/`token_check.py`/
`export_excel.py` đã smoke-test được bằng base env, xem lịch sử cài đặt).
Cần cài thiếu vào 1 env rồi chạy thử:
`python experiments/3.trace_length_performance_cliff/src/run.py --limit 2`
