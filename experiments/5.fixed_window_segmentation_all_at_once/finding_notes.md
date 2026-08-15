# Experiment 5: Fixed Window Segmentation — All-at-once — Finding Notes

Smoke-test `--limit 1` (2 file, 4 window_size, không lỗi), output đã xóa
trước khi chạy full.

## Thiết kế

- Khác exp4 (stride-1, center-sliding, 1 LLM call/step): exp5 chia
  trajectory thành **đoạn overlap 50%** (stride = window_size // 2). Mỗi
  đoạn 1 LLM call duy nhất, hỏi "đoạn này có step nào lỗi không, step_id
  nào" — trả step_id thẳng thay vì quét từng step. Overlap giúp step gần
  biên đoạn vẫn được đánh giá trong ngữ cảnh đầy đủ ở đoạn kế tiếp (đoạn
  cuối clip nếu chạy quá cuối trajectory, không đệm giả; đoạn nào là tập
  con của đoạn trước — vượt quá cuối trajectory — bị bỏ).
- Không có context_mode (before/prev/next) — cả đoạn đã tự chứa toàn bộ
  nội dung cần xét, không có khái niệm "current step" cần ngữ cảnh quanh
  nó.
- `window_size ∈ {5,7,9,11}` — giữ giống exp4 để so sánh cost/accuracy
  công bằng ở cùng budget k.
- Quét tuần tự các đoạn, early-stop ở đoạn đầu tiên `error_found=True`.
  Nếu model trả `step_id` không thuộc đoạn hiện tại (hallucination), coi
  như miss, tiếp tục đoạn sau (không crash, không tính điểm sai lệch).
- Biên trajectory: đoạn cuối bị clip nếu không chia hết window_size,
  không đệm giả.
- Dataset: `who_and_when__hand-crafted` (58 file) + `trace_elephant`
  (220 file), giống exp4.

## Cấu trúc code

`datasets.py`/`llm.py` copy nguyên từ exp4 (không thay đổi logic).
`segments.py` (thay `windows.py`), `system_prompt.py` (1 prompt thay vì
3), `methods.py`, `export_excel.py` (bỏ cột `context_mode`), `run.py`
viết mới theo đúng pattern exp4.

## Chưa làm

- Chưa chạy full 278 file × 4 window_size (~1112 lượt file×window, mỗi
  lượt trung bình vài LLM call/trajectory tới khi early-stop hoặc hết
  đoạn). Lệnh chạy:
  `python "experiments/5.fixed_window_segmentation_all_at_once/src/run.py"`
- Chưa có script vẽ biểu đồ so sánh với exp4 (step_accuracy/agent_accuracy
  theo window_size, cùng số so sánh cost giữa 2 cơ chế).
