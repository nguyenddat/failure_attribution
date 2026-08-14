# Experiment 1 (baseline) — code riêng cho từng dataset

## Bối cảnh

`experiments/single_fault/experiments/baseline/` hiện chạy 2 method
(`all_at_once`, `step_by_step`) trên `who_and_when__hand-crafted` và
`who_and_when__algorithm-generated`, dùng chung 1 bộ prompt/parser/driver
(`shared.py`, `methods/baselines/*.py`, `system_prompt/*.py`).

Yêu cầu: mở rộng baseline sang thêm `trace_elephant` (đã có loader, cùng
schema `trajectory`/`mistake_agent`/`mistake_step` như who&when) và
`telbench` (schema span-based, không có agent, multi-fault — khác hẳn).
Đồng thời tách code — mỗi dataset có prompt/parser/method/driver riêng thay
vì dùng chung 1 bộ như hiện tại, kể cả 2 dataset cùng schema
(who&when, trace_elephant).

## Kiến trúc

Mỗi dataset là 1 thư mục con độc lập trong
`experiments/single_fault/experiments/baseline/`:

```
experiments/single_fault/experiments/baseline/
  who_and_when/
    system_prompt.py   # prompt + parser cho all_at_once và step_by_step
    methods.py          # all_at_once_single_file, step_by_step_single_file
    run.py               # driver: loop 2 subset (hand-crafted, algorithm-generated)
    output/
  trace_elephant/
    system_prompt.py
    methods.py
    run.py
    output/
  telbench/
    system_prompt.py
    methods.py
    metrics.py           # FEA, precision/recall/f1 suy biến
    results.py           # CSV writer riêng (schema khác who&when/trace_elephant)
    run.py
    output/
```

`experiments/single_fault/experiments/baseline/run.py` (file cũ) trở thành
entrypoint gộp: gọi lần lượt `who_and_when/run.py`, `trace_elephant/run.py`,
`telbench/run.py`.

Các file dùng chung không đổi: `utils/schema.py`, `utils/results.py`,
`utils/accuracy.py`, `utils/file.py`, `utils/experiment_paths.py`,
`get_chat_completion.py`, `chat_models.py`. `methods/baselines/*.py` và
`system_prompt/*.py` (bộ cũ, dùng chung) được giữ nguyên không xoá — các
method khác (`task_decomposition`, `subtask_alignment`,
`step_based_multi_step`, `token_based_multi_step`) trong `methods/baselines/`
vẫn import từ đó, không thuộc phạm vi thay đổi này.

## who_and_when/ và trace_elephant/

Schema giống hệt nhau (`trajectory` list step có `agent_name`+`content`,
`mistake_agent` string, `mistake_step` int). Tách code theo yêu cầu, nhưng
dùng chung:

- `utils/schema.py::AccuracyMetrics/CostMetrics/Metadata`
- `utils/results.py` (writer CSV: `{method}_agent_acc`, `{method}_step_acc`,
  `{method}_pred_agent`, `{method}_pred_step`, cost columns)
- `utils/file.py::format_agent_behaviors`, `load_json`
- `utils/accuracy.py::agent_names_match`

Mỗi dataset có `system_prompt.py` riêng (nội dung prompt độc lập, không
import từ `system_prompt/all_at_one.py`/`step_by_step.py` cũ) và
`methods.py` riêng cài lại logic `all_at_once_single_file`/
`step_by_step_single_file` (logic behavior giữ nguyên như bản gốc hiện tại:
all_at_once = 1 LLM call trên toàn trajectory; step_by_step = đọc tuần tự,
dừng khi LLM báo `error_found`).

`run.py` mỗi dataset gọi thẳng `run_method_configs_for_dataset` từ
`shared.py` (driver resumable, ghi CSV incremental — vẫn dùng chung, chỉ
`MethodConfig.run_single_file` trỏ vào `methods.py` riêng của dataset).

Output: `output/{dataset_key}.csv`, `output/{dataset_key}_cost.csv` — giữ
tên file hiện có (`ww_hand_crafted.csv`, `ww_algorithm_generated.csv`) để
không phá các script phân tích downstream đang đọc từ
`experiments/baseline/output/`. `trace_elephant.csv`/`_cost.csv` là file
mới.

## telbench/

Đọc thẳng JSON gốc `data/error_localization/multi_fault/telbench/*.json`
(schema `schemas/telbench.py`: `spans: [{id, raw}]`,
`gold: {error_span_ids: [...]}`). Không convert sang schema
who&when — driver và writer riêng hoàn toàn.

**Ground truth mỗi sample:**
- `gold_spans` = `gold.error_span_ids` (list)
- `gt_first_error` = span có index nhỏ nhất trong `gold_spans` theo thứ tự
  xuất hiện trong `spans` (dùng cho FEA)

**`all_at_once`:** 1 LLM call, đưa toàn bộ `spans` (id + raw, giữ thứ tự)
+ `question`, hỏi span lỗi sớm nhất. Parser trả `{span_id: str}`.

**`step_by_step`:** đọc `spans` tuần tự (giống step_by_step who&when hiện
tại), mỗi lần hỏi LLM "span này có phải lỗi không" (yes/no), dừng ngay khi
gặp span đầu tiên = yes → đó là pred. Hết spans mà không có yes nào →
"Not Found".

**Xử lý exceed token limit (cả 2 method):** bọc try/except quanh từng lần
gọi LLM. Bắt exception context-length-exceeded từ provider → dừng ngay
(break vòng lặp nếu đang ở step_by_step), giữ nguyên `accuracy_metrics`
đang có tại thời điểm đó (có thể đã tìm thấy span lỗi ở step trước, hoặc
vẫn là "Not Found" nếu chưa tìm thấy gì) — không ghi đè/reset — chỉ set
thêm `exceeded_max_token_limit = True`.

**Metric (`metrics.py`):**
- `fea` = 1.0 nếu `pred_span == gt_first_error` else 0.0
- predicted set = `{pred_span}` (size 1, rỗng nếu "Not Found")
- `precision` = 1.0 nếu `pred_span in gold_spans` else 0.0
- `recall` = `1/len(gold_spans)` nếu `pred_span in gold_spans` else 0.0
- `f1` = `2*precision*recall/(precision+recall)` nếu hit else 0.0

**CSV (`results.py` riêng, không dùng `utils/results.py` cũ vì shape khác):**

Cột: `file, gold_spans, gt_first_error, {method}_pred_span, {method}_fea,
{method}_precision, {method}_recall, {method}_f1,
{method}_exceeded_max_token_limit, {method}_latency,
{method}_input_tokens, {method}_output_tokens`.

Cùng nguyên tắc resumable/incremental-write như `shared.py` hiện tại (skip
sample đã có đủ cột method, ghi CSV sau mỗi sample) — viết lại tương tự
trong `telbench/run.py` vì writer khác.

Output: `output/telbench.csv` (accuracy+metric gộp 1 file, không tách
accuracy/cost riêng như 2 dataset kia — vì cột ít, gộp cho gọn).

## Ngoài phạm vi

- Không đổi `methods/baselines/{task_decomposition,step_based_multi_step,
  token_based_multi_step}.py` hay các experiment khác
  (`step_based_segmentation`, `token_based_segmentation`,
  `step_based_context_mode_comparison`) — chỉ đụng
  `experiments/baseline/`.
- Không convert TELBench sang schema single_fault — giữ nguyên schema gốc,
  code riêng đọc thẳng.
- Không thêm agent_accuracy cho telbench (không có khái niệm agent).
- Model mặc định vẫn `gpt-4o-mini` (theo `DEFAULT_MODEL_NAME` hiện tại) trừ
  khi người dùng chỉ định khác lúc chạy.
