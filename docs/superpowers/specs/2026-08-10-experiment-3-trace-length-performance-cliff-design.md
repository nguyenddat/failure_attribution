# Experiment 3: Trace Length Performance Cliff — Design Spec

Ngày viết: 2026-08-10

## 1. Assumption / câu hỏi kiểm định

Khi độ dài trajectory (số step) tăng, độ chính xác định vị lỗi (agent/step
accuracy) của phương pháp all-at-once giảm dần và có thể có một điểm "gãy"
(cliff) — điểm mà accuracy giảm mạnh đột ngột thay vì giảm tuyến tính.
Experiment 3 thu thập dữ liệu accuracy theo độ dài trace, trên nhiều model,
để làm cơ sở phân tích điểm gãy này (phân tích thống kê nằm ngoài phạm vi
spec này — spec này chỉ cover phần thu thập dữ liệu/chạy experiment).

## 2. Dataset dùng

Chuẩn hóa bộ 3 dataset dùng cho experiment 3 và mọi experiment tiếp theo:

| Dataset key | Đường dẫn data |
|---|---|
| `who_and_when__hand-crafted` | `data/error_localization/single_fault/who_and_when__hand-crafted/` |
| `trace_elephant` | `data/error_localization/single_fault/trace_elephant/` |
| `telbench` | `data/error_localization/multi_fault/telbench/` |

`trail` và `aegis` bị loại khỏi bộ dataset đang dùng (không xóa file/data/
schema trong repo — chỉ không đưa vào mapping của experiment 3).

`telbench` giữ nguyên vị trí hiện tại (`multi_fault/`), không di chuyển.

Experiment 3 định nghĩa mapping dataset-directory **riêng**, cục bộ trong
`src/datasets.py` của nó, trải trên cả hai path
(`error_localization/single_fault/` và `error_localization/multi_fault/`).
Không tái sử dụng hay sửa `experiments/single_fault/utils/datasets.py`'s
`DATASET_DIRS`.

## 3. Model dùng

2 model, chạy phương pháp all-at-once (không segmentation):

- `gpt-4o-mini` — đã có sẵn trong `experiments/chat_models.py`.
- `deepseek-v4-flash` — key mới, map sang OpenRouter model id
  `deepseek/deepseek-v4-flash-0731`. Thêm vào `models` dict và
  `model_names` list trong `experiments/chat_models.py`.

## 4. Kiến trúc / cấu trúc thư mục

Theo convention `docs/how_to_run_an_experiment.md`, thuộc track experiment
đánh số (numbered assumption-verification track):

```
experiments/3.trace_length_performance_cliff/
  src/
    datasets.py       # mapping dataset key -> thư mục data (riêng, local)
    token_check.py     # đếm token (tiktoken cl100k) + kiểm tra ngưỡng theo model
    run.py              # runner độc lập, chạy toàn bộ ma trận model x dataset x file
    export_excel.py     # ghi/đọc lại 2 file .xlsx resumable
  results/
    figures/            # để trống ở bước này, dành cho phân tích sau
    tables/
      accuracy.xlsx
      cost.xlsx
  finding_notes.md
```

Script chạy trực tiếp (`python experiments/3.trace_length_performance_cliff/src/run.py`),
không dùng `python -m` (tên folder bắt đầu bằng số). Bootstrap `sys.path`
tới repo root ở đầu `run.py`:

```python
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
```

### Runner độc lập — không dùng `shared.py`

`run.py` gọi trực tiếp:
- `all_at_once_single_file` từ
  `experiments.single_fault.methods.baselines.all_at_once`
- `get_chat_completion` từ `experiments.single_fault.get_chat_completion`
- `get_model` từ `experiments.chat_models`

**Không** dùng `experiments/single_fault/experiments/shared.py`'s
`run_method_configs_for_dataset` hay
`experiments/single_fault/utils/experiment_paths.py`. Lý do: shared.py gắn
với format CSV wide (1 cột/method) và cấu trúc output riêng của track
method-evaluation; experiment 3 cần format long/tidy khác (xem mục 5) và
vòng lặp thêm chiều model + dataset mà shared.py không có sẵn.

### Fix bug import có sẵn

`experiments/single_fault/methods/baselines/all_at_once.py` dòng 21 hiện
có import sai:

```python
from experiments.get_chat_completion import get_chat_completion
```

Đây là module không tồn tại (đúng phải là
`experiments.single_fault.get_chat_completion`). Vì `all_at_once_single_file`
được experiment 3 gọi trực tiếp, bug này chặn hẳn việc chạy. Sửa tại chỗ
(file gốc trong `single_fault/`, không copy/re-implement riêng), vì đây rõ
ràng là bug — sửa có lợi cho mọi chỗ khác đang/sẽ dùng module này:

```python
from experiments.single_fault.get_chat_completion import get_chat_completion
```

## 5. Data flow + Excel schema

### Vòng lặp chính

```
for model in [gpt-4o-mini, deepseek-v4-flash]:
    for dataset in [who_and_when__hand-crafted, trace_elephant, telbench]:
        for file in sorted(dataset_dir.glob("*.json"), key=int(stem)):
            # xem mục 6 cho chi tiết resume + token check + error handling
```

Với mỗi (model, dataset, file):
1. Load JSON, lấy `trajectory`, `mistake_agent`, `mistake_step`.
2. Tính `num_steps = len(trajectory)`.
3. Format trace thành chat content (dùng `format_agent_behaviors`, giống
   `all_at_once_single_file` nội bộ đang làm).
4. Đếm token của chat content bằng `tiktoken.get_encoding("cl100k_base")`.
5. Tính `avg_token_count` = tổng token của chat content chia `num_steps`
   (token trung bình mỗi step trong trace — xem ghi chú làm rõ ambiguity
   ở mục 9).
6. Kiểm tra tổng token (chat content + phần system prompt/format
   instructions ước lượng) so với ngưỡng riêng của model đó (mục 6).
7. Nếu vượt ngưỡng: skip, ghi row với status `token_overflow` (mục 6).
8. Nếu không vượt: gọi `all_at_once_single_file(data, metadata)` →
   `AccuracyMetrics`, `CostMetrics`. Ghi row với status `ok`.

### Excel schema — long/tidy (1 row / model / dataset / file)

Lý do dùng long format thay vì wide format gốc của `results.py`: method cố
định (all_at_once), nhưng model VÀ dataset đều biến thiên — wide format
(mỗi method 1 cột) sẽ nổ tổ hợp cột nếu ép model+dataset vào tên cột.

**`accuracy.xlsx`** — mirror cấu trúc `accuracy.csv` gốc + 3 cột yêu cầu,
bỏ hoàn toàn các cột giá tiền (`$`):

| Cột | Nguồn |
|---|---|
| `model` | model key |
| `dataset` | dataset key |
| `file` | tên file JSON |
| `gt_agent` | `AccuracyMetrics.gt_agent` |
| `gt_step` | `AccuracyMetrics.gt_step` |
| `pred_agent` | `AccuracyMetrics.pred_agent` (null nếu `token_overflow`) |
| `pred_step` | `AccuracyMetrics.pred_step` (null nếu `token_overflow`) |
| `agent_accuracy` | `AccuracyMetrics.agent_accuracy` (null nếu `token_overflow`) |
| `step_accuracy` | `AccuracyMetrics.step_accuracy` (null nếu `token_overflow`) |
| `num_steps` | số step trong trace (cột mới) |
| `avg_token_count` | token trung bình mỗi step trong trace (cột mới) |
| `status` | `ok` \| `token_overflow` |

**`cost.xlsx`** — mirror cấu trúc `cost.csv` gốc + 3 cột yêu cầu, bỏ
`input_cost`/`output_cost`/`total_cost`:

| Cột | Nguồn |
|---|---|
| `model` | model key |
| `dataset` | dataset key |
| `file` | tên file JSON |
| `latency` | `CostMetrics.latency` (null nếu `token_overflow`) |
| `input_tokens` | `CostMetrics.input_tokens` (null nếu `token_overflow`) |
| `output_tokens` | `CostMetrics.output_tokens` (null nếu `token_overflow`) |
| `num_steps` | số step trong trace (cột mới) |
| `avg_token_count` | token trung bình mỗi step trong trace (cột mới) |
| `status` | `ok` \| `token_overflow` |

3 cột mới (`dataset`, `num_steps`, `avg_token_count`) xuất hiện ở **cả
hai** file, giống cách `accuracy_df`/`cost_df` gốc đều tự chứa base columns
(`file`, `gt_agent`, `gt_step`) — để mỗi file .xlsx tự đủ nghĩa, đọc độc
lập được.

## 6. Error handling — token overflow

Ngưỡng token tính riêng theo model (không dùng 1 số cố định chung), trừ
buffer cho phần output + format instructions:

```python
MODEL_TOKEN_THRESHOLDS = {
    "gpt-4o-mini": 128_000 - 2_000,       # 126,000 — context 128k, trừ buffer output/format
    "deepseek-v4-flash": 1_050_000 - 2_000,  # 1,048,000 — context ~1.05M+, trừ buffer
}
```

Buffer 2,000 token dành cho system prompt + format instructions của
parser (`OutputFixingParser`) + output. Token đếm bằng
`tiktoken.get_encoding("cl100k_base")` trên chat content đã format (bước
ước lượng — không phải token count chính xác của tokenizer thật của từng
model, nhưng đủ để chặn trace quá dài).

Khi vượt ngưỡng:
1. Log lỗi ra console (Python `logging`, level ERROR): tên file, model,
   dataset, số token đo được, ngưỡng.
2. Ghi row vào cả `accuracy_df` và `cost_df` với `status="token_overflow"`,
   các cột prediction/cost để `None`/`NaN`, `num_steps`/`avg_token_count`
   vẫn ghi (tính được, không phụ thuộc việc gọi model).
3. **Không dừng chương trình** — tiếp tục file kế tiếp.

Đây là per-file skip-and-log, không phải hard abort.

## 7. File output — resumable

2 file `.xlsx` riêng (`results/tables/accuracy.xlsx`,
`results/tables/cost.xlsx`), tự viết logic resume cục bộ trong
`export_excel.py` (không tái sử dụng `results.py`/`shared.py` vì khác
schema):

- Đầu chương trình: nếu file `.xlsx` đã tồn tại, đọc bằng `pd.read_excel`
  vào DataFrame; nếu chưa, khởi tạo DataFrame rỗng với đúng cột schema
  (mục 5).
- Khóa để xác định 1 row đã hoàn thành: tổ hợp (`model`, `dataset`,
  `file`). Nếu row đã có với `status` không null (đã `ok` hoặc
  `token_overflow`) → skip, không gọi lại model.
- Sau mỗi file xử lý xong (thành công hoặc overflow): ghi đè lại cả 2
  `.xlsx` ngay (giống pattern ghi CSV sau mỗi vòng lặp trong
  `shared.py`), để có thể dừng giữa chừng và chạy tiếp không mất tiến độ.

## 8. Model key

Thêm vào `experiments/chat_models.py`:

```python
model_names = ["gpt-4o-mini", "deepseek-v4-flash"]
models = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash-0731",
}
```

## 9. Testing

Không viết bộ test tự động (pytest suite) cho experiment này — phù hợp
quy mô 1 script thu thập dữ liệu, chạy 1 lần. Thay vào đó: smoke test thủ
công — chạy `run.py` với giới hạn nhỏ (vd. flag `--limit N` giới hạn số
file mỗi dataset, hoặc chỉnh tạm trong lúc test) trên 1 model x 1 dataset
x vài file, kiểm tra bằng mắt: 2 file `.xlsx` sinh ra đúng cột, giá trị
hợp lý, cơ chế resume hoạt động (chạy lại không gọi lại model cho row đã
xong).

`run.py` hỗ trợ optional CLI flag `--limit N` (giới hạn số file/dataset,
mặc định None = chạy hết) để phục vụ smoke test này.

---

## Ghi chú tự-review (self-review)

- **Làm rõ ambiguity — `avg_token_count`**: yêu cầu gốc "average token
  count per trace" mơ hồ vì 1 trace/row không có khái niệm "trung bình"
  nếu hiểu là tổng token của cả trace. Đã chọn nghĩa: **token trung bình
  mỗi step trong trace** (tổng token chat content / `num_steps`) — cho
  thêm thông tin bổ sung ngoài `num_steps` (phân biệt trace có nhiều step
  ngắn vs ít step dài), hữu ích khi phân tích điểm gãy sau này. Tổng token
  toàn trace (dùng để check ngưỡng overflow) không lưu thành cột riêng vì
  suy ra được (`avg_token_count * num_steps`) và không nằm trong 3 cột
  được yêu cầu.
- **Không có TBD/TODO** trong spec.
- **Nội bộ nhất quán**: mục 4 (kiến trúc) khớp mục 5/6/7 (data flow, error
  handling, output) — cùng dùng 1 mapping dataset cục bộ, cùng 1 schema
  long/tidy, cùng cơ chế resume dựa trên (model, dataset, file).
- **Phạm vi**: giới hạn ở việc thu thập dữ liệu (chạy experiment, ghi
  Excel). Phân tích thống kê tìm điểm gãy (cliff detection, vẽ
  figures/) là bước sau, ngoài phạm vi spec này — sẽ cần spec/plan riêng
  khi tới lúc.
