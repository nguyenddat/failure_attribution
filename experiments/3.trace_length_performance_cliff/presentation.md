# Experiment 3 — Breaking Point Performance of Baselines

## Slide 1-2 — Câu hỏi nghiên cứu

- Các baseline attribution hiện tại (đưa cả trace vào 1 prompt) **hỏng ở độ dài trace nào**?
- Đo trực tiếp `step_accuracy` theo `num_steps` để tìm điểm gãy (breaking point), thay vì chỉ báo cáo 1 con số trung bình.
- Nếu tồn tại điểm gãy rõ ràng → chứng minh nhu cầu phải **segment** trace trước khi attribute (động lực cho Exp 4–5).

## Slide 3 — Setup (1/4): Bài toán

- **Bài toán: error step localization** — cho một task và toàn bộ trace của hệ multi-agent, chỉ ra **step đầu tiên xảy ra lỗi quan trọng** (lỗi dẫn tới kết quả cuối sai).
- **Input**: `question` (task gốc) + `trajectory` = chuỗi step, mỗi step có `agent_name` + `content`.
- **Output**: một chỉ số step duy nhất `pred_step ∈ [0, n−1]`.
- **Ground truth**: `mistake_step` (và `mistake_agent` với dataset single-fault).
- Đây là bài toán **định vị** (localization), không phải phân loại có/không lỗi — mọi trace đều đã biết là có lỗi, chỉ hỏi lỗi ở đâu.
- Đánh giá **zero-shot**: không fine-tune, không few-shot example — đo trần năng lực của baseline hiện tại.

## Slide 4 — Setup (2/4): Model & tham số

| Hạng mục | Giá trị |
|---|---|
| Model | `gpt-4o-mini` (`openai/gpt-4o-mini`) |
| Provider | OpenRouter, qua LangChain `ChatOpenAI` |
| Temperature | **không set** → dùng mặc định của API (1.0) |
| top_p / max_tokens / n | không set (mặc định) |
| Context window | 128k; ngưỡng chặn thực dùng 126k (chừa 2k đệm) |
| Structured output | `PydanticOutputParser` + `OutputFixingParser` |
| Số model | 1 (cố định cho toàn bộ thí nghiệm) |

- Dùng **một model duy nhất** để độ dài trace là biến độc lập duy nhất — mọi chênh lệch kết quả đều quy về độ dài, không lẫn hiệu ứng model.
- `OutputFixingParser`: khi model trả JSON sai format, gọi lại để tự sửa → **không mất mẫu** vì lỗi parse.
- **Token guard** bằng `tiktoken` (cl100k_base): trace vượt 126k bị đánh dấu `token_overflow` và loại khỏi phân tích — thực tế chỉ 2 trace telbench (4 row), còn **2.550/2.554 lượt chạy thành công (99,8%)**.
- *Lưu ý khi bị hỏi*: vì temperature = 1.0 và chỉ chạy 1 lần/trace, kết quả có nhiễu ngẫu nhiên; xu hướng đủ mạnh (chênh ~20×) nên kết luận không đổi, nhưng con số tuyệt đối nên đọc là ước lượng.

## Slide 5 — Setup (3/4): Datasets

| Dataset | #trace | TB step | Median | Max | Loại lỗi |
|---|---|---|---|---|---|
| `who_and_when__hand-crafted` | 58 | 51,6 | 32,5 | 130 | single-fault |
| `trace_elephant` | 219 | 27,1 | 21 | 94 | single-fault |
| `telbench` | 1.000 | 11,9 | 10 | 47 | multi-fault |
| **Tổng** | **1.277 trace** | | | | **2.554 lượt chạy** (× 2 baseline) |

- Chọn 3 dataset vì **phủ ba dải độ dài khác nhau** (~12 / ~27 / ~52 step trung bình) → có đủ mẫu ở cả hai phía của điểm gãy, thay vì chỉ đo được một vùng.
- **Chuẩn hoá về một schema chung** để cùng một baseline chạy được trên cả ba nguồn:
  `{question, trajectory[{step, agent_name, content}], mistake_agent, mistake_step}`
  - `trace_elephant`: `content` lấy từ `output.choices[0].message.content`, fallback sang tóm tắt `tool_calls`; `mistake_step` gốc 1-based → shift về 0-based. 1 file thiếu nhãn → loại (220 → 219).
  - `telbench`: 1 span = 1 step, `agent_name = "agent"` (giả), GT = span lỗi **đầu tiên** trong `gold.error_span_ids`.
- Trace đưa vào prompt dưới dạng phẳng, thống nhất: `- step {i} - {agent_name}: {content}`.
- **Paired design**: mỗi trace chạy qua *cả hai* baseline → so sánh trực tiếp trên cùng tập mẫu.

## Slide 6 — Setup (4/4): Metrics

**Metric chính**

- `step_accuracy` = **exact match** `pred_step == gt_step`, lấy trung bình trên tập trace.
  - Khắt khe: không có partial credit, lệch 1 step cũng tính sai.
  - Là metric dùng chung được cho cả 3 dataset → dùng để dựng đường cong theo `num_steps`.

**Metric phụ**

- `agent_accuracy` = khớp tên agent gây lỗi — chỉ có ý nghĩa với 2 dataset single-fault (trên telbench luôn = 1,0 do agent name là giả).
- `latency`, `input_tokens`, `output_tokens` — cộng dồn **toàn bộ lời gọi** của một trace, dùng để so chi phí giữa hai baseline.

**Đường tham chiếu (quan trọng)**

- **Random guess = 1/`num_steps`**: accuracy tuyệt đối giảm tự nhiên khi trace dài ra, nên không thể chỉ nhìn con số giảm.
- Chỉ kết luận "gãy" khi accuracy **tụt xuống ngang mức random** — đó là định nghĩa breaking point dùng trong bài này.

**Cách gộp**

- Đơn vị phân tích: 1 trace = 1 điểm dữ liệu; gộp theo bin `num_steps` (≤20 / 21–30 / 31–40 / >40) để dựng đường cong.
- Hai dataset single-fault được **pool chung** (cùng dạng bài toán, cùng dạng GT); telbench báo cáo riêng vì là multi-fault.

## Slide 7 — Hai baseline được đánh giá

- `all_at_once` — **1 lần gọi LLM**: nhận toàn bộ trace, trả về `step_number` (int) của lỗi quan trọng đầu tiên.
- `step_by_step` — **tối đa n lần gọi**: duyệt tuần tự step 0 → n−1, mỗi lần nhận step hiện tại + context tính đến đó, trả về `error_found` (bool), **dừng ngay tại `true` đầu tiên**. Duyệt hết mà không `true` → `pred_step = -1` (tính sai).
- Hai baseline khác nhau **chỉ ở cách chia context**, dùng chung model, chung cách format trace, chung metric → chênh lệch phản ánh đúng tác động của chiến lược chia.

## Slide 8 — Kết quả tổng thể: cả hai baseline đều yếu

| Dataset | all_at_once | step_by_step |
|---|---|---|
| telbench (11,9 step) | 17,0% | 11,2% |
| trace_elephant (27,1 step) | 21,9% | 15,1% |
| who_and_when (51,6 step) | 15,5% | 15,5% |

- Không baseline nào vượt 22% — attribution ở mức step vẫn là bài toán chưa giải được.
- `step_by_step` **không** tốt hơn `all_at_once` dù tốn hơn nhiều: 2,0× input token, 4,1× latency trung bình.

## Slide 9 — Điểm gãy: ~30 step (single-fault, pooled)

`step_accuracy` theo độ dài trace (who_and_when + trace_elephant):

| num_steps | all_at_once | step_by_step | random (1/n) | #trace |
|---|---|---|---|---|
| ≤ 20 | 35,9% | 20,5% | ~9% | 117 |
| 21–30 | 24,5% | 16,3% | 4,2% | 49 |
| 31–40 | 6,5% | 9,7% | 2,8% | 31 |
| > 40 | 1,3% | 8,8% | ~1,5% | 80 |

- **Cliff nằm ở khoảng 30 step**: `all_at_once` rơi từ 24,5% → 6,5% → 1,3%, tức **giảm ~20× chỉ trong một bước nhảy độ dài**.
- Trên 40 step, `all_at_once` gần như **bằng hoặc dưới random guess** — tín hiệu bị "loãng" hoàn toàn trong context dài.
- `step_by_step` suy giảm thoải hơn (giữ ~9%) nhưng cũng chỉ nhỉnh hơn random, và trả giá bằng chi phí bùng nổ.
- Trace > 30 step chiếm 111/277 trace single-fault (~40%) — đây không phải trường hợp biên hiếm gặp.

## Slide 10 — Đảo chiều thứ hạng ở trace dài

- Trace ngắn (≤ 30 step): `all_at_once` thắng rõ (35,9% vs 20,5%).
- Trace dài (> 30 step): **đảo chiều** — `all_at_once` 2,7%, `step_by_step` 9,0%.
- Diễn giải: điểm mạnh của `all_at_once` là *nhìn toàn cục*; khi trace vượt ngưỡng, khả năng định vị trong context dài sụp đổ nhanh hơn lợi ích toàn cục.
- Hệ quả: **không có baseline nào thắng ở mọi độ dài** → cần cơ chế phụ thuộc độ dài, tức là segmentation.

## Slide 11 — Cliff không phải do "hết context"

- Trace > 40 step chỉ dùng TB ~13k input token (100+ step: ~41k) — còn xa giới hạn 128k của gpt-4o-mini.
- Vậy điểm gãy **không phải do tràn context window**, mà do **suy giảm khả năng suy luận/định vị** khi số ứng viên step tăng.
- Kết luận quan trọng: mở rộng context window **không** cứu được bài toán này; phải giảm số ứng viên mà model phải cân nhắc mỗi lần.

## Slide 12 — Chi phí tăng siêu tuyến tính (step_by_step)

Latency / input token trung bình mỗi trace:

| num_steps | all_at_once | step_by_step |
|---|---|---|
| 1–10 | 1,6 s / 2,5k tok | 4,5 s / 5,0k tok |
| 21–30 | 2,1 s / 9,9k tok | 11,1 s / 18,6k tok |
| 31–50 | 2,2 s / 10,4k tok | 17,4 s / 32,1k tok |
| 51–100 | 2,7 s / 12,9k tok | **36,7 s / 70,6k tok** |

- `step_by_step` scale ~O(n²) token (n lần gọi × prompt dài dần) — ở trace 51–100 step tốn **5,5× token và 13× thời gian** so với `all_at_once`, để đổi lấy ~9% accuracy.
- Tổng chi phí toàn thí nghiệm: 7,0M vs 14,3M input token.

## Slide 13 — Kết luận & chuyển tiếp

- **Có tồn tại breaking point rõ ràng ở ~30 step**: dưới ngưỡng baseline còn dùng được (~25–36%), trên ngưỡng rơi xuống mức random.
- Nguyên nhân là số lượng ứng viên / độ pha loãng tín hiệu, **không** phải giới hạn context.
- Duyệt tuần tự (`step_by_step`) chỉ làm phẳng đường cong chứ không nâng trần, và chi phí tăng bậc hai.
- → Hướng đi: **chia trace dài thành các đoạn ngắn dưới ngưỡng gãy rồi mới attribute** — chính là thiết kế của Experiment 4/5 (fixed-window segmentation).

## Ghi chú giới hạn (nếu bị hỏi)

- `telbench` là multi-fault, được map 1 span = 1 step và lấy span lỗi **đầu tiên** làm GT → `step_accuracy` là ước lượng bảo thủ; `agent_accuracy` = 1,0 do agent name giả.
- Một vài file có `mistake_step >= num_steps` (3/58 who_and_when, 5/220 trace_elephant) → các trace này luôn tính sai, làm accuracy thấp hơn thực tế một chút.
- Bin ≥ 100 step chỉ có 10 trace → đọc là xu hướng, không phải ước lượng chắc chắn.
- Chỉ chạy trên gpt-4o-mini → vị trí chính xác của cliff có thể dịch chuyển với model mạnh hơn, nhưng dạng đường cong là kết quả chính.
