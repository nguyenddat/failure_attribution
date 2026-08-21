# Segment First, Attribute Second
## Error Localization in Long Multi-Agent Traces

---

# 0. Problem Setup & Experimental Protocol

## Slide 0.1 — Bài toán: Error Step Localization

- **Input**: `question` (task gốc) + `trajectory` = chuỗi step, mỗi step có `agent_name` + `content`.
- **Output**: một chỉ số step duy nhất `pred_step` — **step đầu tiên xảy ra lỗi quan trọng** (lỗi dẫn tới kết quả cuối sai) — kèm `pred_agent` là agent thực hiện step đó.
- **Ground truth**: `mistake_step` (0-based) và `mistake_agent`.
- Đây là bài toán **định vị (localization)**, không phải phát hiện có/không lỗi: mọi trace trong dataset đều đã biết là hỏng, câu hỏi duy nhất là **hỏng ở đâu**.
- Quy ước **`pred_step = -1` = "Not Found"**: phương pháp quét hết trajectory mà không báo lỗi ở bất kỳ đâu. Luôn tính là sai, nhưng được theo dõi riêng vì đó là một *kiểu* sai khác hẳn (bỏ sót vs. chỉ nhầm chỗ).
- Đánh giá hoàn toàn **zero-shot**: không fine-tune, không few-shot example → đo trần năng lực sẵn có của model dưới từng chiến lược đưa context.

## Slide 0.2 — Model & tham số khởi tạo

| Hạng mục | Giá trị |
|---|---|
| Model | `gpt-4o-mini` (`openai/gpt-4o-mini`) |
| Provider | OpenRouter, qua LangChain `ChatOpenAI` |
| Temperature | **không set** → mặc định API (1.0) |
| `top_p` / `max_tokens` / `n` | **không set** → mặc định API |
| Context window | 128k (ngưỡng chặn thực dùng 126k, chừa 2k đệm) |
| Structured output | `PydanticOutputParser`, fallback `OutputFixingParser` |
| Số lần chạy / trace | 1 (không self-consistency, không voting) |
| Số model | **1 duy nhất** cho toàn bộ experiment 3–5 |

- Dùng **một model duy nhất** để biến độc lập chỉ còn là *chiến lược đưa context* (full trace vs. sliding window vs. segment) và *độ dài trace* — mọi chênh lệch kết quả đều quy về đó, không lẫn hiệu ứng model.
- Prompt đưa vào **system message**, có phần `{format_instructions}` sinh tự động từ Pydantic schema; output là JSON có `error_found: bool` + chỉ số step.
- `OutputFixingParser`: khi model trả JSON sai format, gọi model lần nữa để tự sửa → **không mất mẫu** vì lỗi parse (0 lỗi parse còn sót ở cả 3 experiment).
- *Lưu ý khi bị hỏi*: temperature 1.0 + chạy 1 lần/trace ⇒ có nhiễu ngẫu nhiên. Xu hướng đủ mạnh để kết luận không đổi, nhưng con số tuyệt đối nên đọc là **ước lượng điểm**, không phải giá trị có khoảng tin cậy.
- **Token guard** (`tiktoken`, `cl100k_base`): trace vượt 126k token bị đánh dấu `token_overflow` và loại khỏi phân tích — thực tế chỉ **4/2.554 lượt chạy (0,2%)** ở experiment 3, các experiment sau không có ca nào (window luôn nhỏ).

## Slide 0.3 — Datasets

| Dataset | #trace | TB step | Median | Min–Max | Loại lỗi | Dùng ở |
|---|---|---|---|---|---|---|
| `who_and_when__hand-crafted` | 58 | 51,6 | 32,5 | 5–130 | single-fault | Exp 3, 4, 5 |
| `trace_elephant` | 219 | 27,1 | 21 | 5–94 | single-fault | Exp 3, 4, 5 |
| `telbench` | 1.000 | 11,9 | 10 | 3–47 | multi-fault | **chỉ Exp 3** |

- Chọn 3 nguồn vì chúng **phủ ba dải độ dài khác nhau** (~12 / ~27 / ~52 step trung bình) → có đủ mẫu ở cả hai phía của điểm gãy, thay vì chỉ đo được một vùng.
- **Chuẩn hoá về một schema chung** để mọi phương pháp chạy được trên mọi nguồn:
  `{question, trajectory[{step, agent_name, content}], mistake_agent, mistake_step}`
  - `trace_elephant`: `content` lấy từ `output.choices[0].message.content`, fallback sang tóm tắt `tool_calls`/`tool_logs`; `mistake_step` gốc 1-based → shift về 0-based. 1 file thiếu nhãn → loại (220 → **219**).
  - `telbench`: 1 span = 1 step, `agent_name = "agent"` (giả), GT = span lỗi **đầu tiên** trong `gold.error_span_ids`.
- Trace đưa vào prompt dưới dạng phẳng, thống nhất giữa mọi phương pháp: `- step {i} - {agent_name}: {content}`.
- **Vì sao Exp 4–5 bỏ `telbench`**: nhãn agent là giả (`"agent"`) nên `agent_accuracy` vô nghĩa, và trace ngắn (median 10) nằm gọn trong một window — không phải vùng mà segmentation được thiết kế để cứu.
- **Paired design**: mỗi trace chạy qua *tất cả* phương pháp/cấu hình → mọi so sánh đều trên cùng tập mẫu, không có nhiễu do khác mẫu.

## Slide 0.4 — Quy mô chạy

| Experiment | Tập mẫu | Cấu hình | Số lượt chạy (file × config) |
|---|---|---|---|
| **Exp 3** — baselines | 1.277 trace (3 dataset) | 2 baseline (`all_at_once`, `step_by_step`) | **2.554** |
| **Exp 4** — sliding window | 277 trace (2 dataset) | 4 window size × 3 context mode = 12 | **3.324** |
| **Exp 5** — segment all-at-once | 277 trace (2 dataset) | 4 window size | **1.108** |

- Window size `k ∈ {5, 7, 9, 11}` giữ **giống hệt nhau ở Exp 4 và Exp 5** → so sánh cost/accuracy công bằng ở cùng "budget" ngữ cảnh.
- Mỗi lượt chạy trong Exp 4/5 gồm **nhiều LLM call** (một call/step hoặc một call/đoạn, dừng sớm khi báo lỗi) — nên số LLM call thực tế lớn hơn số lượt chạy nhiều lần; đó chính là trục cost ở section 4.
- Kết quả ghi tăng dần vào `accuracy.xlsx` / `cost.xlsx` với khoá `(model, dataset, method, file)` → **chạy lại là idempotent**, ngắt giữa chừng không mất dữ liệu và không chạy trùng.

## Slide 0.5 — Metrics

**Metric chính — độ chính xác**

- `step_accuracy` = **exact match** `pred_step == mistake_step`, trung bình trên tập trace.
  - Khắt khe: không partial credit, lệch 1 step vẫn tính sai.
  - Là metric dùng chung được cho mọi dataset và mọi phương pháp → dùng để dựng đường cong theo `num_steps`.
- `agent_accuracy` = `pred_agent == mistake_agent` (so khớp sau khi chuẩn hoá: lowercase, gộp khoảng trắng, bỏ hậu tố trong ngoặc).
  - Dễ hơn nhiều (không gian nhãn nhỏ) → dùng như *metric mềm*: model có tìm đúng "thủ phạm" không, kể cả khi trượt step.

**Metric phụ — chẩn đoán kiểu lỗi**

- **Not-Found rate** = tỉ lệ `pred_step == -1` (quét hết mà không báo lỗi). Tách bạch "bỏ sót" khỏi "nhầm chỗ".
- **Step offset** = `pred_step − gt_step`, xem phân phối: lệch sớm (báo lỗi quá sớm) hay lệch muộn, và lệch bao xa.

**Metric chi phí**

- `input_tokens` / `output_tokens`: **cộng dồn toàn bộ LLM call** của một trace (kể cả call của `OutputFixingParser`).
- **Cost (USD/trace)** = `input_tokens × $0,15/1M + output_tokens × $0,60/1M` (bảng giá `gpt-4o-mini`).
- **Latency (s/trace)** = tổng wall-clock (`time.perf_counter`) của mọi LLM call trong trace — đo tuần tự, chưa tính khả năng chạy song song các window.
- Section 4 đọc hai trục này cùng nhau qua **Pareto frontier** (cost ↔ accuracy, latency ↔ accuracy) và tỉ số **accuracy trên mỗi cent**.

---

# 1. Where Baselines Break: The Long-Trace Cliff

## Slide 1.1 — The gap this experiment closes

- Everything downstream — segmentation, window sizing, cost budgeting — assumes we can tell a **short (easy)** trace from a **long (hard)** one. So far we had **no principled boundary**: "long" was an intuition, not a measured quantity.
- Picking the boundary by convention (median length, a round number like 20, dataset averages) would make every later result an artifact of that arbitrary choice.
- **This step therefore measures the boundary instead of assuming it**: run both existing baselines across the full length range and locate where accuracy actually starts to collapse.
- Definition used throughout: the **breaking point** is the trace length beyond which `step_accuracy` falls to the **random-guess line (1/`num_steps`)** — not merely "where the number goes down".
  - Absolute accuracy *must* decay with length regardless of model quality, because the candidate set grows. Only a drop **to random** means the method has stopped carrying information.

## Slide 1.2 — The two baselines under test

- **`all_at_once`** — **1 LLM call**. Receives the entire trace, returns the step index of the first critical error.
- **`step_by_step`** — **up to n LLM calls**. Walks step 0 → n−1; each call sees the current step plus context up to that point and returns `error_found: bool`; **stops at the first `true`**. Scanning to the end without firing → `pred_step = -1` (counted wrong).
- The two differ **only in how context is partitioned**. Same model, same trace formatting, same metrics ⇒ any gap is attributable to the partitioning strategy, nothing else.
- Scale: 1.277 traces × 2 baselines = **2.554 runs**, 2.550 usable (4 dropped as `token_overflow`).

## Slide 1.3 — Headline: both baselines are weak everywhere

| Dataset (avg. length) | `all_at_once` | `step_by_step` |
|---|---|---|
| `telbench` (11,9 steps) | 17,0% | 11,2% |
| `trace_elephant` (27,1 steps) | 21,9% | 15,1% |
| `who_and_when` (51,6 steps) | 15,5% | 15,5% |

- **No baseline clears 22%.** Step-level attribution is an open problem, not a solved one we are merely optimizing.
- `step_by_step` is **not** better than `all_at_once` despite costing far more — **2,0× input tokens** and **4,1× total latency** across the whole experiment (14,3M vs 7,0M input tokens; 9.783 s vs 2.363 s).
- A single average per dataset hides the real structure — which is exactly why we need the length-resolved curve on the next slide.

## Slide 1.4 — The cliff sits at ≈30 steps

`step_accuracy` vs. trace length (`who_and_when` + `trace_elephant` pooled, single-fault):

| `num_steps` | `all_at_once` | `step_by_step` | random (1/n) | #traces |
|---|---|---|---|---|
| ≤ 20 | **35,9%** | 20,5% | 9,4% | 117 |
| 21–30 | **24,5%** | 16,3% | 4,2% | 49 |
| 31–40 | 6,5% | 9,7% | 2,9% | 31 |
| > 40 | **1,2%** | 8,8% | 1,7% | 80 |

- The cliff is sharp and it is located **around 30 steps**: `all_at_once` falls 24,5% → 6,5% → 1,2%, a **~20× collapse** across one length step.
- Beyond 40 steps `all_at_once` is **at or below the random line** — the signal is fully diluted in long context.
- `step_by_step` decays more gently (plateaus near 9%) but never rises above a few points over random, and pays superlinear cost for it.
- **This is not an edge case**: 111 of 277 single-fault traces (**40%**) sit past the 30-step boundary.
- ⇒ The operational split we lacked: **≤30 steps = "short/tractable", >30 steps = "long/broken"**.

## Slide 1.5 — The ranking flips across the boundary

| Regime | `all_at_once` | `step_by_step` | #traces |
|---|---|---|---|
| ≤ 30 steps | **32,5%** | 19,3% | 166 |
| > 30 steps | 2,7% | **9,0%** | 111 |

- Short traces: seeing everything at once wins decisively (+13 points).
- Long traces: the order **inverts** — chunking the context beats the global view by 3,3×, even though both are near the floor.
- Reading: `all_at_once`'s advantage is *global visibility*; past the boundary its ability to localize inside long context collapses faster than that advantage is worth.
- Consequence: **no single baseline wins at every length**, so the right method must be **length-aware** — i.e. it must segment. This is the direct motivation for Sections 2 and 3.

## Slide 1.6 — The cliff is not a context-window limit

- Traces > 40 steps use only **~14k input tokens** on average; even 100+ step traces use **~41k** — far below `gpt-4o-mini`'s 128k window.
- So the failure is **not** context overflow. It is degraded reasoning/localization as the **number of candidate steps** grows.
- Critical implication: **a bigger context window would not fix this.** The fix must *reduce the number of candidates the model weighs per decision* — which is what fixed-window segmentation does.

## Slide 1.7 — And `step_by_step` pays superlinearly for its flatter curve

Average per-trace cost (single-fault datasets):

| `num_steps` | `all_at_once` | `step_by_step` |
|---|---|---|
| 1–10 | 1,6 s / 1,9k tok | 7,5 s / 8,1k tok |
| 11–20 | 1,7 s / 3,0k tok | 14,7 s / 19,5k tok |
| 21–30 | 1,8 s / 4,3k tok | 18,9 s / 28,8k tok |
| 31–50 | 2,1 s / 7,9k tok | 23,2 s / 43,5k tok |
| 51–100 | 2,7 s / 12,9k tok | **36,7 s / 70,6k tok** |

- `step_by_step` scales ~**O(n²)** in tokens (n calls × a prompt that grows with n): at 51–100 steps it burns **5,5× the tokens and 13,6× the wall-clock** of `all_at_once` to buy ~9% accuracy.
- Its early-stop also produces a distinct failure mode: **Not-Found rate 22,2%** on short traces (scans the whole trace, never fires) vs. **0%** for `all_at_once`, which always names a step.
- Together: the flatter long-trace curve of `step_by_step` is bought at a price that does not scale — it is not a usable answer to the cliff, only evidence that *partitioning direction* is the right one.

## Slide 1.8 — Takeaways → why Sections 2 and 3 exist

1. **A breaking point exists and is measurable: ≈30 steps.** Below it baselines are usable (25–36%); above it they fall to random. We now have a data-derived short/long split instead of an assumed one.
2. **The cause is candidate dilution, not context capacity** — 41k of a 128k window at the worst length. Scaling context is the wrong lever.
3. **Sequential scanning flattens the curve but does not raise the ceiling**, and costs O(n²).
4. ⇒ The remaining lever: **cut long traces into pieces below the breaking point, then attribute inside each piece.** Two ways to do that — scan step-by-step inside a sliding window (Section 2), or judge a whole chunk in one call (Section 3).

## Caveats (if asked)

- `telbench` is multi-fault, mapped as 1 span = 1 step with the **first** error span as GT ⇒ its `step_accuracy` is a conservative estimate, and its `agent_accuracy` is meaningless (fake agent names). Its length-binned curve is also unreliable past 30 steps (only 29 traces there).
- Some files have `mistake_step >= num_steps` (3/58 `who_and_when`, 5/220 `trace_elephant`) → always scored wrong, so absolute accuracy is slightly understated for every method equally.
- The 100+ step bin holds ~10 traces — read it as a trend, not an estimate.
- Single model (`gpt-4o-mini`), temperature at the API default, one run per trace: the **exact location** of the cliff may shift with a stronger model, but the **shape** of the curve is the result that carries.
