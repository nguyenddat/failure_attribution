# Trajectory dài đến đâu thì Failure Attribution gặp khó — và Phân đoạn (Segmentation) giúp được gì?

> Báo cáo tổng hợp toàn bộ mạch nghiên cứu (không chỉ 1 experiment): trạng thái các phần đã chạy xong đến 2026-08-15. Một số cấu hình chưa chạy hết, ghi rõ ở mục Giới hạn.

## 1. Motivation

Bài toán failure attribution (xác định `mistake_agent` — ai gây lỗi, `mistake_step` — lỗi ở bước nào) trong multi-agent trajectory thường được benchmark trên trajectory ngắn. Trajectory thực tế có thể dài hàng chục đến hàng trăm bước, vượt xa context window hiệu quả của LLM-as-judge. Đưa toàn bộ trajectory vào 1 prompt (`all_at_once`) là cách đơn giản nhất nhưng có nguy cơ "chìm" tín hiệu lỗi trong nhiễu khi trajectory dài. Câu hỏi trung tâm của toàn bài:

1. Trajectory dài đến mức nào thì độ chính xác định vị lỗi bắt đầu suy giảm đáng kể?
2. Các phương pháp phân đoạn trajectory (chia nhỏ theo số bước hoặc theo ngân sách token, chỉ đưa 1 phần ngữ cảnh vào mỗi lần gọi LLM) có khắc phục được sự suy giảm đó không, và đánh đổi chi phí ra sao?

## 2. Problem

Đo và so sánh độ chính xác định vị lỗi (`agent_accuracy`, `step_accuracy`) theo độ dài trajectory (`num_steps` / `trajectory_length`), giữa các nhóm phương pháp:

- **Baseline không phân đoạn**: `all_at_once` (1 lần gọi, toàn bộ trajectory), `step_by_step` (duyệt tuần tự, ngữ cảnh tích luỹ toàn bộ lịch sử đến bước hiện tại).
- **Phân đoạn theo số bước** (`step_based_multi_step`): mỗi lần gọi chỉ thấy 1 cửa sổ cố định `w` bước quanh vị trí hiện tại.
- **Phân đoạn theo ngân sách token** (`token_based_multi_step`): mỗi lần gọi chỉ thấy 1 cửa sổ giới hạn theo số token (thay vì số bước).
- **Chế độ ngữ cảnh** (context mode): với cửa sổ cố định, so sánh lấy ngữ cảnh 2 chiều (`surrounding`), chỉ quá khứ (`previous_only`), hay chỉ tương lai (`next_only`).

## 3. Method

**Cơ chế từng phương pháp** (`single_fault/methods/baselines/`):

| Phương pháp | Số lần gọi LLM | Ngữ cảnh mỗi lần gọi |
|---|---|---|
| `all_at_once` | 1 | toàn bộ trajectory |
| `step_by_step` | 1 mỗi bước, dừng khi tìm thấy lỗi | tích luỹ: bước 0..hiện tại |
| `step_based_multi_step_w{N}` | 1 mỗi bước, dừng khi tìm thấy lỗi | cửa sổ cố định N bước (N=5,6,7,8), 3 biến thể: surrounding (trước+sau), previous_only, next_only |
| `token_based_multi_step_{P}pct` | 1 mỗi bước, dừng khi tìm thấy lỗi | cửa sổ mở rộng dần trước/sau đến khi chạm ngân sách token = P% × max_token của model (P=25,40,50) |

Prompt hệ thống (`single_fault/system_prompt/`): `all_at_one.py` và `step_by_step.py` tương ứng 1-1 với 2 baseline; các phương pháp phân đoạn tái dùng logic vòng lặp của `step_by_step` nhưng thay ngữ cảnh đầu vào.

**Dataset** (`who_and_when`, 2 biến thể — độ dài rất khác nhau, xem mục 4):
- `ww_algorithm_generated`: trajectory ngắn, sinh tự động.
- `ww_hand_crafted`: trajectory dài, biên soạn thủ công.

**Model**: `gpt-4o-mini`. **Metric**: `agent_accuracy`, `step_accuracy` (khớp chính xác ground truth), `input_tokens` (chi phí).

## 4. Experiment

### 4.1 Phân bố độ dài trajectory (`dataset_analysis`)

| Dataset | N | trajectory_length min/median/mean/max |
|---|---|---|
| `ww_algorithm_generated` | 126 | 5 / 10 / 8.72 / 10 |
| `ww_hand_crafted` | 58 | 5 / 32.5 / 51.6 / 130 |

Hai dataset này tự nhiên tạo thành 1 phép so sánh ngắn-vs-dài: `algorithm_generated` gần như bị chặn trần ở 10 bước, `ww_hand_crafted` trải dài đến 130 bước, trung vị 32.5.

### 4.2 Đường cong suy giảm theo độ dài chi tiết (Experiment 3, gộp 3 dataset gồm cả telbench/trace_elephant)

Bucket `step_accuracy` theo `num_steps`, `all_at_once`, `gpt-4o-mini` (n=1030 file):

| num_steps | step_accuracy | N |
|---|---|---|
| 1–5 | 33.7% | 101 |
| 6–10 | 16.5% | 278 |
| 11–15 | 12.3% | 204 |
| 16–20 | 11.2% | 214 |
| 21–30 | 8.8% | 102 |
| 31–50 | 6.2% | 97 |
| 51+ | 2.3% | 43 |

### 4.3 Segmentation vs baseline (`step_based_segmentation`, `token_based_segmentation`), trên `ww_hand_crafted` (n=58, dài, trung vị 32.5 bước)

| method | agent_accuracy | step_accuracy |
|---|---|---|
| all_at_once | 0.500 | 0.086 |
| step_by_step | 0.483 | 0.172 |
| step_based_multi_step_w5 | **0.552** | 0.138 |
| step_based_multi_step_w6 | 0.466 | 0.103 |
| step_based_multi_step_w7 | 0.431 | 0.069 |
| step_based_multi_step_w8 | 0.397 | 0.069 |
| token_based_multi_step_25pct | 0.362 | 0.069 |
| token_based_multi_step_40pct | 0.0 (dữ liệu chưa chạy hết, không tin cậy) | 0.0 |

Trên `ww_algorithm_generated` (n=126, ngắn) — mọi phương pháp phân đoạn đều **kém hơn** `all_at_once` (0.500 / 0.325):

| method | agent_accuracy | step_accuracy |
|---|---|---|
| step_by_step | 0.198 | 0.119 |
| step_based_multi_step_w5 | 0.254 | 0.111 |
| token_based_multi_step_25pct | 0.246 | 0.056 |

### 4.4 Chi phí (mean `input_tokens`, `ww_hand_crafted`)

| method | mean input_tokens |
|---|---|
| all_at_once | 17,336 |
| step_based_multi_step_w5 | 35,393 |
| step_by_step | 48,015 |
| step_based_multi_step_w6 | 47,851 |
| step_based_multi_step_w7 | 44,840 |
| step_based_multi_step_w8 | 56,251 |
| token_based_multi_step_25pct | 184,037 |

### 4.5 Chế độ ngữ cảnh (`step_based_context_mode_comparison`, w=5, `ww_hand_crafted`, n=58)

| method | agent_accuracy | step_accuracy |
|---|---|---|
| all_at_once | 0.500 | 0.086 |
| step_by_step | 0.483 | 0.172 |
| step_based_w5 surrounding | 0.552 | 0.138 |
| step_based_w5 previous_only | 0.414 | **0.172** (bằng step_by_step) |
| step_based_w5 next_only | 0.448 | 0.086 (bằng all_at_once, kém nhất) |

44/58 case không phương pháp nào trong 3 chế độ giải được.

## 5. Results — trả lời câu hỏi trung tâm

**(1) Ngưỡng trajectory dài gây khó:** không có 1 điểm gãy duy nhất được kiểm định thống kê chính thức, nhưng dữ liệu bucket (mục 4.2) cho thấy suy giảm bắt đầu rõ từ khoảng **6 bước** (33.7% → 16.5%) và trở nên nghiêm trọng từ khoảng **20 bước trở lên** (11.2% → 2.3% ở 51+ bước) — giảm hơn 14 lần từ đầu đến cuối phổ độ dài quan sát được.

**(2) Phân đoạn có giúp không? Kết quả trộn lẫn, không phải lời giải rõ ràng:**

- Trên trajectory dài (`ww_hand_crafted`), **chỉ `step_based_multi_step_w5` (cửa sổ nhỏ, 2 chiều)** vượt cả 2 baseline — nhưng chỉ ở `agent_accuracy` (0.552 so với 0.500/0.483), còn `step_accuracy` (0.138) vẫn thấp hơn `step_by_step` (0.172).
- Mở rộng cửa sổ (w6→w8) hoặc chuyển sang ngân sách token đều làm accuracy **giảm dần, không tăng** — nhiều ngữ cảnh hơn không tự động tốt hơn.
- Ngữ cảnh **chỉ-quá-khứ** (`previous_only`, thiên về gần đây) đạt `step_accuracy` ngang `step_by_step` (0.172) dù dùng ít ngữ cảnh hơn hẳn — tín hiệu hứa hẹn nhất tìm được: phần lớn thông tin hữu ích nằm ở lịch sử gần, không cần toàn bộ lịch sử. Ngữ cảnh chỉ-tương-lai (`next_only`) không giúp gì.
- Trên trajectory ngắn (`ww_algorithm_generated`), phân đoạn **luôn kém hơn** `all_at_once` — cắt nhỏ trajectory vốn đã ngắn chỉ làm mất ngữ cảnh, không có lợi ích gì để đánh đổi.
- Về chi phí: phân đoạn theo bước với cửa sổ nhỏ (w5) rẻ hơn `step_by_step` (35k vs 48k token), nhưng cửa sổ càng rộng chi phí càng tiệm cận hoặc vượt `step_by_step`. Phân đoạn theo token là **đắt nhất trong mọi phương pháp thử nghiệm** (184k token/`w5` — do ngân sách 25% của 128k token bị nhân dồn qua nhiều lần gọi lặp).

**Kết luận:** Segmentation không phải giải pháp toàn diện. Nó chỉ có lợi thế hẹp (agent_accuracy trên trajectory dài, cửa sổ nhỏ, thiên quá khứ) và trả giá bằng chi phí tăng hoặc step_accuracy không cải thiện. Không có cấu hình nào trong số đã thử nghiệm giải quyết được vấn đề suy giảm accuracy theo độ dài một cách dứt điểm.

## 6. Giới hạn / việc chưa xong

- `token_based_multi_step_40pct`/`50pct` trên `ww_hand_crafted` chưa chạy đầy đủ (phần lớn NaN, số 0.0/0.0 trong bảng không phải kết quả thật) — không dùng để kết luận.
- Experiment 3 (mục 4.2) dùng thêm `telbench`/`trace_elephant` để mở rộng phổ `num_steps` lên đến 51+, nhưng `telbench` mới chạy `all_at_once`, chưa có `step_by_step`/segmentation — chưa thể nối trực tiếp đường cong suy giảm với bảng segmentation (mục 4.3–4.5), vốn chỉ có trên `ww_hand_crafted`/`ww_algorithm_generated`.
- Cỡ mẫu `ww_hand_crafted` nhỏ (n=58) — khác biệt giữa các phương pháp phân đoạn (vài điểm % accuracy) chưa được kiểm định ý nghĩa thống kê.
- `twin_comparison_segment` (phát hiện ranh giới đoạn bằng khoảng cách embedding giữa các bước liền kề) là hướng khám phá riêng, chưa benchmark accuracy — chưa tích hợp vào bảng so sánh trên.
- Prompt `subtask_alignment`/`task_decomposition` tồn tại trong `system_prompt/` nhưng chưa xác nhận được dùng ở component nào trong pipeline segmentation hiện tại — cần rà lại nếu định dùng.
