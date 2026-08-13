# AEGIS: phân tích framework x injected error_type

Nguồn: `data/error_localization/multi_fault/aegis/` (reload full-column từ
`Fancylalala/AEGIS`, script `data/error_localization/multi_fault/aegis.py`),
phân tích bằng `analyze_aegis_correlation.py` + `plot_aegis_*.py`. Cùng
format với [[mast_data_finding]] / [[trace_elephant_data_finding]].
**Không phát hiện bug dạng index-collision như MAST** — nhưng có 2 đặc
điểm riêng cần hiểu trước khi đọc số: metadata không đáng tin, và AEGIS là
dataset **tiêm lỗi nhân tạo** chứ không quan sát lỗi tự nhiên.

## 0. Khác biệt bản chất với MAST/TraceElephant — quan trọng khi diễn giải

MAST và TraceElephant **quan sát** lỗi xảy ra tự nhiên khi chạy hệ thống.
AEGIS **tiêm lỗi nhân tạo** vào trace vốn thành công (LLM-based manipulator
chỉnh sửa 1-4 agent theo chiến lược prompt_injection/response_corruption).
Vì vậy "framework X dính error_type Y nhiều hơn" ở AEGIS phản ánh **hành
vi của injection pipeline** (nó chọn tiêm gì cho framework nào), không
phải xu hướng lỗi tự nhiên của framework đó như 2 dataset kia.

## 1. Metadata không đáng tin — không dùng `num_agents`/`num_injected_agents`

- `metadata.num_agents` khớp số agent unique thực tế trong
  `conversation_history` chỉ **36.1%** số trace.
- `metadata.num_injected_agents` khớp `len(ground_truth.injected_agents)`
  thực tế chỉ **78.3%** số trace.

=> Luôn tính trực tiếp từ list thật (`conversation_history`,
`injected_agents`), không dùng 2 field đếm sẵn trong `metadata`.

## 2. magentic_one / smoagents: chế độ tiêm-1-lỗi-duy-nhất, không phải bug

Resolve TODO cũ (từng ghi trong `finding_1_framework_architecture.md`,
file đó nay không còn — nội dung chuyển vào đây): nghi vấn "SmolAgents/
Magentic-One luôn đúng 1 fault/trace, không giải thích được bằng số lượng
agent" — nay có đủ field để trả lời dứt điểm:

| framework | n | luôn đúng 1 injected agent? | injection_strategy dùng |
|---|---|---|---|
| magentic_one | 449 | Có (100%) | luôn `prompt` (1 loại duy nhất) |
| smoagents | 481 | Có (100%) | luôn `prompt_injection` (1 loại duy nhất) |
| dylan (đối chứng) | 6322 | Không | trộn 5 tổ hợp strategy khác nhau, 1-4 agent/trace |

Cả 2 framework này còn **confound tuyệt đối với benchmark riêng**:
magentic_one chỉ chạy benchmark `magentic+gaia` (model ghi `unknown-model`
— đáng ngờ, có thể là placeholder), smoagents chỉ chạy `smol+gaia`. 4
framework còn lại (agentverse/dylan/llm_debate/macnet) dùng chung 5
benchmark (GSM8K/HumanEval/MATH/MMLU/SciBench) + toàn bộ `gpt-4o-mini`.

=> Kết luận: đây là **giới hạn kỹ thuật của pipeline injection** cho 2
framework này (chỉ hỗ trợ tiêm 1 lỗi/1 strategy), không phải bug ngẫu
nhiên. Loại 2 framework này khỏi so sánh pooled với 4 framework kia — coi
là 1 regime riêng. Ảnh confound: `data/figures/aegis_framework_x_benchmark.png`.

## 3. Chi-square framework x error_type (4 framework pooled) — có ý nghĩa nhưng effect size không đáng kể

Trên 4 framework thật (agentverse/dylan/llm_debate/macnet, đã loại 2
framework single-injection): **13/14 code có p<0.05**, nhưng Cramér's V
toàn bộ nằm trong khoảng **0.019 – 0.077** — dưới ngưỡng "effect nhỏ"
(0.1) theo quy ước Cohen. Đây là artifact cỡ mẫu lớn: pooled ~19000+ lượt
tag, nên chỉ cần lệch cực nhỏ giữa framework cũng đủ ra p<0.05, nhưng độ
mạnh liên hệ thực tế gần như không đáng kể.

| code (V cao nhất → thấp nhất) | p-value | Cramér's V |
|---|---|---|
| FM-2.3 Task Derailment | 4.5e-29 | 0.077 |
| FM-3.3 Incorrect Verification | 6.4e-28 | 0.075 |
| FM-2.1 Conversation Reset | 2.0e-21 | 0.066 |
| FM-1.1 Disobey Task Specification | 8.8e-13 | 0.051 |
| ... 9 code khác | — | 0.019–0.044 |
| FM-3.2 No/Incomplete Verification | 0.85 (không ý nghĩa) | 0.006 |

Nhìn pie chart theo 3 nhóm (`data/figures/aegis_pie_group_by_framework.png`):
cả 6 framework (kể cả 2 framework single-injection) có tỷ lệ System
Design/Inter-Agent/Verification **gần như giống hệt nhau** (34-42% /
40-48% / 15-23%) — mắt thường không phân biệt được, khớp đúng với Cramér's
V cực nhỏ đo được. Injection pipeline của AEGIS tiêm lỗi **gần như đồng
đều bất kể framework**, khác hẳn MAST/TraceElephant nơi phân bố lỗi lệch
rõ theo framework/kiến trúc (V thường 0.1–0.3).

## Kết luận

1. Không có bug data quality kiểu MAST (không index-collision, mỗi trace
   có nhãn riêng biệt hợp lý theo nội dung).
2. Nhưng metadata counts (`num_agents`, `num_injected_agents`) không đáng
   tin — luôn tính lại từ list thật.
3. magentic_one/smoagents là 1 regime riêng (tiêm đúng 1 lỗi, 1 strategy,
   1 benchmark riêng) — không pool chung với 4 framework kia.
4. Trên 4 framework thật: framework có ảnh hưởng tới loại lỗi bị tiêm,
   nhưng **effect size rất nhỏ** — AEGIS injection gần như framework-
   agnostic, không giống MAST/TraceElephant nơi kiến trúc thật sự tạo ra
   khác biệt rõ rệt về loại lỗi.
5. Vì AEGIS là **tiêm lỗi nhân tạo** chứ không quan sát tự nhiên, không
   nên dùng kết quả này để claim "framework X dễ lỗi Y hơn framework khác
   trong thực tế" — chỉ nói được về hành vi injection pipeline.

## File liên quan

- Loader (giữ đủ cột, không drop): `data/error_localization/multi_fault/aegis.py`
- Confusion matrix framework x benchmark: `data/error_localization/multi_fault/plot_aegis_framework_vs_benchmark.py`
- Correlation + chi-square + metadata reliability + single-injection report: `data/error_localization/multi_fault/analyze_aegis_correlation.py`
- Pie 3 nhóm lỗi/framework: `data/error_localization/multi_fault/plot_aegis_pie_by_group.py`
- Makefile target: `make load_aegis`

## TODO

- [ ] Thử tách riêng `model = "unknown-model"` (magentic_one) xem có phải
      placeholder lỗi hay giá trị hợp lệ.
- [ ] Nếu cần so sánh công bằng framework thật (loại confound benchmark):
      lọc theo từng benchmark riêng (vd. chỉ MATH) rồi so 4 framework pooled
      trên cùng 1 benchmark, xem effect size có tăng lên không.
