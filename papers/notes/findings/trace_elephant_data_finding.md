# TraceElephant: phân tích system_name x mistake_agent/mistake_step

Nguồn: `data/error_localization/single_fault/trace_elephant/` (load từ HF
`TraceElephant/TraceElephant`, script
`data/error_localization/single_fault/trace_elephant.py`), phân tích bằng
`analyze_trace_elephant_correlation.py` + `plot_trace_elephant_*.py`. Cùng
format với [[mast_data_finding]], nhưng **không phát hiện bug data quality**
— dataset này sạch.

## 1. Confound: system_name x benchmark_name

Gần như tuyệt đối confound, **trừ GAIA**:

| system_name | benchmark_name | n |
|---|---|---|
| captain-agent | assistantbench | 12 |
| captain-agent | gaia | 73 |
| magentic-one | assistant-bench | 17 |
| magentic-one | gaia | 74 |
| swe-agent | swe-bench | 44 |

GAIA là điểm so sánh công bằng duy nhất (captain-agent n=73 vs magentic-one
n=74, cùng benchmark). Ảnh: `data/figures/trace_elephant_system_x_benchmark.png`.

## 2. Role association (Orchestrator/Worker/Verification) — có ý nghĩa thật

`mistake_agent` là tên riêng theo từng system (không chung taxonomy như
MAST), gom về 3 role dùng chung qua
`trace_elephant_agent_roles.py` (theo đúng cách paper §4.2.2 tự phân
nhóm): Orchestrator (planner trung tâm), Worker (agent thao tác môi
trường/tool), Verification (riêng captain-agent có 2 role kiểm định
DataVerification_Expert/Verification_Expert, không hệ nào khác có).

Chi-square `system_name x role`: **p=1.05e-7, Cramér's V=0.294** (có ý
nghĩa, effect size trung bình).

| system_name | Orchestrator | Worker | Verification |
|---|---|---|---|
| captain-agent | 17.6% | 65.9% | 16.5% |
| magentic-one | 29.7% | 70.3% | 0% |
| swe-agent | 2.3% | 97.7% | 0% |

Đọc: swe-agent gần như thuần Worker (hợp lý — về cơ bản single-agent, chỉ
1/44 trace đổ lỗi cho "Orchestrator"). magentic-one có tỷ lệ Orchestrator
cao nhất (30%) — khớp kiến trúc có 1 Orchestrator trung tâm rõ ràng.
captain-agent là hệ duy nhất có role Verification riêng biệt. Ảnh:
`data/figures/trace_elephant_pie_role_by_system.png`.

## 3. Vị trí lỗi trong trajectory (early/mid/late) — KHÔNG tái hiện được claim paper

Paper §4.2.2 (mục "Decisive Failure Steps") claim: CaptainAgent (dynamic
team formation) → lỗi phân tán khắp timeline; Magentic-One/SWE-Agent (fixed
central orchestrator) → lỗi dồn ở early step.

Tự tính `mistake_step / n_steps` (0=bước đầu, 1=bước cuối), chia 3 phần
bằng nhau (early/mid/late), chi-square `system_name x bucket`:
**p=0.33, Cramér's V=0.103 — KHÔNG có ý nghĩa thống kê.**

| system_name | mean position | early | mid | late |
|---|---|---|---|---|
| captain-agent | 0.462 | 33 | 31 | 21 |
| magentic-one | 0.487 | 36 | 26 | 28 |
| swe-agent | 0.451 | 18 | 19 | 7 |

Mean position gần như giống hệt cả 3 hệ (0.45-0.49) — không hệ nào lệch
hẳn về phía early hay late. Nhìn histogram (`trace_elephant_step_position_by_system.png`):
magentic-one thực ra **bimodal** (đỉnh ở 0.1-0.2 VÀ đỉnh ở 0.6-0.7), không
"dồn sớm" thuần như paper mô tả; captain-agent nghiêng giữa nhẹ; swe-agent
hơi nghiêng sớm nhưng đuôi dài tới cuối.

**Giả thuyết vì sao lệch claim paper** (chưa verify):
- Paper có thể tính effort/step distribution trên toàn bộ 380 trace (kể cả
  160 trace thành công), trong khi dataset công khai chỉ release 220 trace
  fail — nếu trace thành công có pattern khác, loại bỏ chúng có thể đổi
  hình dạng phân bố.
- Cách chia "early/mid/late" của paper có thể không phải chia đều 3 phần
  theo tỷ lệ vị trí — có thể theo absolute step count hoặc theo cụm sự
  kiện khác.
- `n_steps` ở đây đếm theo `step_records.json` (mỗi record = 1 lần agent
  hành động) — có thể không khớp đơn vị "step" paper dùng khi vẽ Figure 6.

## Kết luận

- Không giống MAST, **dataset này không có bug** — mọi trace có nhãn riêng
  biệt, không phát hiện pattern trùng lặp bất thường (mỗi `task_id` unique,
  `mistake_agent`/`mistake_step` không lặp lại theo vị trí file).
- Phần "role gây lỗi khác nhau theo kiến trúc" của paper **tái hiện được**
  và có ý nghĩa thống kê thật trên data công khai.
- Phần "vị trí lỗi theo timeline khác nhau theo kiến trúc" của paper
  **không tái hiện được** trên chính data công khai của họ — cần nghi ngờ
  claim này hoặc phương pháp đo lại (xem giả thuyết ở mục 3), không nên
  trích dẫn lại claim "CaptainAgent dispersed, Magentic-One/SWE-Agent
  early-concentrated" như sự thật đã kiểm chứng.

## File liên quan

- Loader (giữ đủ cột, không drop): `data/error_localization/single_fault/trace_elephant.py`
- Role mapping: `data/error_localization/single_fault/trace_elephant_agent_roles.py`
- Confusion matrix: `data/error_localization/single_fault/plot_trace_elephant_system_vs_benchmark.py`
- Correlation + chi-square (role, step position): `data/error_localization/single_fault/analyze_trace_elephant_correlation.py`
- Pie role/system: `data/error_localization/single_fault/plot_trace_elephant_role_by_system.py`
- Histogram vị trí lỗi/system: `data/error_localization/single_fault/plot_trace_elephant_step_position.py`
- Makefile target: `make load_trace_elephant`
- Bổ sung mục "TraceElephant" trong [[framework_aware]] (đã note claim gốc
  từ paper trước khi verify — nay cần gắn thêm cảnh báo "vị trí lỗi không
  tái hiện được bằng data thật").

## TODO

- [ ] Update [[framework_aware]] mục TraceElephant: thêm cảnh báo claim
      "Decisive Failure Steps" không tái hiện được (mục 3 ở trên).
- [ ] Thử tính lại trên toàn bộ 380 trace nếu tìm được cách lấy 160 trace
      thành công (hiện dataset công khai chỉ có 220 trace fail).
- [ ] Đối chiếu lại đúng cách paper định nghĩa "early/mid/late" (đọc lại
      Appendix nếu có chi tiết hơn §4.2.2).
