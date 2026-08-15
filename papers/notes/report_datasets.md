# Khảo sát Datasets cho Failure Attribution trong LLM Multi-Agent Systems

> Tổng hợp 7 dataset đã đọc/note trong `papers/notes/datasets/`: Who&When, TraceElephant, TRAIL, MAST, AEGIS, AgentErrorBench, TELBENCH. Mục tiêu: hiểu vì sao mỗi dataset ra đời, xây dựng thế nào, cấu trúc field ra sao, và dataset nào phù hợp cho bài toán failure attribution mà project đang làm.

## 1. Motivation

Bài toán chung của cả 7 nghiên cứu: **agent/multi-agent system thất bại ở đâu, do ai, và làm sao quy trách nhiệm (attribution) một cách tự động, đáng tin cậy?** Nhưng mỗi dataset chọn một điểm nhấn khác nhau — mức độ quan sát được của trace (chỉ output vs full trace), đơn vị gán nhãn (agent/step/span/module), cách sinh lỗi (tự nhiên vs tiêm nhân tạo), và cách mở rộng quy mô (thủ công vs LLM-as-a-Judge). Khảo sát này nhằm định vị dataset `who_and_when` (dataset chính project dùng) trong bức tranh chung, và xác định khoảng trống mà các thí nghiệm của project có thể khai thác.

## 2. Bảng tổng quan

| Dataset | Quy mô | Đơn vị lỗi | Cách sinh lỗi | Cách gán nhãn | Điểm đặc trưng |
|---|---|---|---|---|---|
| **Who&When** | 184 lỗi (~380 trace, 220 fail) | (agent, step) | Tự nhiên (chạy thật, fail thật) | 3 expert, 3 vòng, đồng thuận | Trace **output-only** (black-box), first visible mistake |
| **TraceElephant** | 220 trace fail / 380 tổng | (component, step) — decisive/inevitable error | Tự nhiên | 3 expert, 3 vòng | **Full trace** + reproducible execution environment; đối lập triết lý với Who&When |
| **TRAIL** | 148 trace / 1987 span (paper) / 4626 span (thực tải) / 841 lỗi | span (OpenTelemetry) | Tự nhiên, ràng buộc prompt để dẫn dụ lỗi | 4 annotator chuyên môn SE | Trace **cây phân cấp**, cực dài (median ~400K ký tự, max 8.8M) → long-context benchmark trá hình |
| **MAST** | 1642 trace (+21 human-annotated) | trace-level (multi-label 14 mã lỗi) | Tự nhiên | LLM-as-a-Judge (o1), grounded từ 150 trace ban đầu | Không định vị step, chỉ có/không có mỗi failure mode trong cả trace |
| **AEGIS** | 9533 trace (train 7146 + val 1787 + test 600) | (agent, error_type, injection_strategy) | **Tiêm lỗi nhân tạo** (LLM Adaptive Manipulator) | Sinh tự động + IAA kiểm chứng (κ=0.81–0.85) | Duy nhất dùng error injection để scale lên 10K; nhãn suy dẫn trực tiếp từ injection plan |
| **AgentErrorBench** | 200 trace (đều fail) | (step, module, error_type) | Tự nhiên | 10 annotator, 3 vòng, κ=0.55 | Phân loại theo **module nội tại** (memory/reflection/planning/action) thay vì theo agent; đổi quy mô lấy độ sâu (kèm feedback sửa lỗi) |
| **TELBENCH** | 2790 trajectory → Verified-1K (600 easy/400 hard) | semantic span | Tự nhiên | LLM đề xuất + expert thẩm định | Đơn vị là **semantic span** (đoạn hành vi nhất quán, không phải step thô) trong deep-research trajectory dài |

## 3. Từng dataset

### 3.1 Who&When (benchmark chính của project)

- **Vấn đề nhắm tới**: hình thức hóa bài toán *Automated Failure Attribution* — tìm decisive error `(i*, t*)` qua counterfactual intervention: sửa action tại `(i,t)` mà đổi kết quả trajectory từ fail sang success thì đó là decisive error; nếu nhiều decisive error thì chọn cái **sớm nhất**.
- **Xây dựng**: câu hỏi từ GAIA/AssistantBench; 2 loại hệ — CaptainAgent (tự động dựng team, GPT-4o, ≤10 vòng) và Magentic-One (5 agent chuyên biệt, ≤30 vòng). Chú thích 3 vòng bởi 3 chuyên gia (cá nhân → thảo luận bất đồng → cross-validate).
- **Field chính**: `question`, `ground_truth`, `history` = `[(content, name, role)]`, `mistake_agent`/`mistake_step`/`mistake_reason`.
- **Giới hạn tự thân**: trace chỉ có output agent (content), **thiếu input/prompt gốc, tool call, env state** → black-box. Ép constraint "lỗi đầu tiên" dù thừa nhận có thể multi-fault.
- **Baseline finding quan trọng cho project**: all-at-once thắng agent-level nhưng thua step-level; step-by-step ngược lại; cả ba suy giảm mạnh theo độ dài trajectory, hội tụ gần 0% ở L5 (93–130 bước) — đây chính là tiền đề cho các thí nghiệm "trace length performance cliff" của project.

### 3.2 TraceElephant

- **Đối lập trực tiếp với Who&When**: phân tích 184 case của Who&When thấy ≥21% không thể quy trách nhiệm đáng tin cậy nếu chỉ có log output → TraceElephant cung cấp **full execution trace** (input/output NL, tool/env interaction, config, kiến trúc hệ thống) + **reproducible execution environment** để hỏi counterfactual thật (không chỉ giả định trên giấy).
- **Đơn vị lỗi mở rộng**: không nhất thiết là "agent" riêng biệt mà là **functional component** — áp dụng được cho cả single-agent scaffold (planning/orchestration/tool-use module).
- **Nguyên tắc role-aware & recoverability-aware**: khác Who&When (dùng first visible mistake), TraceElephant gán decisive error cho bước mà lỗi trở nên **không thể cứu vãn** — nếu có verifier ở bước sau lẽ ra phát hiện được mà bỏ lỡ, lỗi quyết định gán cho verifier đó, không phải cho agent gây lỗi gốc.
- **3 hệ nguồn**: Captain-Agent×GAIA/AssistantBench, Magentic-One×GAIA/AssistantBench, SWE-Agent×SWE-Bench → 380 trace, giữ lại 220 fail.
- **Finding liên quan trực tiếp tới project**: full trace > only-output (agent-level 62%→51%, step-level 28%→16% khi mất input/output chi tiết — step-level nhạy hơn nhiều); **all-at-once tốt hơn step-by-step** vì trace TraceElephant dài hơn Who&When (20.5–29.3 vs 9.6–28.8 lời gọi LLM/trace) → context tích luỹ của step-by-step vượt quá dài, giảm hiệu năng. Đây là dữ liệu chéo-kiểm chứng cho việc chọn all_at_once làm baseline mạnh trên trajectory dài trong project.

### 3.3 TRAIL

- **Hướng ngược MAST**: đi từ taxonomy có sẵn (literature, 3 trục: Reasoning, Planning & Coordination, System Execution) xuống trace, thay vì quy nạp từ dữ liệu.
- **Trace có cấu trúc chuẩn OpenTelemetry** (span cây, không phẳng như Who&When) — điểm khác biệt lớn về format.
- **Kết quả chính (RQ1)**: mọi metric tương quan **âm** với độ dài input (Pearson r location acc = -0.379, Spearman ρ = -0.508); trace dài luôn vượt ≥2 lần context limit model → TRAIL về bản chất là bài toán long-context. Củng cố thêm giả thuyết "trace càng dài, attribution càng khó" mà project đang kiểm chứng trên who_and_when.
- **RQ3**: nhóm lỗi Context Handling Failures gần như mọi model F1=0.00 — lỗi mang tính cross-segment, khó định vị bằng cách chia nhỏ trace.

### 3.4 MAST (Why Do Multi-Agent LLM Systems Fail?)

- Không định vị step/agent cụ thể — chỉ multi-label 14 failure mode ở **mức toàn trace**. Vai trò chính: cung cấp taxonomy nền (dùng lại trong AEGIS, tham chiếu trong AgentErrorBench, TRAIL).
- Scale bằng LLM-as-a-Judge (o1) sau khi validate generalization ngoài phạm vi phát triển — mô hình mở rộng quy mô phổ biến nhất trong nhóm dataset này.
- Finding: không kiến trúc nào thắng toàn diện — MetaGPT (SoP-driven) ít lỗi specification nhưng nhiều lỗi verification hơn ChatDev (có phase test/review riêng) → gợi ý attribution nên "architecture-aware", điều TraceElephant cũng nhấn mạnh độc lập.

### 3.5 AEGIS

- **Duy nhất dùng error injection** để giải quyết sự khan khiếm dữ liệu (Who&When 184 lỗi, MAST 150 task, TRAIL 148 trace — đều quá nhỏ để train). LLM-based Adaptive Manipulator tiêm lỗi context-aware (prompt injection hoặc response corruption) vào baseline chạy đúng, chỉ giữ lại nếu can thiệp thực sự gây fail (`Z=1`).
- Nhãn suy dẫn trực tiếp từ injection plan → rẻ, nhưng là lỗi **nhân tạo**, không phải lỗi tự nhiên của model — đánh đổi giữa quy mô (9533 trace) và tính thực tế (ecological validity) so với 6 dataset còn lại.
- IAA cao (Program-Human κ=0.81, gần bằng Human-Human κ=0.85) — cho thấy pipeline tự động label khá đáng tin, nhưng đây là tiêm-rồi-xác-nhận chứ không phải phát hiện lỗi tự nhiên.

### 3.6 AgentErrorBench

- Khác biệt lớn nhất: phân loại lỗi theo **module nội tại của agent** (memory/reflection/planning/action/system) thay vì theo agent hay giai đoạn thời gian — chỉ áp dụng tốt cho single-agent scaffold có kiến trúc 4-module tường minh (ReAct-biến-thể).
- Trade quy mô lấy độ sâu: chỉ 200 trace nhưng annotator viết luôn **feedback sửa lỗi** tự nhiên ngữ (dù bản phát hành công khai thiếu field này so với mô tả paper).
- Có nhiều điểm lệch giữa paper và dữ liệu phát hành (nhãn rỗng 13.5%, tên nhãn không chuẩn hóa, 1 case step vượt phạm vi trace) — cảnh báo hữu ích khi làm việc với dataset dạng "small expert-annotated, chưa qua kiểm định công khai kỹ".

### 3.7 TELBENCH

- Bài toán: trong deep-research trajectory (multi-hop, dài), lỗi thường không phải câu trả lời sai cuối mà là **claim thiếu căn cứ commit sớm** rồi bị các đoạn sau kế thừa mà không xác minh lại.
- Đơn vị gán nhãn: **semantic span** — đoạn hành động liên tục xoay quanh 1 mục tiêu cục bộ (không phải step thô/turn thô) — do trace deep-research không có ranh giới agent/turn rõ như Who&When.
- Method đi kèm (DRIFT): 4-bước LLM pipeline (Claim Keeper → Support Seeker → Specialist Auditors → Dependency Tracer) cải thiện ~30 điểm % F1/FEA so với bare LLM, nhưng vẫn khẳng định "phát hiện vùng lỗi" và "định vị chính xác điểm bắt đầu lỗi" là hai năng lực khác nhau — first-error localization vẫn rất khó dù có framework tốt.
- **Giới hạn liên quan trực tiếp tới project**: xây semantic span cần log đầy đủ (tool call, intermediate reasoning); nếu chỉ có output log kiểu Who&When thì không đủ để phân đoạn theo span — nghĩa là kỹ thuật segmentation kiểu TELBENCH không áp dụng thẳng được lên `who_and_when`.

## 4. So sánh chéo theo trục quan trọng cho project

**Mức độ quan sát của trace** (ảnh hưởng trực tiếp tới việc segmentation có khả thi không):
- Output-only (black-box): **Who&When**
- Full trace + structured: TraceElephant, TRAIL (span cây), TELBENCH (semantic span), AgentErrorBench (4-module XML), AEGIS (phase-tagged)
- Raw unstructured string: MAST

**Cách sinh lỗi**:
- Tự nhiên (chạy thật, fail thật): Who&When, TraceElephant, TRAIL, MAST, AgentErrorBench, TELBENCH
- Tiêm nhân tạo: chỉ AEGIS

**Cách mở rộng quy mô nhãn**:
- Thủ công thuần (không LLM-as-Judge): Who&When, TraceElephant, TRAIL, AgentErrorBench
- LLM-as-a-Judge / LLM-hỗ trợ: MAST (o1 gán toàn bộ), AEGIS (injection tự sinh nhãn), TELBENCH (LLM đề xuất + expert thẩm định)
=> Đánh đổi rõ: dataset thủ công nhỏ (148–380 trace) nhưng nhãn "sạch" hơn; dataset có LLM hỗ trợ scale lên nghìn (1642–9533) nhưng phụ thuộc chất lượng judge.

**Bằng chứng độ dài trajectory làm suy giảm attribution** (xuất hiện lặp lại ở 3 dataset độc lập, không riêng project):
- Who&When: cả 3 baseline hội tụ gần 0% ở trajectory 93–130 bước.
- TraceElephant: all-at-once thắng step-by-step vì trace dài hơn Who&When.
- TRAIL: mọi metric tương quan âm với độ dài input, location accuracy suy giảm nhanh nhất.
=> Củng cố mạnh cho hướng nghiên cứu "trace length performance cliff" hiện tại của project: đây không phải hiện tượng riêng của who_and_when mà là pattern chung của cả literature.

## 5. Vị trí của `who_and_when` trong project và khoảng trống còn lại

- Project dùng `who_and_when` vì có **nhãn step-level tường minh** (`mistake_agent`, `mistake_step`) và **history dạng phẳng dễ xử lý** — đánh đổi lấy hạn chế output-only mà chính paper TraceElephant chỉ ra (≥21% case không đủ tin cậy nếu thiếu input/tool call).
- Khoảng trống chưa khai thác trong 7 dataset trên so với hướng project đang đi (segmentation theo độ dài):
  - Không dataset nào cung cấp sẵn nhãn phân đoạn kiểu window/token-budget để so sánh trực tiếp — project phải tự tạo (`step_based_multi_step`, `token_based_multi_step`).
  - TELBENCH cho thấy semantic-span segmentation cần log giàu hơn who_and_when đang có → nếu muốn áp dụng ý tưởng "claim commit sớm, kế thừa không xác minh" của DRIFT lên who_and_when, cần bổ sung field còn thiếu (input/tool call) — hiện chưa có trong pipeline project.
  - AgentErrorBench gợi ý một trục attribution khác (theo module nội tại) chưa được project thử nghiệm trên who_and_when (hiện chỉ attribute theo agent/step).

## 6. Nguồn

Toàn bộ chi tiết xây dựng dataset, cấu trúc field trên HuggingFace/GitHub, và các điểm lệch giữa paper – bản phát hành nằm trong `papers/notes/datasets/note_*.md` (7 file, mỗi file ứng với 1 dataset). Bài viết này là bản tổng hợp/so sánh chéo, không lặp lại chi tiết field-by-field đã có ở đó.
