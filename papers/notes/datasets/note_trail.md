# TRAIL: Trace Reasoning and Agentic Issue Localization

- Sự phát triển của LLM -> agentic systems có khả năng tự động hóa những tác vụ nhiều bước, phức tạp trên nhiều lĩnh vực khác nhau: kỹ thuật phần mềm và truy hồi thông tin đa bước (multi-hop IR). 
- Khác với các mô hình sinh truyền thống, tác tử có thể tương tác với nhiều loại công cụ và điều hướng môi trường một cách linh động, thường với sự giám sát tối thiểu từ con người.
=> Phức tạp gia tăng này của hệ thống == Quy trình đánh giá thách thức và đa chiều hơn.

Việc đánh giá và gỡ lỗi hiệu năng của chúng vẫn là một thách thức lớn:
- Tính bất định của tác tử
- Đặc thù giải quyết tác vụ nhiều bước đòi hỏi observability cao hơn nhiều so với các đánh giá đầu-cuối (end-to-end) đơn giản mà các benchmark hiện có cung cấp. 
=> Đòi hỏi Failure Taxonomy chi tiết + Trace được chú thích kỹ lưỡng đồng thời bám sát các ứng dụng thực tế chứ không xoay quanh dữ liệu giả lập.

Tóm tắt limitation của related works:
- Các nghiên cứu trước đây chủ yếu tập trung vào những trace đã được phân tích cú pháp dưới dạng văn bản phi cấu trúc (mast.md) => không phản ánh đầy đủ đầu ra thường gặp của các framework tác tử có cấu trúc và được ghi nhận theo các định dạng chuẩn hóa (OpenTelemetry)
- Việc xử lý dữ liệu có cấu trúc vẫn là bài toán khó với LLM

Contributions:
- Failure Taxonomy ở mức chi tiết
- benchmark TRAIL: mở rộng từ SWE-Bench và GAIA chú thích ở mức từng lượt (turn-level), tuyển chọn cẩn thận, nhằm chứng minh tính hợp lệ và tính hữu dụng thực tiễn của taxonomy được đề xuất.
=> :
- Phân tích ở mức từng bước (step-level)
- Bám sát kịch bản thực tế bằng cách tạo ra các trace có cấu trúc theo chuẩn OpenTelemetry, với độ dài vượt quá giới hạn context length của các mô hình hiện tại
- So với các benchmark chỉ tập trung vào suy luận và phối hợp giữa các tác tử, TRAIL chú trọng tính hợp lệ thông qua việc bổ sung vào taxonomy các nhóm lỗi thực thi hệ thống và lỗi lập kế hoạch chi tiết, sát thực tế hơn — chẳng hạn API errors và Task Orchestration Errors 
=> Những nhóm lỗi này không chỉ có ý nghĩa với nhà phát triển mô hình mà còn với người dùng và kỹ sư đang tối ưu các ứng dụng AI đơn tác tử lẫn đa tác tử.

Nghiên cứu chỉ rõ contribution họ mạnh ở: execution & planning

## 1. Cách xây dựng dataset (TRAIL)

Hướng ngược với MAST: 
- MAST đi từ *trace → taxonomy* (quy nạp, grounded theory)
- TRAIL đi từ *taxonomy → trace* (diễn dịch, top-down).

- Bước 1: Failure Taxonomy từ literature, không từ dữ liệu: các nghiên cứu trước -> 3 trục lớn: Reasoning, Planning & Coordination, System Execution. Trục System Execution là phần MAST không có.

- Bước 2: Sinh trace theo thiết kế — mục tiêu: ecological validity + trace đủ dài + lỗi nảy sinh hữu cơ:
    + Chọn 2 tác vụ nền buộc phải khám phá môi trường: GAIA (open-world search) và SWE-Bench-Lite (vá lỗi repo GitHub).
    + Chạy 2 kiến trúc khác nhau để phủ cả 2 chế độ: hierarchical multi-agent (OpenDeepResearch, backbone o3-mini) cho GAIA; single-agent (CodeAct + sandbox + interpreter + gitingest, backbone claude-3.7-sonnet) cho SWE-Bench.
    + Cố tình cài ràng buộc trong prompt (giới hạn độ dài output, ép exploration) để dẫn dụ lỗi xuất hiện tự nhiên — không tiêm lỗi nhân tạo.

- Bước 3: Toàn bộ trace thu qua OpenTelemetry / OpenInference => span có cấu trúc phân cấp (`span_id`, `parent_span_id`) ><  tuyến tính như Who&When

- Bước 4: Chú thích thủ công ở mức span (không dùng LLM-as-a-Judge như MAST): 4 annotator chuyên môn software engineering + log debugging. Mỗi lỗi được gán 4 trường: `category` (theo taxonomy) + `location` (span_id) + `evidence` (trích đoạn) + `description` + `impact` (HIGH/MEDIUM/LOW). Đo IAA trên 63 trace riêng.

=> 148 traces / 1987 spans (575 span chứa ≥1 lỗi) / 841 lỗi: trung bình 5.68 lỗi/trace

Tổng kết các trường dữ liệu trong datasets (lấy từ hugging face):
2 config `gaia` (117) + `swe_bench` (31) = 148 trace, mỗi row chỉ có 2 cột JSON string:
- trace: `{trace_id, spans}` — spans là **cây** OpenTelemetry (con nằm trong `child_spans`), không phẳng như Who&When. Mỗi span:
    + `span_id` / `parent_span_id` / `timestamp` / `duration`: định danh và vị trí trong cây
    + `span_name`: tên bước (`LiteLLMModel.__call__`, `Step 1`, `CodeAgent.run`, `SearchInformationTool`...)
    + `status_code`: Ok / Unset / Error
    + `span_attributes`: nội dung thật — `openinference.span.kind` (LLM / CHAIN / TOOL / AGENT), `input.value`, `output.value`, `llm.input_messages[i].{role,content}`, `llm.output_messages[i]`, `llm.model_name`, `llm.token_count.*`
    + `logs`: `body` = `{function.name, function.arguments, function.output}`
- labels: `{trace_id, errors, scores}`
    + errors: `category` (taxonomy) + `location` (span_id, **không phải step index**) + `evidence` (trích đoạn) + `description` + `impact` (HIGH/MEDIUM/LOW)
    + scores: 1 bản/trace, 4 cặp `{X_score 1-5, X_reasoning}` với X = reliability / security / instruction_adherence / plan_opt, kèm `overall`
- Không có `question` / `ground truth` riêng: đề bài GAIA, SWE-Bench chìm trong `input.value` của span gốc

Đếm lại trên bản tải về:
- Flatten cả cây được **4626 span**, không phải 1987 như paper; 841 lỗi nằm trên 577 span (paper 575)
- Cây sâu tới 8 mức; gaia 30.6 span/trace, swe_bench 33.8 span/trace nhưng nhiều lỗi hơn hẳn (8.26 vs 5.0 lỗi/trace)
- Trace rất dài: median ~400K ký tự JSON, max ~8.8M => không nhét trọn vào context được
- `category` chưa chuẩn hóa: 31 chuỗi cho ~21 nhãn (số ít/số nhiều, hoa thường, typo `Instruction non complience`) => phải normalize
- Phân bố lệch: `Formatting Errors` (196) + `Instruction Non-compliance` (153) chiếm ~40%, đuôi dài nhiều lớp chỉ 1-3 mẫu
- 2 lỗi có `location` trỏ tới span_id không tồn tại trong trace

## Điểm đáng chú ý

- Phân bố lỗi cực kỳ lệch
    + Formatting Errors + Instruction Non-compliance = 353/841 ≈ 42%
    + System Execution Errors rất hiếm
=> :
    + LLM hiện tại vẫn yếu ở suy luận mức cao và hiểu tham số tác vụ, ngay cả khi đã prompt-engineering cẩn thận
    + Lỗi hiếm (API failures) lại catastrophic và khó phục hồi — khác với goal deviation hay tool misinterpretation vốn còn cứu được

- Phần lớn lỗi ở mức HIGH hoặc MEDIUM >< 44% lỗi Output Generation là LOW impact
=> category và impact không tương quan: Lỗi thường gặp ≠ lỗi quan trọng.

- Metrics:
    + Cat. F1: phân loại đúng nhóm lỗi
    + Loc. Acc: định vị đúng span
    + Joint: đúng cả hai => 11%

- Long-context: 
    + O1, O3, Claude-3.7-Sonnet đều **CLE (Context Length Exceeded)** trên toàn bộ split SWE-Bench
    + Chỉ mô hình 1M+ context (Gemini-2.5, GPT-4.1, Llama-4) chạy được
=> Split SWE-Bench hiện **không đánh giá được** với phần lớn mô hình

## 3. 3 RQs

### RQ1 — Long context ảnh hưởng thế nào?

- Input: trace dài nhất **luôn gấp ≥2 lần** giới hạn context của mô hình
    + GAIA/gemini-2.5: max 8.25M = 8 x 1M max context length
    + Claude-3.7 (limit 200K), ngay cả giá trị TRUNG BÌNH cũng vượt limit: GAIA mean 262.67K, SWE-Bench mean 727.75K
- Output: annotation của human dài trung bình >1K token, max ~3.7–5.4K.

=> TRAIL  là bài toán long-context, không phải bài toán suy luận riêng biệt.

- Mọi metric đều tương quan ÂM với độ dài input
| | Location Acc | Joint Acc | Categ. F1 |
|---|---|---|---|
| Pearson r | **-0.379** | -0.291 | -0.296 |
| Spearman ρ | **-0.508** | -0.349 | -0.225 |

- Localization chịu ảnh hưởng nặng nhất: định vị suy giảm nhanh hơn phân loại khi trace dài ra.

### RQ2 — Có lợi từ reasoning nhiều hơn không?

- **Có, rõ rệt.** Mọi mô hình reasoning (trừ O1) đều vượt non-reasoning ở cả Cat. F1 và Loc. Acc.
- Khoảng cách **giãn rộng nhất ở Joint Acc**: reasoning models đạt 1.5–8× mô hình non-reasoning tốt nhất.
- Ablation trên cùng một mô hình (O3, reasoning.effort high/med/low): Cat. F1 giảm đơn điệu 0.296 → 0.277 → 0.264
=> Kết luận then chốt: cùng model, chỉ đổi effort => mạnh hơn so sánh chéo mô hình.

### RQ3 — Nhóm lỗi nào dễ/khó dự đoán? (§5.1.6, Hình 4)

- Khó nhất:
+ Context Handling Failures — gần như *mọi* mô hình F1 = 0.00. Ngoại lệ duy nhất: Claude-3.7 (0.18).
+ Tool Selection Errors — đa số 0.00–0.08; chỉ O3 nổi bật (0.53), Gemini-2.5-Pro (0.26), Claude-3.7 (0.27).
+ Task Orchestration — đồng loạt 0.00–0.08, ngoại lệ lạ: Gemini-2.5-Flash đạt 0.47 (cao hơn cả Pro).

- Dễ nhất:
+ Language-Only (một dạng hallucination) — mọi mô hình đều làm được (0.14–0.59) → không cần reasoning nâng cao.

Tuy nhiên, nhóm tác giả đề cập rằng:
- Nếu TRAIL về cơ bản chỉ là một long-context benchmark trá hình, thì cần hỏi: "vậy chỉ cần đợi mô hình context dài hơn là xong?"
- Context Handling Failures = 0.00 gần như toàn bộ => nhóm lỗi này theo định nghĩa là lỗi về việc giữ ngữ cảnh xuyên suốt, tức nó là lỗi liên đoạn (cross-segment)