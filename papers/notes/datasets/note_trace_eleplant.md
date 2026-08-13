- Failure attribution trong LLM-based MAS khó hơn debug truyền thống (state không rời rạc, non-deterministic, log ghi bằng ngôn ngữ tự nhiên mơ hồ).
- Kỹ thuật MAS-specific đã có: ECHO, AgenTracer, GraphTracer, FAMAS.

- Gap của Who&When (benchmark duy nhất hiện có cho bài toán này):
    + Chỉ cung cấp trace partially observable — đầu ra agent thôi, thiếu input/prompt/context gốc.
    + Phù hợp kịch bản black-box, nhưng lệch thực tế debug: dev thường có đủ instruction, prompt, message trung gian, tool call, env state.
    + Phân tích 184 case lỗi Who&When: ít nhất 21% không quy trách nhiệm đáng tin cậy được nếu chỉ có log đầu ra.

- Contributions: tinh thần "nhìn toàn bộ con voi" — full execution trace + reproducible execution environment.
    + Đơn vị quy lỗi: functional component (không nhất thiết agent riêng biệt) → áp dụng luôn cho single-agent scaffold (module plan/orchestrate/tool-use).
    + Khác Who&When 2 điểm:
        1. Trace step-by-step từ nhiều hệ agent tiêu biểu, ghi hành động cấp agent, input/output NL, tool/env interaction, config agent, kiến trúc hệ thống.
        2. Kèm reproducible execution environment → chạy lại kiểm soát được, kiểm tra state, hỏi counterfactual ("nếu agent này nhận input khác thì sao?").
    + Dataset: 220 trace lỗi, 3 hệ agent tiêu biểu (multi-agent orchestration + single-agent tool-focused scaffold), mỗi trace gán nhãn component chịu trách nhiệm + step gây lỗi quyết định.

- So sánh:
    + Who&When = benchmark nền, output-only, black-box.
    + TraceElephant = full-trace, reproducible, developer-centric — đối lập trực tiếp về triết lý thiết kế.
=> Nên tận dụng full trace khi có thể; lĩnh vực cần thêm benchmark đa dạng để đánh giá attribution từ nhiều góc.

## 1. Problem Formulation

- MAS `M` gồm tập `A = {a1, ..., aN}`, mỗi `ai` là 1 functional component (agent tường minh, hoặc module chức năng như planning/orchestration/tool-use trong single-agent scaffold).
- Tuân theo turn-based protocol: mỗi step chỉ đúng 1 component active.

Trace đầy đủ: `T = <o1, ..., oT>`, mỗi `oi` gồm:
- `xi`: input component nhận tại step i
- `yi`: output component sinh ra tại step i
- `step_i`: chỉ số thứ tự step
- `agent_i`: định danh component active tại step i

Task Outcome và Failure Attribution:
- Task outcome function `Ω(T) ∈ {0,1}` (1 = success, 0 = failure).
- Failure attribution 2 cấp: step-level (tìm step gây lỗi quyết định) + agent-level (component chịu trách nhiệm tại step đó).

Định nghĩa Inevitable Decisive Error:
- Step `t` là inevitable nếu sau bước đó, có làm gì tiếp theo thì vẫn lỗi — mọi continuation khả thi từ `t` đều dẫn tới fail (`∀T' ∈ C(T≤t): Ω(T') = 0`).
- Decisive error = step `t` sớm nhất thỏa điều kiện đó → component active tại step này là component chịu trách nhiệm.

Nguyên tắc role-aware và recoverability-aware (khác Who&When — Who&When dùng first visible mistake, TraceElephant dùng inevitable, không phải visible):
- Nếu mistake ở step `t` vẫn còn recoverable — vì có verifier/orchestrator ở step sau chịu trách nhiệm kiểm tra & sửa — thì decisive failure step gán cho điểm bỏ lỡ cơ hội phục hồi đó, không phải bước gây mistake gốc.
- Ví dụ: agent hallucinate 1 fact ở step `t`, nhưng verifier ở step `t+2` có trách nhiệm phát hiện lỗi này mà không phát hiện → lỗi quyết định gán cho verifier tại step `t+2` (vì hệ vẫn recoverable cho tới lúc đó), chứ không gán cho agent hallucinate ban đầu.
- Mục tiêu bài toán: xác định đồng thời step-level + agent-level responsibility, given full failure trace.

## 2. Data Construction

B1: Nguồn dữ liệu: framework x dataset
- Captain-Agent x GAIA → 126 traces, 73 fail
- Captain-Agent x AssistantBench → 21 traces, 12 fail
- Magentic-One x GAIA → 119 traces, 74 fail
- Magentic-One x AssistantBench → 30 traces, 17 fail
- SWE-Agent x SWE-Bench → 84 traces, 44 fail
==> Tổng: 380 traces, 220 fail

B2: Thu thập trace bằng api middleware

B3: Tiền xử lý & lọc dữ liệu
- Trích xuất các thuộc tính cơ bản (agent name, step type…) để giữ độ trung thực với luồng thực thi gốc.
- Lọc tự động để loại bỏ local file paths, thông tin cá nhân/nhạy cảm.
- Rà soát thủ công từng trace trước khi công bố.
- Chỉ giữ lại 220 traces bị fail (loại bỏ trace thành công) làm instance chính thức cho benchmark.

B4: Gán nhãn thất bại (Human Annotation) — hoàn toàn thủ công, không dùng LLM-as-judge
- 3 annotator chuyên gia (≥1 năm kinh nghiệm debug hệ multi-agent) xác định cho mỗi trace fail:
  - Agent nào chịu trách nhiệm chính (agent-level attribution)
  - Bước nào là điểm quyết định khiến thất bại không thể cứu vãn (step-level attribution) — theo nguyên tắc role-aware & recoverability-aware
- Quy trình 3 vòng: làm độc lập → thảo luận nhóm → thống nhất các case tranh cãi đến khi đạt đồng thuận.

B5: Xây dựng môi trường thực thi tái lập (Reproducible Execution Environment)
- Mỗi trace đi kèm executable system (code, config gốc) để chạy lại, kiểm tra state, và đặt câu hỏi giả định kiểu "what if agent này nhận input khác?".

B6: Đánh giá các kỹ thuật attribution tự động
- Chạy nhiều phương pháp automated failure attribution trên benchmark dưới các cấu hình khác nhau (full trace vs. chỉ output; có/không có môi trường chạy lại).
- Kết quả: full trace → accuracy 65.9% (agent-level), 30.3% (step-level); cải thiện +22%/+76% so với chỉ dùng output; có running environment → step-level accuracy tăng thêm ~10%.

B7: Công bố benchmark
- Phát hành traces + annotation theo license CC BY 4.0, kèm code thu thập trace và công cụ đánh giá (open-source).

Tổng kết các trường dữ liệu trong datasets (lấy từ hugging face):
`TraceElephant/TraceElephant` chỉ có 1 file `data.zip` (597MB), HF auto-detect nhầm thành "imagefolder" (do có ảnh) → viewer hỏng, phải tải giải nén tay mới thấy field thật.

Cấu trúc: `data/{system}-runs-{benchmark}/{task_id}/{trace_metadata.json, step_records.json, images/*.png?}`. 220 thư mục = đúng 220 trace fail công bố (không kèm 160 trace pass còn lại của 380 tổng), đếm theo hệ khớp cột "# Failed" Table 1 paper. Không tách file label riêng như AEGIS/TRAIL — nhãn nằm luôn trong metadata.

- `trace_metadata.json`: `task_id/task_instruction/system_name/run_id` + `agent_configuration` (format khác hẳn theo hệ: Captain-Agent = prompt đầy đủ per-agent do MetaAgent tự sinh, Magentic-One = chỉ import statement (agent cố định sẵn), SWE-Agent = `{}` rỗng) + `agent_system_intro` (docs kiến trúc, cố định theo hệ) + nhãn lỗi `mistake_agent/mistake_step/mistake_reason` + `ground_truth` (GAIA/AssistantBench) **hoặc** `tests_status` (SWE-Bench: `FAIL_TO_PASS`/`PASS_TO_PASS`)
- `step_records.json`: list step `{step_id, agent_id, agent_name, input, output, tool_logs?}` — `input`/`output` là **raw request/response OpenAI API** (`model,messages,temperature...` / `choices,usage...`), không gọn như `input_context`/`output_content` paper mô tả; `tool_logs` chỉ có request tool call, thiếu kết quả thực thi — phải tự tìm ở message role `tool` của step sau
- Screenshot: chỉ Magentic-One (WebSurfer) có, file rời `images/step_{id}_image_1.png`, không nhúng trong JSON, khớp qua tên file trùng `step_id` — 2070 ảnh tổng, chính là lý do HF nhận nhầm "imagefolder"
- Step/trace đo thực tế: Captain-GAIA 21.4, Captain-AssistantBench 15.6, Magentic-GAIA 27.8, Magentic-AssistantBench 35.5, SWE-Agent 35.2 (max 94)

## 3. Experiment setup:
- Static: Trace -> dự đoán
- Dynamic: đưa vào runtime thực tế
- Ablation: full/ with metadata/ with input/ with output

## 4. Các điểm đáng chú ý:
- Full > Only output:
    + Chỉ dùng output (giống Who&When): agent-level rớt từ 62% → 51%, step-level rớt từ 28% → 16%
    + Step-level nhạy cảm với thông tin thiếu hơn agent-level nhiều (giảm 76% vs 22%) — vì xác định đúng bước lỗi cần hiểu ngữ cảnh chi tiết hơn xác định đúng agent

- Dynamic > Static nhưng chủ yếu giúp step-level
    + Dynamic cải thiện step-level ~10% nhờ replay + counterfactual check để lọc bỏ ứng viên sai
    + Agent-level ít cải thiện hơn vì thông tin về vai trò/coordination đã có sẵn trong static trace

- Agentic technique > prompting-based: Xây dựng agentic technique > Refine prompt
- Baseline khác (ECHO, AgenTracer, GraphTracer, FAMAS) không đưa vào so sánh: không có code chạy được (không release runnable implementation), tác giả cố reproduce nhưng không ra số khớp paper gốc → loại khỏi eval, chỉ dùng 5 kỹ thuật tự triển khai (3 prompting + 2 agentic)
- All-at-Once tốt hơn Step-by-Step: trace của TraceElephant dài hơn nhiều (trung bình 20.5–29.3 lời gọi LLM/trace vs 9.6–28.8 ở Who&When) → khi đưa từng bước incremental, context quá dài về sau vượt context window → giảm hiệu năng

- Có ground truth luôn tốt hơn không có
- Backbone LLM ảnh hưởng rõ rệt

- Vị trí lỗi khác nhau theo kiến trúc
    + Với Magentic-One: lỗi ở giai đoạn đầu rất khó attribute (accuracy chỉ 8%), giai đoạn cuối dễ hơn nhiều (52%) — do hệ thống có chu kỳ explore/re-plan kéo dài, lỗi sớm không lộ ra ngay
    + Với Captain-Agent và SWE-Agent: accuracy ổn định qua các giai đoạn
    + Kiến trúc cố định thủ công (Magentic-One, SWE-Agent): lỗi tập trung nhiều ở giai đoạn đầu <=> lập kế hoạch/định tuyến ban đầu của orchestrator
    + Tự động Captain-Agent: lỗi phân tán đều hơn qua các bước, vì team agent được xây dựng động cho từng task nên lỗi có thể phát sinh ở nhiều điểm (chọn agent, lập kế hoạch, điều phối, gọi tool)
==> kỹ thuật attribution "architecture-aware", tùy biến theo kiểu kiến trúc MAS (centralized vs dynamic team, tool-heavy vs planning-heavy)

- Agent tương tác môi trường ngoài dễ gây lỗi nhất
    + Web browsing/thu thập thông tin hoặc chỉnh sửa code trực tiếp: > 50% tổng số lỗi do phụ thuộc vào hệ thống ngoài
    + Orchestrator/planner: 18-29%: lỗi phân rã task, chọn agent sai, hoặc logic điều phối kém, thường không lộ ra ngay mà lan truyền và khuếch đại qua các bước sau

## 5. Limitations & Future Work

- Limitations (paper tự nhận):
    + Chỉ dùng 3 hệ agent (Captain-Agent, Magentic-One, SWE-Agent) → không cover hết kiến trúc MAS khác hay kịch bản black-box, finding có thể không generalize toàn bộ
    + Tự biện minh: 3 hệ chọn cố ý đa dạng (dynamic team assembly / centralized orchestration / single-agent SWE scaffold) → giảm bias kiến trúc, tăng tính đại diện

- Không có mục "Future Work" riêng, nhưng phần Implications and Takeaways (4.2.4) đóng vai trò đó, 5 hướng đề xuất:
    1. Architecture-Aware Attribution: kỹ thuật attribution cần tận dụng prior knowledge về thiết kế MAS (centralized vs dynamic team, tool-heavy vs planning-heavy) để tập trung vào điểm dễ lỗi nhất
    2. Enhancing Static Attribution: Static Agentic hiện chỉ dùng tool-use cơ bản → cần reasoning phức tạp hơn (hypothesis generation/testing cycle, graph-based reasoning trên agent interaction network)
    3. Leveraging Dynamic Environment sâu hơn: hiện chỉ replay + counterfactual 1 bước → có thể mở rộng đọc code suy luận node lỗi, tái dựng control-flow, state-space exploration, fault injection tự động, causal discovery đa biến
    4. Specializing Models via Fine-Tuning: model reasoning mạnh (Claude-4.5, DeepSeek-R1) tương quan thuận với accuracy → fine-tune model nhỏ chuyên biệt (theo hướng GraphTracer, AgenTracer) dùng trace + running environment của TraceElephant, thêm structural feature (agent graph, tool-call sequence) làm auxiliary signal
    5. Toward Integrated Debugging Tools: xây tool tích hợp tự động capture trace, visualize agent interaction flow, gợi ý điểm lỗi tiềm năng bằng attribution model → giảm overhead debug trong dev framework