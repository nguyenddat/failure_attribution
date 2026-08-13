- Multi-agent systems – MAS => khả năng mới trong việc giải quyết các bài toán phức tạp, quy mô lớn bằng cách phân rã nhiệm vụ cho các tác tử chuyên biệt hợp tác với nhau: suy luận toán học nâng cao, khám phá khoa học, và kỹ thuật phần mềm. 
- Sai sót của một tác tử đơn lẻ có thể lan truyền qua các tương tác và tạo ra lỗi quan sát được ở đầu ra nằm rất xa so với sai sót gốc
=> Phân tích nguyên nhân gốc rễ và gỡ lỗi có hệ thống cực kỳ khó khăn, và thúc đẩy nhu cầu về các phương pháp có thể quy trách nhiệm một lỗi hệ thống về đúng các tác tử gây ra cùng các chế độ lỗi (error modes) tương ứng.

Sự khan hiếm dữ liệu: Các benchmark hiện có đều nhỏ đến mức đáng ngạc nhiên: 
- Who&When: 184 lỗi được gán nhãn
- MASFT: chỉ hơn 150 nhiệm vụ để rút ra 14 chế độ lỗi
- TRAIL: 148 trajectory với 841 lỗi được gán nhãn
=> Việc chú giải thủ công tốn kém do chuyên gia thực hiện trên các nhật ký thực thi phức tạp. 

Contributions:
- benchmark Aegis: 10 nghìn trajectory lỗi được gán nhãn
- Quy trình có thể tái lập cho việc sinh lỗi tự động trong MAS

## 1. Cách xây dựng datasets (Aegis)

- Bước 0: Dùng 14 Failure Taxonomy của MAST + 6 MAS frameworks (MacNet, DyLAN, LLM-Debate, AgentVerse, Magentic-One, SmolAgents) × 6 benchmarks (MATH, GSM8K, HumanEval, SciBench, MMLU-Pro, GAIA).

- Bước 1: Thu thập baseline thành công có tính tất định: chạy MAS trên các task đúng (temperature=0, fix seed)

- Bước 2: Tiêm lỗi bằng LLM-based Adaptive Manipulator: manipulator sinh lỗi *context-aware* (code → infinite loop; math → phép tính sai nhưng hợp lý), theo 2 chiến lược chọn ngẫu nhiên:
    + Prompt Injection: sửa input/context của agent trước khi nó hành động.
    + Response Corruption: sửa output của agent sau khi nó hành động.
    + Mỗi `τ_corr` sinh ra K biến thể lỗi qua các Injection Plan: `P_inj = {(agent*, error_modes*), ...}` → chính plan này là nhãn ground-truth.

- Bước 3: Validation & gán nhãn: chỉ giữ lại trajectory nếu can thiệp thật sự gây thất bại hệ thống (`Z(τ_inj)=1`). Khi đó `G(τ_inj) = P_inj` — nhãn suy dẫn trực tiếp, tái lập được. Với hệ động như DyLAN cần post-hoc label refinement.

- Bước 4: Kiểm chứng chất lượng nhãn (IAA): 100 trajectory ngẫu nhiên, 3 expert gán nhãn mù → Human-Human κ=0.85, Program-Human κ=0.81 (nhãn tự động gần bằng chất lượng người).

Tổng kết các trường dữ liệu trong datasets (lấy từ hugging face):
`Fancylalala/AEGIS` chia 2 phần: Aegis-Training (train 7146 + val 1787) và Aegis-Bench (test 600) = 9533 record, không phải 10k như paper.
- id: `{benchmark}_{model}_{framework}_{k}_{hash}`
- metadata: `framework` / `benchmark` / `model` / `num_agents` / `num_injected_agents` / `task_type` (math, reasoning, code_generation)
- input:
    + `query`: đề bài gốc
    + `conversation_history`: trajectory phẳng = `[(step, agent_name, content, phase)]` — `phase` ∈ initialization / reasoning / discussion / evaluation, đây là field Who&When không có
    + `final_output`: đáp án cuối của hệ
- output: `faulty_agents` = `[(agent_name, error_type, injection_strategy)]` — `error_type` là mã MAST `FM-1.1` ... `FM-3.3`, `injection_strategy` ∈ prompt_injection / response_corruption
- ground_truth: `correct_answer` (đáp án đúng của benchmark) + `injected_agents` (= output, thêm `malicious_action_description`) + `is_injection_successful`

## 2. Các điểm đáng chú ý

- Case studies:
    + "Tôi sẽ tiết kiệm được bao nhiêu nếu mua vé năm cho cả gia đình (2 người lớn, 1 trẻ 5 tuổi, 1 trẻ 2 tuổi) tại Seattle Children Museum so với mua vé lẻ, nếu đi 4 lần trong năm?": Sai nằm ở agent cung cấp giá vé cung cấp giá sai => tính chi phí sai => chuỗi sai về sau. Gemini-2.5-pro nhầm triệu chứng với nguyên nhân gốc (đánh giá verification sai khi đánh giá nhiều bước dư thừa)
    + Case study 2: 1 agent quy đổi sai 1 phép toán đơn giản: 1 dặm = 20 thay vì 16 block => 1 chuỗi về sau đều sai. Cho thấy các phép sai đơn giản, nhỏ khó phát hiện hơn các phép sai lớn nhưng phép sai nhỏ lại là root cause.