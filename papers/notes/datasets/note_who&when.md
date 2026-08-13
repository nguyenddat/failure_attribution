# Which Agent Causes Task Failures and When On Automated Failure Attribution of LLM Multi-Agent Systems

- Xu hướng tích hợp LLM thành agent và xây dựng multi-agent system đã thu hút sự chú ý 
=> Tiềm năng đáng chú ý trên nhiều lĩnh vực, bao gồm lập trình (Hong et al., 2023), khám phá khoa học (Ghafarollahi & Buehler, 2024), và giải quyết các thách thức phức tạp trong thế giới thực (Fourney et al., 2024).

- **Failure Attribution** — tức xác định các thành phần của hệ thống trực tiếp dẫn đến thất bại của tác vụ — là một bước then chốt, đóng vai trò nền tảng cho việc định hướng cải tiến. Tuy quan trọng như vậy, quy trình này phần lớn:
    + Bị bỏ qua trong dòng nghiên cứu chính do đòi hỏi nhiều công sức (phân tích các log lịch sử phức tạp, xoay xở với những chi tiết kỹ thuật rắc rối của hệ thống)
    + Áp dụnh xạ từ kết quả đánh giá benchmark sang các thành phần gây lỗi phụ thuộc nặng nề vào chuyên môn lĩnh vực 

=> Khi các hệ thống ngày càng phức tạp, thách thức này càng trở nên khó khăn hơn do số lượng thành phần cần xem xét khi quy kết lỗi ngày một tăng.

- Nguyên tắc "đánh giá không phải là mục đích tự thân, mà là phương tiện để cải tiến". 
=> Nghiên cứu đề xuất và hình thức hóa một bài toán nghiên cứu mới: Automatic Failure Attribution in MAS

Contributions:
- Benchmark Who&When: log thất bại được chú thích chi tiết về lỗi khi xử lý các tác vụ thực tế
- Một số phương pháp quy kết lỗi tự động trên Who&When: All-at-once, Step-by-step, Binary search

## 1. Problem Formulation

Formal hóa bài toán **Automated Failure Attribution in Multi-Agent Cooperation**, dựa trên turn-based LLM multi-agent protocol (Hong et al., 2023; Li et al., 2023a; Wu et al., 2023).

**Background — mô hình hệ MAS**:
- Hệ _M_ gồm _N_ agent = {1, 2, ..., N}, hoạt động ở các discrete time step, turn-based: đúng 1 agent hành động mỗi step.
- Formal: _S_ = tập state; _A_ = global action set (agent _i_ chỉ dùng subset _Ai ⊆ A_); _ϕ(t)_ = agent nào active tại step _t_; _P(st+1 | st, at, ϕ(t))_ = state-transition probability khi chỉ agent _ϕ(t)_ act tại _t_.
- Trajectory đầy đủ: _τ = (s0, a0, s1, a1, ..., sT)_, _T_ = terminal step.

**Decisive Error và Objective**:
- Mistake ký hiệu tuple _(i, t)_: agent _i_ active tại _t_, action _at_ bị coi là sai. Một trajectory có thể chứa nhiều mistake, nhưng không phải mistake nào cũng dẫn tới failure tổng thể.
- _Z(τ)_ = kết quả trajectory (1 = failure).
- Định nghĩa decisive error qua **counterfactual intervention**: nếu _τ_ failure (_Z(τ)=1_), thay _at_ tại _(i,t)_ bằng action "đúng" _ãt_ (giữ nguyên các step trước _t_, điều chỉnh lại các step sau _t_ cho hợp lý) → được trajectory mới _τ^(i,t)_ qua intervention _I(i,t)_.
- Nếu _Z(τ^(i,t)) = 0_ (thành công) → _(i,t)_ là **decisive error**. Định nghĩa formal qua decisive error indicator _Δi,t(τ)_: _Δi,t(τ)=1 ⇐⇒_ sửa mistake của agent _i_ tại _t_ đổi _Z(τ)_ từ fail sang success.
- Decisive error = cặp _(i*, t*)_ thỏa _Δi*,t*(τ)=1_ → _i*_ = **failure-responsible agent**, _t*_ = **decisive error step**.

**Ràng buộc multi-fault → earliest-error objective**:
- Một trajectory có thể có nhiều decisive error. Nghiên cứu chọn error **sớm nhất theo thời gian** làm nguyên nhân chính (objective chọn cặp _(i*, t*)_ có decisive error indicator cao nhất với time step sớm nhất).
- Bài toán nghiên cứu = tự động xác định cặp _(i*, t*)_ này trong LLM-powered MAS.

## 2. Cách xây dựng datasets

- Bước 1: Question - Groundtruth lấy từ GAIA và AssistantBench
- Bước 2 — Sinh hệ tác tử và thu thập failure log. Hai loại hệ:
    + Algorithm-generated:  CaptainAgent (AG2) tự động dựng một team agent riêng cho từng truy vấn. Thuật toán lặp tối ưu cấu hình; nhóm tác giả chỉ giữ cấu hình cuối cùng kèm lịch sử thực thi. Toàn bộ agent và cả CaptainAgent đều chạy trên GPT-4o (bản 2024-08-01-preview). Giới hạn 10 vòng lặp.
    + Hand-crafted: dùng Magentic-One, một hệ generalist trưởng thành gồm 5 agent chuyên biệt (duyệt web, thao tác file...). Chạy trên toàn bộ AssistantBench validation + 30 instance lấy ngẫu nhiên từ GAIA (phần còn lại của GAIA bị loại vì log quá dài, khó chú thích). Giới hạn 30 vòng.
- Bước 3 — Chú thích lỗi quyết định. Ba chuyên gia AI agent, quy trình 3 vòng:
    + I: Chia đều log cho 3 người, chú thích theo guideline chuẩn: (a) agent gây lỗi, (b) bước lỗi, (c) lý do bằng ngôn ngữ tự nhiên. Tự phân loại chú thích thành chắc chắn / không chắc chắn
    + II: Thảo luận tập thể các chú thích không chắc chắn cho đến khi đồng thuận thực sự — không dùng nguyên tắc đa số thắng
    + III: Cross-validation: mỗi người rà lại chú thích của người khác để kiểm tra tính nhất quán chuẩn; có bất đồng thì thảo luận / chú thích lại

Tổng kết các trường dữ liệu trong datasets (lấy từ hugging face):
- question: câu hỏi gốc
- ground truth: kết quả chính xác cho câu hỏi
- history: trajectory = [(content, name, role)]:
    + content: system prompt + result from last response
    + name: tên agent
    + role: assistant/ user
- mistake agent/ mistake step/ mistake reason: agent sai/ bước sai/ mistake reason

## 3. Điểm cần chú ý
Về dataset:
- Việc đánh giá agent sai không chỉ nằm ở log: ví dụ với agent search cần kiểm tra web xem thật sự có nội dung đó không? 
- Thừa nhận về multi-fault trong trajectory vẫn ép về constraint "lỗi đầu tiên"

Về baselines:
- All-at-once thắng agent-level nhưng thua cả random ở step-level. 
- Step-by-step ngược lại. 
- Binary search luôn nằm giữa ở cả hai metric.
=> Lý do tác giả đưa ra: agent-level cần ngữ cảnh rộng để so sánh hành vi nhiều agent; step-level bị "space-in-the-needle"  — LLM không truy xuất được thông tin cụ thể trong ngữ cảnh dài.

- Hybrid = all-at-once chọn agent trước, rồi step-by-step chỉ quét các bước của agent đó. Step-level tăng từ 7.90 → 12.28 (+55% tương đối) nhưng đắt gấp 8.7 lần so với all-at-once

- Cả 3 phương pháp suy giảm theo độ dài, step-level nhạy hơn agent-level, và step-by-step suy giảm mạnh nhất. Ở Level 5 (93–130 bước), cả ba hội tụ gần 0% (Phân cấp độ dài của họ: L1 = 5–17, L2 = 19–29, L3 = 31–49, L4 = 51–91, L5 = 93–130).
