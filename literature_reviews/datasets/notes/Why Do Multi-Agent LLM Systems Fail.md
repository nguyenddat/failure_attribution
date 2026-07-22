# Why Do Multi-Agent LLM Systems Fail? (MAST / MAST-Data, Berkeley, NeurIPS 2025 D&B track)

## 1. Bối cảnh & động lực nghiên cứu (nhấn mạnh)

Multi-Agent System (MAS) là hệ thống gồm nhiều agent LLM phối hợp với nhau (mỗi agent có prompt riêng, có state riêng, có thể dùng tool) để cùng giải quyết một tác vụ — ví dụ ChatDev có agent CEO, agent CTO, agent Programmer... cùng viết phần mềm. Ý tưởng dùng nhiều agent thay vì một agent duy nhất đang được kỳ vọng lớn: chia nhỏ việc (task decomposition), chạy song song, cô lập ngữ cảnh (context isolation), kết hợp nhiều model chuyên biệt, tranh luận đa góc nhìn.

Nhưng động lực chính khiến nhóm tác giả làm nghiên cứu này là một **nghịch lý quan sát được**: dù MAS được kỳ vọng cao, hiệu năng thực tế của chúng trên benchmark phổ biến lại cải thiện rất ít so với:
- một agent đơn lẻ (single-agent), hoặc
- một baseline đơn giản như best-of-N sampling (chạy nhiều lần, chọn kết quả tốt nhất).

Nhóm tự đo thực nghiệm và phát hiện **tỷ lệ thất bại (failure rate) từ 41% đến 86.7%** trên 7 hệ MAS SOTA (state-of-the-art) khác nhau. Đây là con số rất lớn, cho thấy MAS "hỏng" thường xuyên chứ không phải hiếm gặp.

Vấn đề gốc rễ hơn nữa: **không có sự đồng thuận (no clear consensus)** trong cộng đồng về cách xây MAS sao cho đáng tin cậy (robust, reliable). Người ta biết MAS hay fail, nhưng không ai có một khung thống nhất để mô tả **fail như thế nào, vì đâu**. Từ đó câu hỏi trung tâm của bài báo là: **"Vì sao MAS thất bại?" (Why do MAS fail?)**

Đây là câu hỏi mang tính nền tảng (foundational), không phải một bug cụ thể — mục tiêu là xây một "khoa học thất bại" (science of failure) cho MAS, tương tự cách ngành phần mềm truyền thống có các loại lỗi được phân loại rõ ràng (null pointer, race condition, off-by-one...) để từ đó biết cách phòng tránh và fix hệ thống.

## 2. Khoảng trống trong các nghiên cứu trước (research gap)

Tác giả điểm qua 3 nhóm nghiên cứu liên quan và chỉ ra khoảng trống của từng nhóm:

1. **Các nghiên cứu về khó khăn của agentic system** (Agent Workflow Memory, DSPy, StateFlow, các survey về rủi ro MAS): các công trình này giải quyết từng vấn đề cụ thể (ví dụ trí nhớ dài hạn, cách lập trình luồng agent...) hoặc chỉ đưa ra tổng quan mức cao (high-level overview). **Không có công trình nào đưa ra một phân loại chi tiết, được rút ra từ dữ liệu thực nghiệm (empirically grounded taxonomy)** giải thích cụ thể vì sao MAS thất bại trên nhiều hệ thống khác nhau.

2. **Các benchmark đánh giá agent** (SWE-bench, các benchmark về độ tin cậy/bảo mật...): các benchmark này đo hiệu năng tổng (aggregate performance) hoặc mục tiêu cấp cao (trustworthiness, security) theo hướng **top-down** — tức là chỉ cho biết hệ thống đúng/sai bao nhiêu %, không cho biết **cơ chế thất bại cụ thể bên trong** là gì.

3. **Các dataset/taxonomy liên quan gần nhất**:
   - AgentEval: đưa ra tiêu chí đánh giá đa chiều theo góc nhìn người dùng cuối, không phải taxonomy lỗi.
   - AGDebugger: công cụ tương tác để debug/chỉnh sửa hội thoại agent, mang tính công cụ (tool) hơn là nghiên cứu phân loại lỗi.
   - **Who&When dataset** (Zhang et al. — bài liên quan trực tiếp đến pipeline hiện tại của dự án này): tập trung vào việc **quy trách nhiệm (attribution)** — tức với một trace đã fail, xác định agent nào và bước (step) nào gây ra lỗi. Đây là bài toán "ai gây lỗi, lỗi ở đâu", khác với MAST.

→ **Khoảng trống MAST lấp vào**: chưa có ai xây một **taxonomy (hệ thống phân loại) các kiểu thất bại** được rút ra bài bản, có cơ sở thực nghiệm, áp dụng được rộng rãi cho nhiều loại MAS khác nhau (khác framework, khác model, khác domain tác vụ). Nói cách khác: Who&When trả lời "ai/lúc nào gây lỗi", còn MAST trả lời "lỗi đó thuộc **loại (kiểu)** gì, bản chất là gì".

## 3. Cách họ giải quyết — Phương pháp & điểm cải tiến/mới (nhấn mạnh)

### 3.1 Xây dựng MAST (Multi-Agent System Failure Taxonomy) bằng Grounded Theory

**Grounded Theory (GT)** là phương pháp nghiên cứu định tính: thay vì đặt giả thuyết trước rồi đi kiểm chứng, người nghiên cứu đọc dữ liệu thô, để các khái niệm/pattern **tự nổi lên (emerge organically)** từ dữ liệu.

Quy trình cụ thể:
- Thu thập 150 trace (mỗi trace là log hội thoại đầy đủ giữa các agent, trung bình hơn 15.000 dòng text) từ 5 framework MAS khác nhau (HyperAgent, AppWorld, AG2, ChatDev, MetaGPT), phủ 2 loại tác vụ (lập trình, giải toán) — chọn có chủ đích (theoretical sampling) để đảm bảo bao phủ đa dạng kiến trúc và kiểu tương tác.
- 6 chuyên gia annotate tay, tốn hơn 20 giờ/người cho 150 trace này.
- Áp dụng các kỹ thuật GT: open coding (gắn nhãn mô tả hành vi lỗi quan sát được), constant comparative analysis (so sánh liên tục giữa các trace để tinh chỉnh định nghĩa), memoing (ghi chú lại insight), theorizing (khái quát hóa thành các "failure mode" có định nghĩa rõ).
- Lặp đến khi đạt **theoretical saturation** — tức phân tích thêm dữ liệu không còn phát hiện failure mode mới nữa.

Sau đó, để đảm bảo các định nghĩa **áp dụng nhất quán giữa các annotator khác nhau** (không mơ hồ, không tùy người hiểu một kiểu), họ chạy 3 vòng **Inter-Annotator Agreement (IAA)**: 3 chuyên gia độc lập gắn nhãn cùng một tập trace nhỏ, so sánh kết quả, thảo luận bất đồng, rồi tinh chỉnh lại định nghĩa taxonomy. Đo mức đồng thuận bằng **Cohen's Kappa (κ)** — chỉ số thống kê đo mức đồng ý giữa nhiều người đánh giá, có trừ hao khả năng đồng ý ngẫu nhiên (κ càng gần 1 càng đồng thuận cao). Kết quả cuối: **κ = 0.88** — mức đồng thuận rất mạnh.

**Kết quả: 14 failure mode (kiểu lỗi), gom vào 3 category (nhóm nguyên nhân):**

- **FC1 — System Design Issues (44.2% tổng số lỗi)**: lỗi do chính cách thiết kế hệ thống, đặc tả (specification) mập mờ hoặc kém từ đầu (giai đoạn pre-execution), gồm:
  - 1.1 Disobey Task Specification (11.8%) — agent không làm đúng yêu cầu tác vụ.
  - 1.2 Disobey Role Specification (1.5%) — agent làm sai vai trò được giao (ví dụ agent CPO tự ý kết thúc hội thoại mà không cần agent CEO đồng ý).
  - 1.3 Step Repetition (15.7%) — lặp lại bước đã làm, không tiến triển.
  - 1.4 Loss of Conversation History (2.8%) — mất ngữ cảnh hội thoại trước đó.
  - 1.5 Unaware of Termination Conditions (12.4%) — không nhận biết được khi nào tác vụ đã hoàn thành để dừng.

- **FC2 — Inter-Agent Misalignment (32.3%)**: lỗi do **luồng thông tin quan trọng bị đứt gãy** giữa các agent trong lúc thực thi (execution), gồm:
  - 2.1 Conversation Reset (2.2%) — hội thoại bị reset ngoài ý muốn.
  - 2.2 Fail to Ask for Clarification (6.8%) — agent cứ giả định thay vì hỏi lại khi thông tin chưa rõ.
  - 2.3 Task Derailment (7.4%) — đi lệch hướng khỏi mục tiêu ban đầu.
  - 2.4 Information Withholding (0.8-0.85%) — một agent giữ thông tin quan trọng không chia sẻ cho agent khác cần nó (ví dụ minh họa: agent Phone không nói rõ định dạng username cần là số điện thoại, khiến agent Supervisor cứ login sai lặp lại).
  - 2.5 Ignored Other Agent's Input (1.9%) — bỏ qua input của agent khác.
  - 2.6 Reasoning-Action Mismatch (13.2%) — agent "nghĩ" một đằng (trong phần suy luận/reasoning) nhưng "làm" một nẻo (hành động thực tế không khớp).

- **FC3 — Task Verification (23.5%)**: lỗi do **khâu kiểm tra chất lượng đầu ra không đầy đủ** (giai đoạn post-execution), gồm:
  - 3.1 Premature Termination (6.2%) — dừng sớm khi tác vụ chưa xong.
  - 3.2 No or Incomplete Verification (8.2%) — không kiểm tra, hoặc kiểm tra hời hợt (ví dụ: chỉ check code compile được, có còn "TODO" hay không, chứ không chạy thử logic thật — dẫn ví dụ ChatDev sinh chương trình cờ vua qua được các check bề mặt nhưng chơi sai luật, không dùng được).
  - 3.3 Incorrect Verification (9.1%) — kiểm tra nhưng kết luận sai (verifier tự nó bị lỗi).

**14 mode này còn được gắn nhãn theo 3 giai đoạn hội thoại (conversation stage): Pre-Execution / Execution / Post-Execution** — cho biết lỗi thường "sinh ra" ở giai đoạn nào của quy trình chạy MAS, có mode trải dài qua nhiều giai đoạn.

### 3.2 LLM-as-a-Judge — pipeline annotate tự động (điểm cải tiến về khả năng mở rộng — scalability)

Gắn nhãn tay 1600+ trace là quá tốn công. Họ xây một **LLM annotator**: đưa cho model OpenAI o1 gồm (a) toàn bộ trace, (b) định nghĩa MAST, (c) vài ví dụ mẫu (few-shot) lấy từ dữ liệu người annotate — để model tự phân loại failure mode cho trace mới.

So sánh hiệu năng LLM annotator với nhãn người thật:
- o1 zero-shot: accuracy 0.89, κ = 0.58 (đồng thuận trung bình).
- o1 few-shot (bản dùng chính thức): **accuracy 0.94, κ = 0.77** (đồng thuận mạnh) — đây là cấu hình được chọn để scale annotate toàn bộ dataset.

Họ còn kiểm tra khả năng tổng quát hóa: áp dụng MAST + LLM annotator cho 2 MAS mới (OpenManus, Magentic-One) và 2 benchmark mới (MMLU, GAIA — GAIA đại diện domain "general agent task" hoàn toàn mới) chưa từng dùng lúc xây taxonomy ban đầu → vẫn đạt κ = 0.79 khi so với người → chứng minh **MAST không bị "quá khớp" (overfit) với 5 framework ban đầu**, mà khái quát hóa tốt sang hệ thống/domain khác.

### 3.3 MAST-Data — bộ dữ liệu công bố

Kết quả cuối: **MAST-Data** gồm 1642 trace đã gắn nhãn, từ 7 framework MAS (ChatDev, MetaGPT, HyperAgent, AppWorld, AG2/MathChat, Magentic-One, OpenManus), nhiều họ model (GPT-4o, GPT-4o-mini, Claude-3.7-Sonnet, Qwen2.5-Coder-32B, CodeLlama-7B), phủ 3 domain tác vụ (coding, math, general agent). Kèm theo **MAST-Data-human**: tập nhỏ hơn có annotate tay bởi nhiều chuyên gia (dùng cho nghiên cứu IAA).

Cả taxonomy (MAST), dataset (MAST-Data) và LLM annotator đều được **open-source** — đây bản thân cũng là một đóng góp/điểm mới: trước đây cộng đồng không có tài nguyên công khai nào ở quy mô này để nghiên cứu failure MAS.

### Tổng kết các điểm mới/cải tiến chính
1. **Taxonomy đầu tiên** (MAST) về failure MAS được xây bằng phương pháp định tính bài bản (Grounded Theory) và kiểm định thống kê nghiêm ngặt (IAA, Cohen's Kappa), thay vì liệt kê chủ quan.
2. **Dataset lớn nhất, đa dạng nhất** (MAST-Data — 1642 trace) về loại chủ đề này tính đến thời điểm bài báo, phủ nhiều framework/model/domain.
3. **Pipeline LLM-as-a-Judge** để annotate failure mode tự động, đạt độ chính xác cao (94%), giải quyết bài toán tốn kém khi annotate tay ở quy mô lớn.
4. **Case study can thiệp thực tế (intervention study)**: không chỉ dừng ở phân loại lý thuyết, họ còn **chứng minh MAST hữu dụng để cải thiện hệ thống thật** — đây là điểm mới quan trọng vì biến taxonomy từ công cụ mô tả (descriptive) thành công cụ hành động (actionable).

## 4. Kết quả (nhấn mạnh)

### 4.1 Kết quả về mặt phân loại/đo lường
- Tỷ lệ 3 category trên tổng 210 trace phân tích chi tiết: **System Design Issues ~41.8-44.2%, Inter-Agent Misalignment ~32.3-36.9%, Task Verification ~21.3-23.5%** (số liệu dao động nhẹ tùy tập con dùng để tính, nhưng thứ tự ưu tiên luôn là: thiết kế hệ thống > phối hợp giữa agent > kiểm tra chất lượng).
- Failure mode phổ biến nhất theo profile riêng từng hệ thống — **không có một "kiểu lỗi chung" cho mọi MAS**, mỗi framework có "điểm yếu đặc trưng" phản ánh đúng kiến trúc của nó:
  - AppWorld: hay bị Premature Termination (3.1) — do kiến trúc dạng "star topology" (một agent trung tâm điều phối) và không có luồng làm việc (workflow) định sẵn rõ ràng, nên điều kiện dừng không rõ ràng.
  - OpenManus: hay bị Step Repetition (1.3).
  - HyperAgent: yếu ở cả Step Repetition (1.3) và Incorrect Verification (3.3).
- So sánh model cùng framework MetaGPT: **GPT-4o ít lỗi FC1 hơn Claude-3.7-Sonnet tới 39%** dù nhìn chung GPT-4o "performance" tốt hơn Claude trong bối cảnh này — cho thấy loại lỗi cũng phụ thuộc model nền, không chỉ kiến trúc.
- So sánh 2 framework cùng benchmark (MetaGPT vs ChatDev trên ProgramDev): MetaGPT ít lỗi FC1+FC2 hơn ChatDev **60-68%**, nhưng lại có **gấp 1.56 lần** lỗi FC3 (task verification) — minh họa rõ **trade-off giữa các nhóm lỗi tùy theo lựa chọn kiến trúc**, không có thiết kế nào miễn nhiễm hoàn toàn.

### 4.2 Kết quả từ case study can thiệp (chứng minh tính ứng dụng — quan trọng nhất để trả lời "họ giải quyết có hiệu quả không")
- Sửa **role specification** của ChatDev (đảm bảo agent CEO có quyền quyết định cuối cùng, thay vì agent CPO tự ý kết thúc hội thoại) → **tăng +9.4% success rate**, cùng model (GPT-4o), cùng prompt người dùng.
- Thêm một bước **verification cấp cao (high-level task objective verification)** vào ChatDev (không chỉ check code compile mà check có đáp ứng đúng mục tiêu tác vụ) → **tăng +15.6% success rate** trên benchmark ProgramDev.

→ Đây là bằng chứng thực nghiệm cho luận điểm cốt lõi của bài báo: **thất bại của MAS phần lớn đến từ vấn đề thiết kế hệ thống/tổ chức (organizational design), không chỉ từ giới hạn của bản thân model LLM.** Họ so sánh với lý thuyết tổ chức con người (organization theory): ngay cả một tổ chức gồm toàn cá nhân giỏi vẫn có thể thất bại thảm hại nếu cấu trúc tổ chức (ai báo cáo ai, ai quyết định cuối) bị sai — MAS cũng vậy.

### 4.3 Insight tổng hợp được nêu bật trong bài
- **Insight 1** (ứng với FC1): MAS thất bại không chỉ vì model nền yếu — một MAS thiết kế tốt vẫn có thể tăng hiệu năng đáng kể dùng chung một model.
- **Insight 2** (ứng với FC2): Các giải pháp chỉ tập trung vào ngữ cảnh (context) hay chuẩn hóa giao thức giao tiếp (như Model Context Protocol, Agent-to-Agent protocol) là **chưa đủ** cho lỗi loại này — vì lỗi FC2 vẫn xảy ra ngay cả khi agent giao tiếp bằng ngôn ngữ tự nhiên trong cùng framework. Cần agent có khả năng **"social reasoning"/lý thuyết tâm trí (theory of mind)** sâu hơn — tức khả năng mô hình hóa được agent khác đang cần thông tin gì, đang hiểu gì — đây là năng lực chưa được train sẵn trong LLM nền hiện tại, nên cần cả cải tiến kiến trúc MAS lẫn cải tiến ở tầng model/training.
- **Insight 3** (ứng với FC3): Cần **verification nhiều tầng (multi-level verification)** — không thể chỉ dựa vào một bước kiểm tra cuối cùng, hời hợt. Cần học từ cách kỹ sư phần mềm truyền thống test code trước khi commit: dùng external knowledge, thu thập kết quả test xuyên suốt quá trình sinh (không chỉ ở cuối), kiểm cả correctness mức thấp (code chạy được) lẫn mục tiêu mức cao (code có đúng ý đồ bài toán không).

## 5. Limitations & Future Work (bài báo tự nêu)

- **MAST không tự nhận là đầy đủ (not exhaustive)**: 14 failure mode được rút ra từ phân tích 150 trace ban đầu (5 framework, 2 domain) — có thể còn kiểu lỗi khác chưa xuất hiện trong tập này, đặc biệt với framework/domain rất khác biệt (ví dụ agent vật lý/robot, agent multi-modal...).
- **FC2 (Inter-Agent Misalignment) là nhóm khó giải nhất theo nhận định của tác giả**: giải pháp hiện tại (chuẩn hóa giao thức giao tiếp, quản lý context tốt hơn) chỉ là biện pháp bề mặt; gốc rễ là agent thiếu khả năng suy luận xã hội/theory of mind — đây là hướng nghiên cứu mở, đòi hỏi có thể phải **train lại model** theo hướng "giao tiếp có chủ đích" (communicative intelligence), chứ không chỉ chỉnh sửa prompt hay kiến trúc.
- **FC3 (Task Verification)**: cần xây dựng verifier tốt hơn hẳn hiện tại — hướng future work được gợi ý là multi-level verification, học theo quy trình test phần mềm chuyên nghiệp — nhưng bài báo chưa tự làm việc này, chỉ dừng ở một case study đơn giản (thêm 1 bước check cấp cao).
- **Case study can thiệp chỉ là "bước đầu" (first step interventions)**: dù có tăng success rate (+9.4%, +15.6%), tác giả thừa nhận **không phải mọi failure mode đều được giải quyết**, và **tỷ lệ hoàn thành tác vụ tổng thể (task completion rate) vẫn còn thấp** sau khi can thiệp — nghĩa là các fix này chỉ vá được một phần vấn đề, chưa phải giải pháp triệt để.
- **Hướng đi tương lai được đề xuất rõ trong bài**: đạt độ tin cậy cao cho MAS cần **thay đổi tổ hợp (combinatorial changes)** ở nhiều tầng cùng lúc — từ cách tổ chức hệ thống agent (agent system organization) đến cải tiến ở tầng model — chứ không có một fix đơn lẻ nào đủ. Tác giả định vị MAST như một **framework/công cụ chẩn đoán nền tảng** để cộng đồng dùng tiếp, chứ bản thân bài báo không giải quyết triệt để vấn đề "how to fix" — chỉ giải quyết triệt để vấn đề "how to diagnose and categorize".
- Ý nghĩa cho hướng nghiên cứu của dự án hiện tại (single-fault failure attribution, method CHIEF...): MAST bổ sung góc nhìn **"loại lỗi là gì"** bên cạnh góc nhìn **"lỗi ở agent nào, step nào"** (vốn là trọng tâm của Who&When/CHIEF) — hai hướng này bổ trợ nhau: biết được kiểu lỗi (MAST) có thể giúp thiết kế heuristic/oracle tốt hơn khi backtracking tìm root cause (agent + step) trong pipeline CHIEF.
