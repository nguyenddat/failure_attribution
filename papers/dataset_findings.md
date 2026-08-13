# Tổng hợp Findings liên quan đến Segmentation trong Failure Attribution

Note này tổng hợp xuyên suốt 7 paper (MAST, Who&When, TRAIL, Aegis, TraceElephant,
AgentErrorBench, TELBENCH/DRIFT) theo đúng hướng nghiên cứu segmentation: chia
trajectory dài (>15 step) thành các segment nhỏ để phục vụ decisive error
localization và (multi) fault detection.

---

## 1. Phân bố lỗi phụ thuộc vào framework/kiến trúc

Finding lặp lại độc lập ở 3 paper — đủ mạnh để coi là đồng thuận.

**MAST** (bằng chứng trực tiếp nhất — cùng backbone GPT-4o, khác framework):
- MetaGPT vs ChatDev trên ProgramDev: MetaGPT ít lỗi FC1/FC2 hơn 60–68% nhưng
  nhiều lỗi FC3 gấp 1.56 lần.
- Lý do: MetaGPT dựa vào SoP (Standard Operating Procedures) → tuân thủ đặc tả
  tốt; ChatDev có phase testing/reviewing riêng → verification tốt hơn.
- ⇒ Error distribution là hệ quả trực tiếp của **design choice** (có/không có
  SoP, có/không có verification phase), không phải thuộc tính chung của backbone
  model.

**TraceElephant** (mở rộng sang vị trí lỗi theo giai đoạn thực thi, phân theo
*kiểu* kiến trúc):
- Kiến trúc cố định thủ công (Magentic-One, SWE-Agent): lỗi tập trung ở giai
  đoạn đầu (planning/routing của orchestrator).
- Kiến trúc dynamic-team tự động (Captain-Agent): lỗi phân tán đều hơn qua các
  bước (team dựng lại theo từng task → nhiều điểm có thể lỗi).
- Magentic-One riêng: accuracy attribution giai đoạn đầu chỉ 8% (rất khó), cuối
  52% (dễ hơn nhiều) — do chu kỳ explore/re-plan kéo dài che lỗi sớm.
- ⇒ Tác giả đề xuất tường minh **"architecture-aware attribution"**: kỹ thuật
  quy lỗi phải biết trước kiểu kiến trúc (centralized vs dynamic, tool-heavy vs
  planning-heavy) để tập trung tìm đúng chỗ dễ lỗi.

**AgentErrorBench** (biến độc lập là task/environment, không phải kiến trúc,
nhưng cùng logic):
- GAIA: nghiêng hẳn về planning (26/50)
- WebShop: plan + memory (19+15/50)
- ALFWorld: trải đều hơn (plan 33, reflection 22, memory 18, action 14, system 13)
- ⇒ Cấu trúc bài toán quyết định module nào dễ hỏng.

**Tổng hợp**: cả 3 paper ủng hộ meta-finding "không có kiến trúc/model nào thắng
toàn diện, chỉ có trade-off theo thiết kế" (câu nói thẳng trong MAST). Đây là cơ
sở đặt câu hỏi: segmentation có nên "aware" kiến trúc khi chọn điểm ngắt, thay
vì dùng một tiêu chí ngắt đoạn chung cho mọi framework?

---

## 2. Độ dài trace tỉ lệ nghịch với độ chính xác → cơ sở cho segmentation

| Paper | Bằng chứng |
|---|---|
| TRAIL | Location Acc: Pearson r = −0.379, Spearman ρ = −0.508 với độ dài input; localization suy giảm nhanh hơn categorization |
| Who&When | 5 mức độ dài (L1=5–17 → L5=93–130 bước), **cả 3 baseline** (all-at-once/step-by-step/binary search) đều suy giảm, hội tụ gần 0% ở L5 |
| TraceElephant | Trace dài hơn hẳn Who&When (20.5–29.3 lời gọi LLM/trace vs 9.6–28.8) khiến step-by-step vượt context window → All-at-Once thắng, ngược xu hướng của Who&When |

- Ngưỡng L1 của Who&When (5–17 bước) gần trùng mốc ">15" đang dùng cho
  segmentation — có thể trích trực tiếp để justify threshold.
- TELBENCH là ứng dụng thực tế nhất: chủ động segment vì lý do tương tự — trace
  dài, không đồng nhất, prompt LLM trực tiếp trên toàn trace không ổn định
  (nhầm triệu chứng với root cause, bỏ sót commitment sớm). Kết quả: trung bình
  11.95 semantic span / trajectory.

---

## 3. Full trace > Only output

Ablation trực tiếp và rõ nhất — **TraceElephant**:
- Chỉ dùng output (giống setup Who&When): agent-level rớt từ 62%→51% (−22%
  tương đối), step-level rớt từ 28%→16% (−76% tương đối).
- Step-level nhạy cảm với thông tin thiếu hơn agent-level nhiều — vì xác định
  đúng bước lỗi cần hiểu ngữ cảnh chi tiết hơn xác định đúng agent.
- Chính TraceElephant dùng finding này để phê phán Who&When: "chỉ cung cấp
  trace partially observable... ít nhất 21% case không quy trách nhiệm đáng tin
  cậy được nếu chỉ có log đầu ra" (tự phân tích trên 184 case của Who&When).

TRAIL gián tiếp ủng hộ cùng hướng: input đầy đủ (structured OpenTelemetry span,
gồm cả `input.value`/`output.value`/tool logs) là lý do RQ1 khẳng định đây là
bài toán long-context chứ không chỉ reasoning thuần túy — thông tin càng đầy đủ
càng cần cơ chế xử lý tốt, không phải cứ full trace là lợi miễn phí (đánh đổi
với độ dài).

---

## 4. Các finding đồng thuận khác

**a) Localization (step-level) luôn khó hơn classification (category/agent-level)**
— xuất hiện ở mọi paper đo cả hai:
- TRAIL: Joint Acc (11%) << Cat. F1 riêng lẻ
- Who&When: All-at-once thắng agent-level nhưng thua cả random ở step-level;
  ngược lại với step-by-step
- TraceElephant: agent-level 65.9% vs step-level chỉ 30.3% (full trace)
- TELBENCH: phân biệt rõ "phát hiện vùng lỗi" (dễ hơn) và "định vị chính xác
  điểm lỗi đầu tiên" (khó, dù DRIFT đã cải thiện mạnh)

**b) Lỗi lan truyền — sửa đúng 1 lỗi gốc thường đủ lật cả trajectory:**
- AgentErrorBench: "sửa MỘT lỗi gốc thường đủ để lật một trajectory thất bại
  thành thành công"
- Aegis case study: agent cung cấp giá vé sai → toàn bộ tính toán sau đó sai
  theo; agent quy đổi đơn vị sai (1 dặm = 20 thay vì 16 block) → chuỗi sai lan
- TELBENCH: claim thiếu căn cứ được các span sau "tái sử dụng như sự thật đã
  xác lập"

**c) Reasoning/effort của model giúp attribution rõ rệt, nhưng scale thuần túy
thì không đảm bảo:**
- TRAIL (RQ2): mọi reasoning model (trừ O1) vượt non-reasoning; ablation cùng
  model chỉ đổi effort (high/med/low) cũng cho xu hướng đơn điệu
- TELBENCH: "scale mô hình lớn hơn KHÔNG đảm bảo chẩn đoán tốt hơn"
- ⇒ Hai finding **bổ sung nhau chứ không mâu thuẫn**: TRAIL nói về reasoning
  effort trong cùng 1 model, TELBENCH nói về việc thay model lớn hơn nói chung
  — khác biến số.

**d) Annotation thủ công bởi expert vẫn là nền tảng chất lượng, kể cả khi có
scale bằng LLM-as-judge** — mọi dataset (MAST, TRAIL, Who&When, Aegis,
TraceElephant, AgentErrorBench, TELBENCH) đều có vòng IAA/pilot annotation
nhiều round với con người trước khi mở rộng quy mô (bằng LLM-judge hoặc
injection tự động). Không paper nào bỏ qua bước con người hoàn toàn.

---

## 5. Các finding CONFLICT — cần chỉ rõ

### a) Xung đột về định nghĩa "decisive error" (xung đột nền tảng nhất)

- **Who&When**: nguyên tắc *earliest visible mistake* — lỗi sớm nhất theo thời
  gian mà counterfactual intervention sửa được sẽ đổi outcome.
- **TraceElephant**: phản bác trực tiếp, dùng nguyên tắc *inevitable +
  recoverability-aware*: nếu lỗi ở step t vẫn còn cơ hội phục hồi (có
  verifier/orchestrator ở step sau chịu trách nhiệm kiểm tra) thì decisive
  error KHÔNG gán cho step t mà gán cho step bỏ lỡ cơ hội phục hồi đó.
- Ví dụ minh họa: agent hallucinate ở step t, verifier ở t+2 không phát hiện →
  TraceElephant gán lỗi quyết định cho verifier tại t+2; Who&When sẽ gán cho
  agent hallucinate ban đầu tại t.
- ⇒ Đây là mâu thuẫn **phương pháp luận thật sự**, không do khác domain. Khi
  làm segmentation + decisive error localization, cần chọn đứng về bên nào
  (hoặc định nghĩa lại), vì hai định nghĩa cho ra ground-truth khác nhau trên
  cùng một trace.

### b) Xung đột (bề mặt) về All-at-once vs Step-by-step

- Who&When: step-by-step thắng ở step-level, all-at-once thắng ở agent-level
  (trace ngắn hơn, L1–L5 tối đa 130 bước)
- TraceElephant: All-at-Once thắng toàn diện, vì trace dài hơn (20.5–29.3 lời
  gọi LLM/trace) khiến step-by-step incremental vượt context window
- ⇒ Không hẳn trái ngược về bản chất — **cùng một trục (độ dài trace) tạo ra
  kết luận ngược nhau**. Đây chính là luận cứ mạnh nhất ủng hộ hướng nghiên cứu
  segmentation: hai baseline "thắng" tùy ngưỡng độ dài, nên cần cơ chế trung
  gian (chia đoạn) thay vì chọn cứng một trong hai chiến lược.

### c) Căng thẳng giữa "single root cause" và "multi-fault"

- AgentErrorBench: nhấn mạnh sửa 1 lỗi gốc là đủ, phản bác cách "duyệt lại toàn
  bộ và sửa mọi thứ trông sai" (self-refine kiểu duyệt toàn bộ).
- Who&When: tự thừa nhận multi-fault tồn tại thật trong trajectory, nhưng *ép*
  constraint chọn lỗi sớm nhất để đơn giản hóa bài toán — không phủ nhận
  multi-fault, chỉ né nó.
- TELBENCH/DRIFT: ngược lại hẳn — track nhiều claim song song qua Claim Ledger,
  cho phép nhiều span "lỗi" cùng tồn tại (span nào commit/reuse/amplify claim
  thiếu căn cứ).
- ⇒ Không hẳn mâu thuẫn dữ liệu, mà là **mâu thuẫn về giả định thiết kế bài
  toán**: một bên giả định lỗi có tính đơn nguyên nhân (đáng tối ưu tìm 1 root
  cause), một bên giả định lỗi có thể đa nguồn/lan tỏa qua nhiều claim. Vì
  hướng nghiên cứu segmentation + "(multi) fault detection" đụng trực tiếp vào
  đây — cần định vị rõ lập trường, vì tài liệu hiện có không thống nhất.

---

## Bảng tóm tắt nhanh

| # | Finding | Đồng thuận / Conflict | Paper liên quan |
|---|---|---|---|
| 1 | Phân bố lỗi phụ thuộc framework/kiến trúc | Đồng thuận | MAST, TraceElephant, AgentErrorBench |
| 2 | Độ dài trace tỉ lệ nghịch độ chính xác | Đồng thuận | TRAIL, Who&When, TraceElephant, TELBENCH |
| 3 | Full trace > only output | Đồng thuận | TraceElephant, TRAIL |
| 4a | Step-level khó hơn agent/category-level | Đồng thuận | TRAIL, Who&When, TraceElephant, TELBENCH |
| 4b | Lỗi lan truyền từ 1 root cause | Đồng thuận | AgentErrorBench, Aegis, TELBENCH |
| 4c | Reasoning effort giúp ích; scale thuần túy thì không chắc | Bổ sung nhau (không mâu thuẫn) | TRAIL, TELBENCH |
| 4d | Expert annotation vẫn là nền tảng | Đồng thuận | Tất cả 7 paper |
| 5a | Định nghĩa "decisive error" | **Conflict thật sự** | Who&When vs TraceElephant |
| 5b | All-at-once vs Step-by-step tốt hơn | Conflict bề mặt (do độ dài trace khác nhau) | Who&When vs TraceElephant |
| 5c | Single root cause vs multi-fault | **Conflict về giả định thiết kế** | AgentErrorBench/Who&When vs TELBENCH/DRIFT |