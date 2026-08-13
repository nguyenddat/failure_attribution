# Framework-aware / environment-aware failure pattern

Nguồn: paper MAST §5.1, §5.3, §4 FC3, Figure 4; paper "Seeing the Whole Elephant"
(TraceElephant) §4.2.2, §4.2.4; paper "Where LLM Agents Fail..." (AgentErrorBench) §5.2,
Appendix A.4 (Figure 11/12/13); paper "Where Do Deep-Research Agents Go Wrong" (DRIFT/TELBENCH)
§3.3; paper AEGIS §7.

Ghi lại pattern lỗi gắn với đặc điểm kiến trúc/framework cụ thể, paper MAST tự nêu (không phải
số tự tính lại) — bổ sung cho `finding_1_framework_architecture.md`.

## MAST

- **AppWorld — star topology, không có workflow cố định** => hay premature termination
  (FM-3.1), do topology này thiếu điều kiện dừng rõ ràng.
- **OpenManus — thiên hướng lặp bước** => step repetition (FM-1.3) là failure mode chủ đạo.
- **HyperAgent — thiếu cơ chế verification tốt** => dominant ở cả step repetition (FM-1.3) và
  incorrect verification (FM-3.3), paper gợi ý nên fix 2 mode này trước.
- **MetaGPT — dùng SoP (Standard Operating Procedures)** => tuân thủ đặc tả tốt, ít lỗi FC1
  (System Design) + FC2 (Inter-Agent) hơn ChatDev 60–68%, nhưng nhiều lỗi FC3 (Verification)
  gấp 1.56 lần ChatDev — SoP giúp tuân thủ nhưng không thay được bước kiểm tra output.
- **ChatDev — có phase testing/reviewing riêng** => verification tốt hơn MetaGPT (FC3 thấp
  hơn), nhưng đổi lại nhiều lỗi FC1/FC2 hơn (thiếu ràng buộc kiểu SoP).
- **MetaGPT & ChatDev — đều có explicit verifier** => tổng số lỗi (Figure 4) thấp hơn hẳn các
  framework không có verifier rõ ràng — nhưng verifier không phải "silver bullet": ChatDev sinh
  chess program pass compile check (superficial) vẫn fail luật chơi thật (FM-3.2, no/incomplete
  verification).
- **Cùng framework MetaGPT, đổi backbone GPT-4o → Claude 3.7 Sonnet** => Claude nhiều lỗi FC1
  hơn GPT-4o 39%. Backbone model cũng ảnh hưởng phân bố lỗi trong cùng 1 kiến trúc, không chỉ
  framework quyết định.

## TraceElephant ("Seeing the Whole Elephant", §4.2.2, §4.2.4)

- **CaptainAgent — hệ tự động dựng team on-the-fly cho từng task** => lỗi **phân tán đều** theo
  timeline (Figure 6), vì có thể phát sinh ở nhiều điểm: agent selection, iterative planning,
  inter-agent coordination, tool-calling — không có 1 giai đoạn cố định gây lỗi.
- **Magentic-One & SWE-Agent — hệ dựng thủ công, orchestrator trung tâm cố định** => lỗi **tập
  trung ở early step**, gắn chặt với quyết định task planning/routing ban đầu của orchestrator.
- **Agent tương tác môi trường ngoài (web browsing ở CaptainAgent/Magentic-One, code editing ở
  SWE-Agent)** => chiếm >50% tổng lỗi toàn benchmark — do phụ thuộc hệ thống ngoài noisy (API,
  website, filesystem): malformed request, parse lỗi, output bất ngờ.
- **Orchestrator/planner agent** => chiếm 18–29% lỗi còn lại — do task decomposition sai, chọn
  agent kém tối ưu, coordination logic lỗi; lỗi loại này không lộ ra ngay mà propagate/amplify
  dần trong quá trình chạy.
- **Magentic-One — accuracy attribution lệch theo giai đoạn**: early phase accuracy rất thấp
  (8%), giữa vừa (19%), cuối cao (52%) — do hệ có chu kỳ exploratory/re-planning kéo dài nên lỗi
  early-phase (decompose/assumption sai) không lộ ngay, chỉ rõ khi execution sau đó fail.
- **Architecture-Aware Attribution (§4.2.4)** — kết luận thẳng của paper: loại agent gây lỗi
  + vị trí bước lỗi lệch đáng kể theo kiến trúc MAS (centralized vs dynamic team formation,
  tool-heavy vs planning-heavy) => cần phương pháp attribution biết trước kiến trúc hệ thống,
  không dùng chung 1 pipeline cho mọi framework.

## AgentErrorBench — environment-aware, không phải framework-aware (Appendix A.4)

Khác 2 mục trên: kiến trúc agent (ReAct 4-module: memory/reflection/plan/action) **giữ cố
định** qua cả 3 benchmark, chỉ đổi environment/domain — nên đây là bằng chứng "domain quyết
định phân bố lỗi" tách biệt khỏi biến số framework.

- **ALFWorld (Figure 11, n≈100)** — embodied control, trải khá đều 5 module: Plan 33
  (Inefficient Plan 16, Impossible Action 12), Reflection 22, Memory 18, Action 14, System 13
  (Step Limit 10). Action/System vẫn chiếm tỷ trọng đáng kể — hợp lý vì domain đòi thao tác vật
  lý cụ thể (mở tủ, di chuyển vật), nhiều cơ hội để action/step-limit lỗi hơn 2 domain kia.
- **WebShop (Figure 12, n≈49)** — nghiêng hẳn Memory + Plan (15 + 19 = 69% tổng lỗi), Action
  gần như biến mất (2/49). Domain e-commerce search/browse ít thao tác "hành động" phức tạp,
  lỗi chủ yếu do nhớ sai thông tin sản phẩm (Hallucination 8, Over Simplification 7) và lập kế
  hoạch tìm kiếm kém (Inefficient Plan 14).
- **GAIA (Figure 13, n=50)** — Plan áp đảo tuyệt đối (26/50 = 52%), riêng Inefficient Plan
  chiếm 18/26 (69% trong nhóm Plan). Domain open-domain reasoning + tool-use nên lỗi chủ yếu ở
  bước lập chiến lược tra cứu/dùng tool, không phải thiếu memory (chỉ 5/50) hay action (6/50).
- **§5.2 Error Propagation** (không tách theo domain, áp dụng chung cả 3): lỗi Memory/Reflection
  hay phát sinh ở step giữa trajectory (5–15) và cascade nặng nhất — agent nhớ sai/đánh giá sai
  tiến độ sớm thì planning sau đó bị bóp méo hệ thống. Lỗi Action lộ rõ và đôi khi phục hồi
  được; lỗi System (tool crash, step-limit) thường kết thúc trajectory ngay lập tức, không kịp
  cascade.
- **Kết luận riêng của bộ này**: cùng 1 kiến trúc agent, đổi environment vẫn đủ để đảo thứ tự
  module hay lỗi nhiều nhất (WebShop/GAIA: Plan+Memory áp đảo, Action gần như 0; ALFWorld: trải
  đều, Action/System vẫn đáng kể) — bổ sung cho kết luận MAST/TraceElephant rằng biến số
  framework quyết định phân bố: ở đây framework cố định, biến số là **task/environment**.

## DRIFT / TELBENCH ("Where Do Deep-Research Agents Go Wrong", §3.3, Appendix C.1)

2 framework deep-research (MiroFlow, OAgent) chạy cùng 3 backbone (GPT/Claude/Gemini) cùng 3
benchmark (GAIA/XBench/BrowseComp) — tách được ảnh hưởng framework khỏi model/benchmark rõ hơn
MAST/AEGIS (không confound cả 3 trục cùng lúc).

- **OAgent** => "evidence-error fingerprint" mạnh hơn — first-error hay là loại lỗi thiếu/hiểu
  sai bằng chứng (evidence).
- **MiroFlow** => nhiều first-error thuộc loại **constraint** và **search-related** hơn OAgent.
- Trục benchmark (không phải framework) cũng lệch riêng: GAIA nghiêng hẳn về **processing
  error** (lỗi sau khi đã thu thập xong thông tin) — khác OAgent/MiroFlow (lệch theo *loại*
  first-error) và khác model family (GPT thiên evidence-heavy, Gemini thiên constraint-heavy,
  Claude cân bằng) — 3 trục framework/benchmark/model lệch theo 3 chiều **độc lập nhau**, không
  trùng pattern.
- **Effort profile (Appendix C.1, Figure 10)**: MiroFlow sinh trajectory dài hơn, nhiều span
  trung gian hơn (đặc biệt GPT trên BrowseComp) => decomposition/search mở rộng hơn. OAgent giữ
  trajectory ngắn hơn nhưng tool-call vẫn có thể cao ở vài cặp model-benchmark => ít bước suy
  luận không đồng nghĩa ít hành động ra ngoài.

## AEGIS (§7 Results — "Task and MAS Influence")

Khác các mục trên: nói về **độ khó attribution** lệch theo kiến trúc, không phải phân bố loại
lỗi lệch theo kiến trúc — cần đọc tách biệt, không gộp chung kết luận.

- **Debate, MacNet** (topology đơn giản/cấu trúc rõ) => model attribute lỗi tốt hơn hẳn.
- **Dylan, AgentVerse** (topology phức tạp) => attribution khó hơn nhiều, fine-tune trên Aegis
  cho gain lớn nhất đúng ở 2 framework này (chỗ model gốc yếu nhất).
- **Magentic-One, SmolAgents** (framework nhỏ/ít đại diện trong tập train) => xu hướng ngược:
  Aegis-SFT làm **giảm** performance so với base model (nghi over-specialization vào framework
  phổ biến trong train set), còn Aegis-GRPO vẫn giữ ổn định/robust hơn trên các framework
  underrepresented này.

## Tổng hợp chéo 5 nghiên cứu

**Xu hướng lặp lại:**
- Orchestrator trung tâm cố định => lỗi dồn sớm (planning/routing). TraceElephant
  (Magentic-One/SWE-Agent), AgentErrorBench (GAIA — Plan 52%), MAST (AppWorld thiếu workflow
  rõ => premature termination).
- Team hình thành động (CaptainAgent-family) => lỗi phân tán khắp timeline, coordination lỗi
  nhiều. TraceElephant (CaptainAgent dispersed), MAST (AG2/CaptainAgent avg fault/trace +
  Inter-Agent cao nhất — [[finding_1_framework_architecture]]).
- Có phase verification/test riêng => giảm lỗi loại khác, nhưng verification tự nó vẫn luôn
  rủi ro cao nhất. MAST (ChatDev test phase giảm FC1/2, vẫn dính superficial-check FM-3.2),
  DRIFT (decision-making/finalization normalized error rate cao nhất 60.5%/51.8%, mọi
  framework).
- Không kiến trúc nào thắng toàn diện, luôn trade-off. MAST (MetaGPT ↔ ChatDev đổi FC1/2 lấy
  FC3), TraceElephant (Architecture-Aware Attribution — cần method riêng theo kiến trúc), AEGIS
  (topology đơn giản dễ attribute, phức tạp khó).
- Backbone model góp phần độc lập với framework. MAST (GPT-4o ↔ Claude cùng MetaGPT lệch FC1
  39%), DRIFT (GPT/Gemini/Claude lệch loại first-error khác trục framework).

**Pattern chung theo kiểu kiến trúc** (không phải tên framework riêng lẻ):
- Centralized/manual-designed (Magentic-One, SWE-Agent, MetaGPT-SoP) => lỗi quy về
  early-stage decision, dễ trace nguồn khi đã fail.
- Dynamic/auto-formed team (CaptainAgent, AG2, Dylan, AgentVerse) => lỗi phối hợp nhiều hơn,
  phân tán hơn, khó attribute hơn.
  **3 nghiên cứu cùng xác nhận pattern này**: TraceElephant (CaptainAgent — lỗi dispersed khắp
  timeline), MAST (AG2/CaptainAgent — avg fault/trace + Inter-Agent cao nhất, xem
  [[finding_1_framework_architecture]]), AEGIS (Dylan/AgentVerse — topology phức tạp nên
  attribution khó nhất trong 6 framework).
- Có explicit verifier/test phase (ChatDev, MetaGPT) => tổng lỗi thấp hơn hệ không có, nhưng
  verification tự nó luôn là nhóm lỗi lớn bất kể đâu.

## Kết luận: kiểm chứng thực nghiệm (Experiment 0–2)

Toàn bộ nội dung phía trên là literature review — số paper tự báo cáo, chưa verify trên
data thật. [`experiments/0.framework_topology_taxonomy`](../../../experiments/0.framework_topology_taxonomy/finding_notes.md)–[`2.error_step_position_by_framework`](../../../experiments/2.error_step_position_by_framework/finding_notes.md)
verify trực tiếp 2 giả thuyết rút ra từ đó trên dữ liệu gốc.

**Experiment 0 (nền)** — chuẩn hoá 1 bảng topology chung (6 bucket: Hierarchical/Pipeline/
Centralized/Decentralized/Single-agent/Variable) cho mọi framework xuất hiện ở Exp 1–2, tránh
mỗi experiment tự đoán topology riêng (cả 2 exp sau đều note từng tự đoán sai rồi phải sửa lại
theo bảng này).

**Experiment 1 — assumption "framework/kiến trúc quyết định *loại* lỗi": không trụ được.**
Test trên MAST, TraceElephant, AEGIS (3 dataset có field framework + nhãn lỗi cấu trúc):
- MAST — **không verify được**: `mast_annotation` là hàm của `trace_index`, không phải nội
  dung trace (bug ở nguồn HuggingFace `mcemri/MAST-Data`). Claim MetaGPT/ChatDev tradeoff
  FC1–FC3 ở mục MAST phía trên không thể xác nhận đúng hay sai.
- TraceElephant — tách 2 claim. Role gây lỗi khác theo kiến trúc: tái hiện được, có ý nghĩa
  (chi-square p=1.05e-7, Cramér's V=0.294). Vị trí lỗi theo timeline (mục Architecture-Aware
  Attribution §4.2.4 phía trên): **không tái hiện** (p=0.33) — mean position gần giống hệt cả
  3 hệ (0.45–0.49), Magentic-One thực ra bimodal chứ không "dồn sớm" như paper mô tả.
- AEGIS — chi-square có ý nghĩa thống kê (13/14 code p<0.05) nhưng Cramér's V chỉ 0.019–0.077,
  **effect size không đáng kể** (dưới cả ngưỡng "effect nhỏ" 0.1). Gộp theo 4 bucket topology
  chuẩn vẫn phẳng (chênh nhau ≤8 điểm %). Thêm caveat: AEGIS tiêm lỗi nhân tạo, không quan sát
  tự nhiên — không đúng tinh thần claim gốc dù có ép so.

=> Assumption hạ từ "Đồng thuận 3 paper" (`papers/dataset_findings.md` #1) xuống "chưa đủ bằng
chứng thực nghiệm". Phần duy nhất sống sót: framework/kiến trúc ảnh hưởng đến *loại agent gây
lỗi* (role), không phải *loại lỗi cụ thể* hay *vị trí lỗi*.

**Experiment 2 — assumption "topology quyết định vị trí lỗi, nhất quán cross-dataset": không
được ủng hộ.** Nối tiếp Exp 1 bằng metric universal (vị trí lỗi chuẩn hoá [0,1], không cần
chung taxonomy loại lỗi), gộp 5 dataset localization theo topology chuẩn Exp 0:
- Centralized (4 framework/3 dataset, n=2785): 2 hình dạng đối lập hẳn — TraceElephant/Who&When
  (Magentic-One, Captain-Agent) đỉnh sớm (~0.1–0.15), TELBENCH (MiroFlow/OAgent) dồn cuối. Cùng
  1 bucket topology, kết quả ngược nhau.
- Single-agent (3 framework/3 dataset, n=498): cũng không đồng nhất — AgentErrorBench hình U,
  TRAIL gần phẳng, TraceElephant lệch giữa.
- Hierarchical: chỉ 1 dataset nguồn (TRAIL) — không có cross-dataset để so.

=> Bucket topology quá rộng (khác domain/task/đơn vị đo step) để đọc ra 1 pattern chung —
khác biệt domain (đã thấy ở mục AgentErrorBench "environment-aware" phía trên) có vẻ lấn át
khác biệt topology.

**Kết luận chung.** 2 giả thuyết cốt lõi của trang literature review này ("framework quyết
định loại lỗi" và hệ luận "kiến trúc quyết định vị trí lỗi") **đều không trụ được khi verify
trên data thật**, ngoại trừ 1 mảnh hẹp (role gây lỗi khác theo kiến trúc, TraceElephant).
Nguyên nhân đến từ 2 hướng khác nhau, không phải 1 điểm yếu chung: MAST hỏng do lỗi annotation
ở nguồn dữ liệu; TraceElephant/AEGIS/Exp 2 hỏng do effect size thật sự yếu hoặc bị biến
domain/task che khuất — không phải do phương pháp thống kê sai. Nên đọc toàn bộ mục
MAST/TraceElephant/AgentErrorBench/DRIFT/AEGIS phía trên như **giả thuyết từ literature**,
chưa phải finding đã xác lập — muốn dùng cho quyết định thiết kế (vd. segmentation "aware"
kiến trúc) cần dữ liệu framework-level sạch hơn (xem mục TODO ở
[`experiments/1.framework_environment_correlation/finding_notes.md`](../../../experiments/1.framework_environment_correlation/finding_notes.md)
và [`experiments/2.error_step_position_by_framework/finding_notes.md`](../../../experiments/2.error_step_position_by_framework/finding_notes.md)).
