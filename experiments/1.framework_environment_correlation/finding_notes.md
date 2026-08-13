# Experiment 1: tương quan phân bố lỗi vs framework/environment

**Assumption kiểm định** (từ `papers/dataset_findings.md`, mục #1 "Phân bố
lỗi phụ thuộc framework/kiến trúc" — lúc đó đánh giá "Đồng thuận" chỉ dựa
literature review, chưa verify bằng data thật): framework/kiến trúc quyết
định phân bố loại lỗi agent gặp phải.

**Dataset dùng**: MAST, TraceElephant, AEGIS (3 dataset có sẵn field
framework/environment + nhãn lỗi có cấu trúc). Không dùng AgentErrorBench
(biến độc lập là environment cố định-kiến-trúc, không phải framework) hay
TRAIL/TELBENCH/Who&When (chưa có bước phân tích framework-level).

**Cấu trúc**: `src/` = script phân tích (đọc dữ liệu từ `data/` — loader
dùng chung, không nằm trong experiment này), `results/` = ảnh + CSV output.
Mọi script chạy trực tiếp (`python src/<file>.py`, không dùng `-m`, vì tên
thư mục bắt đầu bằng số không hợp lệ làm Python package).

**Phân loại topology** (`plot_*_pie_by_architecture.py`): dùng bảng chuẩn
`experiments/0.framework_topology_taxonomy/finding_notes.md` (6 bucket:
Hierarchical/Pipeline/Centralized/Decentralized/Single-agent/Variable),
không tự đoán per-dataset nữa như bản đầu. Sửa lại (fix theo note chuẩn
exp 0): MetaGPT tách ra Pipeline (không gộp Hierarchical với ChatDev),
AG2 = Variable (không phải Decentralized), smoagents/macnet (AEGIS) = Variable
(không gộp Centralized/Decentralized), swe-agent (TraceElephant) = Single-agent
(không gộp Centralized), captain-agent (TraceElephant) = Centralized (không
phải Hierarchical riêng).

---

## MAST — không verify được, data lỗi ở nguồn

Full: [papers/notes/findings/mast_data_finding.md](../../papers/notes/findings/mast_data_finding.md)

**Phát hiện chính**: cột `mast_annotation` trong file gốc `MAD_full_dataset.json`
(HuggingFace `mcemri/MAST-Data`) là **hàm của `trace_index`** (số thứ tự
lần chạy), không phải hàm của nội dung trace. Verify trên toàn bộ 1242
trace, 206 giá trị index, 11 nhóm framework×benchmark×model — 0 index nào
có quá 1 pattern nhãn khác nhau, dù nội dung log hoàn toàn khác nhau. Verify
cả trên raw HuggingFace row (không qua loader của mình) — bug ở nguồn, không
phải do code.

=> Mọi so sánh "framework nào lỗi nhiều/loại gì" trên MAST public data đều
**không tin cậy được**. Không nói được claim gốc của paper MAST (MetaGPT
ít lỗi FC1/FC2 hơn ChatDev 60-68%, nhiều FC3 gấp 1.56 lần) đúng hay sai —
chỉ nói được: input để verify claim đó đã hỏng.

Phát hiện phụ: 62/1242 trace (5%) trùng lặp nội dung (chủ yếu Magentic, do
batch mở rộng chép lại batch gốc).

**Script**: `src/plot_mas_name_vs_benchmark.py` (confound, vẫn valid — không
phụ thuộc annotation), `src/analyze_mas_annotation_correlation.py` +
`src/plot_mast_pie_by_*.py` + `src/plot_mast_failure_distribution.py`
(chạy được, nhưng **kết quả invalid**, giữ làm tài liệu debug/lịch sử điều
tra — xem chi tiết cách phát hiện bug trong file finding gốc).

---

## TraceElephant — tái hiện được 1 nửa

Full: [papers/notes/findings/trace_elephant_data_finding.md](../../papers/notes/findings/trace_elephant_data_finding.md)

Dataset sạch, không có bug kiểu MAST. 2 claim tách biệt:

- **Role gây lỗi khác theo kiến trúc (Orchestrator/Worker/Verification)**:
  chi-square p=1.05e-7, Cramér's V=0.294 — **có ý nghĩa thật**. swe-agent
  gần như thuần Worker (97.7%), magentic-one có tỷ lệ Orchestrator cao nhất
  (29.7%, khớp có 1 orchestrator trung tâm rõ ràng), captain-agent là hệ
  duy nhất có role Verification riêng.
- **Vị trí lỗi theo timeline (early/mid/late)**: chi-square p=0.33 —
  **KHÔNG có ý nghĩa**. Claim paper "CaptainAgent phân tán, Magentic-
  One/SWE-Agent dồn sớm" không tái hiện được — mean position gần như giống
  hệt cả 3 hệ (0.45–0.49), magentic-one thực ra bimodal chứ không "dồn sớm"
  thuần.

**Gộp theo topology** — dùng bảng chuẩn
`experiments/0.framework_topology_taxonomy/finding_notes.md` thay vì tự
phân loại: captain-agent = **Centralized** (dynamic — Captain là 1 hub
điều phối runtime, không phải đa cấp), magentic-one = **Centralized**
(Orchestrator hub, static), swe-agent = **Single-agent** (1 control loop,
không phải multi-agent), Hierarchical/Decentralized rỗng. Khác bản trước
(từng gộp Hierarchical=captain-agent riêng, Centralized=magentic-one+
swe-agent): nay captain-agent+magentic-one gộp chung Centralized
(n=176 trace: Orchestrator 24%, Verification 8%, Worker 68%), swe-agent
tách riêng thành Single-agent (n=44: Orchestrator 2%, Worker 98%,
Verification 0%) — xem
`results/figures/trace_elephant_pie_role_by_architecture.png`.
**Vẫn không thêm bằng chứng độc lập mới**, dính đúng caveat của MAST
("không đủ framework/bucket để tách hiệu ứng kiến trúc khỏi hiệu ứng
framework cụ thể"): mỗi bucket vẫn chỉ 1-2 hệ. Chart chỉ vẽ lại đúng số
per-system đã có ở mục Role association trên, dưới nhãn topology chuẩn.

**Script**: `src/plot_trace_elephant_system_vs_benchmark.py`,
`src/analyze_trace_elephant_correlation.py`,
`src/plot_trace_elephant_role_by_system.py`,
`src/plot_trace_elephant_step_position.py`,
`src/plot_trace_elephant_pie_by_architecture.py`.

---

## AEGIS — effect size không đáng kể (và bản chất khác 2 dataset kia)

Full: [papers/notes/findings/aegis_data_finding.md](../../papers/notes/findings/aegis_data_finding.md)

AEGIS **tiêm lỗi nhân tạo**, không quan sát tự nhiên như MAST/TraceElephant
— "framework X dính lỗi Y nhiều hơn" ở đây phản ánh hành vi injection
pipeline, không phải xu hướng lỗi thật của framework. Cần đọc kết quả với
tiền đề này.

- Metadata (`num_agents`, `num_injected_agents`) không đáng tin (khớp thực
  tế chỉ 36%/78%) — luôn tính lại từ list thật.
- magentic_one/smoagents luôn tiêm đúng 1 lỗi — **xác nhận là giới hạn
  pipeline** (luôn 1 injection_strategy, confound tuyệt đối với benchmark
  riêng), không phải bug — resolve TODO cũ từ file finding_1 đã mất.
- Chi-square framework x error_type (4 framework thật, loại 2 framework
  single-injection): 13/14 code p<0.05 nhưng Cramér's V chỉ 0.019–0.077 —
  **có ý nghĩa thống kê nhưng effect size không đáng kể** (dưới ngưỡng
  "effect nhỏ" 0.1), do cỡ mẫu quá lớn (~19000 tag). Pie chart 6 framework
  nhìn gần như giống hệt nhau.

**Gộp theo topology chuẩn** (`experiments/0.framework_topology_taxonomy/
finding_notes.md`, không phải tự đoán từ paper §4.1 nữa) — sửa lại so với
bản trước: paper AEGIS §4.1 gọi smoagents "orchestrator-executor" giống
magentic_one và gọi macnet "network topology" giống dylan/llm_debate, nhưng
bảng chuẩn exp 0 tách 2 framework này ra bucket riêng **Variable** vì
smolagents/MacNet là **thư viện** — topology do config quyết định, không
cố định như magentic_one (luôn 1 Orchestrator) hay dylan/llm_debate (luôn
decentralized theo thiết kế). Kết quả 4 bucket, xem
`results/figures/aegis_pie_group_by_architecture.png`:

| Bucket | Framework | n | SysDesign / InterAgent / TaskVerif |
|---|---|---|---|
| Hierarchical | agentverse | 1995 | 39% / 43% / 19% |
| Centralized | magentic_one | 449 | 37% / 48% / 15% |
| Decentralized | dylan, llm_debate | 4249 | 37% / 42% / 20% |
| Variable | macnet, smoagents | 2840 | 41% / 40% / 19% |

**Vẫn gần như giống hệt nhau** (chênh nhau nhiều nhất chỉ ~8 điểm %, ở
Centralized/InterAgent) — không đổi kết luận effect size không đáng kể.
Nhưng bằng chứng "pool nhiều framework" **yếu hơn bản trước**: Decentralized
giờ chỉ còn 2 framework thật (n=4249, mất macnet) thay vì 3 (n=6608) —
Centralized giờ tách hẳn ra 1 framework duy nhất (magentic_one, n=449)
thay vì gộp cùng smoagents (trước n=930). Variable là bucket mới, pool
2 framework nhưng khác bản chất bucket kia (do cùng là "thư viện config-
dependent", không phải cùng 1 topology cố định) — không nên đọc nó như
bằng chứng "topology X không ảnh hưởng" theo cách đọc Hierarchical/
Centralized/Decentralized.

**Script**: `src/plot_aegis_framework_vs_benchmark.py`,
`src/analyze_aegis_correlation.py`, `src/plot_aegis_pie_by_group.py`,
`src/plot_aegis_pie_by_architecture.py`.

---

## Kết luận: Assumption 1 — từ "Đồng thuận" xuống "chưa đủ bằng chứng thực nghiệm"

| Dataset | Claim gốc | Kết quả verify |
|---|---|---|
| MAST | MetaGPT/ChatDev tradeoff FC1-3 rõ rệt | **Không verify được** — data annotation lỗi ở nguồn |
| TraceElephant | Role + vị trí lỗi khác theo kiến trúc | **Role**: tái hiện được (p=1e-7). **Vị trí**: không tái hiện (p=0.33) |
| AEGIS | (không trong claim gốc, dùng thay AgentErrorBench) | Có ý nghĩa thống kê nhưng **effect size không đáng kể** (V<0.08), và bản chất injected không phải observed — kết luận này **được củng cố thêm** khi gộp theo topology chuẩn exp 0 (4 bucket vẫn phẳng, chênh nhau ≤8 điểm %, xem mục AEGIS ở trên) |

Mỗi dataset khi verify thực nghiệm đều gãy ít nhất 1 phần của claim gốc:
MAST gãy hoàn toàn (data lỗi), TraceElephant gãy đúng phần mạnh nhất/cụ thể
nhất của claim (vị trí lỗi) chỉ giữ được phần yếu hơn (loại agent), AEGIS
không so sánh được theo đúng tinh thần claim gốc (injected, không natural)
và ngay cả khi ép so vẫn ra effect size không đáng kể.

**Assumption 1 hạ xuống mức giả thuyết chưa kiểm chứng đầy đủ**, không còn
là finding đã xác lập ở mức "đồng thuận 3 paper" như literature review ban
đầu kết luận trong `papers/dataset_findings.md`.

## TODO (mang theo từ 3 file finding gốc)

- [ ] MAST: nếu cần phân tích framework-level thật — thiết kế lại pipeline
      tự đọc `raw_trajectory` + gán nhãn mới, không dùng `mast_annotation`.
- [ ] TraceElephant: đối chiếu lại đúng cách paper định nghĩa "early/mid/late"
      (đọc Appendix nếu có), thử tính trên toàn bộ 380 trace (không chỉ 220
      trace fail) nếu tìm được 160 trace thành công.
- [ ] AEGIS: lọc theo từng benchmark riêng (vd. chỉ MATH) rồi so 4 framework
      pooled trên cùng 1 benchmark, xem effect size có tăng không.
- [ ] Cập nhật `papers/dataset_findings.md` mục #1 + bảng tóm tắt theo kết
      luận revised ở trên.
