# Cách thực hiện Experiment 1 (walkthrough thực tế)

Toàn bộ các bước đã làm để ra `experiments/1.framework_environment_correlation/`.
Áp dụng quy trình chung ở `docs/how_to_run_an_experiment.md`. Kết luận đầy
đủ: [experiments/1.framework_environment_correlation/finding_notes.md](../experiments/1.framework_environment_correlation/finding_notes.md).

## Bước 1 — Chốt assumption

Lấy từ `papers/dataset_findings.md` mục #1 "Phân bố lỗi phụ thuộc
framework/kiến trúc" — literature review 3 paper (MAST, TraceElephant,
AgentErrorBench) đánh giá "Đồng thuận", nhưng chỉ dựa đọc paper, chưa
verify bằng data thật.

## Bước 2 — Chọn dataset: MAST, TraceElephant, AEGIS

Loại AgentErrorBench (biến độc lập là environment cố định-kiến-trúc, không
phải framework) và TRAIL/TELBENCH/Who&When (chưa có framework-level
label phù hợp lúc đó). Thêm AEGIS thay AgentErrorBench vì có sẵn
framework + mã lỗi dùng chung taxonomy MAST.

## Bước 3 — MAST: reload full-column, phát hiện bug data ở nguồn

1. Loader cũ (`schemas/mast.py`) chỉ giữ `mas_name`+`raw_trajectory`+`faults`,
   drop `llm_name`/`benchmark_name`/`trace_id`/`trace.key`/`trace.index`.
   Đọc raw HF row (`mcemri/MAST-Data`) liệt kê đủ 6 cột gốc, viết lại
   schema+loader giữ hết → `make load_mast`, 1242 trace.
2. Confusion matrix `mas_name x benchmark_name` — phát hiện confound gần
   tuyệt đối (6/7 framework chỉ 1 cặp benchmark+model).
3. Chi-square `mas_name x mast_annotation` (14 mã) trên full 7 framework —
   6/14 mã "có ý nghĩa".
4. Phát hiện bất thường: AppWorld/HyperAgent/OpenManus có rate mỗi mã
   giống hệt nhau dù chạy khác benchmark/model hoàn toàn — verify bằng
   hash `raw_trajectory` (0 trùng) + so khớp theo `trace_index` (khớp
   100%) → nghi bug gán nhãn theo vị trí, ban đầu tưởng chỉ 3 framework.
5. Mở rộng kiểm tra: so MetaGPT (lọc đúng GPT-4o/ProgramDev) với ChatDev —
   2 framework "thật" — vẫn giống hệt nhau. Kiểm tra toàn diện mọi
   `trace_index` (0-205) × mọi nhóm (11 nhóm) → 0 index nào có >1 pattern
   nhãn khác nhau, verify trên raw HuggingFace (không qua loader) →
   **`mast_annotation` là hàm của `trace_index`, không phải nội dung
   trace — bug toàn dataset**, không riêng 3 framework.
6. Phát hiện phụ: 62/1242 trace trùng lặp nội dung (Magentic chủ yếu —
   batch mở rộng chép lại batch gốc).
7. Kết luận: mọi phân tích dựa `mast_annotation` **invalid**, giữ script
   làm tài liệu debug. Confusion matrix (không phụ thuộc annotation) vẫn
   valid.

## Bước 4 — TraceElephant: tìm dataset thật, load, phân tích

1. Tìm nguồn công khai: GitHub `TraceElephant/TraceElephant` (ACL 2026) →
   HF dataset `TraceElephant/TraceElephant` (`data.zip`, đọc trực tiếp từ
   zip không giải nén). Xác nhận phân loại: error_localization +
   single_fault (field `mistake_agent`+`mistake_step`, y hệt tên field
   `schemas/who_and_when.py`).
2. Viết schema+loader giữ đủ cột (`task_instruction`, `agent_configuration`,
   mỗi step giữ nguyên `input`/`output`/`tool_logs` dạng dict lồng, không
   rút gọn) → `make load_trace_elephant`, 220 trace (khớp paper).
3. `mistake_agent` là tên riêng theo system (không chung taxonomy) — gom
   3 role dùng chung (Orchestrator/Worker/Verification) theo cách paper
   §4.2.2 tự phân nhóm, viết `trace_elephant_agent_roles.py`.
4. Chi-square `system_name x role`: p=1.05e-7, V=0.294 — **có ý nghĩa**.
5. Chi-square `system_name x vị trí lỗi (early/mid/late)`: p=0.33 —
   **không có ý nghĩa** — claim cụ thể nhất của paper (CaptainAgent phân
   tán, Magentic-One/SWE-Agent dồn sớm) không tái hiện được.

## Bước 5 — AEGIS: reload full-column, resolve TODO cũ, effect size nhỏ

1. Schema cũ drop nhiều field (`id`, `metadata.model/num_agents/
   num_injected_agents`, `output.faulty_agents`, `ground_truth.
   is_injection_successful`, per-step `is_injected`). Đọc raw HF row
   (`Fancylalala/AEGIS`, 3 split train/val/test) đủ 5 cột gốc, viết lại
   giữ hết → `make load_aegis`, 9533 trace.
2. Metadata reliability check: `num_agents` khớp thực tế chỉ 36.1%,
   `num_injected_agents` khớp chỉ 78.3% — không dùng field metadata, tính
   lại từ list thật.
3. Resolve TODO cũ (nghi vấn "SmolAgents/Magentic-One luôn đúng 1
   fault/trace" ghi trong file finding_1 đã mất): verify magentic_one/
   smoagents luôn dùng đúng 1 injection_strategy + confound tuyệt đối với
   benchmark riêng (`magentic+gaia`/`smol+gaia`) → **giới hạn pipeline**,
   không phải bug.
4. Chi-square `framework x error_type` (4 framework thật, loại 2 framework
   single-injection): 13/14 mã p<0.05 nhưng Cramér's V chỉ 0.019-0.077 —
   **có ý nghĩa thống kê nhưng effect size không đáng kể** (cỡ mẫu quá
   lớn, ~19000 tag).

## Bước 6 — Tổ chức lại thành experiment có cấu trúc

1. Tạo `experiments/1.framework_environment_correlation/{src,results}`.
2. Di chuyển 15 script phân tích (không di chuyển loader — loader ở lại
   `data/`, dùng chung cho experiment sau) vào `src/`.
3. Di chuyển toàn bộ png/csv đã sinh vào `results/`.
4. Sửa import + path trong từng script: tên folder bắt đầu bằng số không
   phải Python package hợp lệ → không dùng `-m` được. Đổi sang chạy trực
   tiếp (`python src/foo.py`), thêm `sys.path` bootstrap tới repo root,
   đổi `json_dir`/`fig_path` từ tương đối-theo-vị-trí-file sang tuyệt đối
   từ `REPO_ROOT`.
5. Test chạy lại toàn bộ 13 script (2 file còn lại là module constant,
   không chạy được) — verify khớp 100% kết quả cũ.
6. Viết `finding_notes.md` gộp 3 file finding gốc + bảng so sánh claim gốc
   vs kết quả verify + kết luận Assumption 1 (hạ từ "Đồng thuận" xuống
   "chưa đủ bằng chứng thực nghiệm").

## Bước 7 — Tách results/figures và results/tables

`results/` ban đầu để lẫn png+csv — tách `results/figures/` (10 png) và
`results/tables/` (3 csv), update `RESULTS_DIR` trong từng script cho
đúng subfolder, test lại toàn bộ.

## Bước 8 — Thêm phân tích gộp theo kiến trúc cho 2 dataset còn thiếu

MAST đã có (`plot_mast_pie_by_architecture.py`, dù kết quả invalid do bug
data). TraceElephant và AEGIS chưa:

- **AEGIS**: phân loại dựa **trực tiếp mô tả kiến trúc của chính paper
  AEGIS §4.1** (không tự đoán) — Hierarchical=agentverse, Centralized=
  magentic_one+smoagents, Decentralized=dylan+llm_debate+macnet (3
  framework pool chung). Viết `plot_aegis_pie_by_architecture.py`.
- **TraceElephant**: không có 1 paper mô tả kiến trúc cả 3 hệ như AEGIS —
  tự phân loại (own judgment, note rõ trong docstring): Hierarchical=
  captain-agent, Centralized=magentic-one+swe-agent, Decentralized=rỗng.
  Viết `plot_trace_elephant_pie_by_architecture.py`.

## Bước 9 — Review lại toàn bộ results, cập nhật finding_notes.md

Đọc lại hết 3 CSV + 12 PNG (không chỉ 2 file mới), so với `finding_notes.md`
hiện có → phát hiện 2 chart mới chưa được note:
- **AEGIS** (Decentralized bucket pool 3 framework thật, n=6608): vẫn ra
  39/43/19% vs 38/46/16% vs 39/41/20% — gần như giống hệt nhau dù đã pool
  nhiều framework/bucket → **củng cố thêm** kết luận "effect size không
  đáng kể", đáng tin hơn kết luận tương tự từ MAST/TraceElephant (thiếu
  mẫu/bucket).
- **TraceElephant** (Hierarchical chỉ 1 hệ, Centralized 2 hệ gộp): **không
  thêm bằng chứng độc lập mới** — dính đúng caveat "không đủ
  framework/bucket" đã nêu ở MAST, chỉ vẽ lại per-system data dưới nhãn
  kiến trúc.

Update `finding_notes.md`: thêm 2 đoạn trên vào mục TraceElephant/AEGIS,
sửa dòng AEGIS trong bảng kết luận cuối, thêm 2 script mới vào danh sách.

## Bước 10 — Sửa phân loại topology theo note chuẩn của Experiment 0

Experiment 0 (`experiments/0.framework_topology_taxonomy/finding_notes.md`)
ra đời sau, chốt 1 bảng topology chuẩn (6 bucket: Hierarchical/Pipeline/
Centralized/Decentralized/Single-agent/Variable) cho toàn bộ framework
dùng trong repo — nhưng 3 script `plot_*_pie_by_architecture.py` ở Bước 8
tự phân loại riêng (3 bucket: Hierarchical/Centralized/Decentralized),
lệch với bảng chuẩn ở vài framework. Đối chiếu và sửa lại:

- **MAST**: MetaGPT trước gộp Hierarchical cùng ChatDev → bảng chuẩn ghi
  Pipeline (SOP assembly-line, không có manager hub). AG2 trước gán
  Decentralized → bảng chuẩn ghi Variable (thư viện, topology theo config).
  4 bucket giờ mỗi bucket đúng 1 framework (không đổi kết luận, MAST vẫn
  invalid do bug annotation).
- **AEGIS**: smoagents/macnet trước gộp theo mô tả linh hoạt trong AEGIS
  paper §4.1 (macnet→Decentralized cùng dylan/llm_debate, smoagents→
  Centralized cùng magentic_one) → bảng chuẩn tách riêng cả 2 vào Variable
  (cùng lý do: thư viện, topology theo config). Decentralized giờ chỉ còn
  2 framework (dylan+llm_debate, n=4249, mất macnet) thay vì 3 (n=6608) —
  bằng chứng "pool nhiều framework vẫn phẳng" yếu hơn bản trước dù kết
  luận effect size không đổi (4 bucket mới vẫn phẳng, chênh ≤8 điểm %).
- **TraceElephant**: captain-agent trước tách riêng bucket Hierarchical →
  bảng chuẩn ghi Centralized (Captain là 1 hub điều phối runtime, dynamic).
  swe-agent trước gộp Centralized cùng magentic-one → bảng chuẩn ghi
  Single-agent (1 control loop, không phải multi-agent). Centralized giờ
  pool captain-agent+magentic-one (n=176), Single-agent tách riêng swe-agent
  (n=44).

Cập nhật docstring 3 script + `finding_notes.md` (thêm bảng số liệu mới cho
AEGIS/TraceElephant, note rõ khác gì so với bản trước). Không đổi kết luận
Assumption 1 ở dưới — chỉ đổi cách gộp bucket và vài con số phần trăm.

## Kết luận cuối — Assumption 1

Từ "Đồng thuận" (literature review) xuống "**chưa đủ bằng chứng thực
nghiệm**": MAST không verify được (data lỗi), TraceElephant tái hiện được
đúng phần yếu hơn (role) và gãy đúng phần mạnh/cụ thể nhất (vị trí lỗi),
AEGIS có ý nghĩa thống kê nhưng effect size không đáng kể — và càng pool
nhiều framework/bucket càng rõ là không đáng kể. Chi tiết đầy đủ:
`experiments/1.framework_environment_correlation/finding_notes.md`.

## Script/output liên quan

| Bước | File |
|---|---|
| Loader MAST | `data/error_categorization/mast.py` |
| Loader TraceElephant | `data/error_localization/single_fault/trace_elephant.py` |
| Loader AEGIS | `data/error_localization/multi_fault/aegis.py` |
| Toàn bộ script phân tích | `experiments/1.framework_environment_correlation/src/*.py` |
| Kết quả | `experiments/1.framework_environment_correlation/results/{figures,tables}/` |
| Tổng hợp | `experiments/1.framework_environment_correlation/finding_notes.md` |
| Chi tiết từng dataset | `papers/notes/findings/{mast,trace_elephant,aegis}_data_finding.md` |
