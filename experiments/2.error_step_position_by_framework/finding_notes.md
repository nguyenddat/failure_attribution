# Experiment 2: phân bố vị trí lỗi (step) theo topology, gộp chéo dataset localization

**Câu hỏi**: nối tiếp Experiment 1 — thay vì "loại lỗi nào" (đã bị chặn bởi
taxonomy không chung giữa các dataset), đo **vị trí lỗi trong trajectory**
(0=bước đầu, 1=bước cuối), vì đây là 1 trục universal — mọi dataset
localization đều tính được, không cần chung taxonomy.

**Phạm vi**: chỉ dataset thuộc `data/error_localization/*` (loại MAST —
`error_categorization`, đã xác nhận data hỏng). Trong 6 dataset localization
đã load, **loại thêm AEGIS**: nhãn AEGIS chỉ định vị theo agent
(`injected_agents[].agent_name`), không có số step — không vẽ được trục x.
Còn lại 5 dataset dùng được: TraceElephant, TRAIL, TELBENCH, Who&When
(hand-crafted), AgentErrorBench.

**Gộp theo topology ngay từ đầu, không theo tên framework**: bản đầu tiên
của experiment này thử gộp theo tên framework chuẩn hoá trước — chỉ tìm
được đúng 1 điểm overlap thật (Magentic-One, giữa TraceElephant và
Who&When), 3/5 dataset (AgentErrorBench, TELBENCH, TRAIL) không có
framework nào trùng tên với dataset khác nên bị loại hẳn khỏi so sánh
chéo. Bỏ hướng đó — gộp thẳng theo **topology chuẩn** (bảng ở
`experiments/0.framework_topology_taxonomy/finding_notes.md`, dùng chung
`framework_architecture.py`) ngay từ đầu: framework khác tên, cùng topology,
vẫn gộp được, và cả 5 dataset đều đóng góp vào ít nhất 1 bucket.

## Cách tính "step" mỗi dataset (không đồng nhất — đọc kỹ trước khi so sánh)

| Dataset | Đơn vị step | Single/multi-fault |
|---|---|---|
| TraceElephant | LLM call / agent action | single (1 điểm/trace) |
| Who&When (hand-crafted) | LLM call / agent action | single (1 điểm/trace) |
| AgentErrorBench | lượt agent (memory→reflection→plan→action = 1 "step") | single (1 điểm/trace) |
| TRAIL | span (OpenTelemetry, sau khi flatten cây theo depth-first) | multi (nhiều điểm/trace) |
| TELBENCH | semantic span (đã gộp sẵn theo mục tiêu cục bộ) | multi (nhiều điểm/trace) |

**Không phải cùng đơn vị đo** — chuẩn hoá về [0,1] (vị trí/tổng số step)
giúp so sánh HÌNH DẠNG phân bố được, nhưng vẫn nên đọc là "xu hướng
tương đối trong trajectory của chính dataset đó", không phải giá trị
tuyệt đối so sánh chéo 1-1.

## Phân loại topology dùng (`framework_architecture.py`)

Lấy thẳng từ bảng chuẩn Experiment 0, không tự đoán riêng cho experiment
này (bản trước của file này từng tự đoán, sai 3/8 entry so với bảng chuẩn
— đã sửa):

| Topology | Framework | Dataset | Căn cứ |
|---|---|---|---|
| Hierarchical | OpenDeepResearch | TRAIL (GAIA) | note_trail.md ghi thẳng "hierarchical multi-agent" |
| Centralized | Captain-Agent | TraceElephant | Exp 0: Centralized (dynamic hub, không phải đa cấp) |
| Centralized | Magentic-One | TraceElephant, Who&When | Exp 0: 1 Orchestrator hub |
| Centralized | MiroFlow, OAgent | TELBENCH | Exp 0 ghi "?" (chưa rõ) — suy đoán riêng, độ tin cậy thấp nhất |
| Single-agent | SWE-Agent | TraceElephant | Exp 0: 1 control loop, không phải multi-agent |
| Single-agent | CodeAct | TRAIL (SWE-bench) | note_trail.md ghi thẳng "single-agent" |
| Single-agent | AgentDebug-ReAct | AgentErrorBench | Exp 0: vòng lặp 4-module đơn |

**Decentralized rỗng** — không dataset nào trong 5 dataset này dùng hệ
kiểu peer-to-peer/debate/graph (kiểu đó chỉ có ở AEGIS's llm_debate/dylan/
macnet, nhưng AEGIS bị loại khỏi experiment này từ đầu — không có step).

Khác bản trước (tự đoán, chưa đối chiếu Experiment 0): Captain-Agent
chuyển từ Hierarchical sang Centralized, SWE-Agent/CodeAct/AgentDebug-ReAct
chuyển từ Centralized sang Single-agent (bucket mới). Hệ quả quan trọng:
**Hierarchical giờ chỉ còn 1 dataset nguồn (TRAIL/OpenDeepResearch,
n=585)** — mất luôn overlap 2-dataset (TraceElephant+TRAIL, n=670) mà bản
trước từng có, vì Captain-Agent không còn ở bucket này. Hierarchical không
còn là bằng chứng cross-dataset thật nữa, giống caveat "1 dataset/bucket"
đã gặp ở MAST/TraceElephant (Experiment 1).

## Kết quả: `results/figures/step_position_grid_architecture_x_dataset.png`

Trục y là **density** (diện tích=1), không phải count thô — n lệch quá xa
giữa các cell (44 → 2552, ~58 lần) nên share raw count làm mọi cell trừ
TELBENCH gần như phẳng/không đọc được; density giữ được hình dạng bất kể n
(n thật vẫn ghi trong title mỗi ô).

- **Hierarchical** (chỉ TRAIL/OpenDeepResearch, n=585): dồn rõ về cuối,
  đỉnh ~0.65-0.75. 1 dataset duy nhất — không so sánh chéo được.
- **Centralized** (n=2785, 4 framework/3 dataset): TraceElephant
  (Captain-Agent+Magentic-One, n=175) đỉnh sớm rõ (~0.15) rồi dao động
  phẳng; Who&When (Magentic-One, n=58) đỉnh sớm hơn nữa (~0.1) rồi giảm
  dần; TELBENCH (MiroFlow+OAgent, n=2552) dồn nhẹ về cuối, ngược hẳn 2
  dataset kia. Cột Tổng hợp (n=2785) hình dạng gần như **chỉ là bản sao
  của TELBENCH** vì TELBENCH chiếm 92% (2552/2785) tổng số điểm trong
  bucket — pool trực tiếp theo raw count làm dataset nhỏ (TraceElephant,
  Who&When) gần như biến mất khỏi đường Tổng hợp dù shape của chúng khác
  hẳn. Đọc cột Tổng hợp của bucket này cần biết rõ điều này, không phải
  "trung bình cân bằng" giữa các dataset.
- **Single-agent** (n=498, 3 framework/3 dataset): AgentErrorBench
  (AgentDebug-ReAct, n=200) hình U rõ — cao đầu (~0.1), giảm giữa, bật lại
  cuối; TRAIL (CodeAct, n=254) tương đối phẳng, hơi tăng về cuối; TraceElephant
  (SWE-Agent, n=44) lệch giữa (~0.15-0.65), không đỉnh sớm/cuối. 3 dataset
  cỡ mẫu cân bằng hơn Centralized (40%/51%/9%) nên Tổng hợp ít bị 1 dataset
  chi phối hơn, nhưng vẫn không có pattern chung rõ giữa 3 hình.

**Không ủng hộ giả thuyết "topology quyết định vị trí lỗi nhất quán"**:
trong Centralized, 4 framework/3 dataset cho ít nhất 2 hình dạng đối lập
(đỉnh sớm ở TraceElephant/Who&When vs dồn cuối ở TELBENCH) — bucket
"Centralized" quá rộng (khác domain/task/đơn vị step) để đọc ra 1 pattern
chung. Single-agent cũng vậy (U-shape vs phẳng vs lệch giữa). Hierarchical
không đủ dữ liệu độc lập để kết luận gì (1 dataset).

## Giới hạn

- Không kiểm định thống kê (chi-square/KS-test) giữa các hình dạng —
  chart này thuần trực quan. Muốn kết luận "khác biệt có ý nghĩa" cần bước
  tiếp theo.
- Hierarchical chỉ có 1 dataset nguồn (TRAIL) — không phải bằng chứng
  cross-dataset thật. Centralized/Single-agent tuy có nhiều dataset hơn
  nhưng cột Tổng hợp có thể bị 1 dataset lớn chi phối (xem caveat
  TELBENCH ở trên) — đọc riêng từng cell theo dataset đáng tin hơn đọc
  cột Tổng hợp.
- TRAIL/AgentErrorBench không có field `framework` thật — dùng
  pseudo-framework (task_source, kiến trúc cố định) thay thế, đã note rõ
  trong `extract_step_positions.py`.
- MiroFlow/OAgent (TELBENCH) xếp Centralized dựa suy đoán riêng, không có
  mô tả kiến trúc trực tiếp từ paper — độ tin cậy thấp nhất trong bảng
  phân loại, và TELBENCH lại là dataset chi phối cột Tổng hợp của
  Centralized (xem trên).

## Script

- `src/extract_step_positions.py` — trích xuất `results/tables/step_positions.csv`
  (long format: dataset, framework, raw_step, n_steps, position). Giữ đủ
  cả 5 dataset (không lọc).
- `src/framework_architecture.py` — bảng phân loại topology, lấy từ
  Experiment 0, dùng chung cho script vẽ.
- `src/plot_step_position_grid_by_architecture.py` — grid theo topology x
  dataset, giữ cả 5 dataset (density, không phải count thô).

## TODO

- [ ] Kiểm định thống kê hình dạng phân bố (KS-test 2 mẫu) cho các cặp
      dataset trong cùng 1 bucket topology.
- [ ] Điều tra giả thuyết "dồn cuối là artifact của span-based multi-fault"
      — thử đếm riêng % lỗi rơi đúng vào span/step cuối cùng của trace, so
      với % kỳ vọng nếu random. (TELBENCH/TRAIL dùng đơn vị span, cả 2 đều
      có xu hướng dồn cuối/tăng dần trong ít nhất 1 bucket.)
- [ ] Cân nhắc thêm AEGIS bằng metric khác (không phải step) — vd. lỗi ở
      agent thứ mấy trong turn-order thay vì global step.
- [ ] Nếu muốn cột Tổng hợp không bị dataset lớn chi phối — cân nhắc
      trung bình hoá theo dataset (weight bằng nhau) thay vì pool raw
      count trước khi tính density.
