- Các LLM + khả năng tương tác với môi trường bên ngoài, tools và memory => LLM Agents và được áp dụng ở nhiều lĩnh vực khác nhau: embodied control, khám phá khoa học, open-ended web interaction và hỗ trợ nghiên cứu
><
- agent hiện tại:
    + Chưa hoàn thiện
    + Chưa đủ robust
    + Thường xuyên mắc lỗi — từ hiểu sai chỉ dẫn, dùng sai công cụ, cho đến sụp đổ trong các chuỗi suy luận dài hạn (long-horizon reasoning). 
=> Câu hỏi: LLM agent thất bại ở đâu?

- Mast/ Who&When: 
    + Chủ yếu tập trung vào việc liệt kê các loại lỗi 
    + Đưa ra các nghiên cứu tình huống mang tính định tính. 
    + Chưa đi xa đến mức cung cấp cơ chế có hệ thống để truy vết thất bại về nguyên nhân gốc rễ (root cause)
    + Không cho phép agent tự sửa những thất bại đã được phát hiện dựa trên các hiểu biết đó

Contributions:
- Mô-đun hóa việc phân tích lỗi: memory, reflection, planning và action

## 1. Cách xây dựng datasets — AgentErrorBench

- Bước 1: Xây dựng Failure Taxonomy: 500+ failed trajectories từ 3 benchmark (ALFWorld, WebShop, GAIA) -> phân tích thủ công tìm các mẫu hình lặp lại -> `AgentErrorTaxonomy`. Điểm khác biệt: không phân loại theo *giai đoạn* mà theo **module nội tại của agent** (memory / reflection / planning / action + system), để lỗi luôn quy được về "bộ phận nào hỏng".

- Bước 2: Chuẩn hóa nhãn qua pilot annotation: biến quan sát định tính thành định nghĩa dùng chung:
    + Vòng lặp: annotator được training + feedback từ tác giả → double-annotation độc lập trên tập con chung → thảo luận giải quyết bất đồng → làm rõ ranh giới định nghĩa (VD: "retrieval failure" thuộc Memory vs "constraint ignorance" thuộc Planning) → lặp lại **3 vòng**.
    + Kết quả: Cohen's κ = 0.55 (tác giả gọi là substantial; thực tế chỉ ở mức moderate).

- Bước 3: Annotate thủ công toàn bộ (KHÔNG dùng LLM-as-a-Judge): 200 trajectories × 10 expert annotators (nghiên cứu sinh có kinh nghiệm NLP/agent).
    + Đơn vị gán nhãn: **decision-step** — mỗi bước duyệt qua cả 4 module, gán error type theo taxonomy.
    + Ràng buộc cốt lõi: annotator phải chỉ ra **minimal set of root-cause failures** giải thích được chuỗi lỗi phía sau, thay vì gắn cờ mọi lỗi bề mặt. Đây là điểm phân biệt chính so với các dataset chỉ liệt kê lỗi.
    + Mỗi trajectory kèm thêm **actionable feedback** — không chỉ nói "sai ở đâu" mà còn "nên sửa thế nào", để dataset dùng được cho cả detection lẫn correction.

Không có bước mở rộng quy mô tự động. Dataset dừng ở 200 traces, đánh đổi **quy mô lấy độ sâu**: mỗi nhãn có nguyên nhân gốc + phản hồi sửa lỗi do người viết. Tác giả tự thừa nhận trong Limitation rằng chi phí annotation là rào cản, và họ không đủ nguồn lực để train một debugging model riêng.

Cấu trúc nhãn của một trajectory:
- trajectory: chuỗi (state, action) có cấu trúc theo 4 module mỗi bước — khác log thô phi cấu trúc
- step-level labels: {error type theo taxonomy} cho từng module ở từng bước
- critical error: (step*, module*, error_type*) — bộ ba nguyên nhân gốc sớm nhất
- feedback: hướng dẫn sửa lỗi dạng ngôn ngữ tự nhiên gắn với critical error

Các field trong datasets (lấy từ github của họ):

**`Original_Failure_Trajectory/{ALFWorld, WebShop, GAIA}/` — 200 trace thô** (100 / 50 / 50; 23.2 + 8.2 + 44.2 MB). Mỗi file JSON chỉ 2 khóa:
- `messages`: hội thoại **phẳng, xen kẽ** `user` / `assistant`, 2 message = 1 step (`messages[2i]` = observation, `messages[2i+1]` = agent output). Không có cây span như TRAIL, không có `agent_name` như Who&When (single-agent).
    + message `user`: prompt hệ thống + task + observation hiện tại + `admissible actions` (ALFWorld/WebShop) hoặc danh sách tool (GAIA) + khung 4 thẻ mà agent phải điền. Từ step 2 trở đi có thêm *compact summary* của các step trước (WebShop: `SearchQuery / PagesVisited / RelevantProducts / Selections / IrrelevantSummary`).
    + message `assistant`: chính là **kiến trúc 4 module hiện ra dưới dạng thẻ XML** `<memory>` → `<reflection>` → `<plan>` → `<action>` trong cùng một message. Step 1 chỉ có `<plan>` + `<action>` (chưa có gì để nhớ / phản tỉnh). GAIA thêm `<answer>`, và có thêm `<code>` / `<end_code>` cho `python_code_generator`.
    => Đây là điểm khiến dataset dùng được cho gán nhãn theo module: **trace đã được cấu trúc hóa sẵn theo module tại thời điểm sinh**, không phải parse ngược từ log thô.
- `metadata`: `model` (gpt-4o / llama-70b / qwen3-8b), `environment`, `steps`, `won` (**False ở cả 200/200** — chỉ có trace thất bại), `batch_idx` / `env_id` / `test_idx`, `timestamp`, riêng ALFWorld có `gamefile` (đường dẫn PDDL), GAIA có `pid`.

**`Label/{alfworld, gaia, webshop}_labels.json` — mảng JSON, 100 / 50 / 50 phần tử**, mỗi phần tử:
- `trajectory_id`: khóa nối sang file trace
- `LLM`: `GPT-4o` (81) / `Llama3.3-70B-Turbo` (57) / `Qwen3-8B` (62)
- `task_type`: `alfworld` / `gaia` / `webshop`
- `critical_failure_step`: chỉ số step 1-based (min 1, max 30, trung bình 7.92)
- `critical_failure_module`: `memory` / `reflection` / `plan` / `action` / `system`
- `step_annotations`: mảng, nhưng **luôn đúng 1 phần tử** = `{step, <module>: {failure_type, reasoning}}`, và luôn trùng khớp `critical_failure_step` / `critical_failure_module` (0 ca lệch).

Taxonomy thực tế xuất hiện trong nhãn (`failure_type`):
| module | failure_type |
|---|---|
| memory | `over_simplification` (20), `hallucination` (12), `memory_retrieval_failure` (2) |
| reflection | `progress_misjudge` (20), `outcome_misinterpretation` (12), `causal_misattribution` (5), `hallucination` (1) |
| plan/planning | `inefficient_plan` (43), `constraint_ignorance` (13), `impossible_action` (10) |
| action | `misalignment` (7), `Parameter_error` (4), `invalid_action` (1) |
| system | `step_limit` (8), `environment_error` (7), `tool_execution_error` (4), `llm_limit` (1) |

Phân bố `critical_failure_module` lệch theo môi trường: GAIA nghiêng hẳn về planning (26/50), WebShop về plan + memory (19 + 15 / 50), ALFWorld trải đều hơn (plan 33, reflection 22, memory 18, action 14, system 13).

Những chỗ lệch giữa paper và dữ liệu phát hành:
- **Không có nhãn step-level đầy đủ.** Paper mô tả mỗi step duyệt qua cả 4 module; bản phát hành **chỉ gán nhãn đúng 1 step × 1 module cho mỗi trajectory** (200 annotation / 200 trace). Nên "Stage 1 — Fine-grained Analysis" không có ground truth để chấm.
- **Không có trường `feedback`.** Actionable feedback được nhấn mạnh trong paper nhưng không nằm trong `Label/`; chỗ gần nhất là `reasoning` (1 câu giải thích, thường chỉ diễn giải lại lỗi).
- **27/200 (13.5%) annotation rỗng**: `failure_type = ""` và `reasoning = ""` — vẫn có `critical_failure_step` + `module` nhưng không có loại lỗi. Muốn dùng cho phân loại phải lọc bỏ, còn **173 mẫu**.
- **Tên nhãn chưa chuẩn hóa**: `plan` vs `planning` (2 tên cho cùng module), `inefficient_plan` vs `plan_inefficient`, `Parameter_error` viết hoa lệch quy ước, `tool_execution_error ` dư khoảng trắng cuối => phải normalize trước khi đếm.
- **1 ca nhãn ngoài phạm vi**: `gaia/GPT-4o_003_memory_b000_t00_e03-21a3a421` có `critical_failure_step = 3` nhưng trace chỉ 1 step.
- Độ dài trace rất lệch: ALFWorld luôn đủ 60 message (30 step, tức 100% thất bại vì cạn step limit), WebShop trung bình 51.8, GAIA chỉ 28.7 message (min 2) nhưng file lại nặng nhất (~44 MB / 50 trace) vì output tool dài.


Ngoài ra họ còn propose:
- kiến trúc agent thay thế ReAct: Bắt agent chạy tuần tự 4 module ở mỗi bước: Memory → Reflection → Planning → Action.
- framework gỡ lỗi 3 giai đoạn (đóng góp chính)
    + Stage 1 — Fine-grained Analysis: quét toàn trajectory, gán error type cho từng module ở từng bước → tạo "module-level error profile".
    + Stage 2 — Critical Error Detection: tìm bước SỚM NHẤT mà sửa nó thì ngăn được thất bại cuối cùng. Trả về bộ ba (step*, module*, error_type*).
    + Stage 3 — Iterative Debugging: sinh feedback gắn với error type, re-rollout NGAY TỪ bước critical (không restart từ đầu), lặp tối đa N=5, mỗi lần fail thì refine feedback cụ thể hơn.

Ý tưởng cốt lõi: sửa MỘT lỗi gốc thường đủ để lật một trajectory thất bại thành thành công — hiệu quả hơn nhiều so với cố sửa mọi lỗi bề mặt.

## 2. Điểm cần chú ý

- Error propagation là nút thắt chính: agent hiếm khi thất bại vì một bước khó, mà vì một sai lầm sớm bóp méo toàn bộ suy luận sau đó.
- Tập trung vào root-cause mới tạo ra cải thiện có ý nghĩa. Đây là phản bác trực tiếp với self-refine kiểu "duyệt lại toàn bộ và sửa mọi thứ trông sai".
- Thất bại chủ yếu là lỗi planning, không phải lỗi action.