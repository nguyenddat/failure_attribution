# MAST-Data: `mast_annotation` không dùng được cho phân tích framework-level

Nguồn: `data/error_categorization/mast/` (reload full-column từ
`mcemri/MAST-Data`, script `data/error_categorization/mast.py`), phân tích
bằng `data/error_categorization/analyze_mas_annotation_correlation.py` +
`plot_mast_*.py`. Bổ sung/thay thế phần "Data quality" trong
[[finding_1_framework_architecture]]. Bản tóm tắt cô đọng từng bước:
`docs/mast_data_analysis.md`.

## Kết luận cuối cùng

**Cột `mast_annotation` trong `MAD_full_dataset.json` (file chính của
`mcemri/MAST-Data` trên HuggingFace) là hàm của `trace_index` (số thứ tự
lần chạy trong hệ thống đó), không phải hàm của nội dung trace thật.**
Verify trên toàn bộ 1242 trace, 206 giá trị `trace_index`, 11 nhóm
(mas_name × benchmark_name × llm_name) khác nhau — **0 index nào có quá 1
pattern nhãn khác nhau**, dù nội dung log (`raw_trajectory`) 11 nhóm hoàn
toàn khác nhau (framework khác, model khác, task khác nhau hết). Verify cả
trên raw HuggingFace row (không qua loader của mình) để loại trừ do code.

=> **Không dùng được cột này để trả lời "framework nào lỗi nhiều/loại
gì"** — mọi khác biệt "theo framework" quan sát được chỉ phản ánh framework
đó có bao nhiêu trace và rơi vào range `trace_index` nào, không phải hành
vi thật. Mọi kết quả chi-square/pie/bar chart đã tính trong quá trình điều
tra (mục "Lịch sử phát hiện" bên dưới) đều **invalid**, giữ lại chỉ để
tham khảo quá trình debug.

**Vẫn dùng được**: `raw_trajectory` (log thô, đa số unique — xem mục
Duplicate trace), `mas_name`/`benchmark_name`/`llm_name`/`trace_id`
(metadata sạch). Muốn phân tích lỗi theo framework từ dataset này phải tự
đọc lại `raw_trajectory` và gán nhãn mới, không dùng `mast_annotation` có
sẵn.

## Duplicate trace (vấn đề riêng, không liên quan bug annotation)

62/1242 trace (5%) có `raw_trajectory` **trùng y hệt nội dung** với 1 trace
khác (hash SHA1 giống nhau):

- **Magentic: 30 cặp trùng.** Luôn là 1 trace `trace_id` thấp (0-29, batch
  gốc 30 trace human-annotated) trùng nội dung với 1 trace `trace_id` cao
  (30-136, batch mở rộng 165 trace LLM-annotated) — batch mở rộng vô tình
  chép lại nguyên 30 trace gốc, gắn `trace_id` mới. Magentic n=195 thực
  chất chỉ ~165 lần chạy khác nhau.
- **AppWorld: 1 cặp trùng** (`trace_id` 9 và 17, cùng trong batch 30 trace
  gốc).

Ảnh hưởng: nếu tính rate/proportion theo trace mà không khử trùng, Magentic
bị đếm lố ~15% số trace.

## Đã kiểm tra lối thoát — không có

Repo `mcemri/MAST-Data` chỉ có 2 file: `MAD_full_dataset.json` (bug trên)
và `MAD_human_labelled_dataset.json` (chỉ 19 trace, 3 annotator người
thật, nhưng dùng **taxonomy nháp khác** bản public 14 mã — không đủ lớn/
không tương thích để thay thế cho phân tích thống kê theo framework).

=> Muốn trả lời câu hỏi gốc ("MAST cho biết gì về phân bố lỗi theo
framework/kiến trúc") phải quay lại đọc `raw_trajectory` tự phân loại, hoặc
chuyển sang dataset khác chưa phát hiện bug tương tự (AEGIS, TRAIL).

---

## Lịch sử phát hiện (giữ tham khảo — kết quả bên dưới đều đã invalid)

### 1. Phát hiện ban đầu: tưởng chỉ 3 framework bị lỗi

**Hiện tượng ban đầu**: `mast_annotation` của AppWorld/HyperAgent/OpenManus
giống hệt nhau tuyệt đối theo từng `trace_index` (khớp suốt 30/30 trace
mỗi framework), dù `raw_trajectory` khác nhau hoàn toàn (0 trùng lặp hash),
chạy trên 3 benchmark + 2 model khác nhau. Ban đầu kết luận đây là bug cục
bộ 3 framework này — **sau đó phát hiện bug lan toàn dataset** (xem "Kết
luận cuối cùng" ở trên).

### 2. Confound: mas_name gần như 1-1 với benchmark_name/llm_name

6/7 framework chỉ chạy đúng **1 cặp (benchmark, model)** duy nhất (xem
`data/figures/mast_mas_name_x_benchmark_name.png`):

| mas_name | benchmark_name | llm_name |
|---|---|---|
| AppWorld | Test-C | GPT-4o |
| ChatDev | ProgramDev | GPT-4o |
| HyperAgent | SWE-Bench-Lite | Claude |
| Magentic | GAIA | GPT-4o |
| MetaGPT | ProgramDev (chính) | GPT-4o + Claude (trộn 2 model!) |
| OpenManus | ProgramDev | GPT-4o |
| AG2 | 4 benchmark khác nhau (GSM/MMLU/Olympiad...) | nhiều model |

Chỉ AG2 không bị confound nặng. Phát hiện phụ: so MetaGPT (n=230, trộn
GPT-4o+Claude) với ChatDev (n=130, GPT-4o) ban đầu tưởng khác biệt thật —
lọc lại đúng subset GPT-4o/ProgramDev cả 2 bên (n=130 mỗi bên) thì ra
**giống hệt nhau tuyệt đối cả 14 code** → chính là dấu hiệu dẫn tới phát
hiện bug lan toàn dataset ở mục 1.

### 3. Chi-square: mas_name x code, sau khi (tưởng là) loại bug — INVALID

Lúc đó tưởng loại AppWorld/HyperAgent/OpenManus là đủ, còn 4 framework
"thật" (AG2, ChatDev, Magentic, MetaGPT, n=1152), ra 2 code có vẻ ý nghĩa
thống kê (2.6 Reasoning-Action Mismatch p=7.8e-8, 3.1 Premature Termination
p=6.9e-5). **Kết quả này giờ biết là artifact** — do 4 framework "thật"
vẫn có range `trace_index` khác nhau, chi-square chỉ đang đo "framework
nào có range index nào", không phải hành vi lỗi thật.

### 4. Thử gộp theo kiến trúc (Hierarchical/Centralized/Decentralized) — INVALID

Gán nhãn kiến trúc: Hierarchical=ChatDev+MetaGPT, Centralized=Magentic-One,
Decentralized=AG2. Pie theo kiến trúc gần như giống pie theo framework gốc
(mỗi bucket chỉ 1-2 framework đại diện) — vốn dĩ đã không đủ mẫu để tách
bạch hiệu ứng kiến trúc; nay còn invalid thêm vì input (`mast_annotation`)
tự nó không đáng tin.

## File liên quan

- Loader (giữ đủ cột, không drop): `data/error_categorization/mast.py`
- Correlation + chi-square (kết quả invalid, giữ làm lịch sử): `data/error_categorization/analyze_mas_annotation_correlation.py`
- Confusion matrix mas_name x benchmark (vẫn valid, không liên quan annotation): `data/error_categorization/plot_mas_name_vs_benchmark.py`
- Bar/pie chart theo mast_annotation (đều invalid, giữ làm lịch sử): `plot_mast_failure_distribution.py`, `plot_mast_pie_by_error_type.py`, `plot_mast_pie_by_group.py`, `plot_mast_pie_by_architecture.py`
- Bug annotation + duplicate trace: constant `data/error_categorization/mast_known_issues.py` (hiện chỉ list 3 framework — **cần cập nhật thành "toàn dataset" hoặc xoá dùng, xem TODO**)
- TODO liên quan trong [[finding_1_framework_architecture]] (mục "Data quality") — cần update lại theo phát hiện mới này.

## TODO

- [ ] Update `finding_1_framework_architecture.md` mục Data quality: từ
      "3 framework bug" → "toàn bộ mast_annotation bug".
- [ ] Cân nhắc: xoá/deprecate 4 script `plot_mast_*` dựa trên
      `mast_annotation` (kết quả invalid), hoặc giữ nhưng thêm cảnh báo to
      ở đầu output.
- [ ] Nếu cần phân tích framework-level thật cho MAST: thiết kế lại pipeline
      tự đọc `raw_trajectory` + gán nhãn mới (LLM-judge riêng hoặc thủ
      công), không dùng `mast_annotation` gốc.
