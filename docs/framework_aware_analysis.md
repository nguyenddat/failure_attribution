# MAST-Data: quy trình phân tích mas_name x mast_annotation

Tóm tắt cô đọng từng bước đã thực hiện trong phiên phân tích MAST-Data
(`mcemri/MAST-Data`). Kết luận chi tiết + số liệu đầy đủ:
[papers/notes/findings/mast_data_finding.md](../papers/notes/findings/mast_data_finding.md).

**Kết quả cuối**: cột `mast_annotation` của dataset gốc bị lỗi ở nguồn
(index-keyed, không gắn với nội dung trace) — không dùng được để phân tích
"framework nào lỗi gì". Các bước dưới đây là quy trình dẫn tới phát hiện
đó, giữ lại làm tài liệu tham khảo cho ai muốn tái hiện hoặc audit lại.

## Bước 1 — Reload dữ liệu, giữ đủ cột gốc

Raw HuggingFace row có 6 cột: `mas_name`, `llm_name`, `benchmark_name`,
`trace_id`, `trace` (`key`/`index`/`trajectory`), `mast_annotation` (dict
14 mã lỗi → 0/1). Loader cũ (`schemas/mast.py` bản đầu) chỉ giữ
`mas_name` + `raw_trajectory` + `faults` (list mã lỗi=1), bỏ hết phần còn
lại.

Viết lại loader giữ đủ mọi cột gốc + field `faults` derive để tương thích
ngược:

```
python -m data.error_categorization.mast
```

→ ghi 1242 file JSON vào `data/error_categorization/mast/`.

## Bước 2 — Kiểm tra confound mas_name x benchmark_name x llm_name

```
python -m data.error_categorization.plot_mas_name_vs_benchmark
```

Kết quả: 6/7 framework chỉ chạy đúng 1 cặp (benchmark, model) duy nhất —
`mas_name` gần như 1-1 với benchmark/model, không tách được hiệu ứng
framework khỏi hiệu ứng task/model. Chỉ AG2 trải nhiều benchmark + model.
→ Ảnh: `data/figures/mast_mas_name_x_benchmark_name.png`.

## Bước 3 — Chi-square: mas_name x từng mã lỗi

```
python -m data.error_categorization.analyze_mas_annotation_correlation
```

Test độc lập `mas_name x code` (contingency table + Cramér's V) cho từng
14 mã. Trên full 7 framework: 6/14 mã "có ý nghĩa thống kê" (p<0.05).

## Bước 4 — Trực quan hoá

3 kiểu chart, đều dựa trên `mast_annotation`:

```
python -m data.error_categorization.plot_mast_failure_distribution   # bar chart, rate/100 trace
python -m data.error_categorization.plot_mast_pie_by_error_type      # pie 14 mã / framework
python -m data.error_categorization.plot_mast_pie_by_group           # pie 3 nhóm lỗi / framework
```

## Bước 5 — Phát hiện bất thường: 3 framework nhãn giống hệt nhau

Quan sát: AppWorld, HyperAgent, OpenManus (3 framework baseline single-
agent trong MAST) có rate mỗi mã lỗi **giống hệt nhau tuyệt đối** dù chạy
3 benchmark + 2 model khác nhau. Verify bằng hash `raw_trajectory` (0 trùng
lặp nội dung) + so khớp theo `trace_index` (khớp 100% cả 30 trace/framework)
→ xác nhận là bug gán nhãn, không phải trùng hợp. Loại 3 framework này,
chi-square chạy lại chỉ còn 2/14 mã có ý nghĩa (2.6, 3.1).

## Bước 6 — Mở rộng kiểm tra: bug lan toàn dataset, không chỉ 3 framework

Nghi vấn tiếp: so MetaGPT (lọc đúng subset GPT-4o/ProgramDev, n=130) với
ChatDev (GPT-4o/ProgramDev, n=130) — 2 framework "thật", không thuộc nhóm
nghi vấn ban đầu — vẫn ra **giống hệt nhau tuyệt đối cả 14 mã**.

Kiểm tra toàn diện: với mọi giá trị `trace_index` (0-205) và mọi nhóm
(mas_name × benchmark_name × llm_name, 11 nhóm), đếm số pattern nhãn khác
nhau xuất hiện ở mỗi index. Kết quả: **0/206 index có hơn 1 pattern** — mọi
nhóm cùng `trace_index` luôn nhận đúng 1 nhãn giống nhau, bất kể framework/
benchmark/model. Verify lại trên raw HuggingFace file (không qua loader
của mình) để loại trừ bug do code — kết quả giống hệt.

→ **Kết luận: `mast_annotation` là hàm của `trace_index`, không phải hàm
của nội dung trace.** Bug ở nguồn dữ liệu HuggingFace, ảnh hưởng toàn bộ
1242 trace, không riêng 3 framework ban đầu.

## Bước 7 — Phát hiện phụ: duplicate trace (Magentic, AppWorld)

Hash toàn bộ `raw_trajectory`: 1211/1242 hash unique, 62 trace (31 cặp)
trùng nội dung — 30 cặp thuộc Magentic (trace_id thấp 0-29 trùng trace_id
cao 30-136, tức batch mở rộng chép lại batch gốc), 1 cặp thuộc AppWorld.

## Bước 8 — Kiểm tra lối thoát

`mcemri/MAST-Data` chỉ có 2 file: `MAD_full_dataset.json` (bug trên) và
`MAD_human_labelled_dataset.json` (19 trace, taxonomy nháp khác bản public,
quá nhỏ để thay thế). Không có version sạch để dùng thay.

## Kết luận + khuyến nghị

1. Không dùng `mast_annotation` cho bất kỳ phân tích framework-level nào
   trên MAST-Data — mọi kết quả từ Bước 2-6 (trừ Bước 2, không phụ thuộc
   annotation) coi như invalid, giữ làm tài liệu debug.
2. `raw_trajectory` + metadata (`mas_name`/`benchmark_name`/`llm_name`/
   `trace_id`) vẫn dùng được cho mục đích khác (đọc case study, đo độ dài
   trace, hoặc tự gán nhãn lại từ đầu).
3. Muốn trả lời câu hỏi "phân bố lỗi theo framework/kiến trúc" — chuyển
   sang dataset khác chưa phát hiện bug tương tự (AEGIS, TRAIL), hoặc tự
   xây pipeline gán nhãn mới trên `raw_trajectory` của MAST.

## Script liên quan

| Script | Vai trò | Trạng thái kết quả |
|---|---|---|
| `data/error_categorization/mast.py` | Loader, giữ đủ cột | OK |
| `data/error_categorization/mast_known_issues.py` | Constant loại trừ framework bug | Cần update (bug rộng hơn phạm vi hiện khai báo) |
| `plot_mas_name_vs_benchmark.py` | Confusion matrix mas_name x benchmark | OK, không phụ thuộc annotation |
| `analyze_mas_annotation_correlation.py` | Chi-square mas_name x code | Invalid |
| `plot_mast_failure_distribution.py` | Bar chart rate/framework | Invalid |
| `plot_mast_pie_by_error_type.py` | Pie 14 mã/framework | Invalid |
| `plot_mast_pie_by_group.py` | Pie 3 nhóm/framework | Invalid |
| `plot_mast_pie_by_architecture.py` | Pie 3 nhóm/kiến trúc | Invalid |
