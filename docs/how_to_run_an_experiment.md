# Cách thực hiện 1 experiment (quy trình chung)

Rút ra từ Experiment 1 (`experiments/1.framework_environment_correlation/`).
Áp dụng cho mọi experiment kiểm định assumption bằng data thật trong repo
này. Xem `docs/experiment_1_walkthrough.md` cho ví dụ cụ thể từng bước.

## 1. Xuất phát từ 1 assumption/câu hỏi cụ thể, có nguồn

Không bắt đầu từ "phân tích dataset X" chung chung. Assumption phải:
- Trích được từ đâu đó cụ thể (vd. mục trong `papers/dataset_findings.md`,
  claim của 1 paper, hoặc câu hỏi user đặt ra rõ ràng).
- Diễn đạt được thành 1 câu kiểm định được (có thể đúng/sai/không đủ bằng
  chứng), không phải câu hỏi mở.

Ví dụ Experiment 1: "framework/kiến trúc quyết định phân bố loại lỗi agent
gặp phải" — trích từ `papers/dataset_findings.md` mục #1, lúc đó đánh giá
"Đồng thuận" chỉ dựa đọc paper, chưa verify bằng data thật.

## 2. Chọn dataset phù hợp — cần field gì

Chỉ chọn dataset có đủ 2 loại field để kiểm định assumption:
- Biến độc lập cần đo (vd. `framework`/`system_name`/`environment`).
- Nhãn lỗi có cấu trúc (không phải free-text) để thống kê được (vd. mã lỗi
  cố định, category, hay ít nhất categorical field).

Loại bỏ dataset không có 1 trong 2 (vd. biến độc lập cố định thay vì thay
đổi, hoặc nhãn chỉ là free-text không gom nhóm được).

## 3. Đảm bảo dataset load đủ cột — không drop bất cứ gì

Trước khi phân tích, luôn kiểm tra loader hiện có có drop cột nào không
(đọc docstring `schemas/*.py`). Nếu có, reload lại:
1. Đọc raw row/schema gốc (HuggingFace `datasets`/`load_dataset` hoặc
   `pd.read_parquet`) — liệt kê **toàn bộ** field, kể cả field trông có vẻ
   redundant. Không đoán từ tài liệu paper, đọc trực tiếp dữ liệu.
2. Viết lại `schemas/<dataset>.py` (Pydantic model) mirror đúng raw shape.
3. Viết lại loader trong `data/<taxonomy>/<subcategory>/<dataset>.py`,
   giữ nguyên convention path đã có (`data/error_localization/single_fault/`,
   `data/error_localization/multi_fault/`, `data/error_categorization/`).
4. Thêm Makefile target `load_<dataset>` nếu chưa có.

Loader **ở lại `data/`**, dùng chung cho mọi experiment sau — không di
chuyển vào folder experiment.

## 4. Kiểm tra data quality TRƯỚC khi tin bất kỳ kết quả nào

Bước hay bị bỏ qua nhất nhưng quan trọng nhất. Luôn làm trước khi phân
tích thống kê:

- **Confound check**: biến độc lập chính (framework) có bị trộn 1-1 với
  biến khác (benchmark/model) không? Nếu có, mọi kết luận "do framework"
  có thể thực ra là "do benchmark/model". Vẽ confusion matrix
  (`pd.crosstab` + heatmap) trước tiên.
- **Tìm pattern bất thường**: nhãn có giống hệt nhau bất thường giữa các
  nhóm khác biệt hoàn toàn không (khác framework/model/benchmark)? Nếu có
  — verify bằng hash nội dung (loại trừ trùng lặp data thật) + so khớp
  theo field định vị (index/id) để xác định có phải bug gán nhãn theo vị
  trí thay vì nội dung không. Luôn verify lại trên **raw data, không qua
  loader của mình** để loại trừ bug do code.
- **Metadata tự khai báo có đáng tin không**: field kiểu `num_agents`,
  `num_steps` có khớp con số đếm thực tế từ list/array không? Nếu lệch
  nhiều — không dùng field đó, luôn tính lại từ dữ liệu thô.

Nếu phát hiện bug nghiêm trọng (data không phản ánh nội dung thật) — dừng
lại, note rõ, không tiếp tục phân tích thống kê trên field đó cho tới khi
có giải pháp (dataset khác, hoặc tự gán nhãn lại).

## 5. Phân tích thống kê — luôn kèm effect size, không chỉ p-value

- Chi-square test of independence cho biến categorical, **luôn tính kèm
  Cramér's V** (effect size). p<0.05 với cỡ mẫu lớn (hàng nghìn+) gần như
  luôn có ý nghĩa dù effect cực nhỏ — chỉ p-value dễ đánh lừa.
- Ngưỡng tham khảo Cramér's V (quy ước Cohen): <0.1 không đáng kể, 0.1-0.3
  nhỏ, 0.3-0.5 trung bình, >0.5 lớn.
- Nếu gộp nhóm nhỏ (vd. gộp framework theo kiến trúc) để tăng cỡ mẫu, kiểm
  tra xem mỗi bucket có đủ nhiều framework đại diện không — bucket chỉ có
  1 framework thì kết luận "do kiến trúc" thực ra chỉ là "do framework đó".

## 6. Cấu trúc thư mục experiment

```
experiments/N.<ten_ngan_mo_ta>/
  src/            # script phân tích (không phải loader)
  results/
    figures/      # .png
    tables/       # .csv
  finding_notes.md
```

- Số thứ tự `N.` tăng dần theo experiment. **Tên folder bắt đầu bằng số
  không phải Python package hợp lệ** — không dùng `python -m` được. Mọi
  script chạy trực tiếp: `python experiments/N.xxx/src/foo.py`.
- Script cần import `schemas.*` hoặc module dùng chung khác trong repo:
  bootstrap `sys.path` tới repo root ở đầu file:
  ```python
  REPO_ROOT = Path(__file__).resolve().parents[3]  # src/ -> N.xxx/ -> experiments/ -> root
  sys.path.insert(0, str(REPO_ROOT))
  ```
- Đường dẫn đọc data: tuyệt đối từ `REPO_ROOT / "data/..."`, không dùng
  `Path(__file__).resolve().parent / "..."` (sai từ khi script không còn
  nằm cạnh thư mục data output của loader).
- Đường dẫn ghi kết quả: `Path(__file__).resolve().parent.parent / "results" / "figures"`
  hoặc `.../ "tables"`.
- Script phụ thuộc lẫn nhau trong cùng experiment (constant, hàm dùng
  chung) — import trực tiếp theo tên file (`from mast_known_issues import
  ...`), không qua đường dẫn package, vì khi chạy trực tiếp Python tự thêm
  thư mục chứa script vào `sys.path`.

## 7. finding_notes.md — tóm tắt cô đọng, không phải toàn bộ chi tiết

Cấu trúc gợi ý:
1. Assumption kiểm định + nguồn.
2. Dataset dùng + lý do chọn/loại.
3. 1 mục/dataset: phát hiện chính, có ý nghĩa gì, script liên quan. Link
   ra file finding chi tiết trong `papers/notes/findings/` nếu có (giữ
   toàn bộ số liệu/quá trình điều tra ở đó, không lặp lại ở đây).
4. Bảng so sánh claim gốc vs kết quả verify.
5. Kết luận về assumption (giữ nguyên / hạ mức / bác bỏ).
6. TODO mang theo từ các file finding gốc.

## 8. Trước khi báo cáo xong — review lại toàn bộ results

Sau khi thêm/sửa script, luôn xem lại **toàn bộ** `results/figures/` +
`results/tables/` (không chỉ file vừa tạo) để kiểm tra finding_notes.md có
đang thiếu/lỗi thời so với kết quả thật không — nhất là khi thêm phân
tích mới (vd. gộp theo kiến trúc) có thể làm mạnh hơn hoặc không thêm gì
so với kết luận cũ, cần nói rõ cái nào.
