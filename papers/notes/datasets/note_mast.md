# Why Do Multi-Agent LLM Systems Fail?

- Agentic systems dựa trên Mô hình Ngôn ngữ Lớn (LLM) thu hút sự chú ý đáng kể trong cộng đồng AI và được khai thác trong nhiều lĩnh vực khác nhau: kỹ nghệ phần mềm, khám phá thuốc, mô phỏng khoa học, và các tác tử đa dụng [5–11]. 

- Ngày càng được ứng dụng rộng rãi >< Mức cải thiện hiệu năng vẫn rất khiêm tốn so với các khung đơn tác tử [18] hoặc các baseline đơn giản như lấy mẫu best-of-N [19]
    + Tỷ lệ thất bại từ 41% đến 86,7% trên 7 hệ MAS SOTA mã nguồn mở
    + Chưa có sự đồng thuận rõ ràng nào về cách xây dựng MAS bền vững và đáng tin cậy
=> Vì sao MAS thất bại?

Contributions:
- benchmark MAST-Data: 1642 execution traces đã được gán nhãn thu thập các vết từ 7 khung MAS phổ biến chạy với hai họ mô hình chính (dòng GPT-4 và dòng Claude). 
- benchmark MAST-Data-human: 21 traces được gán nhãn bởi ba chuyên gia con người trong các nghiên cứu về mức đồng thuận giữa những người gán nhãn (inter-annotator agreement) 
- Failure taxonomy: 14 loại Failure của MAS thuộc 3 nhóm

## 1. Cách xây dựng datasets

- Bước 1: Xâu dựng Failure Taxonomy: 150 traces từ 5 MAS -> sáu experts -> phát hiện càng nhiều chế độ thất bại riêng biệt càng tốt, đồng thời đảm bảo các mẫu hình quan sát được không chỉ là đặc thù của một hệ thống đơn lẻ.

- Bước 2: Chuẩn hóa nhãn qua Inter-Annotator Agreement (IAA): biến quan sát định tính thành định nghĩa dùng chung được cho mọi MAS:
    + Vòng lặp IAA: 3 experts độc lập gán nhãn 5 traces chọn ngẫu nhiên → thảo luận giải quyết bất đồng → tinh chỉnh taxonomy (sửa định nghĩa / thêm mode / gộp - bỏ mode) → lặp lại.

- Bước 3: LLM-as-a-Judge Pipeline (mở rộng quy mô): gán nhãn thủ công 1600+ traces là bất khả thi về thời gian và chi phí:
    + Input cho annotator: execution trace + định nghĩa `MAST` + few-shot examples lấy từ dữ liệu human-annotated.
    + Mô hình: OpenAI o1

- Bước 4: Kiểm chứng generalization: trước khi chạy đại trà, xác nhận `MAST` + LLM annotator hoạt động ngoài phạm vi phát triển. Sau khi xác nhận, mở rộng thu thập → `MAST-Data`: **1642 traces / 7 MAS frameworks** = 5 (initial) + 2 (validation) + Manus.

Tổng kết các trường dữ liệu trong datasets (lấy từ hugging face):
- trace: full raw log của toàn bộ agent (dạng string phi cấu trúc)
- mast_annotation: {mã lỗi: 0/1 | 0 là không lỗi, 1 là có}

## 2. Điểm cần chú ý

- "Nếu không cho LLM biết trace có lỗi không thì nó có báo là không có failure mode không?":
    + Trace thành công vẫn có lỗi
    + Unaware of Termination Conditions và Information Withholding gần như chỉ xuất hiện trong trace thất bại
    + Verification xuất hiện thường xuyên ngay cả trong trace thành công

- Không có kiến trúc nào thắng toàn diện, là bài toán trade-off:
    + MetaGPT vs ChatDev cùng dùng GPT-4o trên ProgramDev, MetaGPT ít lỗi FC1 và FC2 hơn 60–68%, nhưng lại nhiều lỗi FC3 gấp 1.56 lần.
=> Do MetaGPT dựa vào SoP (Standard Operating Procedures) nên tuân thủ đặc tả tốt, còn ChatDev có hẳn phase testing/reviewing riêng nên verification tốt hơn

- Giới hạn tác giả tự nhận:
    + Không claim `MAST` bao phủ mọi failure pattern (chỉ là "foundational first step").
    + Có thừa nhận một phần lỗi đến từ giới hạn LLM nền (hallucination, instruction following), nhưng chủ ý tập trung vào các pattern mà system design có thể cải thiện được.

