- Một trajectory deep-research: một quá trình ra quyết định được ghi lại, chứ không phải một phép tính input–output
- Trajectory dần hình thành các claim (về thực thể, ràng buộc, nguồn, ứng viên trung gian, kết luận...) và các đoạn sau thường tái sử dụng claim trước đó như sự thật đã xác lập.

- Điểm yếu của các nghiên cứu:
    + outcome-level evaluation
    + Bước sai thường không phải là câu trả lời sai rõ ràng ở cuối, mà là một commitment sớm hơn mà các đoạn sau kế thừa mà không xác minh lại
    + Log thô (raw logs) tuy chứa đủ bằng chứng cần thiết nhưng dài, không đồng nhất, phụ thuộc đặc thù từng framework (MAST) ==> khó khai thác trực tiếp
    + Prompt LLM trực tiếp trên toàn bộ trajectory để tìm lỗi là không ổn định, vì mô hình có xu hướng: nhầm triệu chứng với root cause, quá tập trung vào câu trả lời cuối, bỏ sót cam kết thiếu căn cứ xuất hiện sớm nhưng lại định hình toàn bộ lời giải về sau.
==> Tập trung vào: tìm kiếm 1 claim thiếu căn cứ có hệ quả đối với các đoạn về sau

Contribution:
- Trajectory -> Semantic Spans
- Dataset: 2.790 trajectory đưa thành các Semantic Spans, nhãn = human experts + llm-as-a-judge
==> Dataset TELBENCH + DRIFT

## 2. Dataset Construction
B1: Nguồn dữ liệu: benchmark × model × framework
- 3 benchmark deep-research công khai:
    + GAIA-val
    + XBench
    + BrowseComp-test: downsample còn 200 task (tránh chiếm ưu thế corpus)
  ==> Tổng 465 task

- Với mỗi task, chạy tổ hợp:
    + 3 backbone model: GPT-5 × Gemini-2.5-Pro × Claude-Sonnet-4.5
    + 2 agent framework: MiroFlow × OAgent
==> Tổng: 465 task × 3 model × 2 framework = 2.790 trajectory (long-form agent trajectories)

B2: Chuẩn hoá log -> unified execution-unit sequences

B3: Span Segmentation — phân đoạn thành Semantic Span
- Mục tiêu: chuyển trajectory -> dãy các semantic span, mỗi span là 1 đoạn thực thi liên tục xoay quanh MỘT mục tiêu cục bộ nhất quán (planning, retrieval, verification, comparison, finalization)
- Tín hiệu dùng để điểm ngắt: thay đổi đối tượng tìm kiếm, tập ứng viên, time scope, verification criterion, reasoning objective
- Không cắt nếu: viết lại truy vấn, thử lại, thu thập thêm chứng cứ
- Kiểm soát chất lượng :
    + Các trường hợp bất thường được tự động gắn cờ (flag) (tác giả không đề cập chi tiết)
    + LLM hỗ trợ rà soát thêm, nhưng override ranh giới cuối cùng chỉ thực hiện sau khi con người kiểm tra
==> Kết quả: trung bình 11,95 semantic span / trajectory (trên tập Verified-1K)

B4: Gán nhãn lỗi: {span_id: 0/1}:
- đưa ra, dựa vào, khuếch đại, hoặc chốt lại một nhận định sai,  thiếu căn cứ, mâu thuẫn, hoặc chốt sớm (prematurely committed)
==> và ẢNH HƯỞNG đến answer path
- KHÔNG tự động là lỗi: khám phá bình thường, tìm kiếm thất bại, giả thuyết tạm thời, lỗi đã được khắc phục, nhiễu công cụ
- llm đưa ra annotatations + reason + recall -> experts thẩm định

B5: Lọc & xác minh → Verified-1K
- Từ 2.790 trajectory, có 1.890 trajectory chứa ≥1 span lỗi (67,7%)
==> 1.000 instance đã xác minh (Verified-1K)
- Chia độ khó: 600 easy / 400 hard (theo độ phức tạp trajectory + độ tinh vi của lỗi)
    + Easy: bằng chứng trực tiếp hơn, trajectory ngắn hơn, ít span gây nhiễu
    + Hard: trajectory dài hơn, lỗi thưa/ẩn hơn, nhiều exploration lành tính gây 
      nhiễu, pattern khó (evidence overclaim, constraint miss, candidate confusion)
B6: Gán nhãn cơ chế (Mechanism Labels) — chỉ phục vụ phân tích, KHÔNG dùng khi đánh giá mô hình
- Operation-stage (8 giai đoạn, mọi span đều có): planning, retrieval, source verification, extraction, computation, decision-making, recovery, finalization
- Primary-fault (18 loại lỗi chính / 6 fault family: Constraint Handling, Search 
  and Retrieval, Evidence Grounding, Entity Mapping, Information Processing, 
  Process Control) — chỉ error span mới có nhãn này

## 3. DRIFT

DRIFT về bản chất là một multi-agent prompting framework, luồng dữ liệu qua 4 bước:

1. Claim Keeper (LLM)
   Input:  Question + Spans (T = s1...sn)
   Output: Claim Ledger L = {ck}, ck = (a_k, i_k, b_k, U_k, τ_k, σ_k)
           - a_k: nội dung claim
           - i_k: span đưa ra claim
           - b_k: span đầu tiên claim có hệ quả
           - U_k: tập span tái sử dụng claim
           - τ_k: loại claim
           - σ_k: trạng thái (exploratory/tentative/consequential/finalized)
   ==> tương đương: span_id -> [claim_id, claim_id, ...]

2. Support Seeker (LLM)
   Input:  L + Spans
   Output: mỗi claim consequential -> 1 trong 4 nhãn
           {claim_id: DIRECT | WEAK | MISSING | CONFLICTING}
           + span nào cung cấp/không cung cấp bằng chứng

3. Specialist Auditors (LLM, skill-routed theo loại claim)
   Input:  claim + support status
   Output: kiểm tra chuyên biệt theo type: entity / constraint / evidence / 
           retrieval / compute / process

4. Dependency Tracer (LLM)
   Input:  Claim Ledger + Support records
   Output: {span_id: error | non-error}
           span lỗi = span commit/reuse/amplify/finalize 1 claim 
           thiếu căn cứ (WEAK/MISSING) hoặc mâu thuẫn (CONFLICTING)
==> Final: Ê = {sj ∈ T | h(sj) = 1}

## 4. Experiment setup
- Bare LLM
- Codex/ Claude harness + 5 backbone model
- DRIFT
==> chạy 3 lần + dataset Verified-1K

Metrics
- Macro Precision (P), Macro Recall (R), Macro F1: đánh giá ở mức span-level
- First-Error Accuracy (FEA): đánh giá khả năng phát hiện ĐÚNG span lỗi 
  xuất hiện SỚM NHẤT (earliest predicted error)

## 5. Findings đáng chú ý
- DRIFT vượt trội mọi baseline, ở mọi backbone
==> Mức cải thiện: lên tới ~30 điểm % F1 và first-error accuracy so với Bare

- "Bọc" LLM trong agentic workflow phức tạp KHÔNG đủ
- First-error localization vẫn RẤT khó (dù DRIFT đã cải thiện mạnh)
==> "phát hiện được VÙNG có lỗi" và "xác định CHÍNH XÁC lỗi bắt đầu từ đâu" là 2 năng lực chẩn đoán khác nhau
==> Có thể nhận ra trajectory "không đáng tin cậy" nhưng vẫn khó định vị đúng span lỗi ĐẦU TIÊN giữa chuỗi dài các bước tìm kiếm/xác  minh/suy luận trung gian

- Scale mô hình lớn hơn KHÔNG đảm bảo chẩn đoán tốt hơn
- Ablation: mức tăng lớn nhất đến từ Claim Keeper
- Độ phức tạp span càng cao, lợi thế của DRIFT càng rõ

## 6. Limitations (nhận xét thêm, paper không đề cập rõ)

- Việc xây dựng Semantic Span phụ thuộc nhiều vào log có đầy đủ thông tin hay không (tool call, intermediate reasoning...). Nếu chỉ có output log (kiểu Who&When) thì KHÔNG đủ để xây span.
==> Nhờ điểm yếu này lại thấy rõ 2 nghiên cứu TraceElephant và TELBENCH hỗ trợ nhau rất mạnh (dùng chung yêu cầu log đầy đủ, bổ trợ nhau về hướng attribution/localization)