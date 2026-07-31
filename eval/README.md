# Bộ câu thử nghiệm (Eval Set)

**Tổng số câu trong bộ thử nghiệm: 22**

File: [`eval_set.json`](eval_set.json)

## Mô tả

Đây là bộ câu hỏi nhóm tự nghĩ ra để kiểm thử điểm mà AI trong sản phẩm phải
"ra quyết định": **dataset nào thực sự liên quan đến yêu cầu của người dùng để
xếp hạng lên đầu, và khi nào phải hỏi lại thay vì tự bịa câu trả lời** (xem
`back-end/pipeline/rank_candidates.py` và `back-end/pipeline/parse_task.py`).

Mỗi câu trong `eval_set.json` gồm:

- `input` — đưa vào gì (câu truy vấn người dùng nhập).
- `must_answer` — sản phẩm PHẢI trả lời/hành xử thế nào. Đây là tiêu chí đánh
  giá đúng/sai, không phải câu trả lời mẫu để so khớp y nguyên.
- `category` — nhóm hành vi đang kiểm thử.
- `grounded_in` — đoạn code/prompt quyết định hành vi đó, để việc chấm điểm
  không dựa trên suy đoán.

## Các nhóm hành vi (category)

| Category | Số câu | Hành vi bắt buộc |
|---|---|---|
| `subject_relevance` | 5 | Dataset đúng chủ đề phải lên đầu; dataset lạc chủ đề (dù trùng từ khóa) phải bị loại/xếp thấp. |
| `clarification_required` | 5 | Câu hỏi mơ hồ/thiếu thông tin → hệ thống PHẢI hỏi lại, KHÔNG được tự đoán rồi trả danh sách dataset. |
| `off_topic` | 2 | Câu hỏi ngoài phạm vi tìm dataset → không được bịa kết quả hoặc thực hiện yêu cầu ngoài phạm vi. |
| `constraint_filtering` | 3 | Ràng buộc rõ ràng (license, ngôn ngữ, định dạng) → kết quả không thỏa phải bị loại, không chỉ ghi chú suông. |
| `source_availability` | 1 | Khi một nguồn (Kaggle) thiếu credential, hệ thống phải báo lỗi rõ ràng thay vì im lặng trả về rỗng. |
| `paraphrase_consistency` | 2 | Hai cách hỏi (Việt/Anh) cùng ý định phải cho kết quả nhất quán. |
| `multi_turn` | 1 | Sau khi người dùng trả lời câu hỏi làm rõ, hệ thống không được hỏi lại vòng lặp. |
| `llm_outage_fallback` | 1 | Khi provider LLM lỗi/hết quota, hệ thống phải fallback sang heuristic, không crash. |
| `sensitive_domain_flagging` | 2 | Dữ liệu nhạy cảm (quân sự, trẻ em, sinh trắc học...) phải được gắn cảnh báo, không lờ đi. |

## Cách dùng

Bộ này dùng để **chấm tay hoặc bán tự động**: chạy từng `input` qua sản phẩm
(qua UI hoặc `POST /api/v1/search`), rồi đối chiếu kết quả thực tế với
`must_answer`. Kết quả PASS/FAIL nên được ghi lại kèm theo run log
(`back-end/audit_log.py`) để so sánh giữa các lần thay đổi model/prompt.

Bộ này bổ sung cho `back-end/tests/golden_queries.json` (50 câu, tự động hoá
qua `back-end/tests/run_eval.py`, tập trung sâu vào chất lượng subject overlap
khi ranking) — ở đây bao quát rộng hơn các loại quyết định khác của AI
(clarification, constraint, sensitive domain, fallback).
