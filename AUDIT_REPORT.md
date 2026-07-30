# Báo cáo audit DataScout AI

Ngày kiểm tra: 2026-07-31

## Kết quả đã xác nhận

- Python compile thành công cho toàn bộ backend.
- 85/85 automated test pass, gồm API contract, pipeline, LLM router,
  prompt guardrail, audit log, golden harness và frontend interaction contract.
- Golden end-to-end smoke test số 1 pass, trả 9 kết quả verified.
- Không còn button tĩnh/không có action contract trong HTML.

## Lỗi đã sửa

### Frontend

- Nút `Search` trong composer trước đây không có handler. Nay nút này bật/tắt
  `Catch-all web` và phản ánh trạng thái bằng `aria-pressed`.
- `Dataset library` và `Search history` trước đây không có hành vi. Nay chúng
  đọc dữ liệu tìm kiếm thực được lưu trong `localStorage`.
- Ba mục lịch sử mẫu đã bị bỏ. Lịch sử nay được tạo từ request thành công,
  có thể mở lại và xóa.
- Thẻ model, menu `Dataset Research`, nút Tools và Settings đều mở dialog cấu hình.
- `New research` reset state ngay tại client thay vì reload toàn bộ trang.
- Thêm test chống tái xuất hiện button ảo, kiểm tra URL protocol và escaping.

### AI Agent

- Heuristic fallback trước đây hiểu sai `human/person/pedestrian detection`
  thành subject `general`, khiến câu rõ ràng bị hỏi lại hoặc tìm sai.
- Bổ sung nhận diện subject người, người đi bộ, khuôn mặt, văn bản/OCR và chữ viết tay.
- Bổ sung nhận diện các cách diễn đạt detection tiếng Việt và tiếng Anh.
- Các truy vấn registry trước đây chạy tuần tự theo `source × keyword`, có thể
  treo nhiều phút. Nay chạy song song có giới hạn tối đa 8 worker và loại keyword trùng.

## Giới hạn còn lại

- Chưa thể khẳng định golden accuracy đạt 80% trên đủ 50 case. Một case live đã
  pass; chạy cả 50 case phụ thuộc API ngoài, tốn thời gian và quota đáng kể.
- Live smoke test vẫn mất khoảng 84 giây. Phần lớn thời gian còn lại nằm ở
  enrichment và nhiều batch LLM ranking, không còn ở chuỗi registry tuần tự.
- Tailwind và Phosphor Icons đang tải từ CDN, phù hợp prototype nhưng production
  nên build và pin dependency cục bộ.
- Test frontend hiện là contract/static test, chưa phải browser E2E bằng Playwright.
