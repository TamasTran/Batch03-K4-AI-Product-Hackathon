# Prompt cho Codex — Data Finder Chatbot (MVP Hackathon)

## Bối cảnh dự án

Xây một chatbot giúp các team trong lab AI tìm dataset phù hợp cho đề tài được giao.
Pain point thật: khi được giao đề tài, team phải tự tìm data nhưng không biết bắt đầu từ
đâu, mất nhiều thời gian, và nhiều khi tìm sai nguồn hoặc bỏ cuộc. Input là mô tả đề tài
bằng text tự nhiên (tiếng Việt hoặc tiếng Anh). Output là danh sách dataset thật, có xếp
hạng độ phù hợp, kèm giải thích rõ ràng và cảnh báo nếu dữ liệu khó tiếp cận hoặc nhạy cảm.

Thời gian: MVP phải "bấm được" (chạy thật, có UI) trong 1-2 ngày. Ưu tiên độ tin cậy và
có kết quả chạy được sớm hơn là tính năng đầy đủ.

## Nguyên tắc bắt buộc, không được vi phạm

1. **Không bao giờ để LLM tự đề xuất tên dataset cụ thể từ trí nhớ (kiến thức huấn luyện)
   mà không verify qua API/search thật.** LLM chỉ được dùng để: (a) hiểu ý định người dùng,
   (b) sinh từ khóa tìm kiếm, (c) xếp hạng và giải thích trên danh sách candidate ĐÃ CÓ THẬT
   từ API. Mọi dataset hiển thị cho user phải có nguồn gốc từ một lệnh gọi API/search thực
   tế trong phiên làm việc đó, không phải suy ra từ độ quen thuộc của tên dataset.
2. Ngoại lệ duy nhất: với các domain hẹp/khó (3D, robotics, y sinh chuyên sâu...) nơi các
   API tổng quát (Hugging Face, Kaggle, Papers with Code) không phủ tới, LLM được phép gợi ý
   **tên registry/tổ chức chuyên ngành đã biết** (ví dụ: ShapeNet, PhysioNet, TCIA, Open
   X-Embodiment) như một gợi ý "nơi để tìm tiếp", nhưng phải gắn nhãn rõ là "gợi ý chưa
   verify" và khuyến khích user tự kiểm tra, KHÔNG được trình bày như kết quả đã xác nhận.
3. Mọi thông tin về license, quy mô, số lượt tải phải lấy trực tiếp từ response API, không
   được LLM tự bịa hoặc suy diễn nếu API không trả về trường đó (để trống hoặc ghi "không rõ").

## Kiến trúc pipeline (4 bước)

### Bước 1 — Parse ý định (LLM call)
Input: mô tả đề tài dạng text tự do.
Output: JSON có cấu trúc gồm:
- `task_type` (vd: text classification, object detection, NER...)
- `domain` (NLP, computer vision, tabular, audio, 3D, y sinh...)
- `modality` (text/image/audio/video/tabular/3d)
- `language` (vietnamese/english/multilingual/any)
- `search_keywords_en`: 2-4 từ khóa tiếng Anh ngắn để search (hầu hết dataset registry index
  bằng tiếng Anh)
- `needs_labels`: boolean
- `is_narrow_domain`: boolean — true nếu domain thuộc nhóm khó tìm qua nguồn tổng quát (3D,
  robotics, y sinh chuyên sâu, geospatial, audio hiếm...)
- `is_sensitive_domain`: boolean + `sensitive_reason` — true nếu đề tài rơi vào nhóm dữ liệu
  nhạy cảm/cần bảo mật: y tế/y sinh, sinh trắc học (khuôn mặt, vân tay, giọng nói định danh),
  tài chính/giao dịch, dữ liệu vị trí/hành vi cá nhân, quân sự/an ninh, dữ liệu độc quyền
  doanh nghiệp, dữ liệu liên quan trẻ em

### Bước 2 — Tìm kiếm đa nguồn (kiến trúc "source registry", dễ mở rộng)
Thiết kế mỗi nguồn là 1 hàm độc lập, cùng trả về schema chuẩn:
```
{ "id": str, "url": str, "source": str, "downloads": int, "likes": int,
  "tags": list[str], "license": str|None, "raw_metadata": dict }
```
Nguồn cần implement cho MVP, theo thứ tự ưu tiên:
1. **Hugging Face Datasets Hub API** (`https://huggingface.co/api/datasets?search=...`) —
   không cần key, ưu tiên implement đầu tiên vì đây là nguồn chính cho Ngày 1.
2. **Kaggle API** (`https://www.kaggle.com/api/v1/datasets/list`) — cần username + API key
   (user tự đăng ký, nhập vào UI hoặc biến môi trường), thêm ở Ngày 2.
3. **Papers with Code API** (`https://paperswithcode.com/api/v1/datasets/`) — không cần key,
   tốt cho dataset gắn với benchmark/task cụ thể, thêm ở Ngày 2 nếu kịp.

Yêu cầu kỹ thuật:
- Gộp kết quả từ nhiều từ khóa, loại trùng theo `(source, id)`.
- Mỗi nguồn bật/tắt độc lập qua config, để dễ thêm nguồn mới sau này mà không sửa logic core.
- Xử lý lỗi mạng/timeout riêng cho từng nguồn (một nguồn lỗi không được làm sập cả pipeline).

### Bước 3 — Xếp hạng đa tiêu chí (LLM call, dựa trên candidate thật)
Không chấm 1 điểm tổng duy nhất. Với mỗi candidate, LLM trả về JSON:
```
{
  "id": str,
  "task_match": 1-5,        // đúng loại nhãn/tác vụ không
  "domain_fit": 1-5,        // đúng lĩnh vực, ít domain gap không
  "label_overlap": 1-5,     // nhãn có bao phủ nhu cầu bài toán không
  "size_adequacy": 1-5,     // quy mô phù hợp thời gian hackathon không
  "access_type": "open" | "registration" | "paid" | "restricted",
  "reasoning": str          // 1 câu giải thích ngắn gọn
}
```
Tính điểm tổng hợp có trọng số (gợi ý: task_match và domain_fit trọng số cao nhất), nhưng
LUÔN hiển thị điểm từng tiêu chí cho user, không chỉ điểm tổng — để user hiểu vì sao dataset
được xếp hạng như vậy (vd: "domain_fit cao nhưng label_overlap thấp" nghĩa là đúng lĩnh vực
nhưng phải tự gán nhãn lại).

Sắp xếp kết quả: ưu tiên `access_type == "open"` lên đầu trước khi sort theo điểm phù hợp,
vì với hackathon 1-2 ngày, dataset cần đăng ký/trả phí thường không kịp dùng.

Ràng buộc quan trọng trong prompt gửi LLM ở bước này: chỉ được chấm điểm cho các `id` có
trong danh sách candidate đưa vào, không được thêm id nào khác.

### Bước 4 — Gợi ý thay thế khi không có dataset mở phù hợp
Nếu sau Bước 3 không có candidate nào đạt `access_type == "open"` với điểm tổng hợp trên
ngưỡng chấp nhận được (gợi ý ngưỡng: điểm tổng hợp >= 3.5/5), tự động trả thêm khối gợi ý
chiến lược thay thế, không chỉ báo "không tìm thấy":
- Synthetic data / render giả lập (Blender, Unity Perception, NVIDIA Omniverse Replicator)
  cho các bài toán có thể mô phỏng được
- Transfer learning từ model pretrained, fine-tune trên tập nhỏ tự thu thập
- Data augmentation mạnh trên tập nhỏ có sẵn
- Weak supervision / self-supervised dùng dữ liệu không nhãn kết hợp ít nhãn thủ công
- Tìm bản "mini/sample" công khai của benchmark lớn bị hạn chế

Nếu `is_sensitive_domain == true`, thêm cảnh báo riêng: nêu rõ đây là domain có ràng buộc
pháp lý/đạo đức (không chỉ "khó tìm"), ưu tiên gợi ý các dataset đã de-identify chính thức
cho mục đích nghiên cứu (vd MIMIC-III/IV, TCIA cho y tế), và nhắc user kiểm tra yêu cầu
ethics approval/IRB nếu lab có quy định.

Nếu `is_narrow_domain == true` và nguồn tổng quát không ra kết quả tốt, LLM được phép gợi ý
tên registry chuyên ngành (theo nguyên tắc bắt buộc #2 ở trên), kèm rõ ràng nhãn "chưa
verify — vui lòng tự kiểm tra link và tính khả dụng".

## Tech stack và cấu trúc code

- Backend: Python
- LLM: Anthropic API, model `claude-sonnet-4-6`, dùng SDK `anthropic`
- HTTP calls tới các nguồn dữ liệu: `requests`
- UI: Streamlit (ưu tiên vì dựng nhanh, có UI bấm được ngay trong vài giờ)
- Cấu trúc file gợi ý:
  ```
  app.py                  # Streamlit UI + orchestration
  sources/
    huggingface.py         # search_hf_datasets()
    kaggle.py               # search_kaggle_datasets()
    paperswithcode.py       # search_paperswithcode_datasets()
  pipeline/
    parse_task.py           # Bước 1
    rank_candidates.py      # Bước 3
    fallback_suggestions.py # Bước 4
  README.md
  ```
  (Với MVP ngày 1, có thể gộp hết vào 1 file `app.py` cho nhanh, tách file là việc của
  ngày 2 nếu có thời gian refactor.)

## Kế hoạch triển khai theo mốc thời gian

**Ngày 1 — mục tiêu: có bản chạy được, bấm được, chỉ với nguồn Hugging Face**
1. Setup project, cài `streamlit`, `anthropic`, `requests`.
2. Viết và test độc lập hàm gọi Hugging Face Datasets API (test bằng query tay trước khi
   nối với LLM).
3. Viết Bước 1 (parse_task), test với 3-4 mô tả đề tài khác nhau, kiểm tra JSON output.
4. Nối Bước 1 → Bước 2 (chỉ HF): input text → parse → search → danh sách dataset thật.
   Đây là cột mốc quan trọng nhất của Ngày 1, dừng lại test kỹ trước khi làm tiếp.
5. Viết Bước 3 (rank_candidates) với đa tiêu chí như mô tả ở trên.
6. Dựng UI Streamlit tối giản: ô nhập text, nút bấm, hiển thị kết quả dạng card (tên
   dataset, điểm từng tiêu chí, access_type, link, giải thích).
7. Cuối ngày: chọn sẵn 2-3 đề tài mẫu để demo, đảm bảo chạy ổn định end-to-end.

**Ngày 2 — mục tiêu: mở rộng nguồn + Bước 4 + độ tin cậy**
1. Thêm Kaggle API và Papers with Code API vào Bước 2 (theo kiến trúc source registry).
2. Thêm cờ `is_sensitive_domain` và `is_narrow_domain` vào Bước 1, thêm Bước 4 (fallback
   suggestions).
3. Thêm khả năng hỏi lại để lọc kết quả (vd: "cần bộ nhỏ hơn", "chỉ cần open license") mà
   không phải chạy lại toàn bộ pipeline từ đầu — chỉ lọc lại trên danh sách candidate đã có.
4. Polish UI, xử lý edge case (candidates rỗng, lỗi API timeout), chuẩn bị demo cuối cùng.

## Yêu cầu kiểm thử trước khi coi là "xong"

- Thử với ít nhất 5 mô tả đề tài khác nhau, gồm cả đề tài phổ biến (NLP, CV cơ bản) và đề
  tài khó/hẹp (3D, y sinh) để kiểm tra cả 4 bước hoạt động đúng, kể cả nhánh fallback.
- Thử ít nhất 1 đề tài rơi vào domain nhạy cảm (vd: "phát hiện khuôn mặt giả mạo") để kiểm
  tra cờ `is_sensitive_domain` và cảnh báo đi kèm hoạt động đúng.
- Xác nhận không có trường hợp nào dataset hiển thị cho user mà không có `url` thật trả về
  từ một lệnh gọi API trong phiên chạy đó.
- Xác nhận UI không bị crash khi một nguồn dữ liệu bị lỗi mạng/timeout (nguồn khác vẫn phải
  chạy tiếp).

## Việc Codex KHÔNG cần làm ở MVP này (để tránh lan man)

- Không cần vector DB hay RAG phức tạp — LLM trực tiếp là đủ cho scope này.
- Không cần scrape tự động các registry chuyên ngành (ShapeNet, PhysioNet...) — chỉ cần
  gợi ý tên kèm nhãn "chưa verify" theo nguyên tắc #2.
- Không cần cache/database lưu trữ lâu dài — session state trong Streamlit là đủ cho demo.
- Không cần authentication/multi-user — đây là MVP demo cho hackathon, không phải sản phẩm
  production.
