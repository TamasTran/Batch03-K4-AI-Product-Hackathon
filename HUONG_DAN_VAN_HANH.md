# Hướng dẫn cài đặt, vận hành và hiểu mã nguồn DataScout AI

Tài liệu này dành cho cả người chưa từng lập trình. Bạn không cần hiểu Python,
FastAPI hay API trước khi đọc. Chỉ cần làm lần lượt theo hướng dẫn là có thể cài
đặt, chạy thử, kiểm tra lỗi và hiểu mỗi phần của hệ thống đang làm nhiệm vụ gì.

---

## 1. DataScout AI là gì?

DataScout AI là một trợ lý tìm kiếm dataset cho dự án AI.

Ví dụ, người dùng nhập:

> Tìm dataset hình ảnh có nhãn để phát hiện người đi bộ.

Hệ thống sẽ:

1. Hiểu người dùng đang cần dữ liệu ảnh cho bài toán object detection.
2. Tìm kiếm trên Hugging Face, Kaggle, OpenML, Zenodo và Papers with Code.
3. Có thể tìm bổ sung trên web nếu đã cấu hình dịch vụ web search.
4. Kiểm tra metadata như giấy phép, số lượt tải, nhãn và quy mô.
5. Loại bớt kết quả không liên quan hoặc vi phạm ràng buộc.
6. Gộp các dataset bị trùng giữa nhiều nguồn.
7. Xếp hạng và giải thích vì sao mỗi dataset phù hợp.
8. Chia kết quả thành:
   - `Verified`: metadata lấy trực tiếp từ registry.
   - `Unverified`: kết quả tìm thấy trên web, người dùng cần tự kiểm tra.

Nói đơn giản, frontend là quầy giao tiếp với người dùng; backend là nhóm nhân viên
phía sau thực hiện phân tích, tìm kiếm, kiểm tra và xếp hạng.

---

## 2. Sơ đồ tổng thể

```text
Người dùng
    |
    v
Trình duyệt tại localhost:3000
    |
    | Gửi yêu cầu HTTP dạng JSON
    v
FastAPI tại localhost:8000
    |
    v
SearchAgent
    |
    +--> Hiểu yêu cầu
    +--> Tìm nhiều nguồn
    +--> Kiểm tra ràng buộc
    +--> Bổ sung metadata
    +--> Gộp kết quả trùng
    +--> Xếp hạng
    +--> Ghi audit log
    |
    v
Frontend hiển thị kết quả
```

Frontend và backend là hai chương trình độc lập. Vì vậy khi sử dụng local, bạn
phải mở hai cửa sổ PowerShell:

- Một cửa sổ chạy backend.
- Một cửa sổ chạy frontend.

---

## 3. Những thứ cần chuẩn bị

### 3.1 Phần mềm bắt buộc

- Windows 10 hoặc Windows 11.
- Python 3.10 trở lên, khuyến nghị Python 3.11.
- Trình duyệt Chrome, Edge hoặc Firefox.
- PowerShell.

Kiểm tra Python:

```powershell
python --version
```

Nếu hiện `Python 3.11.x` hoặc phiên bản tương đương thì có thể tiếp tục.

### 3.2 API key

Hệ thống vẫn có heuristic fallback khi không có LLM, nhưng chất lượng hiểu ngôn
ngữ và xếp hạng sẽ thấp hơn. Nên cấu hình ít nhất một LLM.

Bạn chỉ cần một trong các nhóm sau:

| Model muốn dùng | Biến môi trường cần có |
|---|---|
| OpenAI, ví dụ `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic, ví dụ Claude | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Model qua OpenRouter | `OPENROUTER_API_KEY` |

Các tích hợp tùy chọn:

| Chức năng | Biến cần cấu hình |
|---|---|
| Tìm dataset Kaggle | `KAGGLE_USERNAME`, `KAGGLE_KEY` |
| Web search bằng SerpAPI | `SERPAPI_API_KEY` |
| Web search bằng Bing | `BING_SEARCH_API_KEY` |
| Google Custom Search | `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_ID` |

Không đưa API key vào `front-end/config.js`, file Python hay GitHub.

---

## 4. Cài đặt lần đầu

Mở PowerShell tại thư mục dự án:

```powershell
cd "C:\Users\tamas\OneDrive\Desktop\Công_việc\Batch03-K4-AI-Product-Hackathon"
```

### Bước 1: Tạo virtual environment

```powershell
python -m venv .venv
```

Virtual environment là một “hộp riêng” chứa thư viện Python của dự án. Nó giúp
thư viện dự án này không xung đột với dự án khác.

### Bước 2: Kích hoạt môi trường

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, chạy một lần:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Sau đó thử kích hoạt lại.

### Bước 3: Cài thư viện

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Các thư viện chính:

- `fastapi`: xây REST API.
- `uvicorn`: chạy FastAPI.
- `pydantic`: kiểm tra dữ liệu request và response.
- `requests`: gọi registry, web search và LLM API.
- `python-dotenv`: đọc file `.env`.
- `anthropic`: hỗ trợ Claude.
- `PyYAML`: đọc cấu hình YAML.
- `httpx`: hỗ trợ kiểm thử API.

### Bước 4: Tạo file cấu hình bí mật

```powershell
Copy-Item .env.example .env
```

Mở `.env` bằng trình soạn thảo và điền key. Ví dụ:

```dotenv
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=điền_key_thật_tại_đây

KAGGLE_USERNAME=
KAGGLE_KEY=

CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Không xóa `.env.example`. File này chỉ là mẫu và có thể commit. File `.env` chứa
key thật, đã được `.gitignore` chặn và tuyệt đối không được commit.

---

## 5. Cách chạy hằng ngày

### 5.1 Chạy backend

Mở PowerShell thứ nhất tại thư mục gốc:

```powershell
.\.venv\Scripts\python.exe .\run_backend.py
```

Khi thấy nội dung tương tự:

```text
Uvicorn running on http://127.0.0.1:8000
```

thì backend đã chạy.

Các địa chỉ quan trọng:

- Health check: `http://localhost:8000/api/v1/health`
- Cấu hình an toàn: `http://localhost:8000/api/v1/config`
- Tài liệu API: `http://localhost:8000/docs`

Trang `/docs` cho phép thử API mà chưa cần mở frontend.

### 5.2 Chạy frontend

Mở PowerShell thứ hai:

```powershell
cd "C:\Users\tamas\OneDrive\Desktop\Công_việc\Batch03-K4-AI-Product-Hackathon\front-end"
..\.venv\Scripts\python.exe -m http.server 3000
```

Mở:

```text
http://localhost:3000
```

Không nên mở trực tiếp `index.html` bằng cách nhấp đúp. Chạy qua HTTP server giúp
trình duyệt xử lý tài nguyên và kết nối API đúng cách.

### 5.3 Dừng hệ thống

Trong từng cửa sổ PowerShell đang chạy server, nhấn:

```text
Ctrl + C
```

---

## 6. Cách sử dụng giao diện

1. Nhập mô tả dataset cần tìm.
2. Mở `Data sources` để chọn nguồn.
3. Bật ít nhất một registry hoặc `Catch-all web`.
4. Nút `Web on/Web off` bật hoặc tắt tìm kiếm web.
5. Nhấn nút mũi tên hoặc Enter để tìm.
6. Dùng Shift + Enter nếu muốn xuống dòng.

Nếu yêu cầu quá chung, Agent sẽ hỏi một câu làm rõ. Ví dụ:

> Bạn muốn dùng dataset cho tác vụ nào và loại dữ liệu nào?

Hãy trả lời câu hỏi đó trong ô nhập. Frontend sẽ gửi cả câu ban đầu, câu hỏi làm
rõ và câu trả lời về backend.

Các mục trong menu:

- `Research assistant`: quay lại cuộc hội thoại.
- `Dataset library`: xem các dataset đã tìm được trước đây.
- `Search history`: xem và mở lại các lượt tìm kiếm.
- Biểu tượng thùng rác: xóa lịch sử lưu trong trình duyệt.
- `Settings`: điều chỉnh số kết quả tối đa và xem trạng thái cấu hình.

Lịch sử chỉ được lưu trong `localStorage` của trình duyệt hiện tại, không được
gửi lên database và không đồng bộ sang máy khác.

---

## 7. Một request được xử lý như thế nào?

### Giai đoạn 1: Frontend tạo request

`front-end/app.js` đọc:

- Nội dung người dùng nhập.
- Danh sách nguồn đã bật.
- Trạng thái web fallback.
- Giới hạn kết quả.
- Ngữ cảnh làm rõ nếu có.

Sau đó gửi:

```json
{
  "query": "Tìm dataset tiếng Việt có nhãn cho sentiment classification",
  "clarification_context": null,
  "enabled_sources": ["Hugging Face", "OpenML", "Zenodo"],
  "web_fallback_enabled": true,
  "limit": 15
}
```

### Giai đoạn 2: API kiểm tra request

`back-end/api.py` dùng schema trong `schemas.py` để kiểm tra:

- Query phải từ 3 đến 2000 ký tự.
- Limit phải từ 5 đến 30.
- Tên nguồn phải nằm trong danh sách cho phép.
- Phải bật ít nhất một nguồn.

Request sai sẽ nhận HTTP 422. Pipeline lỗi bất ngờ sẽ nhận HTTP 500.

### Giai đoạn 3: Phân tích intent

`pipeline/parse_task.py` xác định:

- `task_type`: classification, detection, segmentation...
- `domain`: NLP, computer vision, biomedical...
- `modality`: text, image, audio, 3D...
- `subject`: human, pedestrian, product review...
- Ngôn ngữ bắt buộc.
- Nhãn bắt buộc.
- Số mẫu tối thiểu.
- Miền nhạy cảm.
- Keyword tiếng Anh để tìm registry.

Nếu LLM không hoạt động, heuristic sẽ xử lý các nhóm yêu cầu phổ biến.

Nếu task, modality hoặc domain chưa đủ rõ, pipeline dừng trước khi tìm kiếm và
trả về `clarification_required`. Cơ chế này tránh lãng phí API và tránh trả kết
quả có vẻ đẹp nhưng sai mục tiêu.

### Giai đoạn 4: Tìm kiếm song song

`agent.py` chạy hai nhánh:

- Nhánh verified gọi các registry chính thức.
- Nhánh unverified gọi web search.

Các cặp `nguồn × keyword` được chạy song song, tối đa 8 worker. Nếu một nguồn
lỗi, các nguồn khác vẫn tiếp tục.

### Giai đoạn 5: Kiểm tra và chuẩn hóa

`tools.verify_candidates()` chỉ nhận item có đủ:

- ID.
- URL.
- Tên nguồn.

Kết quả cùng URL được gộp sơ bộ. Metadata verified luôn được ưu tiên hơn kết quả
web trùng URL.

### Giai đoạn 6: Kiểm tra ràng buộc

`pipeline/constraints.py` so sánh dataset với yêu cầu:

- Có đúng loại tác vụ không?
- Có đúng chủ thể không?
- Có đúng ngôn ngữ không?
- Có nhãn yêu cầu không?
- Có đủ số mẫu không?

Mỗi candidate nhận trạng thái:

- `matched`: khớp rõ.
- `partial`: chỉ khớp một phần.
- `unknown`: metadata chưa đủ để kết luận.
- `mismatch`: có bằng chứng rõ là không phù hợp.

### Giai đoạn 7: Enrichment

`sources/enrich.py` lấy thêm metadata cho các dataset cần thiết, đặc biệt từ
Hugging Face:

- Ngôn ngữ.
- Nhãn.
- Số mẫu.
- Dataset card.

Nếu enrichment lỗi, candidate vẫn được giữ lại cùng thông tin lỗi thay vì làm
hỏng toàn bộ request.

### Giai đoạn 8: Deduplicate

`pipeline/deduplicate.py` chuẩn hóa tên, bỏ version/năm và so độ giống nhau.

- Rất giống: tự động gộp.
- Giống vừa phải: có thể nhờ LLM đánh giá.
- Không chắc: giữ riêng và đánh dấu có thể trùng.

Nguyên tắc là thận trọng: khi LLM lỗi, hệ thống không tự gộp.

### Giai đoạn 9: Ranking

`pipeline/rank_candidates.py` xếp hạng theo:

- Độ khớp tác vụ.
- Độ khớp domain/chủ thể.
- Độ phù hợp nhãn.
- Quy mô.
- Độ mở và giấy phép.
- Chất lượng metadata.

Verified và unverified được chấm bằng hai contract khác nhau. Kết quả web không
được giả thành verified.

Nếu LLM bỏ sót candidate hoặc lỗi, heuristic ranking xử lý candidate đó.
Candidate dưới relevance threshold bị loại.

### Giai đoạn 10: Guidance và response

Nếu không có dataset mở đủ tốt, `fallback_suggestions.py` gợi ý:

- Synthetic data.
- Fine-tune với tập nhỏ.
- Augmentation.
- Weak/self-supervision.
- Registry chuyên ngành.

API trả response cho frontend và ghi audit log.

---

## 8. Giải thích từng file Python

### File ở thư mục gốc

#### `run_backend.py`

Điểm khởi động backend thuận tiện nhất.

File này:

1. Tìm thư mục `back-end`.
2. Chuyển working directory vào đó.
3. Thêm backend vào Python import path.
4. Chạy đối tượng `app` trong `api.py` bằng Uvicorn tại port 8000.

Thông thường người vận hành chỉ cần chạy file này, không cần chạy `api.py`.

### Nhóm API và điều phối

#### `back-end/api.py`

Là cửa chính của backend.

Nhiệm vụ:

- Tạo ứng dụng FastAPI.
- Đọc `.env`.
- Cấu hình CORS.
- Cung cấp `/health`, `/config` và `/search`.
- Kiểm tra tên source.
- Ghép lịch sử clarification.
- Tạo `SearchAgent`.
- Chuyển kết quả thành response chuẩn.
- Gắn `X-Request-ID`.
- Ghi audit log cho cả request thành công và thất bại.

#### `back-end/schemas.py`

Là bộ mẫu đơn của API. Nó quy định dữ liệu nào được phép đi vào và đi ra.

- `HealthResponse`: trạng thái dịch vụ.
- `ConfigResponse`: model và nguồn nào đã cấu hình.
- `SearchRequest`: query, source, web fallback và limit.
- `ToolEventResponse`: trace rút gọn của một công cụ.
- `SearchResponse`: kết quả cuối cùng.

Pydantic tự động từ chối request sai kiểu hoặc vượt giới hạn.

#### `back-end/agent.py`

Là người quản lý quy trình tìm kiếm.

Các thành phần:

- `ToolEvent`: ghi lại tên tool, input, trạng thái, output hoặc lỗi.
- `AgentRun`: gói toàn bộ kết quả một lượt chạy.
- `SearchAgent`: thực thi pipeline.
- `has_search_source()`: kiểm tra có nguồn nào được bật hay không.

`SearchAgent.run()` không tự viết lại thuật toán nhỏ; nó gọi tuần tự các tool
chuyên trách và lưu trace để có thể kiểm tra lại.

#### `back-end/audit_log.py`

Ghi hồ sơ đầy đủ của mỗi request vào:

```text
back-end/logs/run_<request_id>.json
```

Trước khi ghi, file này che API key, password, authorization và token bằng `***`.
Nó ghi file tạm rồi đổi tên để tránh log bị dở dang nếu tiến trình bị gián đoạn.

### Nhóm tool registry

#### `back-end/tools/__init__.py`

Là bảng chuyển tiếp giữa Agent và implementation.

`TOOL_FUNCTIONS` ánh xạ:

- `analyze_task` sang parser.
- `search_registry` sang source adapter.
- `verify_candidates` sang kiểm tra candidate.
- `prepare_candidates` sang constraint evaluation.
- `enrich_candidates` sang enrichment.
- `deduplicate_candidates` sang dedup.
- `rank_datasets` sang ranking.

Nhờ lớp này, Agent gọi tool bằng tên thống nhất.

### Nhóm phân tích và pipeline

#### `back-end/pipeline/llm.py`

Là bộ định tuyến LLM.

File này:

- Đọc model và API key từ môi trường.
- Suy ra provider từ tên model.
- Hỗ trợ OpenAI, Anthropic, Gemini, OpenRouter và endpoint tương thích OpenAI.
- Tạo HTTP request đúng format của từng provider.
- Ép temperature về mức deterministic.
- Trích JSON từ câu trả lời.
- Báo lỗi provider rõ ràng.

#### `back-end/pipeline/prompts.py`

Đọc prompt runtime từ `back-end/artifacts/system_prompt.md`.

Mỗi bước chỉ lấy section prompt cần dùng. Việc tách prompt khỏi code giúp review
và chỉnh hành vi Agent mà không sửa thuật toán Python.

#### `back-end/pipeline/parse_task.py`

Chuyển câu tự nhiên thành intent có cấu trúc.

Có hai chế độ:

- LLM mode: linh hoạt và hiểu ngữ nghĩa tốt.
- Heuristic mode: phương án dự phòng khi chưa có key hoặc LLM lỗi.

File còn áp dụng confidence gate để quyết định nên search hay hỏi làm rõ.

#### `back-end/pipeline/constraints.py`

Đóng vai người kiểm tra điều kiện đầu vào.

File chứa alias cho ngôn ngữ, task và subject; đánh giá từng candidate; sau đó
tạo candidate pool cân bằng giữa các nguồn để một registry không chiếm toàn bộ
danh sách.

#### `back-end/pipeline/deduplicate.py`

Phát hiện cùng một dataset xuất hiện ở nhiều nguồn.

Nó chuẩn hóa tên, tính similarity, dùng cấu trúc Union-Find để tạo nhóm và chọn
bản ghi có metadata tốt nhất làm đại diện.

#### `back-end/pipeline/rank_candidates.py`

Là phần chấm điểm và xếp hạng.

File này:

- Chấm verified và unverified riêng.
- Chia candidate thành batch cho LLM.
- Phát hiện response bị cắt hoặc thiếu score.
- Fallback về heuristic theo từng candidate.
- Kiểm tra core keyword để tránh điểm cao giả.
- Loại kết quả dưới threshold.
- Tạo diagnostics phục vụ audit và eval.

Đây là một trong những file quan trọng nhất khi muốn cải thiện độ chính xác.

#### `back-end/pipeline/fallback_suggestions.py`

Tạo lời khuyên thay thế khi không tìm thấy dataset mở đủ tốt.

#### `back-end/pipeline/__init__.py`

Đánh dấu `pipeline` là Python package. Hiện file không chứa business logic.

### Nhóm nguồn dữ liệu

#### `back-end/sources/__init__.py`

Tạo `SOURCE_REGISTRY`, ánh xạ tên hiển thị sang hàm tìm kiếm tương ứng.

#### `back-end/sources/huggingface.py`

Gọi Hugging Face Dataset API và chuyển response về schema chung.

#### `back-end/sources/kaggle.py`

Gọi Kaggle Dataset API bằng username và key. Nếu thiếu credential, adapter báo
lỗi nhưng Agent vẫn tiếp tục các nguồn khác.

#### `back-end/sources/openml.py`

Tìm OpenML dataset. Khi API exact-search trả “không có kết quả”, file dùng catalog
được cache và lọc tên cục bộ.

#### `back-end/sources/zenodo.py`

Tìm các Zenodo record có resource type là dataset, rồi lấy title, license, DOI,
download và URL.

#### `back-end/sources/paperswithcode.py`

Tìm dataset qua Papers with Code API và chuẩn hóa URL tương đối thành URL đầy đủ.

#### `back-end/sources/web_fallback.py`

Là catch-all search.

File tự chọn provider web đã cấu hình, tạo truy vấn từ keyword và domain, sau đó
chỉ giữ title, URL và snippet. Kết quả luôn là `unverified`.

#### `back-end/sources/enrich.py`

Gọi thêm dataset card để bổ sung ngôn ngữ, feature, label và sample count.

### Nhóm test

#### `back-end/tests/test_api.py`

Kiểm tra endpoint, validation, clarification, hai evidence lane, request ID và log.

#### `back-end/tests/test_pipeline.py`

Kiểm tra parser, heuristic, constraints, enrichment, dedup, ranking, fallback và
SearchAgent. Đây là file test nghiệp vụ lớn nhất.

#### `back-end/tests/test_llm.py`

Kiểm tra chọn provider, model, API key, header và request LLM.

#### `back-end/tests/test_prompts.py`

Kiểm tra prompt có đủ section, injection guardrail và contract của từng bước.

#### `back-end/tests/test_audit_log.py`

Kiểm tra log đầy đủ nhưng không làm lộ secret.

#### `back-end/tests/test_eval_harness.py`

Kiểm tra cấu trúc bộ 50 golden case và logic chấm eval.

#### `back-end/tests/test_frontend_contract.py`

Kiểm tra button có action, ID frontend được nối vào JavaScript, URL an toàn,
escaping và history không phải dữ liệu mẫu.

#### `back-end/tests/run_eval.py`

Chương trình chạy golden evaluation end-to-end. Có checkpoint để tiếp tục khi
quá trình dài bị gián đoạn.

#### `back-end/tests/live_intent_cases.py`

Chạy nhanh một nhóm câu hỏi thật để quan sát intent mà Agent phân tích.

---

## 9. Các file không phải Python nhưng rất quan trọng

### `front-end/index.html`

Khung giao diện, sidebar, composer, dialog settings và welcome screen.

### `front-end/app.js`

Toàn bộ hành vi frontend:

- Gọi API.
- Render loading, clarification, error và kết quả.
- Escape nội dung để giảm nguy cơ XSS.
- Kiểm tra URL chỉ cho phép HTTP/HTTPS.
- Quản lý sidebar, setting và composer.
- Lưu history/library bằng localStorage.

### `front-end/styles.css`

CSS component và animation.

### `front-end/config.js`

Chỉ chứa địa chỉ backend:

```javascript
window.DATASCOUT_CONFIG = {
  API_BASE_URL: "http://localhost:8000/api/v1",
};
```

Không đặt secret trong file này vì trình duyệt và mọi người dùng đều đọc được.

### `back-end/artifacts/system_prompt.md`

System prompt dùng khi phân tích, deduplicate và ranking.

### `back-end/artifacts/tools.yaml`

Mô tả contract các tool để con người có thể review.

### `.env`

Chứa cấu hình bí mật trên máy. Không commit.

### `.env.example`

Mẫu tên biến môi trường. Được phép commit vì không có key thật.

---

## 10. Chạy kiểm thử

Từ thư mục `back-end`:

```powershell
cd back-end
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kết quả tốt phải kết thúc bằng:

```text
OK
```

Kiểm tra syntax toàn backend:

```powershell
..\.venv\Scripts\python.exe -m compileall -q .
```

Chạy một golden smoke test:

```powershell
..\.venv\Scripts\python.exe tests\run_eval.py --start 1 --count 1
```

Chạy đủ 50 golden case:

```powershell
..\.venv\Scripts\python.exe tests\run_eval.py
```

Tiếp tục từ checkpoint:

```powershell
..\.venv\Scripts\python.exe tests\run_eval.py --resume
```

Golden eval gọi API thật, có thể chạy lâu và tiêu tốn quota.

---

## 11. Cách đọc kết quả và audit log

Mỗi response có header:

```text
X-Request-ID: <mã>
```

Tìm log tương ứng:

```text
back-end/logs/run_<mã>.json
```

Các phần chính:

- `request`: dữ liệu frontend gửi lên.
- `status`: answered, clarification_required hoặc error.
- `tool_events`: từng bước Agent đã gọi.
- `final_result`: kết quả cuối.
- `error`: lỗi cấp pipeline nếu có.

Nếu giao diện trả kết quả lạ, audit log là nơi đầu tiên cần kiểm tra.

---

## 12. Xử lý lỗi thường gặp

### “API chưa kết nối”

Nguyên nhân:

- Backend chưa chạy.
- Backend chạy sai port.
- `front-end/config.js` trỏ sai URL.
- CORS không cho phép origin frontend.

Kiểm tra:

```text
http://localhost:8000/api/v1/health
```

### `ModuleNotFoundError`

Bạn chưa cài dependency hoặc đang dùng sai Python.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### LLM không hoạt động

Mở:

```text
http://localhost:8000/api/v1/config
```

Nếu `llm_enabled` là `false`, kiểm tra:

- `.env` có đúng vị trí không.
- `LLM_MODEL` có đúng provider không.
- Key tương ứng có giá trị không.
- Có khoảng trắng hoặc dấu ngoặc thừa không.

### Kaggle báo thiếu key

Điền cả:

```dotenv
KAGGLE_USERNAME=
KAGGLE_KEY=
```

Hoặc tắt Kaggle trên giao diện.

### Web search không có kết quả

`Catch-all web` cần ít nhất một provider web search được cấu hình. Nếu không có,
tắt tùy chọn này và dùng registry.

### Request chạy lâu

Nguyên nhân có thể là:

- Nhiều source và nhiều keyword.
- Registry ngoài phản hồi chậm.
- Enrichment nhiều candidate.
- LLM ranking nhiều batch.

Thử:

- Giảm limit xuống 5 hoặc 10.
- Chỉ bật Hugging Face và OpenML.
- Tắt Catch-all web.
- Kiểm tra tool event nào lâu trong audit log.

### Có kết quả nhưng không liên quan

Kiểm tra theo thứ tự:

1. `intent` có hiểu đúng task/modality/subject không.
2. `search_keywords_en` có đúng không.
3. `constraint_notes` nói candidate khớp phần nào.
4. Ranking có dùng LLM hay heuristic.
5. Candidate có bị thiếu metadata không.

---

## 13. An toàn và bảo mật

- Không commit `.env`.
- Không ghi API key vào frontend.
- Không gửi data thật của người dùng vào dịch vụ ngoài.
- Không public `data/vlearn-pack`; đây là dữ liệu khóa học.
- Audit log có thể chứa query và metadata, vì vậy không public log.
- Kết quả web luôn phải được coi là unverified.
- Với dữ liệu y tế, sinh trắc học, trẻ em hoặc vị trí, phải kiểm tra pháp lý và
  đạo đức trước khi sử dụng.

---

## 14. Checklist trước khi demo

- [ ] Backend mở được `/api/v1/health`.
- [ ] `/api/v1/config` hiển thị đúng model.
- [ ] Frontend hiện “API sẵn sàng”.
- [ ] Ít nhất một verified registry hoạt động.
- [ ] Thử một câu rõ ràng.
- [ ] Thử một câu mơ hồ và xác nhận Agent hỏi làm rõ.
- [ ] Mở được link dataset.
- [ ] History và Dataset Library hoạt động.
- [ ] Chạy test và nhận `OK`.
- [ ] `.env`, log và data pack không nằm trong commit.
- [ ] Có phương án demo dự phòng nếu API ngoài bị chậm.

---

## 15. Tóm tắt dành cho người bàn giao

Nếu chỉ cần nhớ năm điều:

1. Chạy `run_backend.py` ở terminal thứ nhất.
2. Chạy `python -m http.server 3000` trong `front-end` ở terminal thứ hai.
3. Secret chỉ nằm trong `.env`.
4. Khi kết quả sai, xem intent rồi xem audit log.
5. Trước khi thay đổi Agent, chạy unit test và golden eval.
