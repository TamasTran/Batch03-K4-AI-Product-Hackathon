# DataScout AI Architecture

```text
Browser
  |
  | HTTP JSON
  v
front-end/                 back-end/
  index.html                 api.py
  styles.css                 schemas.py
  app.js             --->    agent.py
  config.js                  pipeline/
                             sources/
                             tools/
```

Frontend và backend chạy thành hai tiến trình độc lập. API key chỉ tồn tại trong
`.env` ở máy chủ và không được gửi xuống trình duyệt.

## Luồng request

1. Frontend gửi `POST /api/v1/search`.
2. FastAPI xác thực request, ghép clarification context nếu có.
3. Intent completeness gate kiểm tra task, modality và domain.
4. Nếu prompt mơ hồ, API hỏi đúng một câu làm rõ và chưa gọi nguồn tìm kiếm.
5. Nếu intent đủ rõ, `SearchAgent` chạy nhánh verified và catch-all song song.
6. Pipeline enrich, deduplicate, rank và loại candidate dưới relevance threshold.
7. API trả hai evidence lane độc lập: `verified` và `unverified`.

## Chạy local

Terminal 1:

```powershell
python -m pip install -r requirements.txt
python run_backend.py
```

Terminal 2:

```powershell
cd front-end
python -m http.server 3000
```

Mở `http://localhost:3000`. OpenAPI nằm tại `http://localhost:8000/docs`.
