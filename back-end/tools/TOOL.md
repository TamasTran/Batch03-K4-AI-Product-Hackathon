# DataScout tools

- `analyze_task`: dùng LLM nếu có key, nếu không dùng heuristic; luôn chạy clarification gate deterministic.
- `search_registry`: gọi một nguồn tại một thời điểm và cô lập lỗi theo nguồn.
- `verify_candidates`: bắt buộc có ID, source và URL, sau đó loại URL trùng chính xác.
- `prepare_candidates`: đánh giá hard constraints và tạo pool cân bằng giữa các nguồn.
- `enrich_candidates`: đọc metadata chi tiết trước khi ranking, đặc biệt dataset card của Hugging Face.
- `deduplicate_candidates`: dedup xuyên nguồn hai tầng trước khi ranking; ưu tiên verified.
- `rank_datasets`: ranking đa tiêu chí, heuristic fallback và loại candidate dưới relevance threshold.
- `build_fallback_guidance`: giữ nguyên phương án thay thế và registry chuyên ngành.

Tên tool phải đồng bộ với `artifacts/tools.yaml`.
