from .llm import call_json, get_client, llm_label
from .prompts import step_prompt


SYSTEM = step_prompt("STEP_1")

GENERIC_DOMAINS = {"", "any", "general", "other", "unknown"}
FIELD_CONFIDENCE_KEYS = (
    "task_type",
    "modality",
    "domain",
    "required_language",
    "required_labels",
)
FIELD_CONFIDENCE_THRESHOLD = 0.6


def _clarification_question(missing_fields: list[str], intent: dict) -> str:
    if "task_type" in missing_fields:
        modality = str(intent.get("modality") or "any").lower()
        domain = str(intent.get("domain") or "general")
        if modality != "any":
            return (
                f"Bạn muốn dùng dữ liệu {modality} cho tác vụ nào, ví dụ phân loại, "
                "phát hiện, phân đoạn, dự đoán hay sinh dữ liệu?"
            )
        if domain.lower() not in GENERIC_DOMAINS:
            return (
                f"Trong lĩnh vực {domain}, bạn muốn dataset phục vụ tác vụ cụ thể nào, "
                "ví dụ phân loại, dự đoán, phát hiện hay phân tích?"
            )
        return (
            "Bạn muốn dùng dataset cho tác vụ nào, loại dữ liệu nào và thuộc chủ đề "
            "nào? Ví dụ: phân tích cảm xúc bình luận sản phẩm, nhận diện ảnh cây trồng "
            "hoặc dự đoán giá nhà từ dữ liệu bảng."
        )
    if "modality" in missing_fields:
        return (
            "Tác vụ đã rõ, nhưng bạn muốn mô hình xử lý loại dữ liệu nào: văn bản, ảnh, "
            "âm thanh, video, dữ liệu bảng hay dữ liệu 3D?"
        )
    return "Bạn có thể mô tả rõ hơn tác vụ hoặc loại dữ liệu mà mô hình cần xử lý không?"


def apply_confidence_policy(intent: dict) -> dict:
    """Derive clarification solely from structured per-field confidence."""
    normalized = dict(intent)
    raw_confidence = normalized.get("field_confidence")
    if not isinstance(raw_confidence, dict):
        raw_confidence = {}

    field_confidence: dict[str, float] = {}
    for field in FIELD_CONFIDENCE_KEYS:
        try:
            value = float(raw_confidence.get(field, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        field_confidence[field] = round(max(0.0, min(1.0, value)), 2)

    missing_fields: list[str] = []
    if field_confidence["task_type"] < FIELD_CONFIDENCE_THRESHOLD:
        missing_fields.append("task_type")
    if field_confidence["modality"] < FIELD_CONFIDENCE_THRESHOLD:
        missing_fields.append("modality")
    if field_confidence["domain"] < FIELD_CONFIDENCE_THRESHOLD:
        missing_fields.append("domain")
    if (
        normalized.get("required_language") not in (None, "", "any")
        and field_confidence["required_language"] < FIELD_CONFIDENCE_THRESHOLD
    ):
        missing_fields.append("required_language")
    if (
        normalized.get("needs_labels")
        and not normalized.get("required_labels")
        and field_confidence["required_labels"] < FIELD_CONFIDENCE_THRESHOLD
    ):
        missing_fields.append("required_labels")

    needs_clarification = any(
        field_confidence[field] < FIELD_CONFIDENCE_THRESHOLD
        for field in ("task_type", "modality")
    )
    normalized["field_confidence"] = field_confidence
    normalized["intent_confidence"] = round(
        sum(field_confidence[field] for field in ("task_type", "modality", "domain")) / 3,
        2,
    )
    normalized["missing_fields"] = missing_fields
    normalized["needs_clarification"] = needs_clarification
    if needs_clarification:
        existing_question = str(normalized.get("clarification_question") or "").strip()
        normalized["clarification_question"] = (
            existing_question or _clarification_question(missing_fields, normalized)
        )
    else:
        normalized["clarification_question"] = ""
    return normalized


def _heuristic(text: str) -> dict:
    value = text.lower()
    rules = [
        (("sentiment", "cảm xúc"), "sentiment classification", "NLP", "text"),
        (("ner", "thực thể", "entity"), "named entity recognition", "NLP", "text"),
        (("phân loại ảnh", "image classification"), "image classification", "computer vision", "image"),
        (("phân loại văn bản", "text classification"), "text classification", "NLP", "text"),
        (("nhận diện ảnh", "nhận dạng ảnh", "image recognition"), "image recognition", "computer vision", "image"),
        (
            (
                "object detection",
                "vehicle detection",
                "car detection",
                "phát hiện vật",
                "nhận diện vật",
                "phát hiện ô tô",
                "phát hiện xe",
                "nhận diện ô tô",
                "nhận diện xe",
                "phát hiện con người",
                "phát hiện người",
                "phát hiện người đi bộ",
                "khoanh vùng người",
                "person detection",
                "human detection",
                "pedestrian detection",
                "people detection",
            ),
            "object detection",
            "computer vision",
            "image",
        ),
        (("segmentation", "phân đoạn"), "segmentation", "computer vision", "image"),
        (("audio", "giọng nói", "speech"), "audio classification", "audio", "audio"),
        (("3d", "point cloud", "robot"), "3D perception", "3D/robotics", "3d"),
        (("y tế", "medical", "sinh học", "biomedical"), "medical prediction", "biomedical", "tabular"),
        (("phân loại", "classification"), "classification", "general", "any"),
        (("dự đoán", "prediction", "forecast"), "prediction", "general", "any"),
    ]
    task, domain, modality = "machine learning", "general", "any"
    matched_rule = False
    for terms, candidate_task, candidate_domain, candidate_modality in rules:
        if any(term in value for term in terms):
            task, domain, modality = candidate_task, candidate_domain, candidate_modality
            matched_rule = True
            break
    sensitive_terms = ("y tế", "medical", "biometric", "khuôn mặt", "face", "vân tay", "giọng nói",
                       "tài chính", "finance", "trẻ em", "children", "vị trí", "location", "quân sự")
    narrow_terms = ("3d", "robot", "medical", "y tế", "biomedical", "geospatial", "point cloud")
    if any(term in value for term in ("tiếng việt", "vietnamese", "vietnam")):
        language = "vietnamese"
    elif any(term in value for term in ("tiếng anh", "english")):
        language = "english"
    else:
        language = "any"
    subject_rules = [
        (("người đi bộ", "pedestrian"), "pedestrian"),
        (("khuôn mặt", "face"), "face"),
        (("con người", "human", "person", "people", "crowd"), "human"),
        (("chữ viết", "handwritten", "handwriting"), "handwriting"),
        (("văn bản", "document", "ocr", "word image", "text image", "containing text", "text recognition"), "text"),
        (("bình luận sản phẩm", "product review", "e-commerce", "thương mại điện tử"), "product reviews"),
        (("mạng xã hội", "social media", "twitter", "facebook"), "social media"),
        (("tin tức", "news"), "news"),
        (("y tế", "medical", "biomedical"), "medical"),
    ]
    subject = "general"
    for terms, candidate_subject in subject_rules:
        if any(term in value for term in terms):
            subject = candidate_subject
            break
    required_labels = (
        ["positive", "negative", "neutral"]
        if "sentiment" in value or "cảm xúc" in value
        else []
    )
    keywords = []
    if language != "any":
        keywords.append(f"{language} {task}")
    if subject != "general":
        keywords.append(
            f"{language + ' ' if language != 'any' else ''}{subject}"
        )
    keywords.append(task)
    if subject != "general":
        keywords.append(f"{task} {subject}")
    hard_constraints = []
    if language != "any":
        hard_constraints.append(f"language={language}")
    if required_labels:
        hard_constraints.append("labels required")
    broad_fallback_task = task in {"classification", "prediction"}
    domain_only_fallback = task == "medical prediction" and not any(
        term in value
        for term in ("dự đoán", "prediction", "forecast", "phân loại", "classification")
    )
    task_confidence = (
        0.35
        if domain_only_fallback
        else 0.7
        if broad_fallback_task
        else 0.9
        if matched_rule
        else 0.2
    )
    modality_confidence = (
        0.9
        if modality not in {"", "any", "unknown"} and not domain_only_fallback
        else 0.3
    )
    domain_confidence = 0.9 if domain not in GENERIC_DOMAINS else 0.3
    intent = {
        "task_type": task, "domain": domain, "modality": modality, "language": language,
        "subject": subject,
        "required_language": language,
        "required_labels": required_labels,
        "preferred_domain": subject,
        "minimum_samples": None,
        "hard_constraints": hard_constraints,
        "search_keywords_en": keywords[:4], "needs_labels": task != "machine learning",
        "is_narrow_domain": any(x in value for x in narrow_terms),
        "is_sensitive_domain": any(x in value for x in sensitive_terms),
        "sensitive_reason": "Có thể chứa dữ liệu cá nhân hoặc chịu ràng buộc pháp lý/đạo đức."
        if any(x in value for x in sensitive_terms) else "",
        "field_confidence": {
            "task_type": task_confidence,
            "modality": modality_confidence,
            "domain": domain_confidence,
            "required_language": 0.95 if language != "any" else 0.3,
            "required_labels": 0.9 if required_labels else 0.3,
        },
    }
    return apply_confidence_policy(intent)


def parse_task(text: str) -> tuple[dict, str]:
    if get_client():
        try:
            result = call_json(SYSTEM, text, 900)
            if not isinstance(result.get("search_keywords_en"), list):
                raise ValueError("search_keywords_en không hợp lệ")
            result["search_keywords_en"] = result["search_keywords_en"][:4]
            result.setdefault("subject", "general")
            result.setdefault("required_language", result.get("language", "any"))
            result.setdefault("required_labels", [])
            result.setdefault("preferred_domain", result.get("domain", "general"))
            result.setdefault("minimum_samples", None)
            result.setdefault("hard_constraints", [])
            return apply_confidence_policy(result), f"LLM ({llm_label()})"
        except Exception as exc:
            return _heuristic(text), f"Heuristic (LLM lỗi: {exc})"
    return _heuristic(text), "Heuristic (chưa cấu hình LLM)"
