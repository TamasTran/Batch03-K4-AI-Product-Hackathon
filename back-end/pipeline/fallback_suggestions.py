def build_guidance(intent: dict, ranked: list[dict]) -> dict:
    verified = [x for x in ranked if x.get("confidence", "verified") == "verified"]
    good_open = any(
        x.get("access_type") == "open" and x.get("total_score", 0) >= 3.5
        for x in verified
    )
    alternatives = []
    if not good_open:
        alternatives = [
            "Tạo synthetic data bằng Blender, Unity Perception hoặc NVIDIA Omniverse nếu bài toán mô phỏng được.",
            "Fine-tune mô hình pretrained trên một tập nhỏ tự thu thập.",
            "Dùng augmentation mạnh; hoặc weak/self-supervision kết hợp một ít nhãn thủ công.",
            "Tìm phiên bản mini/sample công khai của benchmark bị hạn chế.",
        ]
    registries = []
    if intent.get("is_narrow_domain") and not good_open:
        domain = intent.get("domain", "").lower()
        if "3d" in domain:
            registries = ["ShapeNet — chưa verify, vui lòng tự kiểm tra link và khả dụng."]
        elif "bio" in domain or "medical" in domain:
            registries = ["PhysioNet — chưa verify.", "TCIA — chưa verify."]
        elif "robot" in domain:
            registries = ["Open X-Embodiment — chưa verify."]
    return {"good_open_found": good_open, "alternatives": alternatives, "registries": registries}
