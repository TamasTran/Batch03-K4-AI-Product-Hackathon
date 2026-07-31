"""Fires all eval_set.json cases at the live backend and dumps raw responses.
Judging (PASS/FAIL) is done separately by hand against must_answer criteria.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

BASE = "http://localhost:8000/api/v1"
HERE = Path(__file__).parent
cases = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))
raw = {}


def search(query, **kw):
    payload = {"query": query}
    payload.update(kw)
    r = requests.post(f"{BASE}/search", json=payload, timeout=90)
    return {"http_status": r.status_code, "body": r.json() if r.ok else r.text}


for case in cases:
    cid = case["id"]
    if cid == 19:
        continue  # handled separately: multi-turn
    if cid == 16:
        res = search(case["input"], enabled_sources=["Hugging Face", "Kaggle", "OpenML"])
    else:
        res = search(case["input"])
    raw[cid] = res
    print(cid, case["category"], "->", res["body"].get("status") if isinstance(res["body"], dict) else res["body"])

# Case 19: multi-turn
turn1 = search("tôi cần dữ liệu để train model")
raw["19_turn1"] = turn1
ctx = turn1["body"].get("clarification_context")
turn2 = search("nhận diện ảnh sản phẩm lỗi trên dây chuyền", clarification_context=ctx)
raw["19_turn2"] = turn2
print("19 turn1 status:", turn1["body"].get("status"))
print("19 turn2 status:", turn2["body"].get("status"))

(HERE / "raw_results.json").write_text(
    json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Saved eval/raw_results.json")
