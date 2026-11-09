"""临时冒烟脚本：验证 /docs 里的端点确实可用（跑完即删）。"""

import json
import pathlib

import httpx

BASE = "http://127.0.0.1:8000"
out: dict[str, object] = {}

with httpx.Client(timeout=120) as client:
    spec = client.get(f"{BASE}/openapi.json").json()
    out["paths"] = sorted(spec["paths"])
    out["health"] = client.get(f"{BASE}/health").json()

    ok = client.post(
        f"{BASE}/api/v1/recommend",
        json={"requirement": "适合高血压老人的清淡家常菜", "servings": 2, "max_dishes": 2},
    )
    out["recommend_status"] = ok.status_code
    out["recommend_body"] = ok.json()

    bad = client.post(f"{BASE}/api/v1/recommend", json={"requirement": "   "})
    out["invalid_status"] = bad.status_code
    out["invalid_body"] = bad.json()

pathlib.Path(r"d:\cook\_smoke.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
