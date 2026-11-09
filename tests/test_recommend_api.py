"""推荐接口冒烟测试：路由挂载、入参校验、异常到状态码的映射。

不联网：`ai_service.recommend_recipes` 全部 monkeypatch 掉，
所以不依赖 .env 里的真实密钥，也不会消耗额度。

跑法::

    python -m pytest tests/test_recommend_api.py
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.recommend import DISCLAIMER, DishRecommendation, RecommendationResult
from app.services import ai_service

ENDPOINT = "/api/v1/recommend"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """with 语句触发 lifespan（启动建表），与 test_health.py 保持一致。"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _restore_service() -> object:
    """用 monkeypatch 的用例无需清理；这里只确保 ai_service.get_client 缓存不串场。"""
    ai_service.get_client.cache_clear()
    yield
    ai_service.get_client.cache_clear()


def test_recommend_returns_cleaned_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """正常路径：服务层结果原样成为响应体，字段与入参回显正确。"""
    captured: dict[str, object] = {}

    def _fake_recommend(
        requirement: str, *, servings: int = 2, max_dishes: int = 3
    ) -> RecommendationResult:
        captured.update(
            requirement=requirement, servings=servings, max_dishes=max_dishes
        )
        return RecommendationResult(
            requirement=requirement,
            dishes=[
                DishRecommendation(name="清蒸鲈鱼", reason="少油少盐，富含优质蛋白"),
                DishRecommendation(name="冬瓜虾仁汤", reason="清淡利尿、低脂高蛋白"),
            ],
            advice="全天食盐控制在 5 克以内。",
        )

    monkeypatch.setattr(ai_service, "recommend_recipes", _fake_recommend)

    response = client.post(
        ENDPOINT,
        json={"requirement": "  适合高血压老人的菜  ", "servings": 4, "max_dishes": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requirement"] == "适合高血压老人的菜"  # 首尾空白已被清洗
    assert [dish["name"] for dish in body["dishes"]] == ["清蒸鲈鱼", "冬瓜虾仁汤"]
    assert body["disclaimer"] == DISCLAIMER

    assert captured == {
        "requirement": "适合高血压老人的菜",
        "servings": 4,
        "max_dishes": 2,
    }


def test_recommend_uses_defaults(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """只传 requirement 时，servings=2 / max_dishes=3（与 ai_service 默认值一致）。"""
    captured: dict[str, object] = {}

    def _fake_recommend(
        requirement: str, *, servings: int = 2, max_dishes: int = 3
    ) -> RecommendationResult:
        captured.update(servings=servings, max_dishes=max_dishes)
        return RecommendationResult(
            requirement=requirement,
            dishes=[DishRecommendation(name="小米南瓜粥", reason="易消化、养胃")],
        )

    monkeypatch.setattr(ai_service, "recommend_recipes", _fake_recommend)

    assert client.post(ENDPOINT, json={"requirement": "养胃"}).status_code == 200
    assert captured == {"servings": 2, "max_dishes": 3}


@pytest.mark.parametrize(
    "payload",
    [
        {"requirement": ""},
        {"requirement": "   "},
        {"requirement": "汤", "servings": 0},
        {"requirement": "汤", "servings": 21},
        {"requirement": "汤", "max_dishes": 0},
        {"requirement": "汤", "max_dishes": 11},
        {},
    ],
)
def test_recommend_rejects_invalid_payload(
    client: TestClient, payload: dict[str, object]
) -> None:
    """入参边界由 Pydantic 拦住，返回 422 且不触碰 AI 服务。"""
    assert client.post(ENDPOINT, json=payload).status_code == 422


def test_recommend_without_api_key_returns_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未配置密钥 -> 503 + 可读错误码，而不是 500。"""

    def _raise_config_error(*_: object, **__: object) -> None:
        raise ai_service.AIConfigError()

    monkeypatch.setattr(ai_service, "recommend_recipes", _raise_config_error)

    response = client.post(ENDPOINT, json={"requirement": "适合高血压老人的菜"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "ai_not_configured"


def test_recommend_upstream_error_returns_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DeepSeek 调用失败 -> 502，且不把上游细节泄露给前端。"""

    def _raise_upstream(*_: object, **__: object) -> None:
        raise ai_service.AIUpstreamError("调用 AI 服务失败：APIConnectionError")

    monkeypatch.setattr(ai_service, "recommend_recipes", _raise_upstream)

    response = client.post(ENDPOINT, json={"requirement": "适合高血压老人的菜"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "ai_upstream_error"


def test_recommend_bad_model_output_returns_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型输出不可解析 -> 502 ai_bad_response。"""

    def _raise_bad_response(*_: object, **__: object) -> None:
        raise ai_service.AIResponseError("AI 未返回有效的菜品推荐")

    monkeypatch.setattr(ai_service, "recommend_recipes", _raise_bad_response)

    response = client.post(ENDPOINT, json={"requirement": "适合高血压老人的菜"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "ai_bad_response"


def test_openapi_contains_recommend_endpoint(client: TestClient) -> None:
    """路由确实挂在 /api/v1 前缀下，文档里能看到（浏览器 /docs 可测）。"""
    schema = client.get("/openapi.json").json()

    assert ENDPOINT in schema["paths"]
    request_schema = schema["paths"][ENDPOINT]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert request_schema["$ref"].endswith("/RecommendRequest")
    assert schema["paths"][ENDPOINT]["post"]["tags"] == ["ai"]
