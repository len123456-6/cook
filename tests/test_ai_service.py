"""ai_service 单测：不联网、不消耗额度，用假客户端验证「调用参数 + 输出清洗」。

真实调用 DeepSeek 需要 .env 里配置 DEEPSEEK_API_KEY，因此这里全部用 mock：
  - parse_recommendation：纯函数，覆盖代码块包裹、脏条目、非 JSON、空结果；
  - recommend_recipes：验证模型名、response_format、prompt 参数是否正确透传。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError

from app.core.config import settings
from app.services import ai_service

VALID_CONTENT = json.dumps(
    {
        "dishes": [
            {"name": "清蒸鲈鱼", "reason": "少油少盐，富含优质蛋白，适合控血压的老人"},
            {"name": "冬瓜虾仁汤", "reason": "清淡利尿、低脂高蛋白，减轻心脏负担"},
            {"name": "第三道菜", "reason": "超过 max_dishes 时不应返回"},
        ],
        "advice": "全天食盐控制在 5 克以内，多吃新鲜蔬果。",
    },
    ensure_ascii=False,
)


class _FakeCompletions:
    """记录调用参数，返回预设内容（模拟 client.chat.completions）。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


@pytest.fixture(autouse=True)
def _reset_client_cache() -> Any:
    """get_client 带 lru_cache，用例之间必须清掉，避免 key 串场。"""
    ai_service.get_client.cache_clear()
    yield
    ai_service.get_client.cache_clear()


def test_parse_recommendation_cleans_code_fence_and_dirty_items() -> None:
    """模型常见的「代码块包裹 + 重复项 + 空字段」要能自动清洗。"""
    raw = """```json
    {"dishes": [
        {"name": "清蒸鲈鱼", "reason": "少油少盐，富含优质蛋白"},
        {"name": "清蒸鲈鱼", "reason": "重复项应被丢弃"},
        {"name": "   ", "reason": "空菜名应被丢弃"},
        {"name": "芹菜炒香干", "reason": "高钾高纤维，有助控血压"}
    ], "advice": "全天控盐不超过 5 克。"}
    ```"""

    result = ai_service.parse_recommendation(
        raw, requirement="适合高血压老人的菜", max_dishes=2
    )

    assert [dish.name for dish in result.dishes] == ["清蒸鲈鱼", "芹菜炒香干"]
    assert result.advice == "全天控盐不超过 5 克。"
    assert result.requirement == "适合高血压老人的菜"
    assert result.disclaimer  # 医疗建议必须有免责声明


def test_parse_recommendation_rejects_non_json() -> None:
    with pytest.raises(ai_service.AIResponseError):
        ai_service.parse_recommendation("抱歉，我无法给出建议。", requirement="随便")


def test_parse_recommendation_rejects_empty_dishes() -> None:
    with pytest.raises(ai_service.AIResponseError):
        ai_service.parse_recommendation('{"dishes": []}', requirement="随便")


def test_recommend_recipes_passes_params_to_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(VALID_CONTENT)
    monkeypatch.setattr(ai_service, "get_client", lambda: fake)
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test-only")

    result = ai_service.recommend_recipes(
        "适合高血压老人的菜", servings=3, max_dishes=2
    )

    assert [dish.name for dish in result.dishes] == ["清蒸鲈鱼", "冬瓜虾仁汤"]

    call = fake.chat.completions.calls[0]
    assert call["model"] == settings.deepseek_model
    assert call["response_format"] == {"type": "json_object"}
    user_prompt = call["messages"][1]["content"]
    assert "适合高血压老人的菜" in user_prompt
    assert "用餐人数：3 人" in user_prompt
    assert "不要超过 2 道" in user_prompt


def test_recommend_recipes_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ai_service.AIConfigError) as excinfo:
        ai_service.recommend_recipes("适合高血压老人的菜")

    assert excinfo.value.code == "ai_not_configured"


def test_recommend_recipes_wraps_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**_: Any) -> None:
        raise OpenAIError("connection reset")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_raise))
    )
    monkeypatch.setattr(ai_service, "get_client", lambda: client)
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test-only")

    with pytest.raises(ai_service.AIUpstreamError) as excinfo:
        ai_service.recommend_recipes("适合高血压老人的菜")

    assert excinfo.value.code == "ai_upstream_error"
