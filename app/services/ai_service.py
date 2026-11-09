"""AI 智能推荐服务：调用 DeepSeek（OpenAI 兼容协议）按养生需求推荐菜谱。

调用链::

    recommend_recipes(需求)
        -> 组装 system / user prompt
        -> DeepSeek chat/completions（response_format=json_object 强约束）
        -> json 解析 + Pydantic 清洗校验（脏数据不进业务层）
        -> RecommendationResult

约定：
  - 密钥 / base_url / 模型 / 超时全部来自 `app.core.config`（即 .env 或环境变量），
    代码里不出现任何真实 key，也不允许业务层散落配置；
  - 本模块只负责「推荐」，不写库；是否落库由调用方（API 层）决定；
  - 上游异常统一抛 `AIServiceError`，接口层据此映射 503 / 502。

直接自测（需先配好 .env 里的 DEEPSEEK_API_KEY）::

    python -c "from app.services.ai_service import recommend_recipes; \
print(recommend_recipes('适合高血压老人的菜', max_dishes=3).model_dump_json(indent=2))"
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

from app.core.config import BASE_DIR, settings
from app.schemas.recommend import DishRecommendation, RecommendationResult

# ⚠ 临时调试代码（排完「密钥没生效」就删掉）：
# 显式加载项目根目录的 .env，把 DEEPSEEK_API_KEY 等键灌进 os.environ，
# 这样下面 os.getenv 的调试打印才有值；指定 BASE_DIR 是为了不依赖启动时的 cwd。
load_dotenv(BASE_DIR / ".env")

# ⚠ 临时调试打印（import 阶段就会执行 —— 即 uvicorn 启动、路由 import 本模块时打印）
print("DEBUG KEY (import-time):", os.getenv("DEEPSEEK_API_KEY"))

logger = logging.getLogger(__name__)

# 单次请求的 token 上限：3~5 道菜 + 一句建议足够，防止模型跑题长篇大论
MAX_OUTPUT_TOKENS = 1200
DEFAULT_MAX_DISHES = 3

# ---------------------------------------------------------------------------
# 异常：接口层只需捕获 AIServiceError，按 code 映射 HTTP 状态码
# ---------------------------------------------------------------------------


class AIServiceError(RuntimeError):
    """AI 服务统一异常基类。"""

    def __init__(self, message: str, *, code: str = "ai_error") -> None:
        super().__init__(message)
        self.code = code


class AIConfigError(AIServiceError):
    """未配置密钥等「服务端自身」问题 -> 503。"""

    def __init__(
        self, message: str = "未配置 DEEPSEEK_API_KEY，AI 推荐暂不可用"
    ) -> None:
        super().__init__(message, code="ai_not_configured")


class AIUpstreamError(AIServiceError):
    """DeepSeek 调用失败（鉴权、限流、超时、网络）-> 502。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="ai_upstream_error")


class AIResponseError(AIServiceError):
    """模型返回内容不可用（非 JSON / 结构不符 / 空结果）-> 502。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="ai_bad_response")


# ---------------------------------------------------------------------------
# Prompt：要求「纯 JSON」，并且明确医疗边界，避免模型给诊断/疗效承诺
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位熟悉现代营养学与中式家常烹饪的注册营养师。
你的任务：根据用户的养生需求，推荐真正能在家里做出来的菜，并说明推荐理由。

硬性要求：
1. 只输出一个 JSON 对象；不要 markdown 代码块，不要任何解释性前后缀。
2. JSON 结构固定为：
   {"dishes": [{"name": "菜名", "reason": "推荐理由"}], "advice": "整体饮食建议"}
3. 菜名用常见中文叫法且具体到可执行（例如「清蒸鲈鱼」，不要写「鱼类菜肴」）。
4. reason 必须结合用户需求说明「为什么适合」，60 字以内，可点出关键食材或烹饪方式（如清蒸、少油少盐）。
5. advice 用 1~2 句话给出整体饮食提醒，100 字以内。
6. 涉及高血压、糖尿病、痛风等疾病时，只给饮食建议：不做诊断、不承诺疗效、
   不推荐药物或保健品；必要时提醒遵医嘱。
7. 不要编造用户没有提到的忌口、过敏或病情。"""


def _build_user_prompt(requirement: str, *, servings: int, max_dishes: int) -> str:
    """把结构化参数塞进 user 消息（system 保持稳定，便于命中 prompt 缓存）。"""
    return (
        f"用户的养生需求：{requirement}\n"
        f"用餐人数：{servings} 人\n"
        f"请推荐 {max_dishes} 道菜（不要超过 {max_dishes} 道），"
        "并给出每道菜的推荐理由和整体饮食建议。"
    )


# ---------------------------------------------------------------------------
# 客户端：惰性创建 + 进程内复用（内部维护连接池，别每次请求 new 一个）
# ---------------------------------------------------------------------------


def resolve_api_key() -> str:
    """读取 DeepSeek 密钥。

    `Settings` 已经从 `.env` / 环境变量加载了配置；这里再用 `os.getenv` 兜底，
    是为了兼容「进程启动后由容器/CI 注入环境变量」而配置单例已缓存的情况。
    两处都是读环境，不存在硬编码。
    """
    # ⚠ 临时调试打印（key 会完整出现在终端，排完立刻删除这一行）
    print("DEBUG KEY:", os.getenv("DEEPSEEK_API_KEY"))
    return (settings.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """构造指向 DeepSeek 的 OpenAI 客户端（OpenAI 兼容协议，只换 base_url）。

    测试里改了 key / 需要重建客户端时调用 `get_client.cache_clear()`。
    """
    api_key = resolve_api_key()
    if not api_key:
        raise AIConfigError()

    return OpenAI(
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        timeout=settings.deepseek_timeout_seconds,
        max_retries=2,  # 限流 / 抖动时自动重试，仍失败才抛给上层
    )


# ---------------------------------------------------------------------------
# 响应解析：模型可能包代码块、塞多余文本、返回脏条目，这里逐层清洗
# ---------------------------------------------------------------------------


class _RawDish(BaseModel):
    """模型返回的单条菜品（宽松校验，脏条目在下面统一过滤）。"""

    name: str = ""
    reason: str = ""


class _RawPayload(BaseModel):
    """模型返回的 JSON 顶层结构。"""

    dishes: list[_RawDish] = Field(default_factory=list)
    advice: str | None = None


def _extract_json_text(raw: str) -> str:
    """去掉 markdown 代码块与 JSON 前后的多余文本。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 兜底：模型偶尔会加「好的，以下是我推荐的菜：」这类前缀
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return text.strip()


def parse_recommendation(
    raw: str,
    *,
    requirement: str,
    max_dishes: int = DEFAULT_MAX_DISHES,
) -> RecommendationResult:
    """把模型原始输出解析成 `RecommendationResult`（纯函数，便于单测）。

    Raises:
        AIResponseError: 非 JSON、结构不符、清洗后没有任何有效菜品。
    """
    try:
        payload = _RawPayload.model_validate(json.loads(_extract_json_text(raw)))
    except json.JSONDecodeError as exc:
        logger.warning("AI 返回内容不是合法 JSON: %r", raw[:500])
        raise AIResponseError("AI 返回的不是合法 JSON") from exc
    except ValidationError as exc:
        logger.warning("AI 返回结构不符合约定: %s", exc)
        raise AIResponseError("AI 返回的数据结构不符合约定") from exc

    dishes: list[DishRecommendation] = []
    seen: set[str] = set()
    for item in payload.dishes:
        name, reason = item.name.strip(), item.reason.strip()
        if not name or not reason or name in seen:
            continue  # 空名 / 空理由 / 重复菜名直接丢弃，不污染结果
        seen.add(name)
        dishes.append(DishRecommendation(name=name[:50], reason=reason[:200]))
        if len(dishes) >= max_dishes:
            break

    if not dishes:
        logger.warning("AI 返回中没有任何有效菜品: %r", raw[:500])
        raise AIResponseError("AI 未返回有效的菜品推荐")

    advice = (payload.advice or "").strip()[:300] or None
    return RecommendationResult(requirement=requirement, dishes=dishes, advice=advice)


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def recommend_recipes(
    requirement: str,
    *,
    servings: int = 2,
    max_dishes: int = DEFAULT_MAX_DISHES,
) -> RecommendationResult:
    """根据养生需求推荐菜谱。

    Args:
        requirement: 用户自然语言需求，例如「适合高血压老人的菜」。
        servings: 用餐人数，写入 prompt 供模型把握菜量（1~20）。
        max_dishes: 期望返回的菜品数上限（1~10）。

    Returns:
        RecommendationResult: 菜名 + 推荐理由（+ 整体建议、免责声明）。

    Raises:
        ValueError: 入参不合法（接口层由 Pydantic 先行拦截）。
        AIConfigError: 未配置 DEEPSEEK_API_KEY。
        AIUpstreamError: DeepSeek 调用失败。
        AIResponseError: 模型输出不可解析 / 无有效菜品。
    """
    requirement = (requirement or "").strip()
    if not requirement:
        raise ValueError("养生需求不能为空")
    if not 1 <= servings <= 20:
        raise ValueError("servings 必须在 1~20 之间")
    if not 1 <= max_dishes <= 10:
        raise ValueError("max_dishes 必须在 1~10 之间")

    client = get_client()
    logger.info(
        "请求 DeepSeek 推荐：requirement=%r servings=%s max_dishes=%s model=%s",
        requirement,
        servings,
        max_dishes,
        settings.deepseek_model,
    )

    try:
        completion = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        requirement, servings=servings, max_dishes=max_dishes
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},  # DeepSeek 支持，保证纯 JSON
        )
    except OpenAIError as exc:
        # 鉴权失败 / 限流 / 超时 / 网络异常都在这里，细节只进日志不外泄
        logger.exception("调用 DeepSeek 失败")
        raise AIUpstreamError(f"调用 AI 服务失败：{type(exc).__name__}") from exc

    if not completion.choices:
        raise AIResponseError("AI 未返回任何结果")

    content = completion.choices[0].message.content or ""
    result = parse_recommendation(
        content, requirement=requirement, max_dishes=max_dishes
    )
    logger.info("AI 推荐成功：%s", [dish.name for dish in result.dishes])
    return result
