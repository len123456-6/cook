"""AI 推荐相关的 Pydantic 模型（API 契约层）。

`RecommendRequest` 是 `POST /api/v1/recommend` 的请求体；
`DishRecommendation` / `RecommendationResult` 既是 `ai_service` 的返回类型，
也是该接口的响应结构，保证「服务产物 = 接口出参」不走样。

命名约定（与项目一致）：
    入参：RecommendRequest
    出参：RecommendationResult / DishRecommendation
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

DISCLAIMER = "AI 生成的饮食建议仅供参考，不能替代医生或营养师的专业诊断。"


class RecommendRequest(BaseModel):
    """推荐接口入参：`POST /api/v1/recommend` 的请求体。

    这里的边界（1~20 人、1~10 道）与 `ai_service.recommend_recipes` 的内部校验
    保持一致；接口层的 Pydantic 先拦一道，服务层再兜底一道，避免脏参数打到模型。
    """

    requirement: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="养生需求，自然语言描述，如「适合高血压老人的菜」",
        examples=["适合高血压老人的清淡家常菜"],
    )
    servings: int = Field(
        default=2,
        ge=1,
        le=20,
        description="用餐人数，用于让模型把握菜量（1~20）",
        examples=[2],
    )
    max_dishes: int = Field(
        default=3,
        ge=1,
        le=10,
        description="期望返回的菜品数量上限（1~10），默认与 ai_service.DEFAULT_MAX_DISHES 一致",
        examples=[3],
    )

    @field_validator("requirement")
    @classmethod
    def _strip_requirement(cls, value: str) -> str:
        """去掉首尾空白；纯空白等于没提需求，直接 422。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("养生需求不能为空白字符")
        return stripped


class DishRecommendation(BaseModel):
    """单道推荐菜：只保留前端真正要展示的「菜名 + 理由」。"""

    name: str = Field(
        ..., min_length=1, max_length=50, description="菜名，如「清蒸鲈鱼」"
    )
    reason: str = Field(
        ..., min_length=1, max_length=200, description="推荐理由（结合养生需求）"
    )


class RecommendationResult(BaseModel):
    """一次推荐请求的完整结果。"""

    requirement: str = Field(..., description="用户原始养生需求，回显便于前端展示/埋点")
    dishes: list[DishRecommendation] = Field(
        ...,
        min_length=1,
        description="推荐菜品列表，按推荐优先级排序",
    )
    advice: str | None = Field(
        default=None,
        max_length=300,
        description="整体饮食建议（可选，1~2 句话）",
    )
    disclaimer: str = Field(default=DISCLAIMER, description="固定免责声明")
