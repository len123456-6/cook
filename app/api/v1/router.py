"""v1 汇总路由：把各业务域的子路由挂到一起，由 main.py 统一加 `/api/v1` 前缀。

新增领域时只改这里两行（import + include_router），main.py 不用动。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import recommend

api_router = APIRouter()
api_router.include_router(recommend.router)

__all__ = ["api_router"]
