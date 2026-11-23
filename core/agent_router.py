"""
Backward-compatibility facade for core.router.
==============================================
Re-exports all symbols from core.router for existing callers.
"""

from core.router import *  # noqa: F401, F403
from core.router import execute_deep_research, process_query_with_router

__all__ = [
    "process_query_with_router",
    "execute_deep_research",
]
