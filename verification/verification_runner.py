"""向后兼容 stub：实现已迁移至 eval.verification_runner。

本文件仅做 re-export，保持 verification.verification_runner 导入路径不变。
"""
from eval.verification_runner import *  # noqa: F401,F403
from eval.verification_runner import (  # noqa: F401
    MockConnection,
    MockCursor,
    MockOKXClient,
    MockRedisCache,
    MockSQLiteStorage,
    VerificationRunner,
)
