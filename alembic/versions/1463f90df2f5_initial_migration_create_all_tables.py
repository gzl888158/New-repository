"""Initial migration: create all core tables (authoritative schema).

Revision ID: 1463f90df2f5
Revises:
Create Date: 2026-07-20 19:07:06.535377

以 data/sqlite_storage.Base 为单一事实源：upgrade 通过 create_all 生成与运行时
完全一致的 schema，downgrade 通过 drop_all 反向拆除，避免迁移脚本与模型漂移。
"""

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '1463f90df2f5'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """从零创建与运行时一致的完整核心 schema。"""
    from data.sqlite_storage import Base
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    """反向拆除核心 schema。"""
    from data.sqlite_storage import Base
    Base.metadata.drop_all(bind=op.get_bind())
