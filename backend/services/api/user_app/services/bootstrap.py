"""Bootstrap the authentication schema for a self-contained OSS deployment."""

import logging

from backend.services.api.models.base import Base
from backend.shared.database_manager_v2 import get_db_manager, get_session

logger = logging.getLogger(__name__)


async def ensure_auth_schema_and_admin() -> None:
    """Create authentication/RBAC tables and seed the local administrator.

    OSS deployments start from the shared SQL schema, which intentionally does
    not include the user-app ORM tables.  Creating them here makes a fresh
    local database usable without requiring a separate migration command.
    """
    # Import every auth model before inspecting SQLAlchemy metadata.
    from backend.services.api.user_app.models import oauth, rbac, user  # noqa: F401
    from backend.services.api.user_app.services.seed_data import init_admin_data

    db_manager = get_db_manager()
    engine = db_manager._master_engine
    if engine is None:
        raise RuntimeError("Database engine is not initialized")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with get_session(read_only=False) as session:
        await init_admin_data(session)

    logger.info("Authentication schema and default admin are ready")
