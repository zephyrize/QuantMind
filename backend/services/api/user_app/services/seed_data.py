"""Authentication schema seed data for OSS deployments."""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import bcrypt
from dotenv import dotenv_values
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.api.user_app.models.user import User, UserProfile
from backend.services.api.user_app.services.rbac_service import (
    RBACService,
    init_default_roles_and_permissions,
)

logger = logging.getLogger(__name__)

ADMIN_USER_ID = "00000001"
DEFAULT_TENANT_ID = "default"


@dataclass(frozen=True)
class AdminConfig:
    username: str
    email: str
    password: str


def _load_admin_config() -> AdminConfig:
    """Load administrator credentials from the current .env file."""
    configured_path = os.getenv("ENV_FILE_PATH")
    env_path = Path(configured_path) if configured_path else Path("/app/.env")
    if not env_path.exists():
        env_path = Path(__file__).resolve().parents[5] / ".env"
    values = dotenv_values(env_path) if env_path.exists() else {}

    def get_value(name: str, default: str = "") -> str:
        # Prefer the mounted file so `docker compose restart quantmind` picks
        # up a local .env edit without recreating the container.
        return str(values.get(name) or os.getenv(name) or default).strip()

    username = get_value("ADMIN_USERNAME")
    email = get_value("ADMIN_EMAIL")
    password = get_value("ADMIN_PASSWORD")
    missing = [
        name
        for name, value in {
            "ADMIN_USERNAME": username,
            "ADMIN_EMAIL": email,
            "ADMIN_PASSWORD": password,
        }.items()
        if not value or value.startswith("CHANGE_ME")
    ]
    if missing:
        raise RuntimeError(
            "Missing required administrator configuration in .env: "
            + ", ".join(missing)
        )
    return AdminConfig(username=username, email=email, password=password)


async def init_admin_data(db: AsyncSession) -> None:
    """Initialize RBAC and synchronize the configured administrator account."""
    logger.info("Starting administrator account synchronization...")
    admin_config = _load_admin_config()
    await init_default_roles_and_permissions(db)

    result = await db.execute(
        select(User).where(
            User.user_id == ADMIN_USER_ID,
            User.tenant_id == DEFAULT_TENANT_ID,
        )
    )
    admin_user = result.scalar_one_or_none()

    conflicts = await db.execute(
        select(User).where(
            User.tenant_id == DEFAULT_TENANT_ID,
            (User.username == admin_config.username)
            | (User.email == admin_config.email),
        )
    )
    for user in conflicts.scalars().all():
        if user.user_id != ADMIN_USER_ID:
            raise RuntimeError(
                "ADMIN_USERNAME or ADMIN_EMAIL is already assigned to another user"
            )

    password_hash = bcrypt.hashpw(
        admin_config.password.encode(), bcrypt.gensalt()
    ).decode("utf-8")
    if admin_user is None:
        logger.info("Creating configured administrator account...")
        admin_user = User(
            user_id=ADMIN_USER_ID,
            tenant_id=DEFAULT_TENANT_ID,
            username=admin_config.username,
            email=admin_config.email,
            password_hash=password_hash,
            is_active=True,
            is_verified=True,
            is_admin=True,
            is_deleted=False,
        )
        db.add(admin_user)
    else:
        admin_user.username = admin_config.username
        admin_user.email = admin_config.email
        admin_user.password_hash = password_hash
        admin_user.is_active = True
        admin_user.is_verified = True
        admin_user.is_admin = True
        admin_user.is_deleted = False

    # Guarantee the role even for databases created by a prior release.
    rbac_service = RBACService(db)
    admin_role = await rbac_service.get_role_by_code("admin")
    if admin_role:
        roles = await rbac_service.get_user_roles(admin_user.user_id)
        if not any(role.code == "admin" for role in roles):
            await rbac_service.add_role_to_user(admin_user.user_id, admin_role.id)

    result = await db.execute(
        select(UserProfile).where(
            UserProfile.user_id == admin_user.user_id,
            UserProfile.tenant_id == DEFAULT_TENANT_ID,
        )
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        db.add(
            UserProfile(
                user_id=admin_user.user_id,
                tenant_id=DEFAULT_TENANT_ID,
                display_name="System Administrator",
                preferences={
                    "theme": "dark",
                    "language": "zh-CN",
                    "dashboard_layout": "default",
                },
                notification_settings={
                    "email": True,
                    "push": True,
                    "marketing": False,
                },
            )
        )

    await db.commit()
    logger.info("Administrator account synchronization completed.")


if __name__ == "__main__":
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    async def main() -> None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is not set")
        engine = create_async_engine(database_url)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            await init_admin_data(session)
        await engine.dispose()

    asyncio.run(main())
