"""Single entry point for QuantMind runtime environment configuration.

The repository has two configuration files with intentionally separate roles:

* ``.env`` holds Docker Compose / infrastructure settings.
* ``backend/.env`` holds application settings.

Neither file stores the topology-dependent database or Redis host. Those
defaults are derived at runtime so the same files work for a host Python
process (``127.0.0.1``) and for Docker Compose (``db`` / ``redis``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

from backend.shared.runtime_secrets import load_runtime_env


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
ROOT_ENV_FILE = PROJECT_ROOT / ".env"
BACKEND_ENV_FILE = BACKEND_DIR / ".env"


def resolve_project_path(
    value: str | os.PathLike[str] | None,
    *,
    default: str | os.PathLike[str],
) -> Path:
    """Resolve a configurable path relative to the repository root.

    Absolute environment overrides remain supported.  Relative paths are
    deliberately anchored to ``PROJECT_ROOT`` instead of the process working
    directory so host Python and the Docker image resolve the same layout.
    """

    raw = str(value or "").strip()
    candidate = Path(raw) if raw else Path(default)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def is_container_runtime() -> bool:
    """Return whether the process is running inside a container.

    ``QUANTMIND_RUNTIME`` remains an explicit override for environments where
    the usual Docker marker is unavailable (for example some CI runners).
    """

    runtime = os.getenv("QUANTMIND_RUNTIME", "").strip().lower()
    if runtime in {"docker", "container"}:
        return True
    if runtime in {"local", "host"}:
        return False
    return Path("/.dockerenv").exists()


def _set_from_alias(target: str, *sources: str, default: str) -> None:
    """Set ``target`` only when no caller has provided it explicitly."""

    if os.getenv(target, "").strip():
        return
    for source in sources:
        value = os.getenv(source, "").strip()
        if value:
            os.environ[target] = value
            return
    os.environ[target] = default


def _build_database_url() -> None:
    """Build the canonical async SQLAlchemy URL when one was not supplied."""

    if os.getenv("DATABASE_URL", "").strip():
        return

    driver = os.getenv("DB_DRIVER", "asyncpg").strip() or "asyncpg"
    dialect = (
        "postgresql" if driver.startswith("postgresql") else f"postgresql+{driver}"
    )
    user = quote(os.environ["DB_USER"], safe="")
    password = quote(os.environ["DB_PASSWORD"], safe="")
    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    database = quote(os.environ["DB_NAME"], safe="")
    os.environ["DATABASE_URL"] = (
        f"{dialect}://{user}:{password}@{host}:{port}/{database}"
    )


@lru_cache(maxsize=1)
def bootstrap_environment() -> int:
    """Load both configuration files and resolve portable service defaults.

    Priority is preserved as ``process environment > runtime.env > backend/.env
    > root .env > derived defaults``. The two static files deliberately do
    not define overlapping application keys.
    """

    # ``runtime.env`` is managed by the application UI. Load it before the
    # static files because all later loads use ``override=False``.
    runtime_count = load_runtime_env()
    load_dotenv(BACKEND_ENV_FILE, override=False)
    load_dotenv(ROOT_ENV_FILE, override=False)

    container = is_container_runtime()
    _set_from_alias(
        "DB_HOST",
        "POSTGRES_HOST",
        default="db" if container else "127.0.0.1",
    )
    _set_from_alias("DB_PORT", "POSTGRES_PORT", default="5432")
    _set_from_alias("DB_NAME", "POSTGRES_DB", default="quantmind")
    _set_from_alias("DB_USER", "POSTGRES_USER", default="quantmind")
    _set_from_alias("DB_PASSWORD", "POSTGRES_PASSWORD", default="")
    _set_from_alias(
        "REDIS_HOST",
        default="redis" if container else "127.0.0.1",
    )
    _set_from_alias("REDIS_PORT", default="6379")
    _set_from_alias("REDIS_PASSWORD", default="")
    _set_from_alias("REMOTE_QUOTE_REDIS_HOST", "REDIS_HOST", default="")
    _set_from_alias("REMOTE_QUOTE_REDIS_PORT", "REDIS_PORT", default="6379")
    _set_from_alias(
        "REMOTE_QUOTE_REDIS_PASSWORD", "REDIS_PASSWORD", default=""
    )
    _build_database_url()
    return runtime_count

