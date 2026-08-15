"""Resolve Docker-daemon-visible project paths without user configuration.

Nested containers are created through the Docker socket.  Their bind-mount
sources are interpreted by the Docker daemon, so a container path such as
``/app/backend`` cannot be used directly.  This module derives the matching
host path from the current container's own bind mounts.  It also preserves
Windows path separators returned by Docker Desktop.
"""

from __future__ import annotations

import ntpath
import os
import re
import socket
from pathlib import Path
from typing import Any

from backend.shared.env_loader import PROJECT_ROOT, is_container_runtime


class HostProjectPathResolutionError(RuntimeError):
    """Raised when a Docker child-container mount source cannot be resolved."""


_PROJECT_MOUNT_DESTINATIONS = {
    "/data",
    "/app/db",
    "/app/models",
    "/app/logs",
    "/app/user_pools_local",
}


def _is_windows_path(path: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", path)) or "\\" in path


def _dirname(path: str) -> str:
    return (ntpath if _is_windows_path(path) else os.path).dirname(path)


def join_host_project_path(project_root: str, *parts: str) -> str:
    """Join paths using the separator understood by the Docker daemon."""

    path_module = ntpath if _is_windows_path(project_root) else os.path
    return path_module.join(project_root, *parts)


def _current_container_mounts(docker_client: Any) -> list[dict[str, Any]]:
    container_id = os.getenv("HOSTNAME") or socket.gethostname()
    try:
        container = docker_client.containers.get(container_id)
        return list(container.attrs.get("Mounts", []))
    except Exception as exc:
        raise HostProjectPathResolutionError(
            "无法通过 Docker Socket 查询当前容器挂载；"
            "无法安全启动需要 bind mount 的子容器。"
        ) from exc


def _docker_client() -> Any:
    try:
        import docker

        return docker.from_env()
    except Exception as exc:
        raise HostProjectPathResolutionError(
            "Docker SDK 或 Docker Socket 不可用；"
            "无法启动需要 bind mount 的子容器。"
        ) from exc


def resolve_host_project_path(docker_client: Any | None = None) -> str:
    """Return the project root as a path understood by the Docker daemon.

    Host Python processes can use the repository path directly.  A process
    inside Compose instead inspects its own ``/data``/``/app/db`` etc. bind
    mounts and derives their common project root.  No guessed or user-supplied
    path is used when discovery fails.
    """

    if not is_container_runtime():
        return str(PROJECT_ROOT)

    client = docker_client or _docker_client()
    roots: list[str] = []
    for mount in _current_container_mounts(client):
        if mount.get("Type") != "bind":
            continue
        if mount.get("Destination") not in _PROJECT_MOUNT_DESTINATIONS:
            continue
        source = str(mount.get("Source") or "").strip()
        if source:
            roots.append(_dirname(source))

    if not roots:
        raise HostProjectPathResolutionError(
            "当前容器没有可用于推导项目根目录的 bind mount（/data、/app/db、"
            "/app/models、/app/logs）；无法安全启动需要 bind mount 的子容器。"
        )

    # Compose mounts all of these project subdirectories from the same root.
    # Prefer the first match rather than normalizing its syntax: the Docker
    # daemon expects exactly the path representation returned by its API.
    return roots[0]
