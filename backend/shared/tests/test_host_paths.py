import os
import unittest
from unittest.mock import patch

from backend.shared import host_paths


class _FakeContainer:
    def __init__(self, mounts):
        self.attrs = {"Mounts": mounts}


class _FakeContainers:
    def __init__(self, mounts):
        self._container = _FakeContainer(mounts)
        self.requested_id = None

    def get(self, container_id):
        self.requested_id = container_id
        return self._container


class _FakeDockerClient:
    def __init__(self, mounts):
        self.containers = _FakeContainers(mounts)


class HostProjectPathTests(unittest.TestCase):
    def test_local_runtime_uses_repository_root(self):
        with patch.dict(os.environ, {"QUANTMIND_RUNTIME": "local"}, clear=True):
            self.assertEqual(
                host_paths.resolve_host_project_path(),
                str(host_paths.PROJECT_ROOT),
            )

    def test_container_runtime_derives_root_from_data_bind_mount(self):
        client = _FakeDockerClient(
            [
                {
                    "Type": "bind",
                    "Destination": "/data",
                    "Source": "/opt/quantmind/data",
                }
            ]
        )
        with patch.dict(os.environ, {"QUANTMIND_RUNTIME": "docker"}, clear=True):
            root = host_paths.resolve_host_project_path(client)

        self.assertEqual(root, "/opt/quantmind")
        self.assertEqual(
            host_paths.join_host_project_path(root, "backend"),
            "/opt/quantmind/backend",
        )

    def test_windows_docker_source_keeps_windows_separator(self):
        client = _FakeDockerClient(
            [
                {
                    "Type": "bind",
                    "Destination": "/app/db",
                    "Source": r"C:\Users\alice\QuantMind\db",
                }
            ]
        )
        with patch.dict(os.environ, {"QUANTMIND_RUNTIME": "docker"}, clear=True):
            root = host_paths.resolve_host_project_path(client)

        self.assertEqual(root, r"C:\Users\alice\QuantMind")
        self.assertEqual(
            host_paths.join_host_project_path(root, "models"),
            r"C:\Users\alice\QuantMind\models",
        )

    def test_container_runtime_rejects_unknown_mount_layout(self):
        client = _FakeDockerClient(
            [
                {
                    "Type": "volume",
                    "Destination": "/data",
                    "Source": "quantmind-data",
                }
            ]
        )
        with patch.dict(os.environ, {"QUANTMIND_RUNTIME": "docker"}, clear=True):
            with self.assertRaises(host_paths.HostProjectPathResolutionError):
                host_paths.resolve_host_project_path(client)


if __name__ == "__main__":
    unittest.main()
