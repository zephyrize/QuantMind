import os
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.shared import env_loader


class EnvironmentLoaderTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.root_env = patch.object(
            env_loader, "ROOT_ENV_FILE", Path("/missing/root.env")
        )
        self.backend_env = patch.object(
            env_loader, "BACKEND_ENV_FILE", Path("/missing/backend.env")
        )
        self.runtime_env = patch.object(
            env_loader, "load_runtime_env", return_value=0
        )
        self.root_env.start()
        self.backend_env.start()
        self.runtime_env.start()
        env_loader.bootstrap_environment.cache_clear()

    def tearDown(self):
        env_loader.bootstrap_environment.cache_clear()
        self.runtime_env.stop()
        self.backend_env.stop()
        self.root_env.stop()
        self.environment.stop()

    def test_local_runtime_uses_loopback_defaults(self):
        os.environ["QUANTMIND_RUNTIME"] = "local"

        env_loader.bootstrap_environment()

        self.assertFalse(env_loader.is_container_runtime())
        self.assertEqual(os.environ["DB_HOST"], "127.0.0.1")
        self.assertEqual(os.environ["REDIS_HOST"], "127.0.0.1")
        self.assertIn("@127.0.0.1:5432/quantmind", os.environ["DATABASE_URL"])

    def test_docker_runtime_uses_compose_service_defaults(self):
        os.environ["QUANTMIND_RUNTIME"] = "docker"

        env_loader.bootstrap_environment()

        self.assertTrue(env_loader.is_container_runtime())
        self.assertEqual(os.environ["DB_HOST"], "db")
        self.assertEqual(os.environ["REDIS_HOST"], "redis")
        self.assertIn("@db:5432/quantmind", os.environ["DATABASE_URL"])

    def test_explicit_host_override_beats_runtime_default(self):
        os.environ["QUANTMIND_RUNTIME"] = "docker"
        os.environ["DB_HOST"] = "postgres.internal"
        os.environ["REDIS_HOST"] = "redis.internal"

        env_loader.bootstrap_environment()

        self.assertEqual(os.environ["DB_HOST"], "postgres.internal")
        self.assertEqual(os.environ["REDIS_HOST"], "redis.internal")


if __name__ == "__main__":
    unittest.main()
