from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.services.api.routers.admin.admin_training_utils import _normalize_payload
from backend.services.engine.training.local_docker_orchestrator import (
    LocalDockerOrchestrator,
)


def _payload(**overrides):
    payload = {
        "display_name": "alpha158",
        "model_type": "lightgbm",
        "feature_mode": "qlib_alpha158",
        "train_start": "2022-01-01",
        "train_end": "2023-12-31",
        "valid_start": "2024-01-15",
        "valid_end": "2024-06-30",
        "test_start": "2024-07-15",
        "test_end": "2024-12-31",
        "features": ["not_in_snapshot_catalog"],
        "num_boost_round": 100,
        "early_stopping_rounds": 10,
        "context": {"market": "CN", "benchmark": "SH000300"},
    }
    payload.update(overrides)
    return payload


def test_native_alpha158_skips_snapshot_feature_catalog():
    normalized = _normalize_payload(_payload(), allowed_features=["custom_only"])

    assert normalized["feature_mode"] == "qlib_alpha158"
    assert normalized["features"] == ["not_in_snapshot_catalog"]


def test_native_alpha158_rejects_remote_or_non_lgb():
    with pytest.raises(HTTPException, match="local training node"):
        _normalize_payload(_payload(node_id="autodl-1"), allowed_features=[])
    with pytest.raises(HTTPException, match="LightGBM"):
        _normalize_payload(_payload(model_type="xgboost"), allowed_features=[])



def test_local_process_mode_does_not_create_a_docker_client(monkeypatch):
    monkeypatch.setenv("TRAINING_LOCAL_EXECUTION_MODE", "process")
    monkeypatch.setenv("INTERNAL_CALL_SECRET", "test-secret")
    orchestrator = LocalDockerOrchestrator()

    assert orchestrator.execution_mode == "process"
    assert orchestrator.docker is None
    assert orchestrator.api_base == "http://127.0.0.1:8000"


def test_optional_daemon_host_path_uses_only_bind_mounts():
    orchestrator = object.__new__(LocalDockerOrchestrator)
    orchestrator._self_mounts = [
        {
            "Type": "bind",
            "Source": "/host/project/data",
            "Destination": "/data",
        }
    ]

    assert (
        orchestrator._optional_daemon_host_path(
            Path("/app/docker/training/train.py")
        )
        is None
    )
    assert orchestrator._optional_daemon_host_path(Path("/data/example.txt")) == Path(
        "/host/project/data/example.txt"
    )


def test_launch_startup_error_is_marked_failed(monkeypatch):
    orchestrator = object.__new__(LocalDockerOrchestrator)
    marked = []

    async def fail_launch(*_args, **_kwargs):
        raise RuntimeError("pre-start failure")

    async def mark_failed(run_id, exc):
        marked.append((run_id, str(exc)))

    monkeypatch.setattr(orchestrator, "_launch_training_job", fail_launch)
    monkeypatch.setattr(orchestrator, "_mark_startup_failed", mark_failed)

    asyncio.run(orchestrator.launch_training_job("run-local", {}))

    assert marked == [("run-local", "pre-start failure")]


def test_native_alpha158_config_does_not_query_parquet(monkeypatch):
    orchestrator = object.__new__(LocalDockerOrchestrator)

    def parquet_should_not_run(*_args, **_kwargs):
        raise AssertionError("native Qlib mode must not inspect feature snapshots")

    monkeypatch.setattr(orchestrator, "_filter_features_by_parquet", parquet_should_not_run)
    config = orchestrator._build_config_yaml("run-alpha", _payload())

    assert config["data"]["feature_mode"] == "qlib_alpha158"
    assert config["data"]["source_mode"] == "QLIB"
    assert config["data"]["features"] == []
    assert config["data"]["qlib_provider_uri"] == "/tmp/qlib_data"
