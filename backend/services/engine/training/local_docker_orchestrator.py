"""
QuantMind 本地 Docker 训练编排器
==================================
使用本机 docker run 异步执行训练任务，无需云 BatchCompute。

流程：
  1. 生成并挂载 config.yaml
  2. docker run -d 启动训练容器（加入 quantmind-network）
  3. 轮询容器状态，写回 DB
  4. 训练容器完成后通过 callback 回写结果
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import docker
from docker import DockerClient
import yaml

from backend.services.engine.training.training_log_stream import TrainingRunLogStream
from backend.services.engine.training.orchestrator_base import TrainingOrchestrator, REGISTRY
from backend.services.api.training_explain import DEFAULT_EXPLAIN_CFG

logger = logging.getLogger(__name__)

_TRAINING_IMAGE = (os.getenv("TRAINING_IMAGE") or "quantmind-oss:latest").strip()
_CALLBACK_TIMEOUT = int(os.getenv("TRAINING_CALLBACK_TIMEOUT_SECONDS", "600"))
_POLL_INTERVAL = 10  # 秒
_CALLBACK_CHECK_INTERVAL = int(
    os.getenv("TRAINING_CALLBACK_CHECK_INTERVAL_SECONDS", "2")
)
_DOCKER_NETWORK = os.getenv("TRAINING_DOCKER_NETWORK", "quantmind-network")
# ── 路径配置（Docker-in-Docker 场景）────────────────────────────────────────────
# API 容器通过 /var/run/docker.sock 与宿主机 Docker daemon 通信。
# Docker daemon 需要的 volume 路径是 docker-compose.yml 中 bind mount 的
# 宿主机端路径（即 ./data 展开后的绝对路径）。
#
# 已知映射（来自 docker-compose.yml）：
#   ./data:/data        → 宿主机 <compose_dir>/data  ←→ 容器 /data
#   ./backend:/app/backend  → 宿主机 <compose_dir>/backend  ←→ 容器 /app/backend

_LOCAL_DATA_MOUNT_DIR = "/tmp/feature_snapshots"
_QUANTDB_DATA_MOUNT_DIR = "/tmp/quantdb_data"
_QLIB_DATA_MOUNT_DIR = "/tmp/qlib_data"
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_execution_mode() -> str:
    """Use a direct Conda process on the host and Docker only in containers."""
    configured = os.getenv("TRAINING_LOCAL_EXECUTION_MODE", "auto").strip().lower()
    if configured in {"process", "docker"}:
        return configured
    if configured not in {"", "auto"}:
        raise RuntimeError(
            "TRAINING_LOCAL_EXECUTION_MODE must be auto, process, or docker"
        )
    return "docker" if Path("/.dockerenv").exists() else "process"

# ── 训练资源保护：训练期间临时停止其它容器，把内存腾给训练任务 ───────────────────
# 通过 TRAINING_PAUSE_OTHERS=false 可关闭该行为
_PAUSE_OTHERS_ENABLED = os.getenv("TRAINING_PAUSE_OTHERS", "true").strip().lower() not in {
    "0", "false", "no", "off",
}
# 受保护的容器名前缀：训练期间永远不停。可通过环境变量覆盖（逗号分隔）。
# 默认保护 quantmind 全家桶 + 训练容器自身 + 通用基础依赖。
_DEFAULT_PROTECTED_PREFIXES = ("quantmind", "qm-train-")
_PROTECTED_PREFIXES: tuple[str, ...] = tuple(
    p.strip()
    for p in (
        os.getenv("TRAINING_PROTECTED_NAME_PREFIXES")
        or ",".join(_DEFAULT_PROTECTED_PREFIXES)
    ).split(",")
    if p.strip()
)

# 宿主机 compose 工作目录
_qdb_dir = os.getenv("QM_QUANTDB_DATA_DIR", "").strip() or "/data/quantdb"


def _resolve_bind_mount_source(
    container_path: Path, mounts: list[dict[str, Any]]
) -> str | None:
    """Translate a path visible in this container to Docker-daemon source path.

    The Docker SDK talks to the host daemon through ``/var/run/docker.sock``.
    Therefore a child container's volume source must use the source shown by
    ``docker inspect``, not this service container's ``/app`` or ``/data``
    path.  This is essential on Docker Desktop, where the daemon sees paths
    such as ``/run/desktop/mnt/host/c/...``.
    """
    candidates: list[tuple[int, Path, str]] = []
    for mount in mounts:
        if mount.get("Type") != "bind":
            continue
        source = mount.get("Source")
        destination = mount.get("Destination")
        if not source or not destination:
            continue
        destination_path = Path(destination)
        try:
            container_path.relative_to(destination_path)
        except ValueError:
            continue
        candidates.append((len(destination_path.parts), destination_path, source))

    if not candidates:
        return None

    _, destination_path, source = max(candidates, key=lambda item: item[0])
    relative_path = container_path.relative_to(destination_path)
    return str(Path(source) / relative_path)


class LocalDockerOrchestrator(TrainingOrchestrator):
    def __init__(self):
        self.execution_mode = _resolve_execution_mode()
        self.docker = (
            DockerClient.from_env() if self.execution_mode == "docker" else None
        )
        self._self_mounts: list[dict[str, Any]] | None = None
        default_api_base = (
            "http://quantmind-api:8000"
            if self.execution_mode == "docker"
            else "http://127.0.0.1:8000"
        )
        configured_api_base = (
            os.getenv("TRAINING_LOCAL_API_BASE_URL")
            if self.execution_mode == "process"
            else os.getenv("QUANTMIND_API_BASE_URL")
        )
        self.api_base = (configured_api_base or default_api_base).strip()
        self.internal_secret = (os.getenv("INTERNAL_CALL_SECRET") or "").strip()
        # P0-3: 强制 fail-closed。secret 缺失直接抛错，不再用空 secret 走 fail-open。
        if not self.internal_secret:
            raise RuntimeError(
                "INTERNAL_CALL_SECRET not set; cannot start training orchestrator. "
                "Set it in .env or QUANTMIND_ENV=development for auto-generation."
            )
        self.log_stream = TrainingRunLogStream()

    def _get_self_mounts(self) -> list[dict[str, Any]]:
        """Get this service container's bind mounts from the Docker daemon."""
        if self._self_mounts is not None:
            return self._self_mounts

        container_id = os.getenv("HOSTNAME", "").strip()
        if not container_id:
            raise RuntimeError("Cannot determine current container ID for volume mapping")
        try:
            if self.docker is None:
                raise RuntimeError("Docker client is unavailable in process mode")
            container = self.docker.containers.get(container_id)
            self._self_mounts = list(container.attrs.get("Mounts") or [])
        except Exception as exc:
            raise RuntimeError(
                "Cannot inspect the QuantMind container mount mappings; "
                "local Docker training requires access to /var/run/docker.sock"
            ) from exc
        return self._self_mounts

    def _daemon_host_path(self, container_path: Path) -> Path:
        """Return the exact source path the Docker daemon can mount."""
        source = self._optional_daemon_host_path(container_path)
        if source is None:
            raise RuntimeError(
                f"No bind mount maps {container_path} to a Docker host path. "
                "Check the quantmind service volumes before starting training."
            )
        return source

    def _optional_daemon_host_path(self, container_path: Path) -> Path | None:
        """Return a bind-mounted Docker host path, if the path has one."""
        source = _resolve_bind_mount_source(container_path, self._get_self_mounts())
        return Path(source) if source is not None else None

    # ── 训练期间资源保护 ──────────────────────────────────────────────────────────
    @staticmethod
    def _is_protected(name: str) -> bool:
        n = (name or "").lstrip("/")
        return any(n.startswith(p) for p in _PROTECTED_PREFIXES)

    def _pause_others(self, work_dir: Path, run_id: str) -> list[str]:
        """停止所有非保护的运行中容器，把名字写到 work_dir/.paused_containers.json。

        返回被停止的容器名列表。失败时记录 warning 但不抛出，保证训练能继续。
        """
        if not _PAUSE_OTHERS_ENABLED:
            logger.info("[%s] TRAINING_PAUSE_OTHERS disabled, skip", run_id)
            return []

        paused: list[str] = []
        try:
            containers = self.docker.containers.list(filters={"status": "running"})
        except Exception as exc:
            logger.warning("[%s] list running containers failed: %s", run_id, exc)
            return []

        for c in containers:
            name = c.name or ""
            if self._is_protected(name):
                continue
            try:
                # 用 stop 而不是 pause：pause 仍然占内存，stop 才能释放
                c.stop(timeout=20)
                paused.append(name)
                logger.info("[%s] paused container: %s", run_id, name)
            except Exception as exc:
                logger.warning(
                    "[%s] stop container %s failed: %s", run_id, name, exc
                )

        # 落盘，宿主重启 / 进程崩溃后也能从 work_dir 恢复
        try:
            state_path = Path(work_dir) / ".paused_containers.json"
            state_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "paused_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "containers": paused,
                        "protected_prefixes": list(_PROTECTED_PREFIXES),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[%s] write paused-state file failed: %s", run_id, exc)

        if paused:
            logger.info(
                "[%s] paused %d containers to free memory for training: %s",
                run_id,
                len(paused),
                paused,
            )
        return paused

    def _resume_others(self, work_dir: Path, run_id: str) -> list[str]:
        """恢复 _pause_others 停止的容器。幂等：状态文件不存在时直接返回。"""
        state_path = Path(work_dir) / ".paused_containers.json"
        if not state_path.exists():
            return []

        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[%s] read paused-state file failed: %s", run_id, exc)
            return []

        names: list[str] = list(data.get("containers") or [])
        resumed: list[str] = []
        for name in names:
            try:
                c = self.docker.containers.get(name)
                if c.status != "running":
                    c.start()
                resumed.append(name)
                logger.info("[%s] resumed container: %s", run_id, name)
            except docker.errors.NotFound:
                logger.warning(
                    "[%s] cannot resume %s: container no longer exists", run_id, name
                )
            except Exception as exc:
                logger.warning("[%s] start container %s failed: %s", run_id, name, exc)

        # 标记为已处理：保留文件但加 resumed_at，便于事后排查
        try:
            data["resumed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            data["resumed"] = resumed
            state_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        if resumed:
            logger.info(
                "[%s] resumed %d containers after training: %s",
                run_id,
                len(resumed),
                resumed,
            )
        return resumed

    @staticmethod
    def _parse_docker_log_entry(raw_line: str) -> tuple[float, str]:
        """解析 `docker logs --timestamps` 单行，返回 (timestamp, message)。"""
        line = str(raw_line or "").rstrip("\n")
        if not line:
            return 0.0, ""
        if " " not in line:
            return 0.0, line
        ts_part, msg_part = line.split(" ", 1)
        ts_val = 0.0
        try:
            ts_val = datetime.fromisoformat(ts_part.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts_val = 0.0
        return ts_val, msg_part.rstrip("\n")

    @staticmethod
    def _filter_features_by_parquet(
        run_id: str, requested_features: list[str]
    ) -> tuple[list[str], list[str]]:
        """检查请求的特征是否存在于 parquet 中，返回 (valid, missing)。

        不做 return_Nd → mom_ret_Nd 别名回退：features_daily.return_Nd 是未来
        N 日收益，用作特征会泄漏标签。mom_ret_Nd 必须由 l1_factors 提供。
        """
        try:
            import pyarrow.parquet as pq
            from pathlib import Path

            parquet_dir = Path(_LOCAL_DATA_MOUNT_DIR)
            if not parquet_dir.exists():
                logger.warning("[%s] Parquet dir not found: %s", run_id, parquet_dir)
                return requested_features, []

            # 读最新一年的 schema
            parquet_files = sorted(parquet_dir.glob("model_features_*.parquet"))
            if not parquet_files:
                logger.warning("[%s] No parquet files in %s", run_id, parquet_dir)
                return requested_features, []

            schema = pq.ParquetFile(parquet_files[-1]).schema_arrow
            parquet_cols = set(schema.names)

            valid = [f for f in requested_features if f in parquet_cols]
            missing = [f for f in requested_features if f not in parquet_cols]
            return valid, missing
        except Exception as exc:
            logger.warning("[%s] Feature filter failed: %s", run_id, exc)
            return requested_features, []

    @staticmethod
    def _infer_progress_from_log_line(line: str, current: int) -> int:
        text = str(line or "").lower()
        next_progress = int(current)
        if "local data hit:" in text:
            next_progress = max(next_progress, 22)
        if "raw concat size" in text:
            next_progress = max(next_progress, 30)
        if "after date range clip" in text or "data ready:" in text:
            next_progress = max(next_progress, 42)
        if "split mode:" in text or "val_ratio mode:" in text:
            next_progress = max(next_progress, 50)
        # LightGBM / XGBoost / CatBoost early stopping patterns
        if "did not meet early stopping" in text or "early stopping, best iteration" in text:
            next_progress = max(next_progress, 70)
        if "early stopping" in text and ("round" in text or "iteration" in text):
            next_progress = max(next_progress, 70)
        if "training finished" in text:
            next_progress = max(next_progress, 80)
        # Multi-model model save patterns
        if "model saved" in text or "model.lgb" in text or "model.xgb" in text or "model.cbm" in text:
            next_progress = max(next_progress, 85)
        if "predictions saved" in text or "pred.parquet" in text or "pred.pkl" in text:
            next_progress = max(next_progress, 90)
        if "result.json" in text or "result report saved" in text:
            next_progress = max(next_progress, 95)
        if "metadata.json saved" in text or "inference.py" in text:
            next_progress = max(next_progress, 98)
        return min(99, next_progress)

    # ── 构造 config.yaml 内容 ───────────────────────────────────────────────────
    def _build_config_yaml(self, run_id: str, payload: dict) -> dict:
        if payload is None:
            logger.error(
                "[%s] Payload is None in _build_config_yaml, using absolute defaults",
                run_id,
            )
            payload = {}
        context = (
            payload.get("context") if isinstance(payload.get("context"), dict) else {}
        )

        # 默认保持快照 Parquet 训练链路。Qlib 原生模式是显式 opt-in，
        # 不参与快照字段过滤，避免影响既有训练任务。
        feature_mode = str(payload.get("feature_mode") or "snapshot").strip().lower()
        if feature_mode not in {"snapshot", "qlib_alpha158"}:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        data_source_mode = "QLIB" if feature_mode == "qlib_alpha158" else payload.get(
            "data_source_mode", "LOCAL"
        )

        # 过滤掉 parquet 中不存在的特征，避免无效内存分配
        requested_features = payload.get("features", [])
        if feature_mode == "qlib_alpha158":
            valid_features, missing_features = [], []
        else:
            valid_features, missing_features = self._filter_features_by_parquet(
                run_id, requested_features
            )
        if missing_features:
            logger.warning(
                "[%s] %d/%d requested features not in parquet, filtered out: %s...",
                run_id,
                len(missing_features),
                len(requested_features),
                missing_features[:10],
            )
        # 将过滤结果存到 payload 中，供后续返回给前端
        payload["_valid_features"] = valid_features
        payload["_missing_features"] = missing_features

        config: dict[str, Any] = {
            "run_id": run_id,
            "job_name": payload.get("job_name", "unnamed"),
            "data": {
                "train_start": payload.get("train_start", "2022-01-01"),
                "train_end": payload.get("train_end", "2024-12-31"),
                "features": valid_features,
                "feature_mode": feature_mode,
                "source_mode": data_source_mode,
                "local_dir": _LOCAL_DATA_MOUNT_DIR
                if data_source_mode == "LOCAL"
                else None,
                "qlib_provider_uri": _QLIB_DATA_MOUNT_DIR
                if feature_mode == "qlib_alpha158"
                else None,
                "qlib_universe": str(payload.get("qlib_universe") or "all"),
            },
            "model": {
                "type": payload.get("model_type", "lightgbm"),
                "types": payload.get("model_types"),
                "ensemble": payload.get("ensemble", "none"),
                "num_boost_round": payload.get("num_boost_round", 1000),
                "early_stopping_rounds": payload.get("early_stopping_rounds", 100),
                "val_ratio": payload.get("val_ratio", 0.15),
                "params": payload.get("lgb_params", {}),
                "xgb_params": {
                    k: v
                    for k, v in payload.get("xgb_params", {}).items()
                    # LightGBM max_depth=-1 convention is invalid for XGBoost; drop it
                    if not (k == "max_depth" and isinstance(v, (int, float)) and v < 0)
                },
                "catboost_params": payload.get("catboost_params", {}),
                "dl_params": payload.get("dl_params", {}),
            },
            "label": {
                "target_horizon_days": payload.get("target_horizon_days", 1),
                "target_mode": payload.get("target_mode", "return"),
                "label_formula": payload.get("label_formula", ""),
                "effective_trade_date": payload.get("effective_trade_date", ""),
                "training_window": payload.get("training_window", ""),
            },
            "context": {
                "initial_capital": context.get("initial_capital", 1_000_000),
                "benchmark": context.get("benchmark", "SH000300"),
                "commission_rate": context.get("commission_rate", 0.00025),
                "slippage": context.get("slippage", 0.0005),
                "deal_price": context.get("deal_price", "close"),
                "market": context.get("market", "CN"),
                "industry_as_feature": context.get("industry_as_feature", False),
            },
            "explain": payload.get("explain", DEFAULT_EXPLAIN_CFG),
            "output": {
                "result_path": "/workspace/result.json",
                "required_artifacts": payload.get(
                    "required_artifacts",
                    ["model.lgb", "pred.pkl", "metadata.json", "result.json"],
                ),
            },
            "callback": {
                "url": (
                    f"{getattr(self, 'api_base', 'http://quantmind-api:8000')}"
                    f"/api/v1/models/training-runs/{run_id}/complete"
                ),
                "secret": getattr(self, 'internal_secret', ''),
            },
            "cache": {"dir": "/tmp" if data_source_mode == "LOCAL" else None},
        }
        # 显式时间段切分（valid_start/end 优先于 val_ratio）
        split_fields: list[str] = ["valid_start", "valid_end", "test_start", "test_end"]
        if all(payload.get(k) for k in split_fields):
            config["split"] = {
                "train": [payload.get("train_start"), payload.get("train_end")],
                "valid": [payload.get("valid_start"), payload.get("valid_end")],
                "test": [payload.get("test_start"), payload.get("test_end")],
            }
            config["model"]["val_ratio"] = None

        # WFA 稳定性诊断配置（可选，透传给训练脚本）
        if payload.get("wfa") and isinstance(payload.get("wfa"), dict):
            config["wfa"] = payload["wfa"]

        # 训练时长预算（分钟），透传给训练脚本供阶段级超时检查
        try:
            config["max_time_minutes"] = max(10, int(payload.get("max_time_minutes") or 120))
        except Exception:
            config["max_time_minutes"] = 120

        # 特征准入自动化：默认启用 IC/ICIR 因子筛选（剔除无信号特征），
        # 前端/请求显式指定 factor_selection 时以显式配置为准。
        fs_cfg = payload.get("factor_selection")
        if isinstance(fs_cfg, dict):
            config["factor_selection"] = fs_cfg
        elif str(payload.get("auto_feature_filter", "true")).lower() in ("1", "true", "yes", "on"):
            config["factor_selection"] = {
                "method": "ic_icir",
                "n_top": 80,
                "ic_threshold": 0.01,
                "icir_threshold": 0.15,
                "correlation_threshold": 0.9,
            }
        return config

    # ── 启动训练任务 ─────────────────────────────────────────────────────────────
    async def launch_training_job(self, run_id: str, payload: dict = None) -> None:
        """Launch one job and persist every pre-start failure."""
        try:
            await self._launch_training_job(run_id, payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("[%s] training startup failed", run_id)
            await self._mark_startup_failed(run_id, exc)

    async def _mark_startup_failed(self, run_id: str, exc: Exception) -> None:
        """Do not leave a submitted job in provisioning after an early error."""
        from backend.services.api.routers.admin.db import TrainingJobRecord
        from backend.shared.database_manager_v2 import get_session

        message = f"[ERROR] Training startup failed: {exc}"
        tenant_id, user_id = "default", "unknown"
        try:
            async with get_session() as db:
                record = await db.get(TrainingJobRecord, run_id)
                if record:
                    tenant_id = str(record.tenant_id or tenant_id)
                    user_id = str(record.user_id or user_id)
                    if record.status not in {"completed", "failed"}:
                        record.status = "failed"
                        record.progress = 100
                        record.logs = (record.logs or "") + message + "\n"
                        await db.commit()
        finally:
            self.log_stream.append_log(
                run_id=run_id,
                tenant_id=tenant_id,
                user_id=user_id,
                line=message,
                status="failed",
                progress=100,
            )

    async def _launch_local_process(
        self,
        run_id: str,
        payload: dict[str, Any],
        config: dict[str, Any],
        tenant_id: str,
        user_id: str,
    ) -> None:
        """Run the existing training script with this backend's Conda Python."""
        from backend.services.api.routers.admin.db import TrainingJobRecord
        from backend.shared.database_manager_v2 import get_session
        from backend.shared.model_registry import model_registry_service

        model_id = model_registry_service.build_model_id_from_run(run_id)
        workspace = model_registry_service.user_models_root / tenant_id / user_id / model_id
        workspace.mkdir(parents=True, exist_ok=True)
        data_cfg = config.setdefault("data", {})
        data_cfg["local_dir"] = str(_PROJECT_ROOT / "db" / "feature_snapshots")
        if data_cfg.get("feature_mode") == "qlib_alpha158":
            data_cfg["qlib_provider_uri"] = str(_PROJECT_ROOT / "db" / "qlib_data")
        config["output"]["result_path"] = str(workspace / "result.json")
        config_path = workspace / "config.yaml"
        config_path.write_text(
            yaml.dump(config, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

        environment = os.environ.copy()
        environment.update(
            {
                "TRAINING_WORKSPACE_DIR": str(workspace),
                "QLIB_PROVIDER_URI": str(data_cfg.get("qlib_provider_uri") or ""),
                "PYTHONPATH": os.pathsep.join(
                    [str(_PROJECT_ROOT), environment.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep),
            }
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_PROJECT_ROOT / "docker" / "training" / "train.py"),
            "--config",
            str(config_path),
            cwd=str(_PROJECT_ROOT),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        process_id = f"local:{process.pid}"
        async with get_session() as db:
            record = await db.get(TrainingJobRecord, run_id)
            if record:
                record.status = "running"
                record.progress = max(int(record.progress or 0), 12)
                record.instance_id = process_id
                record.logs = (record.logs or "") + f"Local Conda PID: {process.pid}\n"
                await db.commit()
        self.log_stream.append_log(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            line=f"[SYSTEM] Local Conda process started: pid={process.pid}",
            status="running",
            progress=12,
            container_id=process_id,
        )
        REGISTRY.register(
            self._poll_local_process(
                run_id, process, tenant_id=tenant_id, user_id=user_id, payload=payload
            )
        )

    async def _poll_local_process(
        self,
        run_id: str,
        process: asyncio.subprocess.Process,
        *,
        tenant_id: str,
        user_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Forward local stdout and fail closed when a callback is missing."""
        max_minutes = max(10, int(payload.get("max_time_minutes") or 120))
        deadline = time.time() + max_minutes * 60
        progress, tail_logs = 12, []
        stream = process.stdout
        while process.returncode is None:
            if time.time() >= deadline:
                process.terminate()
                await process.wait()
                await self._mark_startup_failed(
                    run_id, RuntimeError(f"Local process exceeded {max_minutes}min limit")
                )
                return
            try:
                raw_line = await asyncio.wait_for(stream.readline(), timeout=1)
            except asyncio.TimeoutError:
                continue
            if not raw_line:
                await process.wait()
                continue
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            tail_logs.append(line)
            tail_logs = tail_logs[-100:]
            progress = self._infer_progress_from_log_line(line, progress)
            self.log_stream.append_log(
                run_id=run_id,
                tenant_id=tenant_id,
                user_id=user_id,
                line=line,
                status="running",
                progress=progress,
                container_id=f"local:{process.pid}",
            )

        from backend.services.api.routers.admin.db import TrainingJobRecord
        from backend.shared.database_manager_v2 import get_session

        async with get_session(read_only=True) as db:
            record = await db.get(TrainingJobRecord, run_id)
            if record and record.status in {"completed", "failed"}:
                return
        detail = "\n".join(tail_logs[-30:])
        if process.returncode == 0:
            error = RuntimeError("Local process exited without training callback")
        else:
            error = RuntimeError(
                f"Local process exited with code {process.returncode}: {detail}"
            )
        await self._mark_startup_failed(run_id, error)

    async def _launch_training_job(self, run_id: str, payload: dict = None) -> None:
        from backend.shared.database_manager_v2 import get_session
        from backend.services.api.routers.admin.db import TrainingJobRecord

        if payload is None:
            logger.error("[%s] Orchestrator received None payload!", run_id)
            payload = {}

        config = self._build_config_yaml(run_id, payload)
        launch_label = (
            "local Conda process" if self.execution_mode == "process" else "container image"
        )
        launch_target = sys.executable if self.execution_mode == "process" else _TRAINING_IMAGE
        async with get_session() as db:
            record = await db.get(TrainingJobRecord, run_id)
            if record:
                record.status = "provisioning"
                record.progress = max(int(record.progress or 0), 5)
                # 增量记录日志，防止覆盖 [SYSTEM] 训练任务已创建
                record.logs = (
                    record.logs or ""
                ) + f"Starting {launch_label}: {launch_target}\n"
                user_id = str(record.user_id or "unknown")
                tenant_id = str(record.tenant_id or "default")

                # 记录系统通知(如日期自动修正)
                notices = payload.get("system_notices") or []
                for msg in notices:
                    record.logs += f"[NOTICE] {msg}\n"

                await db.commit()
                self.log_stream.append_log(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    line=f"[SYSTEM] Starting {launch_label}: {launch_target}",
                    status="provisioning",
                    progress=5,
                )
                # 同时也发到实时日志流
                for msg in notices:
                    self.log_stream.append_log(
                        run_id=run_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        line=f"[NOTICE] {msg}",
                        status="provisioning",
                        progress=5,
                    )
            else:
                logger.warning(
                    "[%s] Training record not found in launch_training_job", run_id
                )
                user_id = "unknown"
                tenant_id = "default"

        if self.execution_mode == "process":
            await self._launch_local_process(
                run_id, payload, config, tenant_id, user_id
            )
            return

        # ── 准备训练工作目录 ────────────────────────────────────────────────────
        # 使用 /data/training_jobs/{run_id} 作为训练容器的工作目录。
        # /data 是 docker-compose 中 ./data:/data 的挂载点，
        # API 容器写入的文件对宿主机和训练容器都可见。
        # 这避免了 _HOST_PROJECT_PATH 在容器内外指向不同文件系统的问题。
        from backend.shared.model_registry import model_registry_service

        model_id = model_registry_service.build_model_id_from_run(run_id)

        # API 容器内的模型注册路径（用于回调后注册模型）
        user_models_root = Path(model_registry_service.user_models_root)
        internal_models_root = (
            user_models_root
            if user_models_root.is_absolute()
            else Path("/app") / user_models_root
        )
        internal_output_dir = internal_models_root / tenant_id / user_id / model_id

        # 训练容器工作目录：使用 /data 挂载点下的路径
        # API 容器内路径：/data/training_jobs/{run_id}（通过 ./data:/data 挂载）
        # 宿主机路径：/opt/quantmind/data/training_jobs/{run_id}（Docker daemon 需要）
        container_work_dir = Path("/data") / "training_jobs" / run_id

        host_output_dir = self._daemon_host_path(container_work_dir)
        if Path("/app/db/feature_snapshots").exists():
            container_local_data_path = Path("/app/db/feature_snapshots")
        else:
            container_local_data_path = Path("/data/feature_snapshots")
        local_data_host_path = self._daemon_host_path(container_local_data_path)
        training_script_host_path = self._optional_daemon_host_path(
            Path("/app/docker/training/train.py")
        )
        feature_mode = str((config.get("data") or {}).get("feature_mode") or "snapshot")

        # 强制创建目录（使用容器内路径，确保 API 容器可写入）
        os.makedirs(internal_output_dir, exist_ok=True)
        os.makedirs(container_work_dir, exist_ok=True)
        logger.info(
            "[%s] Training work directory prepared: %s (host mount: %s)",
            run_id,
            container_work_dir,
            host_output_dir,
        )
        logger.info(
            "[%s] Model registry path prepared: %s",
            run_id,
            internal_output_dir,
        )

        # ── 提前将 config.yaml 写入训练工作目录 ─────────────────────────────
        # 写入 container_work_dir（容器内 /data/training_jobs/{run_id}/），
        # 该目录通过 bind mount 与宿主机 /opt/quantmind/data/training_jobs/{run_id}/ 同步，
        # 会被 Docker 挂载为训练容器的 /workspace
        config_path = container_work_dir / "config.yaml"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                f.flush()
                os.fsync(f.fileno())
            logger.info("[%s] Config saved: %s", run_id, config_path)
            # Verify the file is visible (bind mount propagation)
            if not config_path.exists():
                raise RuntimeError(f"Config file not visible after write: {config_path}")
        except Exception as e:
            logger.error("[%s] Failed to save config: %s", run_id, e)
            raise

        # 始终挂载本地数据目录（宿主机路径，API 容器内 os.path.exists 无法感知）
        volumes: dict[str, dict[str, str]] = {
            str(host_output_dir): {"bind": "/workspace", "mode": "rw"},
            str(local_data_host_path): {"bind": _LOCAL_DATA_MOUNT_DIR, "mode": "ro"},
        }
        if feature_mode == "qlib_alpha158":
            configured_qlib_uri = os.getenv("QLIB_PROVIDER_URI", "").strip()
            qlib_candidates = [
                Path(configured_qlib_uri) if configured_qlib_uri else None,
                Path("/app/db/qlib_data"),
                Path("/data/quantdb/.qlib_cache/cn_data"),
            ]
            qlib_source = next(
                (candidate for candidate in qlib_candidates if candidate and candidate.exists()),
                None,
            )
            if qlib_source is None:
                raise RuntimeError(
                    "Qlib Alpha158 mode requires local Qlib binary data. "
                    "Set QLIB_PROVIDER_URI or provide db/qlib_data."
                )
            qlib_data_host_path = self._daemon_host_path(qlib_source)
            volumes[str(qlib_data_host_path)] = {
                "bind": _QLIB_DATA_MOUNT_DIR,
                "mode": "ro",
            }
            logger.info(
                "[%s] Qlib binary data mounted: %s -> %s",
                run_id,
                qlib_data_host_path,
                _QLIB_DATA_MOUNT_DIR,
            )
        # 挂载 QuantDB 全量数据（6大类：kline/base_sector/financial/bond_etf/technical_derived/ml_datasets）
        # 存在性检查必须针对【容器内】可见的 _qdb_dir，而非宿主机路径
        # （API 容器内 os.path.exists 无法感知宿主机路径，与 _LOCAL_DATA_PATH 同理）
        if Path(_qdb_dir).exists():
            quantdb_data_host_path = self._daemon_host_path(Path(_qdb_dir))
            volumes[str(quantdb_data_host_path)] = {
                "bind": _QUANTDB_DATA_MOUNT_DIR,
                "mode": "ro",
            }
            logger.info(
                "[%s] QuantDB data mounted: %s (host) -> %s",
                run_id,
                quantdb_data_host_path,
                _QUANTDB_DATA_MOUNT_DIR,
            )
        else:
            logger.warning(
                "[%s] QuantDB data dir not visible at %s; skipping mount "
                "(industry code ind_code_l1 will be unavailable)",
                run_id,
                _qdb_dir,
            )
        logger.info(
            "[%s] Training workspace mounted: %s (host) -> /workspace (container writes to %s)",
            run_id,
            host_output_dir,
            container_work_dir,
        )
        logger.info(
            "[%s] Local data path mounted: %s -> %s",
            run_id,
            local_data_host_path,
            _LOCAL_DATA_MOUNT_DIR,
        )
        # 开发环境下用宿主机脚本覆盖镜像内置版；生产镜像未挂载源码时，
        # 直接使用 Dockerfile 已 COPY 的 /app/train.py，不能因此阻止训练启动。
        if training_script_host_path is not None:
            volumes[str(training_script_host_path)] = {
                "bind": "/app/train.py",
                "mode": "ro",
            }
            logger.info(
                "[%s] Local train.py override mounted: %s -> /app/train.py",
                run_id,
                training_script_host_path,
            )
        else:
            logger.warning(
                "[%s] No bind-mounted local train.py found; using the image-built "
                "/app/train.py. Rebuild %s after training-script changes.",
                run_id,
                _TRAINING_IMAGE,
            )
        logger.info(
            "[%s] PERSISTENCE Local output mounted: %s (host) -> /workspace (container: %s)",
            run_id,
            host_output_dir,
            container_work_dir,
        )
        logger.info("[%s] Final volumes config: %s", run_id, volumes)

        # 启动训练容器之前：停掉其它非保护容器，把内存腾出来给训练
        # 用 to_thread 包装：避免 docker.stop（含 SIGTERM 等待）阻塞主 event loop
        try:
            paused = await asyncio.to_thread(
                self._pause_others, container_work_dir, run_id
            )
            if paused:
                self.log_stream.append_log(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    line=f"[SYSTEM] Paused {len(paused)} containers to free memory: "
                    + ", ".join(paused),
                    status="provisioning",
                    progress=10,
                )
        except Exception as pause_err:
            logger.warning("[%s] pause others failed (continuing): %s", run_id, pause_err)

        try:
            # P0-2 孤儿容器策略 A：launch 入口先停 + 删除同名旧容器
            # 场景：API 重启 → recover 重新调度 launch → 旧容器可能仍 Running/Exited
            # - running: stop(timeout=10) 优雅停 → remove
            # - exited: 直接 remove
            # - NotFound: 正常，继续
            container_name = f"qm-train-{run_id}"
            try:
                existing = await asyncio.to_thread(
                    self.docker.containers.get, container_name
                )
                if existing.status == "running":
                    logger.warning(
                        "[%s] orphan container %s still running, stopping first",
                        run_id, container_name,
                    )
                    await asyncio.to_thread(existing.stop, timeout=10)
                await asyncio.to_thread(existing.remove)
                logger.info(
                    "[%s] removed orphan container %s (status=%s)",
                    run_id, container_name, existing.status,
                )
            except Exception as get_exc:
                # NotFound 走这里（容器不存在，正常）；其它异常 warn 但不阻塞
                if "NotFound" in type(get_exc).__name__ or "not found" in str(get_exc).lower():
                    pass  # 正常：没有旧容器
                else:
                    logger.warning(
                        "[%s] check orphan container %s failed (continuing): %s",
                        run_id, container_name, get_exc,
                    )

            # GPU 设备请求：请求所有可用 GPU
            device_requests = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
            container = await asyncio.to_thread(
                self.docker.containers.run,
                _TRAINING_IMAGE,
                command=["python", "/app/train.py", "--config", "/workspace/config.yaml"],
                environment={
                    "INTERNAL_CALL_SECRET": self.internal_secret,
                    "USE_LOCAL_DATA": "true",
                    "TRAINING_LOCAL_DATA_DIR": _LOCAL_DATA_MOUNT_DIR,
                    "TRAINING_CACHE_DIR": "/tmp",
                    "QLIB_PROVIDER_URI": os.getenv("QLIB_PROVIDER_URI", ""),
                    "QUANTDB_DATA_DIR": _QUANTDB_DATA_MOUNT_DIR,
                },
                volumes=volumes,
                network=_DOCKER_NETWORK,
                detach=True,
                name=container_name,
                device_requests=device_requests,
            )
        except Exception as e:
            from backend.shared.database_manager_v2 import get_session
            from backend.services.api.routers.admin.db import TrainingJobRecord

            logger.error("[%s] docker run failed: %s", run_id, e)
            async with get_session() as db:
                record = await db.get(TrainingJobRecord, run_id)
                if record:
                    record.status = "failed"
                    record.logs = (
                        record.logs or ""
                    ) + f"[ERROR] docker run failed: {e}\n"
                    record.progress = 100
                    await db.commit()
            self.log_stream.append_log(
                run_id=run_id,
                tenant_id=tenant_id,
                user_id=user_id,
                line=f"[ERROR] docker run failed: {e}",
                status="failed",
                progress=100,
            )
            # 启动失败，立刻恢复被暂停的容器
            try:
                await asyncio.to_thread(
                    self._resume_others, container_work_dir, run_id
                )
            except Exception as resume_err:
                logger.warning(
                    "[%s] resume others after docker run failure failed: %s",
                    run_id,
                    resume_err,
                )
            return

        logger.info("[%s] Container started: %s", run_id, container.id[:12])
        async with get_session() as db:
            record = await db.get(TrainingJobRecord, run_id)
            if record:
                record.status = "running"
                record.progress = max(int(record.progress or 0), 12)
                record.instance_id = container.id[:12]
                record.logs = (
                    record.logs or ""
                ) + f"Container ID: {container.id[:12]}\n"
                await db.commit()
        self.log_stream.append_log(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            line=f"[SYSTEM] Container ID: {container.id[:12]}",
            status="running",
            progress=12,
            container_id=container.id[:12],
        )

        REGISTRY.register(
            self._poll_container(
                run_id,
                container.id,
                tenant_id=tenant_id,
                user_id=user_id,
                work_dir=container_work_dir,
            )
        )

    # ── 轮询容器状态 ─────────────────────────────────────────────────────────────
    async def _poll_container(
        self,
        run_id: str,
        container_id: str,
        *,
        tenant_id: str,
        user_id: str,
        work_dir: Path | None = None,
    ) -> None:
        from backend.services.api.routers.admin.db import TrainingJobRecord
        from backend.shared.database_manager_v2 import get_session

        async def _try_resume() -> None:
            if work_dir is None:
                return
            try:
                # docker start 可能阻塞数秒，丢到线程池避免卡住 event loop
                resumed = await asyncio.to_thread(
                    self._resume_others, work_dir, run_id
                )
                if resumed:
                    self.log_stream.append_log(
                        run_id=run_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        line=f"[SYSTEM] Resumed {len(resumed)} containers: "
                        + ", ".join(resumed),
                        status=None,
                        progress=None,
                        container_id=container_id[:12],
                    )
            except Exception as exc:
                logger.warning("[%s] resume others failed: %s", run_id, exc)

        # 训练时长预算：默认 120 分钟，可通过 payload.max_time_minutes 配置
        max_time_minutes = 120
        try:
            max_time_minutes = max(10, int(payload.get("max_time_minutes") or 120))
        except Exception:
            max_time_minutes = 120
        deadline = time.time() + max_time_minutes * 60
        log_cursor_ts = max(0.0, time.time() - 2)
        last_log_sig = ""
        current_progress = 12

        while time.time() < deadline:
            await asyncio.sleep(_POLL_INTERVAL)
            try:
                c = self.docker.containers.get(container_id)
                c.reload()
                status = c.attrs["State"].get("Status", "")
                exit_code = c.attrs["State"].get("ExitCode", -1)

                # 增量抓取容器日志并写入回测 Redis，供前端轮询时查看真实进度
                try:
                    raw_logs = c.logs(
                        stdout=True,
                        stderr=True,
                        since=max(0, int(log_cursor_ts) - 1),
                        timestamps=True,
                    ).decode("utf-8", errors="replace")
                    if raw_logs:
                        for raw_line in raw_logs.splitlines():
                            ts_val, msg = self._parse_docker_log_entry(raw_line)
                            if not msg:
                                continue
                            sig = f"{ts_val:.6f}:{msg}"
                            if sig == last_log_sig:
                                continue
                            if ts_val > 0:
                                log_cursor_ts = max(log_cursor_ts, ts_val)
                            last_log_sig = sig
                            current_progress = self._infer_progress_from_log_line(
                                msg, current_progress
                            )
                            self.log_stream.append_log(
                                run_id=run_id,
                                tenant_id=tenant_id,
                                user_id=user_id,
                                line=msg,
                                status="running",
                                progress=current_progress,
                                container_id=container_id[:12],
                            )
                except Exception as log_err:
                    logger.debug(
                        "[%s] incremental log fetch failed: %s", run_id, log_err
                    )

                if status in ("running", "created"):
                    continue

                # 容器已结束，获取最后100行日志
                tail_logs = c.logs(tail=100).decode("utf-8", errors="replace")

                if exit_code == 0:
                    # 容器成功退出：将训练产物复制到模型注册目录，
                    # 供 complete_training_run → register_model_from_training_run 使用。
                    try:
                        import shutil
                        from backend.shared.model_registry import model_registry_service

                        user_models_root = Path(model_registry_service.user_models_root)
                        internal_models_root = (
                            user_models_root
                            if user_models_root.is_absolute()
                            else Path("/app") / user_models_root
                        )
                        internal_model_dir = (
                            internal_models_root / tenant_id / user_id / model_id
                        )
                        internal_model_dir.mkdir(parents=True, exist_ok=True)

                        # 从训练工作目录复制产物到模型注册目录
                        # 支持多框架模型文件：model.lgb (LightGBM), model.xgb (XGBoost),
                        # model.cbm (CatBoost), model.pkl (sklearn/Linear), model.pth (PyTorch)
                        for artifact in ("model.lgb", "model.xgb", "model.cbm", "model.pkl",
                                         "model.pth", "metadata.json", "pred.parquet",
                                         "pred.pkl", "config.yaml", "result.json",
                                         "inference.py", "shap_summary.csv"):
                            src = container_work_dir / artifact
                            if src.exists():
                                shutil.copy2(str(src), str(internal_model_dir / artifact))

                        logger.info(
                            "[%s] Training artifacts copied to %s",
                            run_id,
                            internal_model_dir,
                        )

                        # 可选：同步到生产模型目录（系统内置模型）
                        if payload.get("deploy_to_production"):
                            try:
                                prod_models_root = Path(
                                    model_registry_service.production_models_root
                                )
                                if not prod_models_root.is_absolute():
                                    prod_models_root = Path("/app") / prod_models_root
                                prod_model_dir = prod_models_root / model_id
                                prod_model_dir.mkdir(parents=True, exist_ok=True)

                                for artifact in ("model.lgb", "model.xgb", "model.cbm", "model.pkl",
                                                 "model.pth", "metadata.json", "pred.parquet",
                                                 "pred.pkl", "config.yaml", "result.json",
                                                 "inference.py", "shap_summary.csv"):
                                    src = container_work_dir / artifact
                                    if src.exists():
                                        shutil.copy2(str(src), str(prod_model_dir / artifact))

                                logger.info(
                                    "[%s] Production model artifacts copied to %s",
                                    run_id,
                                    prod_model_dir,
                                )
                            except Exception as prod_err:
                                logger.warning(
                                    "[%s] Failed to copy production artifacts: %s",
                                    run_id, prod_err,
                                )
                    except Exception as copy_err:
                        logger.warning("[%s] Failed to copy artifacts: %s", run_id, copy_err)

                    async with get_session() as db:
                        r = await db.get(TrainingJobRecord, run_id)
                        if r:
                            r.status = "waiting_callback"
                            r.progress = max(int(r.progress or 0), 95)
                            r.logs = (
                                (r.logs or "")
                                + f"[DONE] Container exited 0, waiting callback\n{tail_logs}"
                            )
                            await db.commit()
                    self.log_stream.append_log(
                        run_id=run_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        line="[DONE] Container exited 0, waiting callback",
                        status="waiting_callback",
                        progress=95,
                        container_id=container_id[:12],
                    )
                    # 等 callback；回调一到立即结束等待并清理容器，避免容器长时间停留在 Exited
                    callback_deadline = time.time() + _CALLBACK_TIMEOUT
                    callback_received = False
                    while time.time() < callback_deadline:
                        await asyncio.sleep(max(1, _CALLBACK_CHECK_INTERVAL))
                        async with get_session(read_only=True) as db:
                            r = await db.get(TrainingJobRecord, run_id)
                            if r and str(r.status or "") in {"completed", "failed"}:
                                callback_received = True
                                break
                    if not callback_received:
                        async with get_session() as db:
                            r = await db.get(TrainingJobRecord, run_id)
                            if r and r.status == "waiting_callback":
                                r.status = "failed"
                                r.logs = (
                                    r.logs or ""
                                ) + "[TIMEOUT] Callback not received\n"
                                r.progress = 100
                                await db.commit()
                                self.log_stream.append_log(
                                    run_id=run_id,
                                    tenant_id=tenant_id,
                                    user_id=user_id,
                                    line="[TIMEOUT] Callback not received",
                                    status="failed",
                                    progress=100,
                                    container_id=container_id[:12],
                                )
                else:
                    async with get_session() as db:
                        r = await db.get(TrainingJobRecord, run_id)
                        if r:
                            r.status = "failed"
                            r.logs = (
                                r.logs or ""
                            ) + f"[FAILED] ExitCode={exit_code}\n{tail_logs}"
                            r.progress = 100
                            await db.commit()
                    self.log_stream.append_log(
                        run_id=run_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        line=f"[FAILED] ExitCode={exit_code}",
                        status="failed",
                        progress=100,
                        container_id=container_id[:12],
                    )
                    logger.error("[%s] Training failed, ExitCode=%d", run_id, exit_code)

                try:
                    c.remove(force=True, v=True)
                except Exception:
                    pass
                await _try_resume()
                return

            except docker.errors.NotFound:
                async with get_session() as db:
                    r = await db.get(TrainingJobRecord, run_id)
                    if r and r.status not in ("completed", "failed"):
                        r.status = "failed"
                        r.logs = (r.logs or "") + "[ERROR] Container not found\n"
                        r.progress = 100
                        await db.commit()
                        self.log_stream.append_log(
                            run_id=run_id,
                            tenant_id=tenant_id,
                            user_id=user_id,
                            line="[ERROR] Container not found",
                            status="failed",
                            progress=100,
                            container_id=container_id[:12],
                        )
                await _try_resume()
                return
            except Exception as e:
                logger.warning("[%s] poll error: %s", run_id, e)

        # 超出时长预算
        async with get_session() as db:
            r = await db.get(TrainingJobRecord, run_id)
            if r and r.status not in ("completed", "failed"):
                r.status = "failed"
                r.logs = (r.logs or "") + f"[TIMEOUT] {max_time_minutes}min limit exceeded\n"
                r.progress = 100
                await db.commit()
                self.log_stream.append_log(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    line=f"[TIMEOUT] {max_time_minutes}min limit exceeded",
                    status="failed",
                    progress=100,
                    container_id=container_id[:12],
                )
        await _try_resume()


    # ── 多周期训练编排（一次训练产出多周期模型 + 自动融合）───────────────────────
    async def launch_multi_horizon_job(
        self,
        parent_run_id: str,
        child_run_ids: list[str],
        payload: dict,
    ) -> None:
        """串行跑多个周期的训练任务，全部成功后自动创建 ICIR 加权融合模型。

        每个 child 是一个独立单周期训练任务（已有完整 Docker 容器 + 回调闭环）。
        编排器按顺序依次启动，等待每个 child 完成（或失败），再推进下一个。
        全部成功 → 调 register_ensemble_model 生成「多周期融合模型」。
        """
        from backend.shared.database_manager_v2 import get_session
        from backend.services.api.routers.admin.db import TrainingJobRecord
        from backend.shared.model_registry import model_registry_service

        tenant_id = str(payload.get("_tenant_id") or "")
        user_id = str(payload.get("_user_id") or "")
        # 从 parent record 读取归属
        if not tenant_id or not user_id:
            async with get_session() as db:
                parent_rec = await db.get(TrainingJobRecord, parent_run_id)
                if parent_rec:
                    tenant_id = str(parent_rec.tenant_id or "default")
                    user_id = str(parent_rec.user_id or "")
        display_name = str(payload.get("display_name") or "multi_horizon")

        async def _set_parent(status: str, progress: int, log_line: str) -> None:
            async with get_session() as db:
                r = await db.get(TrainingJobRecord, parent_run_id)
                if r:
                    r.status = status
                    r.progress = progress
                    r.logs = (r.logs or "") + log_line
                    await db.commit()
                self.log_stream.append_log(
                    run_id=parent_run_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    line=log_line.strip(),
                    status=status,
                    progress=progress,
                )

        try:
            await _set_parent("provisioning", 5, f"[MH] 多周期训练启动，共 {len(child_run_ids)} 个周期\n")

            completed_model_ids: list[str] = []
            horizon_labels: list[str] = []
            n_total = len(child_run_ids)

            for idx, child_run_id in enumerate(child_run_ids):
                # 读取 child payload（含固定 target_horizon_days）
                async with get_session() as db:
                    child_rec = await db.get(TrainingJobRecord, child_run_id)
                    if child_rec is None:
                        raise RuntimeError(f"child run not found: {child_run_id}")
                    child_payload = (
                        child_rec.request_payload
                        if isinstance(child_rec.request_payload, dict)
                        else {}
                    )
                horizon = int(child_payload.get("target_horizon_days") or 0)
                horizon_labels.append(f"T{horizon}")

                base_progress = 5 + int((idx / n_total) * 90)
                await _set_parent(
                    "running",
                    base_progress,
                    f"[MH] ({idx + 1}/{n_total}) 训练 T+{horizon} 模型…\n",
                )

                # 启动 child 训练（内部会启容器 + 等回调 + 注册模型）
                await self.launch_training_job(run_id=child_run_id, payload=child_payload)

                # 等待 child 完成
                child_deadline = time.time() + 7200
                while time.time() < child_deadline:
                    await asyncio.sleep(_POLL_INTERVAL)
                    async with get_session(read_only=True) as db:
                        r = await db.get(TrainingJobRecord, child_run_id)
                        if r is None:
                            break
                        st = str(r.status or "")
                        if st == "completed":
                            completed_model_ids.append(
                                model_registry_service.build_model_id_from_run(child_run_id)
                            )
                            break
                        if st == "failed":
                            raise RuntimeError(
                                f"child T+{horizon} training failed: {(r.result or {}).get('error') or (r.logs or '')[-300:]}"
                            )

                if child_run_id not in completed_model_ids:
                    raise RuntimeError(f"child T+{horizon} timed out waiting for completion")

                await _set_parent(
                    "running",
                    5 + int(((idx + 1) / n_total) * 90),
                    f"[MH] T+{horizon} 模型训练完成（{idx + 1}/{n_total}）\n",
                )

            # ── 全部完成 → 创建融合模型 ──
            if len(completed_model_ids) < 2:
                raise RuntimeError("multi-horizon requires at least 2 completed models")

            fusion_name = f"{display_name}_MultiHorizon"
            fusion = await model_registry_service.register_ensemble_model(
                tenant_id=tenant_id,
                user_id=user_id,
                source_model_ids=completed_model_ids,
                display_name=fusion_name,
                weight_strategy="icir",
            )
            fusion_model_id = str(fusion.get("model_id") or "")

            await _set_parent(
                "completed",
                100,
                f"[MH] 融合模型已创建: {fusion_model_id}（ICIR 加权，周期: {'+'.join(horizon_labels)}）\n",
            )

            # 把融合模型信息 + 最丰富的一个 child 完整结果写入 parent result，
            # 保证前端 parseTrainingResult 能正常解析（metrics + artifacts 必需）
            async with get_session() as db:
                r = await db.get(TrainingJobRecord, parent_run_id)
                if r:
                    child_results = []
                    for child_run_id in child_run_ids:
                        child_rec = await db.get(TrainingJobRecord, child_run_id)
                        if child_rec and isinstance(child_rec.result, dict):
                            child_results.append(
                                {
                                    "run_id": child_run_id,
                                    "target_horizon_days": int(
                                        (child_rec.request_payload or {}).get("target_horizon_days") or 0
                                    ),
                                    "result": child_rec.result,
                                }
                            )
                    # 选 metrics 最完整的 child 作为展示基底
                    base_result: dict = {}
                    for cr in child_results:
                        m = (cr.get("result") or {}).get("metrics") or {}
                        if m.get("train") and m.get("val") and m.get("test"):
                            base_result = cr["result"]
                            break
                    parent_result = dict(base_result)
                    parent_result["status"] = "completed"
                    parent_result["multi_horizon"] = {
                        "horizons": horizon_labels,
                        "child_run_ids": child_run_ids,
                        "child_model_ids": completed_model_ids,
                        "fusion_model_id": fusion_model_id,
                        "child_results": child_results,
                    }
                    if isinstance(parent_result.get("metadata"), dict):
                        parent_result["metadata"]["multi_horizon"] = {
                            "horizons": horizon_labels,
                            "fusion_model_id": fusion_model_id,
                        }
                    r.result = parent_result
                    await db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] multi-horizon orchestration failed: %s", parent_run_id, exc)
            await _set_parent(
                "failed",
                100,
                f"[MH] 多周期训练失败: {exc}\n",
            )
