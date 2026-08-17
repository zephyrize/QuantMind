from backend.services.engine.training.training_log_stream import TrainingRunLogStream


class _FakeRedis:
    def __init__(self) -> None:
        self.records: list[tuple[bytes, dict[bytes, bytes]]] = []
        self.values: dict[str, str] = {}

    def xadd(self, _key, fields, maxlen, approximate):
        entry_id = f"{len(self.records) + 1}-0".encode()
        payload = {
            str(key).encode(): str(value).encode() for key, value in fields.items()
        }
        self.records.append((entry_id, payload))
        return entry_id

    def setex(self, key, _ttl, value):
        self.values[str(key)] = str(value)

    def get(self, key):
        value = self.values.get(str(key))
        return value.encode() if value is not None else None

    def xrevrange(self, _key, count):
        return list(reversed(self.records[-count:]))


def test_infer_stage_prefers_terminal_orchestration_statuses():
    assert (
        TrainingRunLogStream._infer_stage("completed", "train's l2: 0.9")
        == "completed"
    )
    assert TrainingRunLogStream._infer_stage("waiting_callback") == "waiting_callback"
    assert TrainingRunLogStream._infer_stage("failed") == "failed"


def test_infer_stage_uses_training_log_markers():
    assert (
        TrainingRunLogStream._infer_stage(
            "running", "Time cost: 10s | Loading data Done"
        )
        == "loading_data"
    )
    assert (
        TrainingRunLogStream._infer_stage(
            "running", "Training until validation scores"
        )
        == "training"
    )
    assert (
        TrainingRunLogStream._infer_stage("running", "result.json saved")
        == "saving_artifacts"
    )


def test_snapshot_preserves_ordered_structured_log_entries():
    stream = TrainingRunLogStream()
    stream._client = _FakeRedis()

    stream.append_log(
        run_id="run-1",
        tenant_id="default",
        user_id="user-1",
        line="[SYSTEM] Starting local Conda process",
        status="provisioning",
        progress=5,
    )
    stream.append_log(
        run_id="run-1",
        tenant_id="default",
        user_id="user-1",
        line="Training until validation scores don't improve",
        status="running",
        progress=60,
    )

    snapshot = stream.fetch_snapshot("run-1")

    assert snapshot is not None
    assert snapshot["stage"] == "training"
    assert snapshot["progress"] == 60
    assert [entry["line"] for entry in snapshot["log_entries"]] == [
        "[SYSTEM] Starting local Conda process",
        "Training until validation scores don't improve",
    ]
    assert [entry["source"] for entry in snapshot["log_entries"]] == [
        "orchestrator",
        "training_stdout",
    ]
