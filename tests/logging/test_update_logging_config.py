from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
LOGGING_CONFIG_PATH = ROOT / "tools" / "logging" / "update_logging_config.py"
LOGGING_ENV_KEYS = (
    "NO_COLOR",
    "VLLM_LOGGING_PREFIX",
    "VLLM_LOGGING_LEVEL",
    "VLLM_LOGGING_COLOR",
    "VLLM_LOGGING_STREAM",
)


def _load_logging_config_module(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str] | None = None,
) -> Any:
    for key in LOGGING_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    module_name = f"update_logging_config_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, LOGGING_CONFIG_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"VLLM_LOGGING_COLOR": "1"}, True),
        ({"VLLM_LOGGING_COLOR": "0"}, False),
        ({"VLLM_LOGGING_COLOR": "1", "NO_COLOR": "true"}, False),
    ],
)
def test_use_color_respects_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    expected: bool,
) -> None:
    module = _load_logging_config_module(monkeypatch, env)

    assert module._use_color() is expected


def test_build_logging_config_uses_import_time_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_logging_config_module(
        monkeypatch,
        {
            "VLLM_LOGGING_PREFIX": "[api] ",
            "VLLM_LOGGING_LEVEL": "debug",
            "VLLM_LOGGING_COLOR": "0",
            "VLLM_LOGGING_STREAM": "ext://sys.stderr",
        },
    )

    config = module._build_logging_config("server.log")

    assert config["omni_logging_format"] is True
    assert config["formatters"]["vllm"]["format"].startswith("[api] ")
    assert config["handlers"]["vllm"]["filename"] == "server.log"
    assert config["handlers"]["vllm"]["level"] == "DEBUG"
    assert config["handlers"]["console"]["formatter"] == "vllm"
    assert config["handlers"]["console"]["stream"] == "ext://sys.stderr"
    assert config["loggers"]["vllm"]["handlers"] == ["vllm", "console"]
    assert config["loggers"]["vllm"]["propagate"] is False


def test_delete_old_log_files_removes_target_and_rotated_backups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_logging_config_module(monkeypatch)
    log_file = tmp_path / "server.log"
    rotated_file = tmp_path / "server.log.1"
    gz_rotated_file = tmp_path / "server.log.2.gz"
    unrelated_file = tmp_path / "other.log"
    for file_path in (log_file, rotated_file, gz_rotated_file, unrelated_file):
        file_path.write_text("old log", encoding="utf-8")

    module.delete_old_log_files(str(log_file))

    assert not log_file.exists()
    assert not rotated_file.exists()
    assert not gz_rotated_file.exists()
    assert unrelated_file.exists()


def test_save_config_writes_json_to_requested_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_logging_config_module(monkeypatch)
    output_path = tmp_path / "logging_config.json"
    config = {"version": 1, "handlers": {"vllm": {"level": "INFO"}}}

    returned_path = module.save_config(config, str(output_path))

    assert returned_path == str(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == config


def test_main_generates_config_and_respects_no_delete_old(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_logging_config_module(monkeypatch, {"VLLM_LOGGING_LEVEL": "warning"})
    log_file = tmp_path / "server.log"
    output_path = tmp_path / "generated_logging_config.json"
    log_file.write_text("old log", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_logging_config.py",
            "--logging-file-name",
            str(log_file),
            "--output",
            str(output_path),
            "--no-delete-old",
        ],
    )

    assert module.main() == 0

    captured = capsys.readouterr()
    config = json.loads(output_path.read_text(encoding="utf-8"))
    assert captured.out == str(output_path)
    assert log_file.exists()
    assert config["handlers"]["vllm"]["filename"] == str(log_file)
    assert config["handlers"]["vllm"]["level"] == "WARNING"
