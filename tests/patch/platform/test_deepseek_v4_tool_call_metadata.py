from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from typing import Any, ClassVar

import pytest

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "ascend_vllm").is_dir())
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "platform" / "patch_deepseek_v4_tool_call_metadata.py"


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_vllm_stubs(monkeypatch: pytest.MonkeyPatch):
    vllm = _make_package("vllm")
    entrypoints = _make_package("vllm.entrypoints")
    openai = _make_package("vllm.entrypoints.openai")
    engine = _make_package("vllm.entrypoints.openai.engine")
    protocol = types.ModuleType("vllm.entrypoints.openai.engine.protocol")

    class ModelStub:
        """Minimal Pydantic-compatible stub needed by the monkey patch."""

        field_defaults: ClassVar[dict[str, Any]] = {}

        def __init__(self, **values: Any) -> None:
            self._fields_set = set(values)
            for name, default in self.field_defaults.items():
                value = values.pop(name, default() if callable(default) else default)
                setattr(self, name, value)
            for name, value in values.items():
                setattr(self, name, value)

        @classmethod
        def model_validate(cls, values: dict[str, Any]):
            return cls(**values)

        def model_dump(self, *, exclude_unset: bool = False) -> dict[str, Any]:
            def serialize(value: Any) -> Any:
                if isinstance(value, ModelStub):
                    return value.model_dump(exclude_unset=exclude_unset)
                if isinstance(value, list):
                    return [serialize(item) for item in value]
                return value

            return {
                name: serialize(getattr(self, name))
                for name in self.field_defaults
                if not exclude_unset or name in self._fields_set
            }

    class DeltaFunctionCall(ModelStub):
        field_defaults = {"name": None, "arguments": None}

    class DeltaToolCall(ModelStub):
        field_defaults = {"id": None, "type": None, "index": None, "function": None}

        def __init__(self, **values: Any) -> None:
            if isinstance(values.get("function"), dict):
                values["function"] = DeltaFunctionCall(**values["function"])
            super().__init__(**values)

    class DeltaMessage(ModelStub):
        field_defaults = {"content": None, "tool_calls": list}

    vars(protocol)["DeltaMessage"] = DeltaMessage
    vars(protocol)["DeltaToolCall"] = DeltaToolCall

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entrypoints)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai", openai)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.engine.protocol", protocol)

    vllm_ascend = _make_package("vllm_ascend")
    vllm_ascend_patch = _make_package("vllm_ascend.patch")
    vllm_ascend_platform = _make_package("vllm_ascend.patch.platform")
    upstream = types.ModuleType("vllm_ascend.patch.platform.patch_deepseek_v4_tool_call_parser")

    class DeepSeekV4ToolParser:
        pass

    def original_pop_pending_delta_message(self):
        return self.pending_message

    vars(upstream)["DeepSeekV4ToolParser"] = DeepSeekV4ToolParser
    vars(upstream)["_pop_pending_delta_message"] = original_pop_pending_delta_message

    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch", vllm_ascend_patch)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.platform", vllm_ascend_platform)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.patch.platform.patch_deepseek_v4_tool_call_parser",
        upstream,
    )
    vars(vllm_ascend_platform)["patch_deepseek_v4_tool_call_parser"] = upstream

    return upstream, DeltaFunctionCall, DeltaMessage, DeltaToolCall


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    upstream, delta_function_call, delta_message, delta_tool_call = _install_vllm_stubs(monkeypatch)
    module_name = f"patch_deepseek_v4_tool_call_metadata_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, upstream, delta_function_call, delta_message, delta_tool_call


def test_argument_delta_omits_none_tool_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _, upstream, delta_function_call, delta_message, delta_tool_call = _load_patch_module(monkeypatch)
    parser = types.SimpleNamespace(
        pending_message=delta_message(
            tool_calls=[
                delta_tool_call(
                    index=0,
                    id=None,
                    type=None,
                    function=delta_function_call(name=None, arguments='{"city":'),
                )
            ]
        )
    )

    result = upstream._pop_pending_delta_message(parser)

    assert result.model_dump(exclude_unset=True) == {
        "tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]
    }


def test_name_delta_keeps_non_none_tool_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _, upstream, delta_function_call, delta_message, delta_tool_call = _load_patch_module(monkeypatch)
    parser = types.SimpleNamespace(
        pending_message=delta_message(
            tool_calls=[
                delta_tool_call(
                    index=0,
                    id="call_123",
                    type="function",
                    function=delta_function_call(name="weather", arguments=""),
                )
            ]
        )
    )

    result = upstream._pop_pending_delta_message(parser)

    assert result.model_dump(exclude_unset=True) == {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {"name": "weather", "arguments": ""},
            }
        ]
    }


def test_patch_replaces_module_global_and_class_method(monkeypatch: pytest.MonkeyPatch) -> None:
    module, upstream, _, _, _ = _load_patch_module(monkeypatch)

    assert upstream._pop_pending_delta_message is module._patched_pop_pending_delta_message
    assert upstream.DeepSeekV4ToolParser._pop_pending_delta_message is module._patched_pop_pending_delta_message

    module._apply_patch()
    assert upstream._pop_pending_delta_message is module._patched_pop_pending_delta_message
