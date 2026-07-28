from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
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

    class OpenAIBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")

    class DeltaFunctionCall(BaseModel):
        name: str | None = None
        arguments: str | None = None

    class DeltaToolCall(OpenAIBaseModel):
        id: str | None = None
        type: Literal["function"] | None = None
        index: int
        function: DeltaFunctionCall | None = None

    class DeltaMessage(OpenAIBaseModel):
        content: str | None = None
        tool_calls: list[DeltaToolCall] = Field(default_factory=list)

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
