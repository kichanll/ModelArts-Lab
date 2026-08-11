from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "platform" / "patch_deepseek_v4_validation.py"


class _FakeTokenizer:
    def __init__(self) -> None:
        self.apply_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        self.apply_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        return "original-result"


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_external_dependency_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    vllm = _make_package("vllm")
    tokenizers = _make_package("vllm.tokenizers")
    deepseek_v4 = types.ModuleType("vllm.tokenizers.deepseek_v4")

    original_get_calls: list[Any] = []

    def original_get_deepseek_v4_tokenizer(tokenizer: Any) -> Any:
        original_get_calls.append(tokenizer)
        return tokenizer

    vars(deepseek_v4)["HfTokenizer"] = _FakeTokenizer
    vars(deepseek_v4)["get_deepseek_v4_tokenizer"] = original_get_deepseek_v4_tokenizer
    vars(tokenizers)["deepseek_v4"] = deepseek_v4

    vllm_ascend = _make_package("vllm_ascend")
    vllm_ascend_patch = _make_package("vllm_ascend.patch")
    vllm_ascend_patch_platform = _make_package("vllm_ascend.patch.platform")
    ascend_deepseek_patch = types.ModuleType("vllm_ascend.patch.platform.patch_deepseek_v4_thinking")

    ascend_original_hook = object()
    vars(ascend_deepseek_patch)["_patched_get_deepseek_v4_tokenizer"] = ascend_original_hook
    vars(vllm_ascend_patch_platform)["patch_deepseek_v4_thinking"] = ascend_deepseek_patch

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", tokenizers)
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers.deepseek_v4",
        deepseek_v4,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.patch",
        vllm_ascend_patch,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.patch.platform",
        vllm_ascend_patch_platform,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.patch.platform.patch_deepseek_v4_thinking",
        ascend_deepseek_patch,
    )

    return {
        "deepseek_v4": deepseek_v4,
        "original_get": original_get_deepseek_v4_tokenizer,
        "original_get_calls": original_get_calls,
        "ascend_deepseek_patch": ascend_deepseek_patch,
        "ascend_original_hook": ascend_original_hook,
    }


def _load_patch_module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, dict[str, Any]]:
    stubs = _install_external_dependency_stubs(monkeypatch)

    module_name = f"patch_deepseek_v4_validation_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PATCH_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module, stubs


def _patched_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, _FakeTokenizer, dict[str, Any]]:
    module, stubs = _load_patch_module(monkeypatch)

    source_tokenizer = _FakeTokenizer()
    tokenizer = module.deepseek_v4_tokenizer.get_deepseek_v4_tokenizer(source_tokenizer)

    assert tokenizer is source_tokenizer
    return module, tokenizer, stubs


def test_apply_patch_wraps_only_vllm_getter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, stubs = _load_patch_module(monkeypatch)
    deepseek_v4 = stubs["deepseek_v4"]

    assert module._VALIDATION_PATCH_APPLIED is True
    assert module._ORIGINAL_GET_DEEPSEEK_V4_TOKENIZER is stubs["original_get"]
    assert deepseek_v4.get_deepseek_v4_tokenizer is module._patched_get_deepseek_v4_tokenizer

    # ModelArts 不再覆盖 vllm-ascend 的私有 hook。
    assert stubs["ascend_deepseek_patch"]._patched_get_deepseek_v4_tokenizer is stubs["ascend_original_hook"]

    # 重复调用 apply_patch 不应该再次包装。
    patched_getter = deepseek_v4.get_deepseek_v4_tokenizer
    module.apply_patch()

    assert deepseek_v4.get_deepseek_v4_tokenizer is patched_getter


@pytest.mark.parametrize(
    "reasoning_effort",
    ["none", "low", "high", "max"],
)
def test_apply_chat_template_delegates_without_rewriting_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort: str,
) -> None:
    _, tokenizer, stubs = _patched_tokenizer(monkeypatch)

    messages = [
        {
            "role": "user",
            "content": "hello",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup"},
        }
    ]

    result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        reasoning_effort=reasoning_effort,
        enable_thinking=True,
        tokenize=False,
        custom_option="keep-me",
    )

    assert result == "original-result"
    assert tokenizer.apply_calls == [
        {
            "messages": messages,
            "tools": tools,
            "kwargs": {
                "reasoning_effort": reasoning_effort,
                "enable_thinking": True,
                "tokenize": False,
                "custom_option": "keep-me",
            },
        }
    ]
    assert len(stubs["original_get_calls"]) == 1


@pytest.mark.parametrize(
    ("messages", "expected_message"),
    [
        (
            [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "invalid"},
            ],
            "system message can only be the first message",
        ),
        (
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "unfinished"},
            ],
            "Invalid consecutive assistant message",
        ),
        (
            [{"role": "unknown", "content": "hello"}],
            "Unkown role",
        ),
    ],
)
def test_apply_chat_template_rejects_invalid_messages(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, Any]],
    expected_message: str,
) -> None:
    _, tokenizer, _ = _patched_tokenizer(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        tokenizer.apply_chat_template(messages)

    assert expected_message in str(exc_info.value)
    assert tokenizer.apply_calls == []


def test_apply_chat_template_rejects_empty_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer, _ = _patched_tokenizer(monkeypatch)

    messages = [
        {
            "role": "assistant",
            "tool_calls": [],
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        tokenizer.apply_chat_template(messages)

    assert "minimum length 1" in str(exc_info.value)
    assert tokenizer.apply_calls == []


def test_apply_chat_template_rejects_mismatched_tool_call_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer, _ = _patched_tokenizer(monkeypatch)

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "result",
        },
    ]

    with pytest.raises(ValueError) as exc_info:
        tokenizer.apply_chat_template(messages)

    assert "responding to each tool call" in str(exc_info.value)
    assert tokenizer.apply_calls == []


def test_apply_chat_template_accepts_matching_tool_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer, _ = _patched_tokenizer(monkeypatch)

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup_a"},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "lookup_b"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "result-a",
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "result-b",
        },
    ]

    result = tokenizer.apply_chat_template(
        messages,
        reasoning_effort="high",
    )

    assert result == "original-result"
    assert tokenizer.apply_calls[0]["messages"] == messages
    assert tokenizer.apply_calls[0]["kwargs"] == {"reasoning_effort": "high"}


def test_apply_chat_template_accepts_terminal_assistant_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer, _ = _patched_tokenizer(monkeypatch)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "prefix",
            "prefix": True,
        },
    ]

    result = tokenizer.apply_chat_template(messages)

    assert result == "original-result"
    assert len(tokenizer.apply_calls) == 1


def test_apply_chat_template_validates_conversation_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer, _ = _patched_tokenizer(monkeypatch)

    messages = [{"role": "user", "content": "valid"}]
    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "invalid"},
    ]

    with pytest.raises(ValueError) as exc_info:
        tokenizer.apply_chat_template(
            messages,
            conversation=conversation,
        )

    assert "system message can only be the first message" in str(exc_info.value)
    assert tokenizer.apply_calls == []
