from __future__ import annotations

from typing import Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage, DeltaToolCall
from vllm_ascend.patch.platform import patch_deepseek_v4_tool_call_parser as upstream_patch

_PATCH_MARKER = "_ascend_vllm_tool_call_metadata_patch_applied"
_original_pop_pending_delta_message = upstream_patch._pop_pending_delta_message


def _without_none_metadata(tool_call: DeltaToolCall) -> DeltaToolCall:
    payload = tool_call.model_dump(exclude_unset=True)
    if payload.get("id") is None:
        payload.pop("id", None)
    if payload.get("type") is None:
        payload.pop("type", None)

    function = payload.get("function")
    if isinstance(function, dict) and function.get("name") is None:
        function.pop("name", None)

    return DeltaToolCall.model_validate(payload)


def _patched_pop_pending_delta_message(self: Any) -> DeltaMessage | None:
    message = _original_pop_pending_delta_message(self)
    if message is None:
        return None

    message.tool_calls = [_without_none_metadata(tool_call) for tool_call in message.tool_calls]
    return message


def _apply_patch() -> None:
    if getattr(upstream_patch, _PATCH_MARKER, False):
        return

    upstream_patch._pop_pending_delta_message = _patched_pop_pending_delta_message
    upstream_patch.DeepSeekV4ToolParser._pop_pending_delta_message = _patched_pop_pending_delta_message
    setattr(upstream_patch, _PATCH_MARKER, True)


_apply_patch()
