from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.tokenizers import deepseek_v4 as deepseek_v4_tokenizer
from vllm_ascend.patch.platform import patch_deepseek_v4_thinking as _ascend_patch  # noqa: F401

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

_ORIGINAL_GET_DEEPSEEK_V4_TOKENIZER = (
    deepseek_v4_tokenizer.get_deepseek_v4_tokenizer
)

_VALIDATION_PATCH_APPLIED = False


def _patched_get_deepseek_v4_tokenizer(tokenizer: deepseek_v4_tokenizer.HfTokenizer):
    dsv4_tokenizer = _ORIGINAL_GET_DEEPSEEK_V4_TOKENIZER(tokenizer)
    tokenizer_cls = type(dsv4_tokenizer)
    original_apply_chat_template = tokenizer_cls.apply_chat_template

    class _ValidatedDeepseekV4Tokenizer(tokenizer_cls):
        @classmethod
        def validate_messages(cls, messages):
            try:
                if len(messages) > 0 and messages[-1].get("role") == "system":
                    return (
                        False,
                        (
                            f"Invalid system message at message index {len(messages) - 1}, "
                            "system message can only be the first message in the conversation"
                        ),
                    )

                if len(messages) > 0 and messages[-1].get("role") == "assistant":
                    if isinstance(messages[-1].get("prefix"), bool) and messages[-1].get("prefix"):
                        return True, ""
                    if messages[-1].get("tool_calls") is not None:
                        return True, ""
                    return False, f"Invalid consecutive assistant message at message index {len(messages) - 1}"

                for index in range(len(messages)):
                    msg = messages[index]
                    role = msg.get("role")
                    if role not in ["system", "developer", "user", "tool", "assistant"]:
                        return False, f"Unkown role: {role}. Role just can be [system, developer, user, tool,assistant]"
                return True, ""
            except Exception:
                return True, ""

        @classmethod
        def validate_sufficient_tools(cls, messages):
            try:
                for index in range(len(messages)):
                    msg = messages[index]
                    role = msg.get("role")
                    if role == "assistant":
                        tool_calls_id = []
                        message_tool_id = []
                        tool_calls = msg.get("tool_calls")
                        if isinstance(tool_calls, list) and len(tool_calls) == 0:
                            return (
                                False,
                                (
                                    f"Invalid message at message[{index}].tool_calls: empty error. "
                                    "Expected an array with minimum length 1, but got an empty array instead."
                                ),
                            )
                        if tool_calls and isinstance(tool_calls, list):
                            tool_counter = 0
                            for tool_call in tool_calls:
                                tool_counter += 1
                                if (
                                    index + tool_counter >= len(messages)
                                    or messages[index + tool_counter].get("role") != "tool"
                                ):
                                    return (
                                        False,
                                        "An assistant message with tool calls must be followed by a tool messages "
                                        "responding to each tool call",
                                    )
                                else:
                                    tool_calls_id.append(tool_call.get("id"))
                                    message_tool_id.append(messages[index + tool_counter].get("tool_call_id"))
                            if sorted(tool_calls_id) != sorted(message_tool_id):
                                return (
                                    False,
                                    "An assistant message with tool calls must be followed by a tool messages "
                                    "responding to each tool call",
                                )
                            index += tool_counter
                return True, ""
            except Exception:
                return True, ""

        def apply_chat_template(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            conversation = kwargs.get("conversation", messages)
            messages_to_validate = conversation.copy()

            valid, err = self.validate_messages(messages_to_validate)
            if not valid:
                raise ValueError(err)

            valid, err = self.validate_sufficient_tools(messages_to_validate)
            if not valid:
                raise ValueError(err)

            return original_apply_chat_template(
                self,
                messages,
                tools=tools,
                **kwargs,
            )

    _ValidatedDeepseekV4Tokenizer.__name__ = (
        f"Validated{tokenizer_cls.__name__}"
    )
    dsv4_tokenizer.__class__ = _ValidatedDeepseekV4Tokenizer
    return dsv4_tokenizer


def apply_patch() -> None:
    global _VALIDATION_PATCH_APPLIED

    if _VALIDATION_PATCH_APPLIED:
        return

    deepseek_v4_tokenizer.get_deepseek_v4_tokenizer = (
        _patched_get_deepseek_v4_tokenizer
    )

    _VALIDATION_PATCH_APPLIED = True


apply_patch()
