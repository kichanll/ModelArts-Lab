from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreRequest, detokenizer
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer, IncrementalDetokenizer

COT_THINK_END_TOKEN = "</think>"
TOOL_START_TOKEN = "<tool_call>"
TOOL_END_TOKEN = "</tool_call>"
SLIDING_WINDOW_SIZE = -16

origin_from_new_request = IncrementalDetokenizer.from_new_request
origin_init = BaseIncrementalDetokenizer.__init__


@classmethod
def from_new_request(
    cls,
    tokenizer: TokenizerLike | None,
    request: EngineCoreRequest,
) -> "IncrementalDetokenizer":

    incremental_detokenizer: IncrementalDetokenizer = origin_from_new_request(tokenizer, request)

    # Adapt: 增加初始化思维链token功能，用于后续识别处理碎片使用
    if tokenizer is not None:
        incremental_detokenizer.set_cot_token_ids(tokenizer)

    # No tokenizer => skipping detokenization.

    return incremental_detokenizer


def __init__(self, request: EngineCoreRequest):
    origin_init(self, request)
    # Adapt: 增加思维链结束标记
    self._cot_end_symbol = False


def set_cot_token_ids(self, tokenizer: TokenizerLike | None):
    """
    Initialization of the CoT end marker</think>
    """

    vocab = tokenizer.get_vocab()
    self.cot_think_end_token_id = vocab.get(COT_THINK_END_TOKEN)


def is_cot_end(self) -> bool:
    """
    Determine whether the current output has completed the output of the CoT.
    """

    if self._cot_end_symbol:
        return True

    if len(self.token_ids) <= 1:
        return False

    cot_end = False

    # but we currently cannot obtain the vllm_config within the detokenizer process.
    check_token_ids = self.token_ids[SLIDING_WINDOW_SIZE:]

    # COT ends with "</think>"
    if self.cot_think_end_token_id in check_token_ids:
        cot_end = True

    if cot_end:
        self._cot_end_symbol = True

    return self._cot_end_symbol


def get_next_output_text(self, finished: bool, delta: bool) -> str:
    """
    Patch to fix the <think> and </think> get truncated when the stop is enabled.
    If delta is True, only new text since the last call to this method is returned
    """

    # We return the full output text if the sequence is finished.

    thinking = not self.is_cot_end()
    is_thinking_end_chunk = COT_THINK_END_TOKEN in self.output_text[self._last_output_text_offset :]
    is_tool_start_chunk = TOOL_START_TOKEN in self.output_text[self._last_output_text_offset :]
    is_tool_end_chunk = TOOL_END_TOKEN in self.output_text[self._last_output_text_offset :]
    # Adapt: The logic for assigning the value of buffer_length is modified to
    # be compatible with the COT terminator, preventing fragmentation.
    buffer_length = (
        0
        if finished or thinking or is_thinking_end_chunk or is_tool_start_chunk or is_tool_end_chunk
        else self.stop_buffer_length
    )
    if not delta:
        return self.output_text[:-buffer_length] if buffer_length else self.output_text
    length = len(self.output_text) - buffer_length
    last_offset = self._last_output_text_offset
    if last_offset < length:
        self._last_output_text_offset = length
        return self.output_text[last_offset:length]
    return ""


detokenizer.IncrementalDetokenizer.from_new_request = from_new_request
detokenizer.BaseIncrementalDetokenizer.__init__ = __init__
detokenizer.BaseIncrementalDetokenizer.get_next_output_text = get_next_output_text
detokenizer.BaseIncrementalDetokenizer.is_cot_end = is_cot_end
detokenizer.BaseIncrementalDetokenizer.set_cot_token_ids = set_cot_token_ids
