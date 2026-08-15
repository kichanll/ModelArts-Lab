# ruff: noqa: I001

from ascend_vllm.patch.platform import patch_health as patch_health
from ascend_vllm.patch.platform import patch_request as patch_request
from ascend_vllm.patch.platform import patch_detokenizer as patch_detokenizer
from ascend_vllm.patch.platform import patch_scheduler as patch_scheduler
from ascend_vllm.patch.platform import (
    patch_deepseek_v4_validation as patch_deepseek_v4_validation,
)
from ascend_vllm.patch.platform import (
    patch_recompute_scheduler as patch_recompute_scheduler,
)
