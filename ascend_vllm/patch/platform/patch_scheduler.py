# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_cached_tokens_patch_applied"


def _patch_scheduler() -> None:
    current_free_request = Scheduler._free_request

    if getattr(current_free_request, _PATCH_MARKER, False):
        return

    @functools.wraps(current_free_request)
    def patched_free_request(
        self: Scheduler,
        request: Request,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        kv_transfer_params = current_free_request(
            self,
            request,
            *args,
            **kwargs,
        )

        # P/D: Pass the prefill node's cached prompt-token count
        # to the decode node through kv_transfer_params.
        if kv_transfer_params is not None and request.prefill_stats is not None:
            kv_transfer_params["num_cached_tokens"] = request.prefill_stats.num_cached_tokens

        return kv_transfer_params

    setattr(patched_free_request, _PATCH_MARKER, True)
    Scheduler._free_request = patched_free_request


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_scheduler()
    _PATCH_APPLIED = True


apply_patch()
