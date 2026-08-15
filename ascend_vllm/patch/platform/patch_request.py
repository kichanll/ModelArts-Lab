# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from typing import Any

from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.request import Request

_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_cached_tokens_patch_applied"


def _patch_request() -> None:
    current_init = Request.__init__

    if not getattr(current_init, _PATCH_MARKER, False):

        @functools.wraps(current_init)
        def patched_init(
            self: Request,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            current_init(self, *args, **kwargs)

            # P/D: Decode receives the number of cached prompt tokens
            # through kv_transfer_params from the prefill node.
            kv_transfer_params = self.kv_transfer_params
            self.num_cached_tokens = (
                kv_transfer_params.get("num_cached_tokens") if kv_transfer_params is not None else None
            )

        setattr(patched_init, _PATCH_MARKER, True)
        Request.__init__ = patched_init

    current_take_prefill_stats = Request.take_prefill_stats

    if not getattr(
        current_take_prefill_stats,
        _PATCH_MARKER,
        False,
    ):

        @functools.wraps(current_take_prefill_stats)
        def patched_take_prefill_stats(
            self: Request,
        ) -> PrefillStats | None:
            prefill_stats = current_take_prefill_stats(self)

            num_cached_tokens = getattr(
                self,
                "num_cached_tokens",
                None,
            )
            if prefill_stats is not None and num_cached_tokens is not None:
                prefill_stats.num_cached_tokens = num_cached_tokens

            return prefill_stats

        setattr(
            patched_take_prefill_stats,
            _PATCH_MARKER,
            True,
        )
        Request.take_prefill_stats = patched_take_prefill_stats


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_request()
    _PATCH_APPLIED = True


apply_patch()
