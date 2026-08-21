from __future__ import annotations

import functools

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.ascend_vllm.patch.worker.patch_spec_decode_drafter")

_PATCH_APPLIED = False


def _patch_propose_draft_token_ids() -> None:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = NPUModelRunner.propose_draft_token_ids

    if getattr(original, "_modelarts_drafter_skip_patch_applied", False):
        return

    @functools.wraps(original)
    def patched_propose_draft_token_ids(
        self,
        valid_sampled_token_ids,
        sampling_metadata,
        scheduler_output,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        positions,
        num_scheduled_tokens,
        hidden_states,
        aux_hidden_states=None,
        sample_hidden_states=None,
        target_model_batch_desc=None,
    ):
        # Second-layer protection: skip drafter when the sequence is too
        # close to max_model_len, preventing out-of-bounds positions in
        # the Triton kernel and avoiding wasted draft computation.
        if self.speculative_config and spec_decode_common_attn_metadata is not None:
            input_fits = self._input_fits_in_drafter(spec_decode_common_attn_metadata)
            if not input_fits:
                # DP sync: when dp_size > 1, all DP ranks must call the
                # drafter to keep collective communication consistent.
                # Use dummy_run as a no-op substitute.
                if self.dp_size > 1:
                    self.drafter.dummy_run(num_tokens=1)

                # Return zero draft tokens so the nested caller in
                # sample_tokens assigns zeros to self._draft_token_ids
                # and _copy_draft_token_ids_to_cpu propagates them.
                # This prevents stale drafts from the previous step.
                num_reqs = len(self.input_batch.req_ids)
                draft_token_ids = torch.zeros(1, device=self.device, dtype=torch.int32).expand(
                    num_reqs, self.num_spec_tokens
                )
                return draft_token_ids

        # Normal path: delegate to the original method.
        return original(
            self,
            valid_sampled_token_ids,
            sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            positions,
            num_scheduled_tokens,
            hidden_states,
            aux_hidden_states,
            sample_hidden_states,
            target_model_batch_desc,
        )

    patched_propose_draft_token_ids._modelarts_drafter_skip_patch_applied = True
    NPUModelRunner.propose_draft_token_ids = patched_propose_draft_token_ids


def apply_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    _patch_propose_draft_token_ids()

    _PATCH_APPLIED = True


apply_patch()
