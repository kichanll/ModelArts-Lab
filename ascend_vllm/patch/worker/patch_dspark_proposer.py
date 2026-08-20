from __future__ import annotations

from typing import Any

import torch
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm_ascend.attention.attention_v1 import AscendAttentionState

from ascend_vllm.patch.worker.patch_spec_decode_utils import (
    copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid,
)

_PATCH_APPLIED = False


def _dspark_set_inputs_first_pass(
    self,
    target_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    token_indices_to_sample: torch.Tensor | None,
    cad: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor | None,
    req_scheduled_tokens=None,
    long_seq_metadata=None,
    num_prefill_reqs=0,
    num_decode_reqs=0,
) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
    # The initial input token of markovHead is the next token
    n = next_token_ids.shape[0]
    self._dspark_seed_buffer[:n].copy_(next_token_ids)
    self._dspark_seed_buffer[n:].fill_(0)
    batch_size = cad.num_reqs
    num_query_total = batch_size * self.num_query_per_req
    num_sample_total = batch_size * self.num_speculative_tokens
    has_num_rejected = num_rejected_tokens_gpu is not None
    primary_gid = getattr(self, "kv_cache_gid", 0)
    self._per_group_block_table_buffers = {
        attn_group.kv_cache_group_id: self._per_group_block_tables[attn_group.kv_cache_group_id]
        for attn_group in self.draft_attn_groups
    }
    self._context_slot_mapping_buffers = None
    self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])
    self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]

    token_indices_to_sample = torch.empty(
        num_sample_total,
        dtype=torch.int32,
        device=self.device,
    )

    # Query block: reuse the DFlash inputs kernel logic (host-side ref)
    # per kv-cache-group to fill positions / input_ids / query slot_mapping
    # / token_indices.
    draft_attn_groups = getattr(self, "draft_attn_groups", [])
    for attn_group in draft_attn_groups:
        gid = attn_group.kv_cache_group_id
        gid_block_table = self._per_group_block_table_buffers.get(gid)
        if gid_block_table is None:
            continue
        kv_block_size = int(attn_group.kv_cache_spec.block_size)
        if batch_size > 0:
            copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[batch_size,](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=self._per_group_slot_mappings[gid],
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._per_group_context_slot_mapping_buffers[gid],
                out_query_slot_mapping_ptr=self._per_group_query_slot_mapping_buffers[gid],
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=gid_block_table,
                block_table_stride=gid_block_table.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=num_rejected_tokens_gpu,
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kv_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                max_model_len=self.max_model_len,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
            )
    # to compute self._context_slot_mapping_buffers from dict to list
    self._context_slot_mapping_buffers = [
        self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
    ]

    effective_seq_lens = cad.seq_lens
    if has_num_rejected:
        effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

    cad.query_start_loc = self.arange_dflash[: batch_size + 1] * self.num_query_per_req
    cad.seq_lens = effective_seq_lens + self.num_query_per_req
    cad.query_start_loc_cpu = (
        torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * self.num_query_per_req
    ).to(torch.int32)

    if hasattr(cad, "actual_seq_lengths_q"):
        cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size
    if hasattr(cad, "decode_token_per_req"):
        cad.decode_token_per_req = self.num_query_per_req

    cad.num_actual_tokens = num_query_total
    cad.num_input_tokens = num_query_total
    cad.max_query_len = self.num_query_per_req
    cad.max_seq_len = cad.max_seq_len + self.num_query_per_req
    cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_query_total]
    cad.positions = self.positions  # this would be sliced in attention backend
    cad.causal = False
    cad.attn_mask = None
    cad.attn_state = AscendAttentionState.ChunkedPrefill

    return num_query_total, token_indices_to_sample, cad, None


def apply_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import vllm_ascend.spec_decode.dspark_proposer as dspark_mod
    from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

    dspark_mod.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid = (
        copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
    )

    AscendDSparkProposer.set_inputs_first_pass = _dspark_set_inputs_first_pass

    _PATCH_APPLIED = True


apply_patch()
