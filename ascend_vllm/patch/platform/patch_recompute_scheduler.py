from __future__ import annotations

_PATCH_APPLIED = False


def _patch_recompute_scheduler() -> None:
    """Patch RecomputeScheduler for HMA invalid-block handling."""
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_ascend.core import recompute_scheduler as rs

    def update_requests_with_invalid_blocks(
        self,
        requests,
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """Reset requests affected by failed KV loads for full recomputation."""
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()

        for request in requests:
            req_id = request.request_id
            req_block_id_groups = self.kv_cache_manager.get_block_ids(req_id)

            # Block IDs are allocated from one global BlockPool, so flattening
            # groups is sufficient for identifying the owning request.
            req_block_ids = {block_id for group in req_block_id_groups for block_id in group if block_id is not None}

            affected_block_ids = req_block_ids & invalid_block_ids
            if not affected_block_ids:
                continue

            # Exclude tokens scheduled in the current step because their
            # outputs have not yet become part of the stable computed prefix.
            req_num_computed_tokens = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            affected_req_ids.add(req_id)
            total_affected_tokens += req_num_computed_tokens
            request.num_computed_tokens = 0

            # If the caller requests eviction, all request blocks are
            # downstream of the new computed-token boundary (zero).
            if evict_blocks:
                blocks_to_evict.update(req_block_ids)

        return (
            affected_req_ids,
            total_affected_tokens,
            blocks_to_evict,
        )

    Scheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks
    rs.RecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks
    rs.AsyncRecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_recompute_scheduler()
    _PATCH_APPLIED = True


apply_patch()
