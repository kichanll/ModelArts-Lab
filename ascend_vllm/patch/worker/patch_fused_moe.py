# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch_npu
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner, FusedMoEEvents
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    npu_stream_switch,
    shared_expert_dp_enabled,
    shared_experts_calculation_stream,
)


def _forward_shared_experts(self, hidden_states: torch.Tensor, fused_moe_evts: FusedMoEEvents):
    if self._shared_experts is None:
        return None

    def maybe_wait_event(evt: torch.npu.Event | None):
        if evt is not None:
            torch.npu.current_stream().wait_event(evt)

    with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap_shared_expert):
        # Only used for int quantization
        has_quantized_shared = hasattr(self._shared_experts.gate_up_proj, "weight_scale") and hasattr(
            self._shared_experts.down_proj, "weight_scale"
        )
        if has_quantized_shared and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
            original_dtype = hidden_states.dtype
            # Execute dynamic quant concurrently with MoE gate.
            torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            # Execute the gate projection and activation concurrently with the
            # dispatch communication.
            maybe_wait_event(fused_moe_evts.after_routed_experts)
            hidden_states = torch_npu.npu_quant_matmul(
                quantized_x,
                self._shared_experts.gate_up_proj.weight,
                self._shared_experts.gate_up_proj.weight_scale,
                pertoken_scale=None,
                bias=None,
                output_dtype=torch.int32,
            )
            # Execute activation concurrently with gmm2.

            maybe_wait_event(fused_moe_evts.before_gmm2)
            quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=self._shared_experts.gate_up_proj.weight_scale_fp32,
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=None,
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=fused_moe_evts.swiglu_limit,
                glu_alpha=fused_moe_evts.swiglu_alpha,
                glu_bias=fused_moe_evts.swiglu_beta,
            )
            # Execute the down projection concurrently with the combine
            # communication.
            maybe_wait_event(fused_moe_evts.before_combine)
            shared_out = torch_npu.npu_quant_matmul(
                quantized_x,
                self._shared_experts.down_proj.weight,
                self._shared_experts.down_proj.weight_scale,
                pertoken_scale=swiglu_out_scale,
                bias=None,
                output_dtype=original_dtype,
            )
        elif has_quantized_shared and self.quant_type == QuantType.W4A8MXFP:
            original_dtype = hidden_states.dtype
            # Execute dynamic quant concurrently with MoE gate.
            torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
            # Execute the gate projection and activation concurrently with the
            # dispatch communication.
            maybe_wait_event(fused_moe_evts.before_dispatch)
            hidden_states = self._shared_experts.gate_up_proj((quantized_x, pertoken_scale))[0]
            # Execute activation concurrently with gmm2.
            maybe_wait_event(fused_moe_evts.before_gmm2)
            quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                hidden_states,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=2,
                clamp_value=fused_moe_evts.swiglu_limit,
            )
            # Execute the down projection concurrently with the combine
            # communication.
            maybe_wait_event(fused_moe_evts.before_combine)
            shared_out = self._shared_experts.down_proj((quantized_x, swiglu_out_scale))[0]
        else:
            # Ensure the shared experts wait for hidden_states to be ready.
            torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
            # Execute the gate projection and activation concurrently with the
            # dispatch communication.
            maybe_wait_event(fused_moe_evts.before_dispatch)
            part1_out = self._shared_experts_part1(hidden_states)
            # Execute the down projection concurrently with the combine
            # communication.
            maybe_wait_event(fused_moe_evts.before_combine)
            shared_out = self._shared_experts_part2(hidden_states, part1_out)

    # Make sure the default stream waits for the shared experts stream to
    # finish.
    if self.multistream_overlap_shared_expert:
        torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

    # NOTE: This is exactly the opposite of
    # `maybe_all_reduce_tensor_model_parallel`
    moe_comm_type = _EXTRA_CTX.moe_comm_type
    if (
        moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
        and not shared_expert_dp_enabled()
    ):
        shared_out = tensor_model_parallel_all_reduce(shared_out)
    return shared_out


AscendMoERunner._forward_shared_experts = _forward_shared_experts
