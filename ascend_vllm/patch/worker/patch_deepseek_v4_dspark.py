# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import typing
from collections.abc import Iterable

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.models.deepseek_v4_dspark import DSparkDeepseekV4ForCausalLM, _EXPERT_SCALE_RE


def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Load the ``mtp.{i}.*`` draft weights from the target checkpoint.

    Non-MTP weights belong to the target model and are skipped, except for
    standalone embedding/head weights used by the Ascend draft loader.
    """
    expert_mapping = self.model.get_expert_mapping()
    expert_scale_suffix = (
        ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" else ".weight_scale_inv"
    )

    # (param_name, checkpoint shard name, shard_id) for non-expert
    # stacked parameters. Ascend keeps wq_a and wkv as separate parameters.
    stacked_params_mapping = [
        ("mlp.gate_up_proj", "mlp.gate_proj", 0),
        ("mlp.gate_up_proj", "mlp.up_proj", 1),
        ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
        ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
    ]

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    n_local_head = self.config.num_attention_heads // tp_size
    head_start = n_local_head * tp_rank
    head_end = n_local_head * (tp_rank + 1)

    for name, loaded_weight in weights:
        if name == "embed.weight" and not self.has_own_embed_tokens:
            name = "model.embed_tokens.weight"
        elif name == "head.weight" and not self.has_own_lm_head:
            name = "lm_head.weight"
        else:
            mapped_name = self._remap_dspark_name(name)
            if mapped_name is None:
                continue
            name = mapped_name

        # Expert scale parameters use a dtype-specific suffix; other
        # quantized parameters use Ascend's ``weight_scale`` convention.
        if name.endswith(".scale"):
            suffix = expert_scale_suffix if _EXPERT_SCALE_RE.search(name) else ".weight_scale"
            name = name.removesuffix(".scale") + suffix

        # E8M0 expert scales must retain their raw exponent bytes.
        if ".experts." in name:
            if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                loaded_weight = loaded_weight.view(torch.uint8)
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                param = params_dict[name_mapped]
                weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_params.add(name_mapped)
                    break
            continue

        # Stacked rules only apply to decoder-layer weights. Head-stack
        # parameters load directly through the fallback below.
        is_layer_param = name.startswith("model.layers.")
        for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
            if not is_layer_param or f".{weight_name}." not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            param.weight_loader(param, loaded_weight, stacked_shard_id)
            loaded_params.add(name)
            break
        else:
            if "attn_sink" in name:
                narrow = loaded_weight[head_start:head_end]
                with torch.no_grad():
                    params_dict[name][: narrow.shape[0]].copy_(narrow)
                loaded_params.add(name)
                continue
            # Skip params not in model
            if name not in params_dict:
                logger.warning_once(
                    "DSpark: skip loading %s - not in model params "
                    "(likely quantization scale for unquantized layer)",
                    name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

    logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
    return loaded_params

DSparkDeepseekV4ForCausalLM.load_weights = load_weights