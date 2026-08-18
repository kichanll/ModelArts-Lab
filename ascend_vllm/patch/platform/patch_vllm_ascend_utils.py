# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import json
from typing import Any
from vllm.platforms.interface import get_assigned_physical_gpu_ids

import vllm_ascend.utils as _utils_module


def setup_ascend_local_comm_res(local_rank: int, kv_transfer_config: Any | None) -> None:
    """Load the local A5 endpoint config into ASCEND_LOCAL_COMM_RES."""
    if kv_transfer_config is None:
        return

    assigned_ids = get_assigned_physical_gpu_ids()
    if assigned_ids is not None:
        devices = assigned_ids
    else:
        visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
        if visible_devices is None:
            from vllm_ascend.cpu_binding import DeviceInfo
            devices = sorted([int(x) for x in DeviceInfo.get_npu_map_info()])
        else:
            devices = [int(x) for x in visible_devices.split(",") if x.strip()]

    extra_config = kv_transfer_config.kv_connector_extra_config or {}
    local_comm_res_path = extra_config.get("ascend_local_comm_res_path")
    if not local_comm_res_path:
        return

    if not devices:
        raise ValueError("No NPU devices found or specified in ASCEND_RT_VISIBLE_DEVICES.")
    if local_rank < 0 or local_rank >= len(devices):
        raise ValueError(f"local_rank {local_rank} is out of bounds for the available NPU devices: {devices}")

    local_comm_res_file = os.path.join(local_comm_res_path, f"ub_endpoint_npu_{devices[local_rank]}.json")
    try:
        with open(local_comm_res_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Endpoint config file not found: {local_comm_res_file}. "
            "Please set ascend_local_comm_res_path in kv_connector_extra_config "
            "to a directory containing ub_endpoint_npu_*.json endpoint configuration files."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse endpoint config file: {local_comm_res_file}") from e

    os.environ["ASCEND_LOCAL_COMM_RES"] = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

_utils_module.setup_ascend_local_comm_res = setup_ascend_local_comm_res
