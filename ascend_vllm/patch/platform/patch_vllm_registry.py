# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import pickle
import subprocess
import tempfile
from collections.abc import Callable
from typing import TypeVar

from vllm.model_executor.models import registry
from vllm.model_executor.models.registry import _SUBPROCESS_COMMAND

_T = TypeVar("_T")


def _run_in_subprocess(fn: Callable[[], _T]) -> _T:
    # NOTE: We use a temporary directory instead of a temporary file to avoid
    # issues like https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
    with tempfile.TemporaryDirectory() as tempdir:
        output_filepath = os.path.join(tempdir, "registry_output.tmp")

        # `cloudpickle` allows pickling lambda functions directly
        import cloudpickle

        input_bytes = cloudpickle.dumps((fn, output_filepath))

        # cannot use `sys.executable __file__` here because the script
        # contains relative imports
        returned = subprocess.run(_SUBPROCESS_COMMAND, input=input_bytes, capture_output=True)

        # check if the subprocess is successful
        try:
            returned.check_returncode()
        except Exception as e:
            if os.path.exists(output_filepath):
                try:
                    with open(output_filepath, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    pass
            # wrap raised exception to provide more information
            raise RuntimeError(f"Error raised in subprocess:\n{returned.stderr.decode()}") from e

        with open(output_filepath, "rb") as f:
            return pickle.load(f)


registry._run_in_subprocess = _run_in_subprocess
