import os
from typing import List

from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(__file__)

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

setup(
    name="ascend_vllm",
    version='6.5.923',
    packages=find_packages(include=("ascend_vllm", "ascend_vllm.*")),
    entry_points={
        "vllm.platform_plugins": ["ascend_vllm = ascend_vllm:register"],
        "vllm.general_plugins": [""],
    },
    install_requires=get_requirements(),
)
