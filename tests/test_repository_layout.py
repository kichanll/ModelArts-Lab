from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist() -> None:
    required_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/lint.yml",
        ".pre-commit-config.yaml",
        "CONTRIBUTING.md",
        "README.md",
        "docs/upstream-sync.md",
        "docs/version-matrix.md",
        "pyproject.toml",
        "requirements-lint.txt",
        "scripts/format.sh",
        "scripts/lint.sh",
    ]

    missing = [path for path in required_files if not (ROOT / path).is_file()]
    assert missing == []


def test_upstream_policy_names_required_upstreams() -> None:
    upstream_doc = (ROOT / "docs/upstream-sync.md").read_text(encoding="utf-8")

    assert "vllm-project/vllm-ascend" in upstream_doc
    assert "vllm-project/vllm.git" in upstream_doc
