# vllm-cloud

This repository is the downstream workspace for vLLM Ascend based cloud
adaptation work.

The intended upstream chain is:

1. `vllm-project/vllm`
2. `vllm-project/vllm-ascend`
3. this repository

The repository is intentionally initialized with project hygiene, CI, lint, and
formatting infrastructure before importing or adapting upstream code. The Python
lint baseline is kept compatible with vLLM Ascend to reduce churn when upstream
source is imported.

## Quick Start

Use a Python 3.11 environment for local validation. The local environment can be
managed by conda or another tool; environment creation is intentionally not
codified in this repository.

```bash
python -m pip install -r requirements-lint.txt
```

Run the local checks:

```bash
bash scripts/lint.sh
python -m pytest
```

Format files:

```bash
bash scripts/format.sh
```

The project records lint and test dependencies, but not a conda environment
definition.

## Upstream Policy

Before importing upstream source, choose and record the exact baseline in
[`docs/version-matrix.md`](docs/version-matrix.md). The sync process is
documented in [`docs/upstream-sync.md`](docs/upstream-sync.md).
