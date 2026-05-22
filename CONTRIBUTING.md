# Contributing

This repository is a downstream workspace for vLLM Ascend based cloud
adaptation work. Keep changes small, attributable, and easy to replay on top of
the selected upstream baseline.

## Local Setup

Use a Python 3.11 environment for validation. The environment may be a local
conda environment, but conda environment requirements are intentionally not
stored in this repository.

```bash
python -m pip install -r requirements-lint.txt
```

Optional pre-commit setup:

```bash
pre-commit install
```

## Checks

Run the same checks used by CI:

```bash
bash scripts/lint.sh
python -m pytest
```

Format files before submitting changes:

```bash
bash scripts/format.sh
```

The lint baseline intentionally follows vLLM Ascend's Python style where it
matters for downstream sync: 120-character lines, Ruff for Python formatting and
linting, and gradual mypy coverage instead of strict repository-wide typing.

## Upstream Changes

Any change that updates the vLLM Ascend or vLLM baseline must also update
`docs/version-matrix.md` and explain the sync in the pull request.

Prefer upstreamable fixes over downstream-only patches. If a downstream-only
patch is necessary, document why it is cloud-specific and what would make it
removable.
