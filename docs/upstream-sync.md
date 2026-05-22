# Upstream Sync

This repository tracks vLLM Ascend as its direct upstream and vLLM as the
transitive runtime upstream.

## Remotes

Use explicit remote names so the sync direction is unambiguous:

```bash
git remote add upstream-ascend https://github.com/vllm-project/vllm-ascend.git
git remote add upstream-vllm https://github.com/vllm-project/vllm.git
```

## Baseline Rule

Before importing source code or syncing upstream changes, record the exact
baseline in `docs/version-matrix.md`.

The baseline must include:

- vLLM Ascend branch, tag, and commit
- vLLM branch, tag, and commit
- Python version
- CANN version
- PyTorch and torch-npu versions
- supported Ascend hardware family

## Sync Strategy

Prefer a clean upstream history over directory-level copy operations.

Recommended flow:

```bash
git fetch upstream-ascend
git fetch upstream-vllm
git checkout vllm/vllm-cloud
git merge --no-ff upstream-ascend/<selected-branch>
```

Use rebase only for private, unpublished branches. For shared integration
branches, prefer merge commits so the upstream boundary remains auditable.

## Downstream Patch Rule

Keep downstream patches focused and documented. A patch should be classified as
one of:

- cloud integration
- ModelArts packaging or deployment
- Ascend runtime compatibility
- temporary workaround for an upstream issue

Temporary workarounds must include the upstream issue, removal condition, and
validation scope.

## Tooling Rule

Keep Python formatting and lint rules compatible with vLLM Ascend unless a
cloud-specific reason requires divergence. Divergences must be documented in the
same change that introduces them.
