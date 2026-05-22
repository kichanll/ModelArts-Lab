#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ -z "${PYTHON:-}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="python"
    else
        echo "python or python3 is required" >&2
        exit 1
    fi
fi

"${PYTHON}" -m ruff format .
"${PYTHON}" -m ruff check --fix .
