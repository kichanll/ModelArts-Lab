from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/lint.yml",
    ".github/dependabot.yml",
    ".pre-commit-config.yaml",
    ".yamllint.yml",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "docs/upstream-sync.md",
    "docs/version-matrix.md",
    "pyproject.toml",
    "requirements-lint.txt",
    "scripts/format.sh",
    "scripts/lint.sh",
)

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
}


def iter_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {"Makefile"}:
            files.append(path)
    return files


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Missing required repository files:\n{joined}")

    failures: list[str] = []
    for path in iter_text_files():
        data = path.read_bytes()
        rel_path = path.relative_to(ROOT)
        if b"\r\n" in data:
            failures.append(f"{rel_path}: contains CRLF line endings")
        if data and not data.endswith(b"\n"):
            failures.append(f"{rel_path}: missing final newline")

        for line_number, line in enumerate(data.splitlines(), start=1):
            if line.rstrip(b" \t") != line:
                failures.append(f"{rel_path}:{line_number}: trailing whitespace")

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
