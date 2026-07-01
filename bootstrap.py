"""
Quant-Lab project bootstrap script.

Idempotently creates the directory layout and package ``__init__.py``
files that the Quant-Lab platform expects. Safe to run multiple times;
existing files and directories are left untouched.

Usage
-----
python bootstrap.py
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

#: Directories that must exist for the platform to run.
DIRECTORIES: tuple[str, ...] = (
    "config",
    "core",
    "utils",
    "data",
    "data/raw",
    "data/processed",
    "data/cache",
    "data/parquet",
    "data/exports",
    "indicators",
    "features",
    "features/cache",
    "backtesting",
    "backtesting/results",
    "backtesting/trades",
    "optimization",
    "optimization/results",
    "ml",
    "ml/models",
    "ml/training",
    "ml/predictions",
    "reports",
    "reports/html",
    "reports/pdf",
    "reports/templates",
    "visualization",
    "visualization/figures",
    "visualization/charts",
    "strategies",
    "tests",
    "tests/data",
    "notebooks",
    "logs",
    "temp",
)

#: Directories that are importable Python packages and need __init__.py.
PACKAGE_DIRS: tuple[str, ...] = (
    "config",
    "core",
    "utils",
    "data",
    "indicators",
    "features",
    "backtesting",
    "optimization",
    "ml",
    "reports",
    "visualization",
    "strategies",
    "tests",
)

#: Root-level files that must exist (created empty if missing).
ROOT_FILES: tuple[str, ...] = (
    "README.md",
    ".gitignore",
    "requirements.txt",
    "pyproject.toml",
    "main.py",
)


def create_directories() -> None:
    """Create every directory in :data:`DIRECTORIES`, if missing."""
    for directory in DIRECTORIES:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"[dir]  {path}")


def create_packages() -> None:
    """Create ``__init__.py`` for every package in :data:`PACKAGE_DIRS`."""
    for package in PACKAGE_DIRS:
        init_file = PROJECT_ROOT / package / "__init__.py"

        if not init_file.exists():
            init_file.write_text('"""Quant-Lab package."""\n', encoding="utf-8")

        print(f"[pkg]  {init_file}")


def create_root_files() -> None:
    """Create empty root files in :data:`ROOT_FILES`, if missing."""
    for file_name in ROOT_FILES:
        path = PROJECT_ROOT / file_name

        if not path.exists():
            path.touch()

        print(f"[file] {path}")


def main() -> None:
    """Run the full bootstrap sequence."""
    print("=" * 60)
    print("Bootstrapping Quant-Lab")
    print("=" * 60)

    create_directories()
    create_packages()
    create_root_files()

    print()
    print("Quant-Lab initialized successfully.")


if __name__ == "__main__":
    main()
