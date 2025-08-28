#!/usr/bin/env python3
"""Mypy type checking script for the collectivecrossing project."""

import subprocess
import sys
from pathlib import Path


def run_mypy() -> int:
    """
    Run mypy type checking on the project.

    Returns:
        Exit code (0 for success, non-zero for errors)

    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    print("🔍 Running mypy type checking...")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Source directory: {src_dir}")
    print("-" * 50)

    # Build the mypy command (matching pre-commit configuration)
    # Note: pre-commit uses mypy v1.7.1 with additional dependencies
    cmd = f"mypy --config-file {project_root / 'pyproject.toml'} {src_dir} {project_root / 'tests'}"

    print(f"🚀 Command: {cmd}")
    print()

    try:
        # Run mypy
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=project_root,
        )

        print("-" * 50)
        if result.returncode == 0:
            print("✅ Mypy type checking passed!")
            return 0
        else:
            print(f"❌ Mypy type checking failed with exit code {result.returncode}")
            return result.returncode

    except FileNotFoundError:
        print("❌ Error: mypy not found. Please install it with:")
        print("   uv add --dev mypy")
        return 1
    except Exception as e:
        print(f"❌ Error running mypy: {e}")
        return 1


def main() -> None:
    """Run mypy type checking on the project."""
    exit_code = run_mypy()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
