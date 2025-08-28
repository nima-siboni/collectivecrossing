# Installation Guide

## Prerequisites

- **Python 3.10+** 🐍
- **[uv](https://docs.astral.sh/uv/) package manager** ⚡

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd collectivecrossing
```

### 2. Install Dependencies

```bash
# Install main dependencies
uv sync

# Install development dependencies
uv sync --dev
```

### 3. Set Up Development Environment

```bash
# Set up pre-commit hooks
uv run pre-commit install
```

### 4. Verify Installation

```bash
# Run tests to verify everything works
uv run pytest

# Check code quality
uv run ruff check . --config tool-config.toml
```

## Alternative Installation Methods

### Using pip (not recommended)

```bash
pip install -e .
```

### Using conda

```bash
conda create -n collectivecrossing python=3.10
conda activate collectivecrossing
uv sync
```

## Troubleshooting

### Common Issues

1. **uv not found**: Install uv from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
2. **Python version**: Ensure you have Python 3.10 or higher
3. **Permission errors**: Use `uv sync --user` or check your Python environment

### Getting Help

If you encounter issues during installation, please:
1. Check the [GitHub Issues](https://github.com/nima-siboni/collectivecrossing/issues)
2. Create a new issue with your error details
3. Include your Python version and operating system
