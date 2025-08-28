# Scripts Directory

This directory contains utility scripts for the collectivecrossing project.

## Mypy Type Checking Scripts

### Python Script (`run_mypy.py`)
Run mypy type checking using Python:
```bash
python scripts/run_mypy.py
```

### Shell Script (`run_mypy.sh`) - Unix/Linux/macOS
Run mypy type checking using bash:
```bash
./scripts/run_mypy.sh
```

### Batch Script (`run_mypy.bat`) - Windows
Run mypy type checking using Windows batch:
```cmd
scripts\run_mypy.bat
```

### Using uv (Recommended)
If you're using uv as your package manager, you can also run:
```bash
uv run mypy src/ tests/
```

## What These Scripts Do

1. **Check Installation**: Verify that mypy is installed
2. **Configure Paths**: Automatically detect project root and source directories
3. **Run Type Checking**: Execute mypy with the project's configuration
4. **Provide Feedback**: Show clear success/failure messages with colored output

## Configuration

The scripts use the mypy configuration from `pyproject.toml` in the project root. This includes:
- Type checking rules
- Module overrides for external libraries
- Ignore patterns for missing imports

## Error Handling

- If mypy is not installed, the scripts will show installation instructions
- If type checking fails, the scripts will show the specific errors
- All scripts return appropriate exit codes for CI/CD integration
